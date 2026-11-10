"""Unit tests for checkpoint provenance inventory (`BASE-05`), side-effect-free loading (`BASE-06`),
hardware numerical tolerances (`BASE-11`), and clean-checkout smoke round-trip (`BASE-12`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.checkpoint_registry import (
    CLI_COMMAND_CLASSIFICATION,
    DEFAULT_INVENTORY_PATH,
    load_checkpoint_inventory,
    load_checkpoint_strict,
    make_experiment_config,
    save_checkpoint_protected,
    validate_checkpoint_inventory,
    validate_experiment_config,
)
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
from aix_zero.tokenizer import AIXZeroTokenizer

ROOT = Path(__file__).resolve().parents[1]


class CheckpointRegistryAndSmokeTests(unittest.TestCase):
    def test_checkpoint_inventory_matches_all_committed_safetensors(self):
        inv = load_checkpoint_inventory(DEFAULT_INVENTORY_PATH, verify_files_on_disk=True)
        self.assertEqual(len(inv["checkpoints"]), 12)
        indexed_paths = {c["path"] for c in inv["checkpoints"]}
        actual_paths = {
            str(p.relative_to(ROOT))
            for p in list((ROOT / "aix_zero/checkpoints").glob("*.safetensors"))
            + list((ROOT / "research/bayesian-pilot/candidates").glob("*/*.safetensors"))
        }
        self.assertEqual(indexed_paths, actual_paths)

        # Verify teacher-assisted checkpoints explicitly disclose provenance gaps
        v11_entry = next(c for c in inv["checkpoints"] if c["checkpoint_id"] == "aix-zero-v11-kip")
        self.assertEqual(v11_entry["promotion_status"], "ACTIVE_BASELINE")
        self.assertEqual(v11_entry["provenance_status"], "teacher_assisted_or_external_web_session")
        self.assertTrue(len(v11_entry["known_provenance_gaps"]) >= 1)

    def test_load_checkpoint_strict_has_no_filesystem_side_effects(self):
        ckpt = ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        before_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        model, tok, info = load_checkpoint_strict(ckpt, expected_sha256=before_sha)
        self.assertFalse(model.training)
        self.assertEqual(info["physical_parameters"], 817410)
        self.assertEqual(info["vocab_size"], 134)
        self.assertEqual(tok.vocab_size, 134)
        self.assertEqual(hashlib.sha256(ckpt.read_bytes()).hexdigest(), before_sha)

        # Wrong expected SHA-256 fails closed
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_checkpoint_strict(ckpt, expected_sha256="0" * 64)

        # Smaller tokenizer fails closed without expanding embeddings
        small_tok = AIXZeroTokenizer(use_word_vocab=False)
        with self.assertRaisesRegex(ValueError, "smaller than checkpoint vocabulary"):
            load_checkpoint_strict(ckpt, tokenizer=small_tok)

    def test_clean_directory_storage_round_trip_and_tolerances(self):
        ckpt = ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        before_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        model, tok, _ = load_checkpoint_strict(ckpt)
        input_ids = torch.tensor([tok.encode("Q:(9+7)%3=")], dtype=torch.long)
        with torch.no_grad():
            logits_before = model(input_ids)["logits"].detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp) / "smoke-cartridges"
            bank = SparseExpertCartridgeBank(tmp_dir)
            names = {f"SOURCE_EXPERT_{i}": i for i in range(model.config.n_experts)}
            manifest = bank.export_all_domain_cartridges(
                model, {k: {} for k in names}, {}, domain_expert_map=names
            )
            self.assertEqual(manifest["registered_cartridges"], 4)
            res = bank.install_cartridge(model, "SOURCE_EXPERT_0", slot=0)
            self.assertTrue(res["installed"])

        with torch.no_grad():
            logits_after = model(input_ids)["logits"]
        # Exact round-trip on same host must preserve logits within float32 tolerance
        self.assertTrue(torch.allclose(logits_before, logits_after, rtol=1e-5, atol=1e-6))
        self.assertEqual(hashlib.sha256(ckpt.read_bytes()).hexdigest(), before_sha)

        # Verify HARDWARE-AND-NUMERICAL-TOLERANCES.md exists
        tol_doc = ROOT / "docs/HARDWARE-AND-NUMERICAL-TOLERANCES.md"
        self.assertTrue(tol_doc.is_file())
        self.assertIn("Never promises bitwise logit equivalence", tol_doc.read_text(encoding="utf-8").replace("never", "Never"))

    def test_cli_command_classification_and_help_fixture(self):
        self.assertGreaterEqual(len(CLI_COMMAND_CLASSIFICATION), 4)
        bench = CLI_COMMAND_CLASSIFICATION["aix_zero.benchmark_baseline"]
        self.assertEqual(bench["tier"], "stable_release_interface")
        self.assertEqual(bench["network_or_paid_calls"], "none")
        kip = CLI_COMMAND_CLASSIFICATION["aix_zero.knowledge_is_power"]
        self.assertEqual(kip["tier"], "experimental_research_command")

    def test_experiment_config_schema_round_trip_and_rejection(self):
        cfg = make_experiment_config(
            {
                "schema_version": "1.0.0",
                "experiment_id": "exp-bayesian-pilot-01",
                "hypothesis": "Single-expert frozen-backbone fine-tuning improves BP/BS exact match without anchor regression.",
                "seed": 42,
                "model_identity": {
                    "base_checkpoint_path": "aix_zero/checkpoints/aix_zero_v11_kip.safetensors",
                    "base_checkpoint_sha256": "a" * 64,
                    "tokenizer_vocab_size": 134,
                    "tokenizer_vocab_sha256": "b" * 64,
                    "trainable_scope": "single_routed_expert",
                    "trainable_parameters": 49152,
                    "target_expert_slot": 3,
                },
                "data_identity": {
                    "dataset_id": "aix-zero-bayesian-pilot-v1",
                    "dataset_sha256": "c" * 64,
                    "train_split": "train",
                    "validation_split": "validation",
                    "test_split": "test",
                },
                "optimizer": {
                    "name": "AdamW",
                    "lr": 0.001,
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "batch_size": 16,
                    "max_steps": 120,
                },
                "hardware_budget": {
                    "device": "cpu",
                    "dtype": "float32",
                    "threads": 1,
                    "max_wall_clock_seconds": 300,
                },
                "selection_and_stop_rule": {
                    "checkpoint_selection_metric": "validation_exact_match",
                    "controls": ["no_test_split_access", "frozen_backbone_sha256_check"],
                    "promotion_gate": "paired_hoeffding_union_bound",
                },
            }
        )
        validate_experiment_config(cfg)

        bad_cfg = dict(cfg)
        bad_cfg["unexpected_field"] = 123
        with self.assertRaises(Exception):
            validate_experiment_config(bad_cfg)

    def test_save_checkpoint_protected_prevents_overwrite_and_supports_nested_dirs(self):
        ckpt = ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        model, _, _ = load_checkpoint_strict(ckpt)

        # Attempting to overwrite the active v11 baseline fails closed
        with self.assertRaisesRegex(PermissionError, "Refusing to overwrite indexed baseline checkpoint"):
            save_checkpoint_protected(model, ckpt)

        with tempfile.TemporaryDirectory() as tmp:
            nested_out = Path(tmp) / "nested" / "sub" / "candidate.safetensors"
            source_ref = Path(tmp) / "source_ref.safetensors"
            source_ref.write_bytes(b"placeholder")

            # Saving to the same path as source_checkpoint_path fails closed
            with self.assertRaisesRegex(PermissionError, "Refusing to overwrite source checkpoint"):
                save_checkpoint_protected(model, source_ref, source_checkpoint_path=source_ref)

            # Saving to a symlink pointing to source_checkpoint_path fails closed
            symlink_path = Path(tmp) / "link.safetensors"
            symlink_path.symlink_to(source_ref)
            with self.assertRaisesRegex(PermissionError, "Refusing to overwrite source checkpoint"):
                save_checkpoint_protected(model, symlink_path, source_checkpoint_path=source_ref)

            # Saving to nested non-existent directory succeeds atomically and returns SHA-256
            res = save_checkpoint_protected(
                model,
                nested_out,
                source_checkpoint_path=ckpt,
            )
            self.assertTrue(nested_out.is_file())
            self.assertEqual(len(res["sha256"]), 64)

    def test_bayesian_pilot_readme_and_specialist_provenance_enforcement(self):
        readme = ROOT / "research/bayesian-pilot/README.md"
        self.assertTrue(readme.is_file())
        text = readme.read_text(encoding="utf-8")
        self.assertIn("REJECTED", text)
        self.assertIn("frozen_tensors_unchanged", text)

        # Verify EXPERT-01 provenance separation in SparseExpertCartridgeBank
        SparseExpertCartridgeBank.validate_specialist_provenance(
            {
                "provenance": "existing_routed_expert_snapshot",
                "promotion_status": "NOT_EVALUATED_SNAPSHOT",
                "verified_accuracy_pct": None,
            }
        )

        with self.assertRaisesRegex(ValueError, "Copied existing_routed_expert_snapshot cannot claim"):
            SparseExpertCartridgeBank.validate_specialist_provenance(
                {
                    "provenance": "existing_routed_expert_snapshot",
                    "promotion_status": "APPROVED_ACTIVE",
                }
            )

        with self.assertRaisesRegex(ValueError, "requires empirical_gate.decision == 'PASS'"):
            SparseExpertCartridgeBank.validate_specialist_provenance(
                {
                    "provenance": "frozen_v11_single_expert_training_candidate",
                    "promotion_status": "APPROVED_ACTIVE",
                    "training": {"frozen_tensors_unchanged": True},
                    "empirical_gate": {"decision": "FAIL"},
                }
            )


if __name__ == "__main__":
    unittest.main()
