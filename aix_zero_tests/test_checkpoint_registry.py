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
    DEFAULT_INVENTORY_PATH,
    load_checkpoint_inventory,
    load_checkpoint_strict,
    validate_checkpoint_inventory,
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


if __name__ == "__main__":
    unittest.main()
