"""Unit tests for checkpoint provenance inventory (`BASE-05`), side-effect-free loading (`BASE-06`),
hardware numerical tolerances (`BASE-11`), and clean-checkout smoke round-trip (`BASE-12`).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.checkpoint_registry import (
    CLI_COMMAND_CLASSIFICATION,
    DEFAULT_INVENTORY_PATH,
    build_experiment_failure_record,
    decompose_fixed_route_vs_learned_weights,
    load_checkpoint_inventory,
    load_checkpoint_strict,
    make_experiment_config,
    make_network_job_contract,
    plan_artifact_retention_cleanup,
    resume_training_from_bundle,
    save_checkpoint_protected,
    save_experiment_failure_record,
    save_training_recovery_bundle,
    select_checkpoint_without_test_leakage,
    step_with_nonfinite_guard_and_recovery,
    validate_checkpoint_inventory,
    validate_experiment_config,
    validate_network_job_contract,
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

    def test_base_09_interrupted_experiment_recovery_and_base_10_failure_records(self):
        from aix_zero.architecture import AIXZeroConfig, AIXZeroModel

        cfg = make_experiment_config(
            {
                "schema_version": "1.0.0",
                "experiment_id": "exp-recovery-test",
                "hypothesis": "Deterministic recovery matches uninterrupted control.",
                "seed": 42,
                "model_identity": {
                    "base_checkpoint_path": "aix_zero/checkpoints/aix_zero_v11_kip.safetensors",
                    "base_checkpoint_sha256": "a" * 64,
                    "tokenizer_vocab_size": 32,
                    "tokenizer_vocab_sha256": "b" * 64,
                    "trainable_scope": "full_model",
                    "trainable_parameters": 1024,
                },
                "data_identity": {
                    "dataset_id": "toy-dataset",
                    "dataset_sha256": "c" * 64,
                    "train_split": "train",
                    "validation_split": "validation",
                    "test_split": "test",
                },
                "optimizer": {
                    "name": "AdamW",
                    "lr": 0.01,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "batch_size": 2,
                    "max_steps": 2,
                },
                "hardware_budget": {
                    "device": "cpu",
                    "dtype": "float32",
                    "threads": 1,
                    "max_wall_clock_seconds": 60,
                },
                "selection_and_stop_rule": {
                    "checkpoint_selection_metric": "validation_loss",
                    "controls": ["uninterrupted_control"],
                    "promotion_gate": "paired_hoeffding_union_bound",
                },
            }
        )
        m_cfg = AIXZeroConfig(vocab_size=32, d_model=16, n_heads=2, n_kv_heads=2, d_ff=32, max_seq_len=16, use_moe=True, n_experts=2, moe_top_k=1)
        batches = [
            torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long),
            torch.tensor([[3, 4, 5, 6], [4, 5, 6, 7]], dtype=torch.long),
        ]

        # Control: 2 uninterrupted steps
        torch.manual_seed(99)
        ctrl_model = AIXZeroModel(m_cfg)
        ctrl_opt = torch.optim.AdamW(ctrl_model.parameters(), lr=0.01, weight_decay=0.0)
        for step_idx, b in enumerate(batches):
            ctrl_opt.zero_grad(set_to_none=True)
            loss = ctrl_model(b, labels=b)["loss"]
            loss.backward()
            ctrl_opt.step()

        # Interrupted after step 1, saved, then resumed for step 2
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(99)
            rec_model = AIXZeroModel(m_cfg)
            rec_opt = torch.optim.AdamW(rec_model.parameters(), lr=0.01, weight_decay=0.0)
            rec_opt.zero_grad(set_to_none=True)
            rec_model(batches[0], labels=batches[0])["loss"].backward()
            rec_opt.step()

            bundle_dir = Path(tmp) / "bundle"
            manifest = save_training_recovery_bundle(
                bundle_dir=bundle_dir,
                experiment_config=cfg,
                step=1,
                data_cursor=1,
                model=rec_model,
                optimizer=rec_opt,
            )
            self.assertFalse(manifest["is_validated_final_checkpoint"])

            # Fresh model & optimizer resumed from bundle
            resumed_model = AIXZeroModel(m_cfg)
            resumed_opt = torch.optim.AdamW(resumed_model.parameters(), lr=0.01, weight_decay=0.0)
            res_info = resume_training_from_bundle(
                bundle_dir=bundle_dir,
                expected_experiment_config=cfg,
                model=resumed_model,
                optimizer=resumed_opt,
            )
            self.assertEqual(res_info["step"], 1)
            resumed_opt.zero_grad(set_to_none=True)
            resumed_model(batches[1], labels=batches[1])["loss"].backward()
            resumed_opt.step()

            for (name, p_ctrl), (_, p_res) in zip(ctrl_model.named_parameters(), resumed_model.named_parameters()):
                torch.testing.assert_close(p_res, p_ctrl, rtol=1e-5, atol=1e-6, msg=name)

            # Mismatched dataset identity refuses resume without overwriting
            bad_cfg = copy.deepcopy(cfg)
            bad_cfg["data_identity"]["dataset_sha256"] = "d" * 64
            bad_cfg = make_experiment_config({k: v for k, v in bad_cfg.items() if k != "config_sha256"})
            with self.assertRaisesRegex(ValueError, "config_sha256 mismatch"):
                resume_training_from_bundle(
                    bundle_dir=bundle_dir,
                    expected_experiment_config=bad_cfg,
                    model=resumed_model,
                    optimizer=resumed_opt,
                )

            # BASE-10 structured failure records
            fail_rec = build_experiment_failure_record(
                experiment_id="exp-recovery-test",
                config_sha256=cfg["config_sha256"],
                status="FAILED_NONFINITE",
                step_reached=1,
                reason="NaN loss encountered during step 2",
            )
            self.assertFalse(fail_rec["is_validated_checkpoint"])
            self.assertFalse(fail_rec["eligible_for_promotion"])
            fail_path = Path(tmp) / "failure.json"
            save_experiment_failure_record(fail_rec, fail_path)
            self.assertTrue(fail_path.is_file())

    def test_base_16_audit_and_network_01_job_contracts_and_expert_02_03_hardening(self):
        audit_doc = ROOT / "docs/M0-FOUNDATION-READINESS-AUDIT.md"
        self.assertTrue(audit_doc.is_file())
        self.assertIn("M0_FOUNDATION_READY", audit_doc.read_text(encoding="utf-8"))

        # NETWORK-01 evaluation and specialist training job contracts
        eval_job = make_network_job_contract(
            {
                "schema_version": "1.0.0",
                "job_id": "job-eval-v11-001",
                "job_type": "OFFLINE_EVALUATION_JOB",
                "experiment_config_sha256": "a" * 64,
                "inputs": {
                    "base_checkpoint_path": "aix_zero/checkpoints/aix_zero_v11_kip.safetensors",
                    "base_checkpoint_sha256": "b" * 64,
                    "dataset_id": "v11-kip-test",
                    "dataset_sha256": "c" * 64,
                    "seed": 42,
                    "target_expert_slot": None,
                },
                "resource_budget": {
                    "max_steps": 0,
                    "max_wall_clock_seconds": 120,
                    "max_memory_mb": 512,
                    "device": "cpu",
                },
                "lease": {
                    "lease_id": "lease-001",
                    "worker_id": "worker-local-01",
                    "issued_at_utc": "2026-11-10T22:00:00Z",
                    "expires_at_utc": "2026-11-10T22:30:00Z",
                    "heartbeat_interval_seconds": 15,
                    "attempt_number": 1,
                },
                "expected_outputs": {
                    "requires_evaluation_result_json": True,
                    "requires_candidate_safetensors": False,
                    "requires_failure_record_on_abort": True,
                },
            }
        )
        validate_network_job_contract(eval_job)

        bad_job = copy.deepcopy(eval_job)
        bad_job["job_type"] = "UNKNOWN_SYNCHRONIZED_ALLREDUCE_JOB"
        with self.assertRaises(Exception):
            validate_network_job_contract(bad_job)

        # EXPERT-02 & EXPERT-03 hardening tests
        ckpt = ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        model, _, info = load_checkpoint_strict(ckpt)
        with tempfile.TemporaryDirectory() as tmp:
            bdir = Path(tmp) / "bank"
            bank = SparseExpertCartridgeBank(bdir)
            bank.export_all_domain_cartridges(
                model, {"DOM_A": {}}, {}, domain_expert_map={"DOM_A": 0}
            )
            # Duplicate JSON key in manifest is rejected (EXPERT-03)
            raw_txt = bank.manifest_path.read_text(encoding="utf-8")
            dup_txt = raw_txt.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,')
            bank.manifest_path.write_text(dup_txt, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                bank.get_manifest()
            bank.manifest_path.write_text(raw_txt, encoding="utf-8")

            # Trained specialist for another base checkpoint SHA-256 is rejected (EXPERT-02)
            manifest = bank.get_manifest()
            manifest["cartridges"]["DOM_A"]["provenance"] = "frozen_v11_single_expert_training_candidate"
            manifest["cartridges"]["DOM_A"]["promotion_status"] = "APPROVED_ACTIVE"
            manifest["cartridges"]["DOM_A"]["training"] = {"frozen_tensors_unchanged": True}
            manifest["cartridges"]["DOM_A"]["empirical_gate"] = {"decision": "PASS"}
            manifest["cartridges"]["DOM_A"]["base_checkpoint_sha256"] = "f" * 64
            bank.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Trained specialist base_checkpoint_sha256 mismatch"):
                bank.install_cartridge(
                    model, "DOM_A", slot=0, active_base_checkpoint_sha256=info["sha256"]
                )

    def test_wave5_base_15_train_02_10_12_and_expert_04_13(self):
        from aix_zero.architecture import AIXZeroConfig, AIXZeroModel

        # TRAIN-02 fixed-route vs learned-weight decomposition
        pilot_json = json.loads((ROOT / "research/bayesian-pilot/results.json").read_text(encoding="utf-8"))
        decomp = decompose_fixed_route_vs_learned_weights(pilot_json)
        self.assertIn("BP", decomp["tasks"])
        self.assertAlmostEqual(
            decomp["tasks"]["BP"]["routing_intervention_effect_delta"]
            + decomp["tasks"]["BP"]["learned_weight_effect_delta"],
            decomp["tasks"]["BP"]["net_delta_vs_unchanged_base"],
            places=5,
        )

        # TRAIN-12 checkpoint selection without test leakage
        sel = select_checkpoint_without_test_leakage(
            initial_validation_loss=8.5,
            validation_logs=[
                {"step": 60, "validation_loss": 2.1},
                {"step": 120, "validation_loss": 0.9},
                {"step": 180, "validation_loss": 1.1},
            ],
        )
        self.assertEqual(sel["selected_step"], 120)
        self.assertTrue(sel["included_step_zero"])
        with self.assertRaisesRegex(PermissionError, "Final-test metrics are prohibited"):
            select_checkpoint_without_test_leakage(
                initial_validation_loss=8.5,
                validation_logs=[{"step": 60, "validation_loss": 2.1, "test_accuracy": 0.5}],
            )

        # BASE-15 retention & TRAIN-10 nonfinite fault recovery
        cfg = make_experiment_config(
            {
                "schema_version": "1.0.0",
                "experiment_id": "exp-wave5-fault",
                "hypothesis": "Fault injection preserves last verified recovery bundle.",
                "seed": 42,
                "model_identity": {
                    "base_checkpoint_path": "aix_zero/checkpoints/aix_zero_v11_kip.safetensors",
                    "base_checkpoint_sha256": "a" * 64,
                    "tokenizer_vocab_size": 32,
                    "tokenizer_vocab_sha256": "b" * 64,
                    "trainable_scope": "full_model",
                    "trainable_parameters": 1024,
                },
                "data_identity": {
                    "dataset_id": "toy-dataset",
                    "dataset_sha256": "c" * 64,
                    "train_split": "train",
                    "validation_split": "validation",
                    "test_split": "test",
                },
                "optimizer": {
                    "name": "AdamW",
                    "lr": 0.01,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "batch_size": 2,
                    "max_steps": 2,
                },
                "hardware_budget": {
                    "device": "cpu",
                    "dtype": "float32",
                    "threads": 1,
                    "max_wall_clock_seconds": 60,
                },
                "selection_and_stop_rule": {
                    "checkpoint_selection_metric": "validation_loss",
                    "controls": ["uninterrupted_control"],
                    "promotion_gate": "paired_hoeffding_union_bound",
                },
            }
        )
        m_cfg = AIXZeroConfig(vocab_size=32, d_model=16, n_heads=2, n_kv_heads=2, d_ff=32, max_seq_len=16, use_moe=True, n_experts=2, moe_top_k=1)
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            model = AIXZeroModel(m_cfg)
            opt = torch.optim.AdamW(model.parameters(), lr=0.01)
            bundle_dir = tdir / "bundle"
            save_training_recovery_bundle(
                bundle_dir=bundle_dir,
                experiment_config=cfg,
                step=1,
                data_cursor=1,
                model=model,
                optimizer=opt,
            )
            saved_w = {k: v.detach().clone() for k, v in model.state_dict().items()}

            # Inject NaN loss at step 2 -> rolls back to step 1 and writes failure record
            nan_loss = torch.tensor(float("nan"), requires_grad=True)
            fail_path = tdir / "nonfinite_fail.json"
            step_res = step_with_nonfinite_guard_and_recovery(
                model=model,
                optimizer=opt,
                loss=nan_loss,
                step=2,
                recovery_bundle_dir=bundle_dir,
                experiment_config=cfg,
                failure_record_path=fail_path,
            )
            self.assertFalse(step_res["uninterrupted_success"])
            self.assertTrue(step_res["recovered_from_nonfinite_fault"])
            self.assertEqual(step_res["rolled_back_to_step"], 1)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, saved_w[k])

            # BASE-15 dry-run and cleanup
            scratch_file = tdir / "scratch_temp.tmp"
            scratch_file.write_bytes(b"reclaim-me")
            plan = plan_artifact_retention_cleanup(tdir, dry_run=True)
            self.assertIn(str(scratch_file), plan["reclaimable_files"])
            self.assertIn(str(fail_path), plan["retained_failure_records"])
            self.assertTrue(scratch_file.exists())

            exec_plan = plan_artifact_retention_cleanup(tdir, dry_run=False)
            self.assertIn(str(scratch_file), exec_plan["deleted_files"])
            self.assertFalse(scratch_file.exists())
            self.assertTrue(fail_path.exists())

            # EXPERT-04 temporary specialist session & EXPERT-13 crash-safe catalog publication
            bank = SparseExpertCartridgeBank(tdir / "expert_bank")
            manifest = bank.export_all_domain_cartridges(
                model, {"DOM_0": {}}, {}, domain_expert_map={"DOM_0": 0}
            )
            orig_router_w = model.latent_reasoning_core.core_block.moe.router.weight.detach().clone()
            with self.assertRaisesRegex(RuntimeError, "simulated error inside session"):
                with bank.temporary_specialist_session(model, "DOM_0", slot=0, force_explicit_route=True):
                    with self.assertRaisesRegex(RuntimeError, "Concurrent temporary_specialist_session"):
                        with bank.temporary_specialist_session(model, "DOM_0", slot=0):
                            pass
                    raise RuntimeError("simulated error inside session")
            # Router weight and session state are restored after exception
            torch.testing.assert_close(
                model.latent_reasoning_core.core_block.moe.router.weight, orig_router_w
            )

            # Simulated interruption during publish leaves previous catalog intact
            with self.assertRaises(InterruptedError):
                bank.publish_verified_catalog_atomic(
                    manifest, simulate_interruption_before_manifest_swap=True
                )
            self.assertEqual(bank.get_manifest()["registered_cartridges"], 1)
            pub_res = bank.publish_verified_catalog_atomic(manifest)
            self.assertTrue(pub_res["published"])

            # EXPERT-05 deterministic explicit task router & EXPERT-14 safe shard cleanup
            r_unpromoted = bank.route_explicit_task_schema("Q:DOM_0:x=1")
            self.assertFalse(r_unpromoted["specialist_activated"])
            self.assertEqual(r_unpromoted["route_target"], "BASE_MODEL_DEFAULT")
            self.assertFalse(bank.route_explicit_task_schema("malformed_prompt")["specialist_activated"])

            orphan_shard = bank.cartridge_dir / "expert_orphan_0000.safetensors"
            orphan_shard.write_bytes(b"orphan")
            gc_dry = bank.cleanup_unreferenced_shards(dry_run=True)
            self.assertIn("expert_orphan_0000.safetensors", gc_dry["reclaimable_shards"])
            self.assertEqual(len(gc_dry["protected_shards"]), 1)
            gc_exec = bank.cleanup_unreferenced_shards(dry_run=False)
            self.assertIn("expert_orphan_0000.safetensors", gc_exec["deleted_shards"])
            self.assertFalse(orphan_shard.exists())


if __name__ == "__main__":
    unittest.main()
