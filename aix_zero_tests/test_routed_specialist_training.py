"""Unit tests for Routed Domain Cartridge Evaluation and Isolated Specialist Training (`[AIX NEURAL-01]` / Issue #428)."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.meta_ai_autopilot import META_AI_EXPERT_DOMAINS, build_meta_ai_self_improvement_curriculum
from aix_zero.routed_specialist_training import (
    audit_model_and_cartridge_tensor_ownership,
    compute_model_state_sha256,
    evaluate_routed_cartridge_arm,
    evaluate_routed_cartridge_paired_comparison,
    train_isolated_domain_cartridge,
)
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
from aix_zero.tokenizer import build_v11_tokenizer


class RoutedSpecialistTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20261124)
        self.tokenizer = build_v11_tokenizer()
        self.model = AIXZeroModel(
            AIXZeroConfig(
                vocab_size=self.tokenizer.vocab_size,
                d_model=128,
                n_heads=4,
                n_kv_heads=2,
                d_ff=256,
                max_seq_len=112,
                use_moe=True,
                n_experts=4,
                moe_top_k=2,
                use_memory_bank=True,
                use_prompt_pointer=True,
                use_key_ngram_conv=True,
            )
        )
        self.model.eval()

    def test_tensor_ownership_audit_distinguishes_aliases_from_trained_specialists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank = SparseExpertCartridgeBank(Path(tmp) / "bank")
            domain_map = {dom: int(meta["expert_slot"]) for dom, meta in META_AI_EXPERT_DOMAINS.items()}
            bank.export_all_domain_cartridges(
                self.model,
                META_AI_EXPERT_DOMAINS,
                {dom: None for dom in META_AI_EXPERT_DOMAINS},
                domain_expert_map=domain_map,
            )
            audit = audit_model_and_cartridge_tensor_ownership(self.model, bank)
            self.assertEqual(audit["n_experts"], 4)
            self.assertEqual(audit["per_slot_expert_parameters"], 49152)
            self.assertEqual(audit["per_slot_expert_bytes"], 49152 * 4)
            self.assertEqual(audit["registered_cartridge_count"], 30)
            # Exporting 4 slots across 30 domains produces at most 4 unique SHA-256 digests (aliases)
            self.assertLessEqual(audit["unique_cartridge_sha256_count"], 4)
            self.assertEqual(audit["trained_specialist_domain_count"], 0)
            self.assertEqual(audit["snapshot_alias_domain_count"], 30)
            self.assertFalse(audit["zero_cross_domain_interference_guaranteed_by_disk_file_alone"])

    def test_two_independently_trained_domain_cartridges_and_paired_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank = SparseExpertCartridgeBank(Path(tmp) / "bank")
            domain_map = {dom: int(meta["expert_slot"]) for dom, meta in META_AI_EXPERT_DOMAINS.items()}
            bank.export_all_domain_cartridges(
                self.model,
                META_AI_EXPERT_DOMAINS,
                {dom: None for dom in META_AI_EXPERT_DOMAINS},
                domain_expert_map=domain_map,
            )
            base_sha_before = compute_model_state_sha256(self.model)

            curr = build_meta_ai_self_improvement_curriculum(cycle_index=1)
            quant_items = [it for it in curr if it["domain"] == "META_QUANTIZATION_FP8_INT4"]
            flash_items = [it for it in curr if it["domain"] == "META_FLASH_ATTENTION_TILING"]
            rloo_items = [it for it in curr if it["domain"] == "META_TRL_RLOO_LEAVE_ONE_OUT"]

            # Notice both META_QUANTIZATION_FP8_INT4 (slot 2) and META_FLASH_ATTENTION_TILING (slot 0)
            # or even two domains sharing the SAME slot index (e.g. slot=2) can be trained independently!
            res_quant = train_isolated_domain_cartridge(
                self.model,
                self.tokenizer,
                bank,
                domain="META_QUANTIZATION_FP8_INT4",
                train_items=quant_items,
                val_items=quant_items,
                domain_meta=META_AI_EXPERT_DOMAINS["META_QUANTIZATION_FP8_INT4"],
                slot=2,
                steps=8,
                lr=4e-3,
                seed=101,
                active_base_checkpoint_sha256=base_sha_before,
            )
            self.assertTrue(res_quant["weights_updated"])
            self.assertTrue(res_quant["frozen_tensors_unchanged"])
            self.assertLess(res_quant["post_train_loss"], res_quant["pre_train_loss"])
            self.assertEqual(res_quant["promotion_status"], "APPROVED_ACTIVE")
            self.assertEqual(compute_model_state_sha256(self.model), base_sha_before)

            # Train a second domain using the SAME physical slot (slot 2) to verify zero optimizer/slot leak
            res_flash = train_isolated_domain_cartridge(
                self.model,
                self.tokenizer,
                bank,
                domain="META_FLASH_ATTENTION_TILING",
                train_items=flash_items,
                val_items=flash_items,
                domain_meta=META_AI_EXPERT_DOMAINS["META_FLASH_ATTENTION_TILING"],
                slot=2,
                steps=8,
                lr=4e-3,
                seed=202,
                active_base_checkpoint_sha256=base_sha_before,
            )
            self.assertTrue(res_flash["weights_updated"])
            self.assertTrue(res_flash["frozen_tensors_unchanged"])
            self.assertNotEqual(res_quant["post_slot_sha256"], res_flash["post_slot_sha256"])
            self.assertEqual(compute_model_state_sha256(self.model), base_sha_before)

            # Verify inactive cartridges remained untouched and manifest tracks 2 trained specialists
            audit_after = audit_model_and_cartridge_tensor_ownership(self.model, bank)
            self.assertEqual(audit_after["trained_specialist_domain_count"], 2)
            self.assertIn("META_QUANTIZATION_FP8_INT4", audit_after["trained_specialist_domains"])
            self.assertIn("META_FLASH_ATTENTION_TILING", audit_after["trained_specialist_domains"])

            # Verify paired comparison (unchanged_baseline vs predicted_prefix vs oracle vs wrong_cartridge)
            paired = evaluate_routed_cartridge_paired_comparison(
                self.model,
                self.tokenizer,
                bank,
                domain_items=quant_items[:2] + flash_items[:2],
                retention_items=rloo_items[:2],
                wrong_cartridge_domain="META_FLASH_ATTENTION_TILING",
                active_base_checkpoint_sha256=base_sha_before,
                max_new_tokens=16,
            )
            self.assertTrue(paired["base_model_restored_unchanged"])
            pred_arm = paired["arms"]["predicted_prefix_routed"]
            self.assertEqual(pred_arm["evaluation_mode"], "neural_only")
            self.assertFalse(pred_arm["routing_uses_gold_answers"])
            self.assertEqual(pred_arm["specialist_activations"], 4)
            # Second example of each domain should hit the LRU hot cache
            self.assertGreaterEqual(pred_arm["cache_hits"], 2)
            for rec in pred_arm["records"]:
                self.assertTrue(rec["route_decision"].startswith("SPECIALIST:"))
                self.assertIsNotNone(rec["cartridge_sha256"])
                self.assertTrue(rec["checkpoint_compatible"])
                self.assertTrue(rec["architecture_compatible"])
                self.assertEqual(rec["trainable_specialist_parameters"], 49152)

    def test_cancellation_and_corrupt_or_incompatible_cartridge_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bank = SparseExpertCartridgeBank(Path(tmp) / "bank")
            domain_map = {dom: int(meta["expert_slot"]) for dom, meta in META_AI_EXPERT_DOMAINS.items()}
            bank.export_all_domain_cartridges(
                self.model,
                META_AI_EXPERT_DOMAINS,
                {dom: None for dom in META_AI_EXPERT_DOMAINS},
                domain_expert_map=domain_map,
            )
            base_sha = compute_model_state_sha256(self.model)
            curr = build_meta_ai_self_improvement_curriculum(cycle_index=1)
            quant_items = [it for it in curr if it["domain"] == "META_QUANTIZATION_FP8_INT4"]

            # 1. Cancellation during isolated training restores base model cleanly
            with self.assertRaises(InterruptedError):
                train_isolated_domain_cartridge(
                    self.model,
                    self.tokenizer,
                    bank,
                    domain="META_QUANTIZATION_FP8_INT4",
                    train_items=quant_items,
                    slot=2,
                    steps=4,
                    cancel_check=lambda: True,
                )
            self.assertEqual(compute_model_state_sha256(self.model), base_sha)

            # 2. Train valid specialist, then test incompatible base_checkpoint_sha256 fallback
            train_isolated_domain_cartridge(
                self.model,
                self.tokenizer,
                bank,
                domain="META_QUANTIZATION_FP8_INT4",
                train_items=quant_items,
                slot=2,
                steps=4,
                active_base_checkpoint_sha256=base_sha,
            )
            incomp_eval = evaluate_routed_cartridge_arm(
                self.model,
                self.tokenizer,
                bank,
                items=quant_items[:1],
                routing_mode="predicted_prefix",
                active_base_checkpoint_sha256="0" * 64,
                max_new_tokens=12,
            )
            self.assertEqual(incomp_eval["fallback_events"], 1)
            self.assertEqual(incomp_eval["records"][0]["route_decision"], "BASE_MODEL_FALLBACK")
            self.assertFalse(incomp_eval["records"][0]["checkpoint_compatible"])
            self.assertEqual(compute_model_state_sha256(self.model), base_sha)


if __name__ == "__main__":
    unittest.main()
