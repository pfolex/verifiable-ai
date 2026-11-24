"""Unit tests for Physical LoRA Capacity and Function-Preserving Migration (`[AIX NEURAL-02]` / Issue #429)."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.physical_lora import (
    LoRALinear,
    attach_physical_lora_to_model,
    expand_model_lora_rank,
    expand_moe_experts_function_preserving,
    load_physical_lora_checkpoint,
    measure_physical_lora_Accounting,
    save_physical_lora_checkpoint,
    unmask_expanded_moe_experts,
)
from aix_zero.tokenizer import build_v11_tokenizer


class PhysicalLoRATests(unittest.TestCase):
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
        self.sample_ids = torch.tensor([[1, 12, 18, 22, 31, 44, 52, 2]], dtype=torch.long)
        self.sample_labels = torch.tensor([[0, 0, 0, 22, 31, 44, 52, 2]], dtype=torch.long)

    def test_attach_and_trained_rank_expansion_preserve_exact_outputs_and_migrate_optimizer(self) -> None:
        with torch.no_grad():
            base_logits = self.model(self.sample_ids)["logits"].detach().clone()

        # 1. Attaching physical LoRA (B=0) preserves forward output identically
        acc_r4 = attach_physical_lora_to_model(self.model, rank=4, alpha=8.0)
        self.assertGreater(acc_r4["physical_lora_parameters"], 0)
        self.assertEqual(acc_r4["trainable_parameters"], acc_r4["physical_lora_parameters"])
        with torch.no_grad():
            lora_init_logits = self.model(self.sample_ids)["logits"]
        self.assertTrue(torch.allclose(base_logits, lora_init_logits, atol=1e-6))

        # 2. Train r=4 for 2 steps so lora_B != 0 and optimizer state is populated
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=5e-3)
        self.model.train()
        for _ in range(2):
            opt.zero_grad()
            loss = self.model(self.sample_ids, labels=self.sample_labels)["loss"]
            loss.backward()
            opt.step()

        self.model.eval()
        with torch.no_grad():
            trained_r4_logits = self.model(self.sample_ids)["logits"].detach().clone()
        # Confirm trained output moved away from base output
        self.assertFalse(torch.allclose(base_logits, trained_r4_logits, atol=1e-4))

        # 3. Expand rank r=4 -> r=8 even with a DIFFERENT alpha/r scaling ratio (e.g. new_alpha=8.0 so scale changes 2.0 -> 1.0)
        exp_r8 = expand_model_lora_rank(self.model, new_rank=8, new_alpha=8.0, optimizer=opt)
        self.assertEqual(exp_r8["lora_ranks_present"], [8])
        self.assertGreater(exp_r8["optimizer_state_memory_bytes"], 0)

        with torch.no_grad():
            expanded_r8_logits = self.model(self.sample_ids)["logits"]
        # Function preservation despite changed alpha/r scale!
        self.assertTrue(torch.allclose(trained_r4_logits, expanded_r8_logits, atol=1e-5))

        # 4. Verify two-step gradient dynamics on added capacity (rows/cols 4..8):
        # Step 1 after expansion: B[:, 4:8] is zero, so grad(A[4:8, :]) == 0 while grad(B[:, 4:8]) != 0
        first_lora: LoRALinear = next(m for m in self.model.modules() if isinstance(m, LoRALinear))
        self.model.train()
        opt.zero_grad()
        loss_step1 = self.model(self.sample_ids, labels=self.sample_labels)["loss"]
        loss_step1.backward()
        self.assertIsNotNone(first_lora.lora_B.grad)
        self.assertIsNotNone(first_lora.lora_A.grad)
        self.assertGreater(float(first_lora.lora_B.grad[:, 4:].abs().sum().item()), 0.0)
        self.assertEqual(float(first_lora.lora_A.grad[4:, :].abs().sum().item()), 0.0)
        opt.step()

        # Step 2 after expansion: now B[:, 4:8] != 0, so grad(A[4:8, :]) becomes non-zero!
        opt.zero_grad()
        loss_step2 = self.model(self.sample_ids, labels=self.sample_labels)["loss"]
        loss_step2.backward()
        self.assertGreater(float(first_lora.lora_A.grad[4:, :].abs().sum().item()), 0.0)
        opt.step()

    def test_expert_count_growth_preserves_router_probabilities_at_boundary(self) -> None:
        self.model.eval()
        with torch.no_grad():
            pre_out = self.model(self.sample_ids)["logits"].detach().clone()

        rep = expand_moe_experts_function_preserving(self.model, new_n_experts=8)
        self.assertEqual(rep["old_n_experts"], 4)
        self.assertEqual(rep["new_n_experts"], 8)
        self.assertTrue(rep["new_slots_masked_at_boundary"])

        with torch.no_grad():
            post_out = self.model(self.sample_ids)["logits"]
        self.assertTrue(torch.allclose(pre_out, post_out, atol=1e-6))

        # Unmasking allows new experts to participate
        unmask_expanded_moe_experts(self.model, initial_logit_offset=0.0)
        with torch.no_grad():
            unmasked_out = self.model(self.sample_ids)["logits"]
        self.assertEqual(unmasked_out.shape, pre_out.shape)

    def test_versioned_checkpoint_roundtrip_and_caps_fail_closed(self) -> None:
        attach_physical_lora_to_model(self.model, rank=4, alpha=8.0)
        expand_moe_experts_function_preserving(self.model, new_n_experts=6)
        self.model.eval()
        with torch.no_grad():
            expected_logits = self.model(self.sample_ids)["logits"].detach().clone()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "physical_lora_v2.pt"
            meta = save_physical_lora_checkpoint(self.model, ckpt_path)
            self.assertEqual(meta["schema_version"], "2.0.0-physical-lora")

            reloaded = load_physical_lora_checkpoint(ckpt_path)
            reloaded.eval()
            with torch.no_grad():
                reloaded_logits = reloaded(self.sample_ids)["logits"]
            self.assertTrue(torch.allclose(expected_logits, reloaded_logits, atol=1e-6))

            # Tamper with a tensor shape and verify load fails closed
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            first_key = next(k for k in payload["state_dict"] if "lora_A" in k)
            payload["state_dict"][first_key] = torch.zeros((99, 128))
            bad_path = Path(tmp) / "corrupt_shape.pt"
            torch.save(payload, bad_path)
            with self.assertRaises(ValueError):
                load_physical_lora_checkpoint(bad_path)

        # Verify rank cap and parameter cap fail closed
        with self.assertRaises(ValueError):
            expand_model_lora_rank(self.model, new_rank=1024, max_lora_rank=64)
        with self.assertRaises(ValueError):
            expand_model_lora_rank(self.model, new_rank=16, max_total_parameters=100)


if __name__ == "__main__":
    unittest.main()
