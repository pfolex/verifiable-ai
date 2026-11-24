"""Unit tests for Equation-Prefix Supervision and Separately Measured ALU-Trigger Accuracy (`[AIX NEURAL-03]` / Issue #430)."""
from __future__ import annotations

import unittest

import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.equation_prefix_supervision import (
    build_arithmetic_prefix_study_splits,
    classify_equation_setup_vs_reference,
    compute_prefix_and_continuation_losses,
    encode_equation_prefix_supervision_batch,
    evaluate_bounded_equation_prefix,
    evaluate_three_way_prefix_and_alu_outcomes,
    find_first_reasoning_equation_equal_char_offset,
)
from aix_zero.tokenizer import build_v11_tokenizer


class EquationPrefixSupervisionTests(unittest.TestCase):
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

    def test_loss_mask_boundaries_ignore_prompt_equals_and_split_prefix_from_continuation(self) -> None:
        comp = "<think>ratio=16/4=4</think><ans>4</ans>"
        idx = find_first_reasoning_equation_equal_char_offset(comp)
        self.assertEqual(comp[: idx + 1], "<think>ratio=16/4=")
        self.assertEqual(comp[idx + 1 :], "4</think><ans>4</ans>")

        splits = build_arithmetic_prefix_study_splits()
        batch = encode_equation_prefix_supervision_batch(
            splits["train"][:2],
            self.tokenizer,
            prefix_weight=5.0,
            continuation_weight=1.0,
            max_seq_len=112,
        )
        p_ids = self.tokenizer.encode(splits["train"][0]["prompt"], add_bos=True, add_eos=False)
        p_len = len(p_ids)

        # Prompt tokens (which contain two '=' signs!) must have label 0 and zero masks/weights
        self.assertTrue(torch.all(batch["labels"][0, :p_len] == 0))
        self.assertTrue(torch.all(batch["prefix_mask"][0, :p_len] == 0.0))
        self.assertTrue(torch.all(batch["continuation_mask"][0, :p_len] == 0.0))
        self.assertTrue(torch.all(batch["token_weights"][0, :p_len] == 0.0))

        # Prefix tokens receive weight 5.0 and continuation tokens receive weight 1.0
        pref_ids = self.tokenizer.encode("<think>ratio=16/4=", add_bos=False, add_eos=False)
        cont_ids = self.tokenizer.encode("4</think><ans>4</ans>", add_bos=False, add_eos=True)
        self.assertTrue(torch.all(batch["prefix_mask"][0, p_len : p_len + len(pref_ids)] == 1.0))
        self.assertTrue(torch.all(batch["token_weights"][0, p_len : p_len + len(pref_ids)] == 5.0))
        self.assertTrue(
            torch.all(
                batch["continuation_mask"][0, p_len + len(pref_ids) : p_len + len(pref_ids) + len(cont_ids)] == 1.0
            )
        )
        self.assertTrue(
            torch.all(
                batch["token_weights"][0, p_len + len(pref_ids) : p_len + len(pref_ids) + len(cont_ids)] == 1.0
            )
        )

        # Verify separate prefix_loss and continuation_loss logging
        losses = compute_prefix_and_continuation_losses(self.model, batch)
        self.assertGreater(float(losses["prefix_loss"].item()), 0.0)
        self.assertGreater(float(losses["continuation_loss"].item()), 0.0)
        self.assertGreater(float(losses["total_loss"].item()), 0.0)

    def test_diagnostic_categories_and_bounded_alu_safety(self) -> None:
        ref = "<think>ratio=16/4=4</think><ans>4</ans>"
        gold = "4"

        # 1. Correct setup, wrong continuation
        c1 = classify_equation_setup_vs_reference("<think>ratio=16/4=99</think><ans>99</ans>", ref, gold)
        self.assertTrue(c1["exact_prefix_match"])
        self.assertTrue(c1["semantic_setup_correct"])
        self.assertFalse(c1["full_neural_answer_correct"])
        self.assertEqual(c1["diagnostic_category"], "correct_setup_wrong_continuation")

        # 2. Correct arithmetic, wrong operand selection (e.g. distractor 99/3=33 or reversed 4/16=1/4)
        c2 = classify_equation_setup_vs_reference("<think>ratio=4/16=1/4</think><ans>1/4</ans>", ref, gold)
        self.assertFalse(c2["semantic_setup_correct"])
        self.assertEqual(c2["diagnostic_category"], "correct_arithmetic_wrong_operand_selection")

        # 3. Malformed ALU trigger
        c3 = classify_equation_setup_vs_reference("<think>ratio=16^4=65536</think><ans>65536</ans>", ref, gold)
        self.assertEqual(c3["diagnostic_category"], "malformed_alu_trigger")

        # 4. Bounded ALU parser safety checks
        div_zero = evaluate_bounded_equation_prefix("<think>ratio=16/0=")
        self.assertTrue(div_zero["alu_triggered"])
        self.assertFalse(div_zero["accepted"])
        self.assertEqual(div_zero["rejection_reason"], "division_by_zero")

        big_num = evaluate_bounded_equation_prefix("<think>ratio=999999999999999/4=")
        self.assertFalse(big_num["accepted"])
        self.assertEqual(big_num["rejection_reason"], "operand_magnitude_limit_exceeded")

        valid_alu = evaluate_bounded_equation_prefix("<think>ratio=16/4=")
        self.assertTrue(valid_alu["accepted"])
        self.assertEqual(valid_alu["computed_value"], "4")
        self.assertEqual(valid_alu["completed_trace"], "<think>ratio=16/4=4</think><ans>4</ans>")

    def test_prefix_supervision_improves_setup_and_reports_three_outcomes_separately(self) -> None:
        splits = build_arithmetic_prefix_study_splits()
        train_items = splits["train"]

        pre_report = evaluate_three_way_prefix_and_alu_outcomes(
            self.model, self.tokenizer, train_items, max_new_tokens=28
        )
        batch = encode_equation_prefix_supervision_batch(
            train_items, self.tokenizer, prefix_weight=5.0, continuation_weight=1.0
        )
        pre_losses = compute_prefix_and_continuation_losses(self.model, batch)

        opt = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.model.train()
        for _ in range(18):
            opt.zero_grad()
            l_dict = compute_prefix_and_continuation_losses(self.model, batch)
            l_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()

        post_losses = compute_prefix_and_continuation_losses(self.model, batch)
        self.assertLess(float(post_losses["prefix_loss"].item()), float(pre_losses["prefix_loss"].item()))
        self.assertLess(float(post_losses["continuation_loss"].item()), float(pre_losses["continuation_loss"].item()))

        post_report = evaluate_three_way_prefix_and_alu_outcomes(
            self.model, self.tokenizer, train_items, max_new_tokens=28
        )
        outcomes = post_report["outcomes"]
        self.assertIn("neural_only_full_performance", outcomes)
        self.assertIn("neural_equation_setup_accuracy", outcomes)
        self.assertIn("alu_assisted_final_performance", outcomes)
        self.assertEqual(outcomes["neural_only_full_performance"]["assistance_mode"], "neural_only")
        self.assertEqual(outcomes["alu_assisted_final_performance"]["assistance_mode"], "bounded_alu_assisted")
        self.assertTrue(outcomes["alu_assisted_final_performance"]["unassisted_neural_claim_prohibited"])
        self.assertGreater(
            outcomes["neural_equation_setup_accuracy"]["semantic_setup_accuracy"],
            pre_report["outcomes"]["neural_equation_setup_accuracy"]["semantic_setup_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
