"""Unit tests for Bounded Multi-Turn Ollama Coaching and Verified Cartridge Distillation (`[AIX PEER-01]` / Issue #431)."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.equation_prefix_supervision import build_arithmetic_prefix_study_splits
from aix_zero.ollama_peer_coaching import (
    discover_ollama_service,
    distill_verified_coaching_into_cartridge,
    run_bounded_ollama_coaching_session,
)
from aix_zero.tokenizer import build_v11_tokenizer


class OllamaPeerCoachingTests(unittest.TestCase):
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
        self.splits = build_arithmetic_prefix_study_splits()

    def test_discovery_reports_actionable_unavailable_when_service_or_model_missing(self) -> None:
        # 1. Service unreachable
        def _offline_getter(url: str, timeout: float):
            raise ConnectionRefusedError("Connection refused")

        rep_offline = discover_ollama_service(
            requested_model="qwen2.5-coder:7b",
            endpoint_url="http://127.0.0.1:11434",
            http_get_json=_offline_getter,
        )
        self.assertEqual(rep_offline["status"], "UNAVAILABLE")
        self.assertFalse(rep_offline["service_available"])
        self.assertFalse(rep_offline["model_available"])
        self.assertIn("No implicit model download was attempted", rep_offline["actionable_message"])

        # 2. Service running, but requested model missing
        def _missing_model_getter(url: str, timeout: float):
            if url.endswith("/api/version"):
                return {"version": "0.5.7"}
            return {"models": [{"name": "llama3:8b", "digest": "sha256:1111", "size": 1000}]}

        rep_missing = discover_ollama_service(
            requested_model="qwen2.5-coder:7b",
            http_get_json=_missing_model_getter,
        )
        self.assertEqual(rep_missing["status"], "UNAVAILABLE")
        self.assertTrue(rep_missing["service_available"])
        self.assertFalse(rep_missing["model_available"])
        self.assertIn("llama3:8b", rep_missing["available_local_models"])

        # 3. Service and model available
        def _ready_getter(url: str, timeout: float):
            if url.endswith("/api/version"):
                return {"version": "0.5.7"}
            return {"models": [{"name": "qwen2.5-coder:7b", "digest": "sha256:abcd1234", "size": 4096}]}

        rep_ready = discover_ollama_service(
            requested_model="qwen2.5-coder:7b",
            http_get_json=_ready_getter,
        )
        self.assertEqual(rep_ready["status"], "AVAILABLE")
        self.assertTrue(rep_ready["service_available"])
        self.assertTrue(rep_ready["model_available"])
        self.assertEqual(rep_ready["model_digest"], "sha256:abcd1234")

    def test_multi_turn_retry_validator_disagreement_and_split_hygiene(self) -> None:
        train_tasks = self.splits["train"]
        heldout_prompts = {it["prompt"] for it in self.splits["heldout_standard"]}

        # 1. Attempting to pass a held-out/development prompt must raise ValueError (#407 guard)
        with self.assertRaises(ValueError):
            run_bounded_ollama_coaching_session(
                training_tasks=train_tasks + [{"prompt": next(iter(heldout_prompts)), "completion": "<think>24/4=6</think><ans>6</ans>", "gold": "6"}],
                reserved_non_training_prompts=heldout_prompts,
            )

        with self.assertRaises(ValueError):
            run_bounded_ollama_coaching_session(
                training_tasks=[{**train_tasks[0], "split": "development"}],
            )

        # 2. Multi-turn retry:
        # - Task 0 Turn 1: malformed response (no tags) -> rejected; Turn 2: valid trace -> accepted!
        # - Task 1 Turn 1: validator disagreement (valid R1 equation & matching gold `<think>2*1=2</think><ans>2</ans>`,
        #   but wrong semantic operands vs `16/8=2`!) -> rejected by independent_semantic_setup_ok!
        #   Turn 2: `<think>ratio=16/8=2</think><ans>2</ans>` -> accepted!
        def _mock_teacher(prompt_sent: str, turn_idx: int, settings: dict) -> dict:
            if "fp16_bits=16;quant_bits=4" in prompt_sent:
                if turn_idx == 1:
                    return {"response": "The compression is 4.", "tokens_used": 8}
                # Ensure reference answer was not leaked into feedback
                self.assertNotIn("<ans>4</ans>", prompt_sent)
                return {"response": "<think>ratio=16/4=4</think><ans>4</ans>", "tokens_used": 12}
            if "fp16_bits=16;quant_bits=8" in prompt_sent:
                if turn_idx == 1:
                    # Passes R1 equation & answer check (2*1=2, ans=2), but disagrees with semantic setup (16/8)!
                    return {"response": "<think>ratio=2*1=2</think><ans>2</ans>", "tokens_used": 12}
                return {"response": "<think>ratio=16/8=2</think><ans>2</ans>", "tokens_used": 12}
            if "fp16_bits=32;quant_bits=4" in prompt_sent:
                return {"response": "<think>ratio=32/4=8</think><ans>8</ans>", "tokens_used": 12}
            return {"response": "<think>ratio=32/8=4</think><ans>4</ans>", "tokens_used": 12}

        session = run_bounded_ollama_coaching_session(
            training_tasks=train_tasks,
            reserved_non_training_prompts=heldout_prompts,
            max_turns_per_task=2,
            max_tokens_per_turn=32,
            max_total_tokens=256,
            teacher_generate_fn=_mock_teacher,
        )
        self.assertEqual(session["session_status"], "COMPLETED")
        self.assertEqual(session["admitted_count"], 4)
        for ex in session["admitted_examples"]:
            self.assertEqual(ex["verification_status"], "verifier_accepted")
            self.assertFalse(ex["guaranteed_universal_correctness"])

        # Verify Task 0 and Task 1 required 2 turns
        self.assertEqual(session["task_logs"][0]["turns_executed"], 2)
        self.assertEqual(session["task_logs"][1]["turns_executed"], 2)
        self.assertFalse(session["task_logs"][1]["turns"][0]["validation"]["independent_semantic_setup_ok"])

        # 3. Distill accepted examples into an isolated cartridge and compare the 4 arms
        with tempfile.TemporaryDirectory() as tmp:
            distill_report = distill_verified_coaching_into_cartridge(
                self.model,
                self.tokenizer,
                domain="META_QUANTIZATION_FP8_INT4",
                coaching_report=session,
                self_train_control_items=train_tasks[:1],
                heldout_eval_items=self.splits["heldout_standard"][:2],
                work_dir=Path(tmp),
                slot=2,
                steps=8,
                lr=4e-3,
                max_new_tokens=20,
            )
            self.assertEqual(distill_report["evaluation_assistance_mode"], "neural_only")
            self.assertFalse(distill_report["teacher_assistance_at_eval_time"])
            self.assertFalse(distill_report["inflight_alu_at_eval_time"])
            self.assertTrue(distill_report["base_model_unchanged"])
            self.assertIn("original_baseline", distill_report["arms"])
            self.assertIn("no_teacher_self_trained_control", distill_report["arms"])
            self.assertIn("teacher_distilled_candidate", distill_report["arms"])
            self.assertIn("teacher_only_reference", distill_report)

    def test_timeout_cancellation_and_token_budget_exhaustion(self) -> None:
        train_tasks = self.splits["train"]

        # 1. Token budget exhaustion
        sess_budget = run_bounded_ollama_coaching_session(
            training_tasks=train_tasks,
            max_turns_per_task=2,
            max_tokens_per_turn=20,
            max_total_tokens=15,
            teacher_generate_fn=lambda _p, _t, _s: {"response": "bad", "tokens_used": 20},
        )
        self.assertEqual(sess_budget["termination_reason"], "token_budget_exhausted")
        self.assertEqual(sess_budget["admitted_count"], 0)

        # 2. Timeout handling
        def _timeout_teacher(_p: str, _t: int, _s: dict) -> dict:
            raise TimeoutError("Teacher timed out")

        sess_timeout = run_bounded_ollama_coaching_session(
            training_tasks=train_tasks,
            teacher_generate_fn=_timeout_teacher,
        )
        self.assertEqual(sess_timeout["termination_reason"], "teacher_call_timeout")
        self.assertEqual(sess_timeout["admitted_count"], 0)

        # 3. Caller cancellation
        sess_cancel = run_bounded_ollama_coaching_session(
            training_tasks=train_tasks,
            teacher_generate_fn=lambda _p, _t, _s: {"response": "<think>16/4=4</think><ans>4</ans>", "tokens_used": 10},
            cancel_check=lambda: True,
        )
        self.assertEqual(sess_cancel["termination_reason"], "cancelled_by_caller")


if __name__ == "__main__":
    unittest.main()
