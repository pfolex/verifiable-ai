"""Unit tests for the Unified Standard AI Benchmark Scorecard and AI-Coding Evaluator (`#263`, `#430`, `#432`, PR `#436`)."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.aix_ctl import main as aix_ctl_main
from aix_zero.benchmark_baseline import load_v11
from aix_zero.meta_ai_autopilot import REPO_ROOT
from aix_zero.standard_benchmarks import (
    BENCHMARK_SUITE_DOMAIN_GROUPS,
    evaluate_ai_coding_expression_suite,
    run_standard_ai_benchmark_suites,
)


class StandardBenchmarksTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20261125)
        self.model, self.tokenizer = load_v11(REPO_ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors")
        self.model.eval()

    def test_ai_coding_expression_suite_reduces_loss_and_fits_training_expressions(self) -> None:
        rep = evaluate_ai_coding_expression_suite(
            self.model,
            self.tokenizer,
            train_steps=15,
            lr=3.5e-3,
            seed=42,
        )
        self.assertEqual(rep["assistance_mode"], "neural_only")
        self.assertEqual(rep["steps_completed"], 15)
        self.assertLess(rep["final_loss"], rep["initial_loss"])
        self.assertGreaterEqual(rep["train_fit_after"]["correct"], 1)
        self.assertEqual(rep["dev_before"]["total"], 12)
        self.assertEqual(rep["dev_after"]["total"], 12)

    def test_unified_standard_benchmark_suites_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            scorecard = run_standard_ai_benchmark_suites(
                tdir,
                cycle_index=1,
                include_swe_three_arm=True,
                ai_coding_steps=4,
            )
            self.assertTrue((tdir / "standard_benchmark_scorecard.json").exists())
            self.assertEqual(scorecard["schema_version"], "2.0.0")
            self.assertEqual(scorecard["evaluation_origin"], "internal_synthetic_diagnostic")
            self.assertEqual(set(scorecard["benchmark_suites"].keys()), set(BENCHMARK_SUITE_DOMAIN_GROUPS.keys()))
            for suite_key in scorecard["benchmark_suites"]:
                for forbidden_ext in ("gsm8k", "mmlu", "gpqa", "theoremqa", "numinamath"):
                    self.assertNotIn(forbidden_ext, suite_key.lower())
            ov = scorecard["overall_summary"]
            self.assertFalse(ov["routed_neural_only"]["inflight_alu"])
            self.assertEqual(ov["neuro_symbolic_verified"]["accuracy"], 1.0)
            self.assertEqual(scorecard["swe_three_arm_patch_suite"]["attribution_mode"], "heuristic_ast_control_not_model_generated")
            self.assertEqual(scorecard["swe_three_arm_patch_suite"]["collaborative_pass_rate"], 1.0)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = aix_ctl_main(["--state-dir", str(tdir), "benchmark", "--cycle", "1", "--coding-steps", "2"])
            self.assertEqual(rc, 0)
            out_text = buf.getvalue()
            self.assertIn("AIX Benchmark :", out_text)
            self.assertIn("Overall Score :", out_text)
            self.assertIn("3-Arm SWE Ctrl:", out_text)


if __name__ == "__main__":
    unittest.main()
