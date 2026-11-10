"""Unit tests for the AIX-Zero to EVAL-001..006 harness bridge (`EVAL-01` / Issue #40)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from evaluation_result import validate_evaluation_result

from aix_zero.benchmark_baseline import load_v11
from aix_zero.eval_harness_bridge import (
    AIXZeroProvider,
    assess_aix_paired_results,
    baseline_report_to_eval_results,
    run_aix_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]


class EvalHarnessBridgeTests(unittest.TestCase):
    def test_committed_v11_baseline_converts_to_valid_eval_001_results(self):
        baseline_report = json.loads((ROOT / "research/baseline/v11-local.json").read_text())
        converted = baseline_report_to_eval_results(baseline_report)
        self.assertEqual(set(converted), {"neural_only", "symbolic_assisted"})

        neural = converted["neural_only"]
        assisted = converted["symbolic_assisted"]
        validate_evaluation_result(neural)
        validate_evaluation_result(assisted)

        self.assertEqual(neural["summary"]["sample_count"], 105)
        self.assertAlmostEqual(neural["summary"]["mean_score"], 9 / 105)
        self.assertFalse(neural["model"]["parameters"]["inflight_alu"])

        self.assertEqual(assisted["summary"]["sample_count"], 105)
        self.assertAlmostEqual(assisted["summary"]["mean_score"], 1.0)
        self.assertTrue(assisted["model"]["parameters"]["inflight_alu"])

    def test_paired_statistical_gate_flags_small_per_domain_n_as_inconclusive(self):
        baseline_report = json.loads((ROOT / "research/baseline/v11-local.json").read_text())
        converted = baseline_report_to_eval_results(baseline_report)
        domains_by_id = {
            f"{row['domain']}-r{row.get('repeat', 0)}-{idx:03d}": row["domain"]
            for idx, row in enumerate(baseline_report["modes"]["neural_only"]["samples"])
        }
        policy = {
            "policy_version": "1.0.0",
            "alpha": 0.05,
            "min_improvement": 0.10,
            "non_regression_margin": 0.05,
            "min_samples_per_domain": 10,
            "max_error_rate": 0.0,
            "independent_samples": True,
        }
        assessment = assess_aix_paired_results(
            converted["neural_only"],
            converted["symbolic_assisted"],
            domains_by_id,
            policy,
        )
        # Overall improvement is large (96/105 = +91.4%), so overall passes even with 22 simultaneous constraints,
        # but each domain only has N=5 < min_samples_per_domain=10, so the gate honestly returns INCONCLUSIVE.
        self.assertEqual(assessment["criteria"]["overall"]["decision"], "PASS")
        self.assertEqual(assessment["decision"], "INCONCLUSIVE")
        self.assertIn("insufficient samples for one or more constraints", assessment["reasons"])

    def test_live_aix_zero_provider_runs_through_benchmark_runner(self):
        ckpt = ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        ckpt_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        model, tokenizer = load_v11(ckpt)
        cases = [
            {"sample_id": "mod-01", "domain": "MODULAR_NT", "prompt": "Q:(9+7)%3=", "gold": "1"},
            {"sample_id": "alg-01", "domain": "ALGEBRA_SOLVE", "prompt": "Q:4x+2=18,x=", "gold": "4"},
        ]
        provider = AIXZeroProvider(
            model,
            tokenizer,
            checkpoint_sha256=ckpt_sha,
            execution_mode="symbolic_assisted",
            num_candidates=1,
            max_new_tokens=32,
        )
        result = run_aix_benchmark(cases, provider, run_id="test-bridge-run")
        validate_evaluation_result(result)
        self.assertEqual(result["summary"]["sample_count"], 2)
        self.assertEqual(result["summary"]["mean_score"], 1.0)
        self.assertTrue(all(s["input_tokens"] > 0 and s["output_tokens"] > 0 for s in result["samples"]))


if __name__ == "__main__":
    unittest.main()
