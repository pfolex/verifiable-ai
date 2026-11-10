"""Unit tests for the AIX-Zero to EVAL-001..006 harness bridge (`EVAL-01`, `EVAL-02`, `EVAL-03`, `EVAL-06`)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import unittest

from evaluation_result import validate_evaluation_result

from aix_zero.benchmark_baseline import load_v11
from aix_zero.eval_harness_bridge import (
    AIXZeroProvider,
    OPEN_WEIGHT_BASELINE_REGISTRY,
    assess_aix_paired_results,
    assert_comparable_conditions,
    baseline_report_to_eval_results,
    build_latency_and_throughput_report,
    build_multidimensional_scorecard,
    compute_energy_and_useful_work,
    evaluate_probability_calibration,
    group_clustered_paired_interval,
    rational_scorer_metadata,
    run_aix_benchmark,
    score_rational_or_probability,
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
        self.assertEqual(neural["model"]["parameters"]["answer_source"], "neural_only")

        self.assertEqual(assisted["summary"]["sample_count"], 105)
        self.assertAlmostEqual(assisted["summary"]["mean_score"], 1.0)
        self.assertTrue(assisted["model"]["parameters"]["inflight_alu"])
        self.assertEqual(assisted["model"]["parameters"]["answer_source"], "arithmetic_repaired")

    def test_multidimensional_scorecard_keeps_raw_metrics_and_separates_modes(self):
        baseline_report = json.loads((ROOT / "research/baseline/v11-local.json").read_text())
        scorecard = build_multidimensional_scorecard(baseline_report)
        self.assertEqual(len(scorecard["entries"]), 2)
        by_source = {e["answer_source"]: e for e in scorecard["entries"]}
        self.assertIn("neural_only", by_source)
        self.assertIn("arithmetic_repaired", by_source)

        n_entry = by_source["neural_only"]
        self.assertEqual(n_entry["capability"]["correct_samples"], 9)
        self.assertEqual(n_entry["capability"]["total_samples"], 105)
        self.assertEqual(n_entry["resources"]["physical_parameters"], 817410)
        self.assertEqual(n_entry["resources"]["active_parameters_per_token"], 719106)
        self.assertIsNone(n_entry["cost_and_energy"]["measured_energy_joules"])

    def test_mixing_assisted_and_unassisted_modes_is_rejected_by_default(self):
        baseline_report = json.loads((ROOT / "research/baseline/v11-local.json").read_text())
        converted = baseline_report_to_eval_results(baseline_report)
        with self.assertRaisesRegex(ValueError, "Cannot compare mixed answer_source"):
            assert_comparable_conditions(converted["neural_only"], converted["symbolic_assisted"])

        # Mismatched sampling budget is also rejected even within the same mode
        mismatched = copy.deepcopy(converted["neural_only"])
        mismatched["model"]["parameters"]["num_candidates"] = 8
        with self.assertRaisesRegex(ValueError, "Incomparable sampling/compute budget"):
            assert_comparable_conditions(converted["neural_only"], mismatched)

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
            allow_cross_mode_ablation=True,
        )
        self.assertEqual(assessment["criteria"]["overall"]["decision"], "PASS")
        self.assertEqual(assessment["decision"], "INCONCLUSIVE")
        self.assertIn("insufficient samples for one or more constraints", assessment["reasons"])

    def test_exact_rational_and_probability_validity_scorer(self):
        meta = rational_scorer_metadata(require_probability_range=True, require_reduced=False)
        self.assertEqual(meta["name"], "aix-exact-rational-probability-scorer")

        # Mathematically equivalent fraction ("2/4" == "1/2")
        eq = score_rational_or_probability("<ans>2/4</ans>", "1/2", is_probability=True, require_reduced=False)
        self.assertEqual(eq["score"], 1.0)
        self.assertEqual(eq["status"], "VALID_EXACT_MATCH")
        self.assertEqual(eq["parsed_fraction"], "1/2")

        # Require reduced rejects unreduced "2/4"
        unred = score_rational_or_probability("2/4", "1/2", is_probability=True, require_reduced=True)
        self.assertEqual(unred["score"], 0.0)
        self.assertEqual(unred["status"], "UNREDUCED_FRACTION")

        # Out-of-range probability (> 1 or < 0)
        gt1 = score_rational_or_probability("5/4", "1/2", is_probability=True)
        self.assertEqual(gt1["score"], 0.0)
        self.assertFalse(gt1["valid_probability"])
        self.assertEqual(gt1["status"], "OUT_OF_RANGE_PROBABILITY")

        neg = score_rational_or_probability("-1/3", "1/3", is_probability=True)
        self.assertEqual(neg["score"], 0.0)
        self.assertFalse(neg["valid_probability"])
        self.assertEqual(neg["status"], "OUT_OF_RANGE_PROBABILITY")

        # Zero denominator
        div0 = score_rational_or_probability("3/0", "1/2", is_probability=True)
        self.assertEqual(div0["score"], 0.0)
        self.assertFalse(div0["valid_rational"])
        self.assertEqual(div0["status"], "ZERO_DENOMINATOR")

        # Impossible conditioning event
        imp = score_rational_or_probability("1/2", "1/2", is_probability=True, conditioning_denominator=0)
        self.assertEqual(imp["score"], 0.0)
        self.assertEqual(imp["status"], "IMPOSSIBLE_CONDITIONING_EVENT")

        # Malformed fraction
        bad = score_rational_or_probability("not_a_number", "1/2", is_probability=True)
        self.assertEqual(bad["score"], 0.0)
        self.assertEqual(bad["status"], "MALFORMED_FRACTION")

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

    def test_eval_05_group_clustered_paired_intervals_prevent_correlated_variant_inflation(self):
        # 5 semantic templates, each repeated 10 times with identical outcomes -> effective G is 5, not 50
        pairs = []
        for g in range(5):
            for r in range(10):
                pairs.append({
                    "sample_id": f"s-{g}-{r}",
                    "group_id": f"template-{g}",
                    "delta": 1.0 if g < 3 else 0.0,
                })
        clustered = group_clustered_paired_interval(pairs, alpha=0.05)
        self.assertEqual(clustered["prompt_count"], 50)
        self.assertEqual(clustered["group_count"], 5)
        self.assertTrue(len(clustered["warnings"]) >= 1)
        self.assertAlmostEqual(clustered["mean_group_delta"], 0.6)
        self.assertLess(clustered["group_lower"], clustered["naive_prompt_lower"])

    def test_eval_07_probability_calibration_and_verifier_confidence(self):
        records = [
            {"predicted_prob": "4/5", "gold_prob": "3/4", "verifier_confidence": 0.9, "correct": True},
            {"predicted_prob": "1/5", "gold_prob": "1/4", "verifier_confidence": 0.8, "correct": False},
            {"predicted_prob": "1/2", "gold_prob": "1/2", "verifier_confidence": 0.6, "correct": True},
        ]
        cal = evaluate_probability_calibration(records, num_bins=5)
        self.assertEqual(cal["event_probability_calibration"]["sample_count"], 3)
        self.assertEqual(cal["verifier_confidence_calibration"]["sample_count"], 3)
        self.assertGreaterEqual(cal["event_probability_calibration"]["brier_score"], 0.0)
        self.assertGreaterEqual(cal["verifier_confidence_calibration"]["ece"], 0.0)

    def test_eval_08_latency_and_eval_10_energy_and_eval_11_external_baselines(self):
        samples = [
            {"latency_ms": 12.0, "output_tokens": 4, "score": 1.0},
            {"latency_ms": 15.0, "output_tokens": 5, "score": 0.0},
            {"latency_ms": 18.0, "output_tokens": 6, "score": 1.0},
            {"latency_ms": 42.0, "output_tokens": 5, "score": 0.0},
        ]
        lat = build_latency_and_throughput_report(
            samples,
            first_request_ms=25.0,
            warmup_excluded_count=1,
            concurrency=1,
            cache_condition="cold_application_cache_not_os_disk_cold",
        )
        self.assertEqual(lat["cache_condition"], "cold_application_cache_not_os_disk_cold")
        self.assertEqual(lat["full_response_latency_ms"]["p95_ms"], 42.0)

        energy_unmeasured = compute_energy_and_useful_work(samples, measured_joules=None)
        self.assertIsNone(energy_unmeasured["measured_energy_joules"])
        self.assertIsNone(energy_unmeasured["estimated_energy"])
        self.assertEqual(energy_unmeasured["verified_useful_samples"], 2)

        energy_est = compute_energy_and_useful_work(samples, estimated_tdp_watts=15.0)
        self.assertIsNone(energy_est["measured_energy_joules"])
        self.assertIsNotNone(energy_est["estimated_energy"])
        self.assertGreater(energy_est["estimated_energy"]["estimated_joules"], 0.0)

        self.assertGreaterEqual(len(OPEN_WEIGHT_BASELINE_REGISTRY), 3)
        self.assertEqual(OPEN_WEIGHT_BASELINE_REGISTRY[0]["baseline_id"], "qwen2.5-0.5b-instruct-cpu")


if __name__ == "__main__":
    unittest.main()
