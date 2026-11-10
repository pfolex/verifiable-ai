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
    EXTERNAL_CODING_BENCHMARK_COMPATIBILITY,
    OPEN_WEIGHT_BASELINE_REGISTRY,
    assess_aix_paired_results,
    assert_comparable_conditions,
    baseline_report_to_eval_results,
    build_latency_and_throughput_report,
    build_multidimensional_scorecard,
    compute_energy_and_useful_work,
    compute_exact_shapley_values,
    evaluate_bayesian_edge_cases,
    evaluate_cooperation_stability,
    evaluate_probability_calibration,
    evaluate_representation_robustness_suite,
    evaluate_scientific_reasoning_suite,
    evaluate_structural_causal_model,
    format_external_coding_fixture,
    group_clustered_paired_interval,
    measure_memory_and_cache_budget,
    rational_scorer_metadata,
    run_aix_benchmark,
    score_cooperative_game_task,
    score_epistemic_abstention_task,
    score_rational_or_probability,
    verify_independent_reproduction,
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

    def test_wave4_eval_09_12_15_and_reason_05_10(self):
        ckpt = ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
        model, _ = load_v11(ckpt)

        # EVAL-09 memory and cache budget separation
        mem_ok = measure_memory_and_cache_budget(
            model,
            cached_cartridge_bytes=200_000,
            max_cached_cartridge_bytes=1024 * 1024,
            temporary_activation_bytes=64_000,
        )
        self.assertEqual(mem_ok["status"], "WITHIN_BUDGET")
        mem_exceeded = measure_memory_and_cache_budget(
            model,
            cached_cartridge_bytes=2 * 1024 * 1024,
            max_cached_cartridge_bytes=1024 * 1024,
        )
        self.assertEqual(mem_exceeded["status"], "BUDGET_EXCEEDED")

        # EVAL-12 external coding benchmark compatibility fixture
        self.assertIn("swe-bench-verified", EXTERNAL_CODING_BENCHMARK_COMPATIBILITY)
        fixture = format_external_coding_fixture(
            "livecodebench-exec",
            [{"prompt": "Q:Code:def f(x):return 2*x+1;f(3)=", "gold": "7"}],
        )
        self.assertFalse(fixture["is_full_leaderboard_run"])

        # EVAL-15 independent reproduction gate
        author_rep = {
            "author_id": "alice-author",
            "checkpoint_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "answer_source": "neural_only",
            "mean_score": 9 / 105,
            "environment": {"os": "macOS"},
        }
        repro_ok = {
            "reproducer_id": "bob-reproducer",
            "checkpoint_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
            "answer_source": "neural_only",
            "mean_score": 9 / 105,
            "environment": {"os": "linux"},
        }
        verified = verify_independent_reproduction(author_rep, repro_ok)
        self.assertEqual(verified["claim_status"], "REPRODUCED_VERIFIED")
        self.assertTrue(verified["eligible_for_release_claim"])

        repro_disagree = dict(repro_ok, mean_score=20 / 105)
        blocked = verify_independent_reproduction(author_rep, repro_disagree)
        self.assertEqual(blocked["claim_status"], "BLOCKED_BY_DISAGREEMENT")
        self.assertFalse(blocked["eligible_for_release_claim"])

        # REASON-10 cooperative game verifier (surplus vs Shapley allocation)
        surplus = score_cooperative_game_task("5", v_a=2, v_b=3, v_ab=10, query_type="grand_coalition_surplus")
        self.assertEqual(surplus["score"], 1.0)
        shap_a = score_cooperative_game_task("9/2", v_a=2, v_b=3, v_ab=10, query_type="shapley_player_a")
        self.assertEqual(shap_a["score"], 1.0)
        # Confusing total surplus (5) with Player A's Shapley value (9/2) scores 0.0
        wrong_shap = score_cooperative_game_task("5", v_a=2, v_b=3, v_ab=10, query_type="shapley_player_a")
        self.assertEqual(wrong_shap["score"], 0.0)

        # REASON-05 epistemic abstention vs false certainty
        useful_abs = score_epistemic_abstention_task("<ans>NON_IDENTIFIABLE</ans>", is_identifiable=False)
        self.assertEqual(useful_abs["epistemic_outcome"], "USEFUL_ABSTENTION")
        self.assertEqual(useful_abs["score"], 1.0)

        false_cert = score_epistemic_abstention_task("<ans>1/2</ans>", is_identifiable=False)
        self.assertEqual(false_cert["epistemic_outcome"], "FALSE_CERTAINTY")
        self.assertTrue(false_cert["false_certainty"])
        self.assertEqual(false_cert["score"], 0.0)

    def test_wave5_eval_13_14_and_reason_02_06_11_12(self):
        # EVAL-13 scientific reasoning with independent solver
        sci = evaluate_scientific_reasoning_suite(
            [
                {"family": "KINEMATICS_DISPLACEMENT", "parameters": {"u": "5", "a": "3", "t": "4"}, "prediction": "44"},
                {"family": "OHMS_LAW_POWER", "parameters": {"V": "12", "R": "8"}, "prediction": "18"},
            ]
        )
        self.assertEqual(sci["mean_score"], 1.0)

        # EVAL-14 representation robustness grouped by semantic identity
        rob = evaluate_representation_robustness_suite(
            [
                {"semantic_group_id": "g1", "prediction": "1/2", "gold": "1/2", "is_probability": True},
                {"semantic_group_id": "g1", "prediction": "1/2", "gold": "1/2", "is_probability": True},
                {"semantic_group_id": "g2", "prediction": "1/3", "gold": "1/3", "is_probability": True},
                {"semantic_group_id": "g2", "prediction": "2/3", "gold": "1/3", "is_probability": True},
            ]
        )
        self.assertEqual(rob["independent_semantic_groups"], 2)
        self.assertEqual(rob["group_all_correct_rate"], 0.5)
        self.assertEqual(rob["group_consistency_rate"], 0.5)

        # REASON-02 Bayesian edge cases (impossible evidence, symmetric likelihoods)
        imp = evaluate_bayesian_edge_cases(p="1/2", a="0", b="0", e="1")
        self.assertEqual(imp["status"], "IMPOSSIBLE_EVIDENCE")
        sym = evaluate_bayesian_edge_cases(p="2/5", a="1/3", b="1/3", e="1")
        self.assertTrue(sym["likelihood_symmetry_preserves_prior"])
        self.assertEqual(sym["posterior"], "2/5")

        # REASON-06 structural causal model (observational vs interventional + cycle rejection)
        scm = {
            "edges": [("Z", "X"), ("Z", "Y"), ("X", "Y")],
            "p_z1": "1/2",
            "p_x1_given_z0": "1/4",
            "p_x1_given_z1": "3/4",
            "p_y1_given_z0_x0": "1/10",
            "p_y1_given_z0_x1": "2/10",
            "p_y1_given_z1_x0": "7/10",
            "p_y1_given_z1_x1": "8/10",
        }
        do_x1 = evaluate_structural_causal_model(scm, query_kind="interventional_do_x", x_value=1)
        obs_x1 = evaluate_structural_causal_model(scm, query_kind="observational_given_x", x_value=1)
        self.assertEqual(do_x1["exact_probability"], "1/2")
        self.assertNotEqual(do_x1["exact_probability"], obs_x1["exact_probability"])

        cyclic_scm = dict(scm, edges=[("X", "Y"), ("Y", "X")])
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            evaluate_structural_causal_model(cyclic_scm, query_kind="interventional_do_x", x_value=1)

        # REASON-11 exact 3-player Shapley with dummy player C
        shap = compute_exact_shapley_values(
            ["A", "B", "C"],
            {
                ("A",): 2,
                ("B",): 4,
                ("C",): 0,
                ("A", "B"): 10,
                ("A", "C"): 2,
                ("B", "C"): 4,
                ("A", "B", "C"): 10,
            },
        )
        self.assertTrue(shap["efficiency_verified"])
        self.assertTrue(shap["dummy_player_axiom_verified"])
        self.assertEqual(shap["shapley_values"]["C"], "0")
        self.assertEqual(shap["shapley_values"]["A"], "4")
        self.assertEqual(shap["shapley_values"]["B"], "6")

        # REASON-12 unstable cooperation (Prisoner's Dilemma) vs stable coordination
        pd = evaluate_cooperation_stability(
            payoff_cc=(3, 3), payoff_cd=(0, 5), payoff_dc=(5, 0), payoff_dd=(1, 1)
        )
        self.assertTrue(pd["socially_optimal_joint_cooperate"])
        self.assertFalse(pd["cooperation_is_nash_stable"])
        self.assertEqual(pd["classification"], "UNSTABLE_HIGH_WELFARE_TRAP")


if __name__ == "__main__":
    unittest.main()
