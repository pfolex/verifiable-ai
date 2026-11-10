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
    assess_cross_representation_transfer,
    assert_comparable_conditions,
    baseline_report_to_eval_results,
    build_latency_and_throughput_report,
    build_multidimensional_scorecard,
    compare_open_base_with_tiny_aix_zero,
    compute_energy_and_useful_work,
    compute_exact_shapley_values,
    enumerate_finite_scm_exact,
    evaluate_bayesian_edge_cases,
    evaluate_bounded_test_time_search,
    evaluate_causal_identifiability_task,
    evaluate_compositional_belief_decision_task,
    evaluate_compute_and_capability_investment_model,
    evaluate_cooperation_stability,
    evaluate_cross_task_family_generalization,
    evaluate_frontier_comparison_milestone,
    evaluate_non_iid_evidence_task,
    evaluate_posterior_predictive_task,
    evaluate_probability_calibration,
    evaluate_quantization_tradeoffs,
    evaluate_representation_robustness_suite,
    evaluate_scientific_reasoning_suite,
    evaluate_structural_causal_model,
    evaluate_verified_retrieval_augmentation,
    format_external_coding_fixture,
    generate_confounded_causal_pairs,
    group_clustered_paired_interval,
    measure_memory_and_cache_budget,
    prepare_reproducible_competitive_model_release,
    publish_final_roadmap_scaling_decision,
    rational_scorer_metadata,
    run_aix_benchmark,
    run_independent_capability_red_team_audit,
    score_cooperative_game_task,
    score_epistemic_abstention_task,
    score_rational_or_probability,
    validate_frontier_comparison_claim,
    verify_independent_reproduction,
    verify_reasoning_trace_steps,
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

    def test_wave6_eval_16_and_reason_03_04_07(self):
        # EVAL-16 frontier comparison claim policy
        with self.assertRaisesRegex(ValueError, "Prohibited frontier claim_scope"):
            validate_frontier_comparison_claim({"claim_scope": "overall_world_best"})

        valid_claim = validate_frontier_comparison_claim(
            {
                "claim_scope": "21-domain-symbolic-assisted-cpu-1thread",
                "dated_open_weight_baselines": ["qwen2.5-0.5b-instruct-cpu"],
                "paired_confidence_interval": {"lower": 0.1, "upper": 0.3},
                "contamination_exclusion_review": "v12-semantic-disjoint",
                "independent_reproduction_report": {"claim_status": "REPRODUCED_VERIFIED"},
            }
        )
        self.assertTrue(valid_claim["claim_approved"])

        # REASON-07 exhaustive finite SCM enumeration agrees with backdoor adjustment
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
        enum_do = enumerate_finite_scm_exact(scm, query_kind="interventional_do_x", x_value=1)
        self.assertTrue(enum_do["independent_backdoor_agreement"])
        self.assertEqual(enum_do["exact_probability"], "1/2")

        # REASON-03 non-IID dependent evidence invalidates naive IID solver
        non_iid = evaluate_non_iid_evidence_task(
            p="1/2", a1="2/3", b1="1/3", a2_given_e1_1="1", b2_given_e1_1="1", e="11"
        )
        self.assertTrue(non_iid["naive_iid_fails_negative_control"])
        self.assertEqual(non_iid["exact_dependent_posterior"], "2/3")
        self.assertEqual(non_iid["naive_iid_posterior"], "4/5")

        # REASON-04 posterior predictive marginalization
        pp = evaluate_posterior_predictive_task(
            p="1/2", a_obs="2/3", b_obs="1/3", e_obs="1", a_next="3/4", b_next="1/4"
        )
        self.assertEqual(pp["latent_posterior_h1"], "2/3")
        self.assertEqual(pp["posterior_predictive_e_next_1"], "7/12")
        self.assertTrue(pp["posterior_differs_from_predictive"])

    def test_wave7_reason_08_13_14_and_scale_07_13(self):
        # REASON-14 reference-verified trace checks reject right answer from invalid intermediate steps
        expected = [
            {"step_name": "unnormalized_h1", "value": "2/5"},
            {"step_name": "unnormalized_h0", "value": "1/10"},
        ]
        bad_trace = verify_reasoning_trace_steps(
            [{"step_name": "unnormalized_h1", "value": "9/10"}, {"step_name": "unnormalized_h0", "value": "1/10"}],
            expected_steps=expected,
            final_prediction="4/5",
            gold_final_answer="4/5",
        )
        self.assertEqual(bad_trace["status"], "REJECTED_INVALID_INTERMEDIATE_STEP")
        self.assertEqual(bad_trace["score"], 0.0)
        self.assertFalse(bad_trace["formal_proof_claimed"])

        good_trace = verify_reasoning_trace_steps(
            expected,
            expected_steps=expected,
            final_prediction="4/5",
            gold_final_answer="4/5",
        )
        self.assertEqual(good_trace["status"], "VERIFIED_TRACE_AND_ANSWER")
        self.assertEqual(good_trace["score"], 1.0)

        # REASON-08 confounding pairs and held-out graph topology
        pairs = generate_confounded_causal_pairs()
        self.assertTrue(pairs["train_pair"]["observational_differs_from_interventional"])
        self.assertTrue(pairs["held_out_test_pair"]["observational_differs_from_interventional"])
        self.assertTrue(pairs["holds_out_graph_topology"])
        self.assertEqual(pairs["train_pair"]["confounder_variable"], "Z")

        # REASON-13 compositional belief + decision evaluation
        comp = evaluate_compositional_belief_decision_task(
            p="1/2",
            a="4/5",
            b="1/5",
            e="1",
            payoff_act_if_h1=10,
            payoff_act_if_h0=-5,
            payoff_wait=0,
            predicted_posterior="4/5",
            predicted_expected_utility_act="7",
            predicted_action="ACT",
        )
        self.assertEqual(comp["stage1_belief_score"], 1.0)
        self.assertEqual(comp["stage2_expected_utility_score"], 1.0)
        self.assertEqual(comp["stage3_final_action_score"], 1.0)
        self.assertEqual(comp["full_compositional_exact_score"], 1.0)

        # SCALE-07 bounded test-time search requires disclosure when comparing multi-candidate/tool vs 1-sample
        with self.assertRaisesRegex(ValueError, "without disclosure"):
            evaluate_bounded_test_time_search(
                single_sample_baseline_accuracy=0.0857,
                search_candidate_accuracy=1.0,
                verifier_kind="inflight_alu",
                num_candidates=4,
                max_candidates_cap=8,
                tokens_used=160,
                max_token_budget=256,
                wall_clock_ms=18.5,
                multi_sample_or_tool_disclosure=False,
            )
        search_ok = evaluate_bounded_test_time_search(
            single_sample_baseline_accuracy=0.0857,
            search_candidate_accuracy=1.0,
            verifier_kind="inflight_alu",
            num_candidates=4,
            max_candidates_cap=8,
            tokens_used=160,
            max_token_budget=256,
            wall_clock_ms=18.5,
            multi_sample_or_tool_disclosure=True,
        )
        self.assertTrue(search_ok["multi_sample_or_tool_disclosure"])

        # SCALE-13 independent red-team evaluation separates failure categories and strips private data
        rt = run_independent_capability_red_team_audit(
            [
                {
                    "finding_id": "RT-01",
                    "reviewer_consent": True,
                    "failure_category": "model_failure",
                    "summary": "Neural-only v11 gets 9/105 without arithmetic repair.",
                    "remediation": "Always report neural-only and ALU-assisted scores separately.",
                    "private_email": "redteam@example.org",
                },
                {
                    "finding_id": "RT-02",
                    "reviewer_consent": True,
                    "failure_category": "data_failure",
                    "summary": "Prompt layout novelty could leak if not held out.",
                    "remediation": "Enforce disjoint semantic group hashes.",
                },
            ]
        )
        self.assertEqual(rt["total_findings"], 2)
        self.assertFalse(rt["contains_private_contributor_data"])
        self.assertEqual(len(rt["failures_by_category"]["model_failure"]), 1)

    def test_wave8_reason_09_and_15(self):
        # REASON-09 causal identifiability negative cases
        non_id_ok = evaluate_causal_identifiability_task(
            "<ans>NON_IDENTIFIABLE</ans>", has_unobserved_confounder_on_backdoor=True
        )
        self.assertEqual(non_id_ok["score"], 1.0)
        self.assertFalse(non_id_ok["invented_causal_effect"])

        invented = evaluate_causal_identifiability_task(
            "<ans>3/5</ans>", has_unobserved_confounder_on_backdoor=True
        )
        self.assertEqual(invented["score"], 0.0)
        self.assertTrue(invented["invented_causal_effect"])
        self.assertEqual(invented["status"], "REJECTED_INVENTED_CAUSAL_EFFECT")

        # REASON-15 cross-representation transfer holding semantic groups fixed
        tr = assess_cross_representation_transfer(
            [
                {
                    "semantic_group_id": "G1",
                    "gold": "4/5",
                    "representations": {
                        "compact_dsl": "4/5",
                        "verbose_kv": "1/5",
                        "json_fields": "not_a_fraction",
                    },
                }
            ]
        )
        self.assertTrue(tr["semantic_groups_held_fixed"])
        self.assertEqual(tr["representations"]["compact_dsl"]["exact_match_accuracy"], 1.0)
        self.assertEqual(tr["representations"]["verbose_kv"]["reasoning_failure_rate"], 1.0)
        self.assertEqual(tr["representations"]["json_fields"]["parser_failure_rate"], 1.0)

    def test_wave9_scale_02_and_08(self):
        # SCALE-02 compare stronger open base with tiny AIX-Zero without conflating capabilities
        cmp_base = compare_open_base_with_tiny_aix_zero(
            aix_zero_metrics={
                "model_id": "aix-zero-v11-817k",
                "license": "MIT",
                "source_revision_sha": "a" * 40,
                "peak_memory_mb": 42.0,
                "execution_cost_ms": 14.0,
                "accuracy": 0.0857,
            },
            external_open_base_metrics={
                "model_id": "qwen2.5-0.5b-instruct",
                "license": "Apache-2.0",
                "source_revision_sha": "b" * 40,
                "peak_memory_mb": 1150.0,
                "execution_cost_ms": 420.0,
                "accuracy": 0.45,
            },
        )
        self.assertTrue(cmp_base["capabilities_strictly_separated"])
        self.assertIn("qwen2.5-0.5b-instruct", cmp_base["inherited_capability_attribution"])

        # SCALE-08 measure generalization across task families with failures and abstentions reported
        gen = evaluate_cross_task_family_generalization(
            predeclared_partition_id="final_cross_family_v1",
            task_family_records=[
                {"family": "BP", "outcome": "EXACT_MATCH"},
                {"family": "SCM", "outcome": "ABSTAINED"},
                {"family": "COOP", "outcome": "FAILED"},
            ],
            shortcut_control_accuracy=0.25,
        )
        self.assertEqual(gen["failure_count"], 1)
        self.assertEqual(gen["abstention_count"], 1)
        self.assertTrue(gen["beats_shortcut_control"])

    def test_wave10_scale_05_and_06(self):
        # SCALE-05 quantization accuracy and efficiency tradeoffs
        q_reg = evaluate_quantization_tradeoffs(
            method_version="dynamic_int8_linear_v1",
            fp32_accuracy=0.0857,
            quantized_accuracy=0.0500,
            fp32_memory_bytes=3269640,
            quantized_memory_bytes=950000,
            fp32_latency_ms=14.0,
            quantized_latency_ms=11.0,
            max_allowed_accuracy_drop=0.01,
        )
        self.assertGreater(q_reg["compression_ratio"], 3.0)
        self.assertTrue(q_reg["capability_regressed"])
        self.assertFalse(q_reg["qualifies_as_model_improvement"])
        self.assertFalse(q_reg["compression_alone_is_improvement"])

        # SCALE-06 verified retrieval augmentation across complete, missing, and conflicting evidence
        ret = evaluate_verified_retrieval_augmentation(
            [
                {"evidence_regime": "complete_evidence", "model_only_correct": False, "retrieved_only_correct": True, "solver_assisted_correct": True},
                {"evidence_regime": "missing_evidence", "model_only_correct": False, "retrieved_only_correct": False, "solver_assisted_correct": False},
                {"evidence_regime": "conflicting_evidence", "model_only_correct": False, "retrieved_only_correct": False, "solver_assisted_correct": True},
            ]
        )
        self.assertTrue(ret["modes_strictly_separated"])
        self.assertAlmostEqual(ret["retrieved_evidence_only_accuracy"], 1 / 3, places=5)
        self.assertAlmostEqual(ret["solver_assisted_accuracy"], 2 / 3, places=5)

    def test_wave11_scale_14(self):
        rel = prepare_reproducible_competitive_model_release(
            [
                {"claim_id": "v11-21-domain-symbolic-assisted", "independently_reproduced": True, "beats_declared_controls": True},
                {"claim_id": "v12-bayesian-single-expert", "independently_reproduced": True, "beats_declared_controls": False},
            ]
        )
        self.assertEqual(rel["confirmed_claim_count"], 1)
        self.assertEqual(rel["withdrawn_or_qualified_claim_count"], 1)
        self.assertTrue(rel["release_ready"])

    def test_wave12_to_16_scale_01_15_and_16(self):
        # SCALE-01 compute and capability investment model
        with self.assertRaisesRegex(ValueError, "cannot assume donated hardware"):
            evaluate_compute_and_capability_investment_model(
                measured_step_ms=6.4,
                measured_or_tdp_watts=35.0,
                electricity_usd_per_kwh_range=(0.12, 0.25),
                hardware_amortization_usd_per_hr=0.05,
                assumes_zero_cost_volunteer_hardware_or_power=True,
            )
        inv = evaluate_compute_and_capability_investment_model(
            measured_step_ms=6.4,
            measured_or_tdp_watts=35.0,
            electricity_usd_per_kwh_range=(0.12, 0.25),
            hardware_amortization_usd_per_hr=0.05,
        )
        self.assertFalse(inv["assumes_zero_cost_volunteer_resources"])

        # SCALE-15 frontier-comparison milestone
        with self.assertRaisesRegex(ValueError, "overall best remains unproven"):
            evaluate_frontier_comparison_milestone(
                evaluation_date_iso="2026-11-10",
                disclosed_budget={"cpu_threads": 1},
                scoped_verified_strengths=["21-domain-symbolic-assisted"],
                failed_dimensions=["neural_only_multi_step"],
                uncertainty_interval_95=(0.95, 1.0),
                claims_unscoped_overall_best=True,
            )
        ms = evaluate_frontier_comparison_milestone(
            evaluation_date_iso="2026-11-10",
            disclosed_budget={"cpu_threads": 1, "memory_mb": 42},
            scoped_verified_strengths=["21-domain-symbolic-assisted"],
            failed_dimensions=["neural_only_multi_step", "bayesian_single_expert_holdout"],
            uncertainty_interval_95=(0.95, 1.0),
        )
        self.assertEqual(ms["overall_best_status"], "UNPROVEN")

        # SCALE-16 final roadmap scaling decision
        self.assertTrue((ROOT / "docs/ROADMAP-FINAL-SCALING-DECISION.md").is_file())
        with self.assertRaisesRegex(ValueError, "Cannot promise world-leading intelligence"):
            publish_final_roadmap_scaling_decision(
                decision="REDIRECT_TO_JOINT_BACKBONE_AND_VERIFIED_TRACE_RESEARCH_BEFORE_VOLUNTEER_SCALING",
                next_bounded_hypothesis="H-V12-JOINT",
                evidence_summary={},
                promises_inevitable_world_leading_intelligence=True,
            )
        final_dec = publish_final_roadmap_scaling_decision(
            decision="REDIRECT_TO_JOINT_BACKBONE_AND_VERIFIED_TRACE_RESEARCH_BEFORE_VOLUNTEER_SCALING",
            next_bounded_hypothesis="H-V12-JOINT-LATENT-CORE-AND-STEP-VERIFIER",
            evidence_summary={"v11_neural_only": "9/105", "v11_symbolic_assisted": "105/105", "bayesian_pilot": "REJECTED"},
        )
        self.assertFalse(final_dec["promises_inevitable_world_leading_intelligence"])


if __name__ == "__main__":
    unittest.main()
