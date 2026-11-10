"""Unit tests for semantic problem identity and disjoint split verification (`DATA-02` / Issue #43)."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from aix_zero.bayesian_tasks import build_dataset
from aix_zero.knowledge_is_power import build_21_domain_kip_v11_curriculum
from aix_zero.semantic_splits import (
    SPLIT_ACCESS_POLICY,
    analyze_bayesian_pilot_errors,
    audit_answer_frequency_shortcuts,
    build_benchmark_exclusion_ledger,
    build_cross_domain_composition_pilot,
    build_reproducible_data_inspection_report,
    build_rotating_test_partition,
    build_validation_curriculum_experiment_set,
    canonical_problem_identity,
    compute_difficulty_factors,
    cross_check_generator_with_independent_oracle,
    decode_rational_representation,
    encode_rational_representation,
    export_training_split_safe,
    generate_generalization_taxonomy_manifest,
    generate_interpretable_negative_examples,
    partition_by_semantic_identity,
    semantic_identity_hash,
    validate_contributor_dataset_submission,
    validate_metamorphic_relations,
    verify_disjoint_splits,
    verify_multistep_bayesian_derivation,
)

ROOT = Path(__file__).resolve().parents[1]


class SemanticSplitsTests(unittest.TestCase):
    def test_formatting_and_layout_changes_cannot_bypass_duplicate_detection(self):
        # Bayesian prompts with reordered fields (train_a vs novel) and unreduced fractions ("2/4" vs "1/2")
        p1 = "Q:BP:p=1/2;a=2/3;b=1/5;e=1;P="
        p2 = "Q:BP:e=0;b=2/10;p=2/4;a=4/6;P="
        self.assertEqual(canonical_problem_identity(p1), canonical_problem_identity(p2))
        self.assertEqual(semantic_identity_hash(p1), semantic_identity_hash(p2))

        # Code trace prompts with renamed formal parameter x vs n and whitespace differences
        c1 = "Q:Code:def f(x):return 4*x+6;f(4)="
        c2 = "Q:Code: def f(n): return 4 * n + 6 ; f(4) ="
        self.assertEqual(semantic_identity_hash(c1), semantic_identity_hash(c2))

        # Causal and game tasks with reordered key=value assignments and unreduced rationals
        causal1 = {"domain": "CAUSAL_DO", "prompt": "Q:Causal:Z=2;X=4/2;Y=5;doX="}
        causal2 = {"domain": "CAUSAL_DO", "prompt": "Q:Causal:X=2;Y=5;Z=2;doX="}
        self.assertEqual(semantic_identity_hash(causal1), semantic_identity_hash(causal2))

        game1 = {"domain": "SYMBIOTIC_GAME", "prompt": "Q:Game:a11=4;a12=2;a21=1;a22=3;eq="}
        game2 = {"domain": "SYMBIOTIC_GAME", "prompt": "Q:Game:a22=3;a21=1;a12=2;a11=4;eq="}
        self.assertEqual(semantic_identity_hash(game1), semantic_identity_hash(game2))

    def test_bayesian_pilot_splits_are_strictly_disjoint(self):
        ds = build_dataset(seed=42)
        for task in ("BP", "BS"):
            splits = {
                split: ds["tasks"][task][split]
                for split in ("train", "validation", "test")
            }
            report = verify_disjoint_splits(splits)
            self.assertTrue(report["disjoint"])
            self.assertEqual(report["split_unique_identities"]["train"], 85)
            self.assertEqual(report["split_unique_identities"]["validation"], 16)
            self.assertEqual(report["split_unique_identities"]["test"], 24)

    def test_v11_legacy_word_problem_surface_name_leakage_is_caught_and_repartitioned(self):
        all_train, all_test, _, _ = build_21_domain_kip_v11_curriculum(seed=42)
        # In the legacy v3/v11 generator, WORD_PROBLEM split on full prompt strings so
        # "Sam has 5 packs of 5 and buys 2" (test) vs "Mia has 5 packs of 5 and buys 2" (train)
        # shared the exact same operand triple (5, 5, 2). Verify that semantic split auditing catches it!
        with self.assertRaisesRegex(ValueError, "Semantic split leakage detected"):
            verify_disjoint_splits({"train": all_train, "test": all_test})

        # And verify that the remaining 20 non-WORD_PROBLEM domains are strictly disjoint:
        non_word_train = [x for x in all_train if x["domain"] != "WORD_PROBLEM"]
        non_word_test = [x for x in all_test if x["domain"] != "WORD_PROBLEM"]
        report = verify_disjoint_splits({"train": non_word_train, "test": non_word_test})
        self.assertTrue(report["disjoint"])

        # And verify that partition_by_semantic_identity cleanly groups surface-name variants into disjoint splits:
        word_pool = [x for x in all_train + all_test if x["domain"] == "WORD_PROBLEM"]
        clean_word_splits = partition_by_semantic_identity(word_pool)
        self.assertTrue(verify_disjoint_splits(clean_word_splits)["disjoint"])

    def test_leaked_problem_across_splits_raises_value_error(self):
        train = [{"prompt": "Q:BP:p=1/3;a=2/3;b=1/5;e=1;P="}]
        # Same semantic probability triple disguised with novel layout and unreduced 2/6
        test = [{"prompt": "Q:BP:e=0;b=1/5;p=2/6;a=2/3;P="}]
        with self.assertRaisesRegex(ValueError, "Semantic split leakage detected"):
            verify_disjoint_splits({"train": train, "test": test})

    def test_eval_04_holdout_protection_and_rotating_slices(self):
        self.assertFalse(SPLIT_ACCESS_POLICY["test"]["includes_gold_answers"])
        self.assertTrue(SPLIT_ACCESS_POLICY["test"]["consumes_final_test_budget"])
        ds = build_dataset(seed=42)
        bp_splits = ds["tasks"]["BP"]

        # Safe training export succeeds on disjoint splits and refuses test/holdout splits
        exported = export_training_split_safe(bp_splits, requested_split="train")
        self.assertEqual(len(exported), 340)
        with self.assertRaisesRegex(PermissionError, "cannot be exported for training"):
            export_training_split_safe(bp_splits, requested_split="test")

        # Rotating test slice partition produces deterministic disjoint active/reserve slices
        rot0 = build_rotating_test_partition(bp_splits["test"], epoch_id="2026-Q4", rotation_seed=0, active_test_groups=6)
        rot1 = build_rotating_test_partition(bp_splits["test"], epoch_id="2026-Q4", rotation_seed=1, active_test_groups=6)
        self.assertEqual(rot0["active_group_count"] + rot0["reserve_group_count"], 24)
        self.assertNotEqual(
            [x["prompt"] for x in rot0["public_test_cases"]],
            [x["prompt"] for x in rot1["public_test_cases"]],
        )
        self.assertTrue(all("gold" not in x for x in rot0["public_test_cases"]))

    def test_data_03_explicit_difficulty_metadata_across_domains(self):
        bp_easy = compute_difficulty_factors({"domain": "BP", "prompt": "Q:BP:p=1/2;a=1/2;b=1/4;e=1;P="})
        bs_hard = compute_difficulty_factors(
            {"domain": "BS", "prompt": "Q:BS:p=3/7;a=4/9;b=2/9;e=101;P="}
        )
        self.assertGreater(bs_hard["operation_depth"], bp_easy["operation_depth"])
        self.assertGreater(bs_hard["max_denominator"], bp_easy["max_denominator"])
        self.assertEqual(bp_easy["difficulty_tier"], "tier_1_basic")
        self.assertEqual(bs_hard["difficulty_tier"], "tier_3_compositional")

    def test_wave4_data_04_06_07_08_13_and_train_01(self):
        ds = build_dataset(seed=42)
        bp_splits = ds["tasks"]["BP"]

        # DATA-04 independent oracle cross-check + injected mistake detection
        clean_check = cross_check_generator_with_independent_oracle(bp_splits["test"])
        self.assertTrue(clean_check["all_verified"])
        corrupted = [dict(bp_splits["test"][0], gold="99/100")]
        bad_check = cross_check_generator_with_independent_oracle(corrupted)
        self.assertFalse(bad_check["all_verified"])
        self.assertEqual(bad_check["mismatch_count"], 1)

        # DATA-06 generalization taxonomy & blocking held-out grammar in train
        tax = generate_generalization_taxonomy_manifest(bp_splits)
        self.assertIn("representation_or_layout_novelty", tax["split_category_counts"]["test"])
        with self.assertRaisesRegex(ValueError, "Held-out grammar category"):
            generate_generalization_taxonomy_manifest({"train": bp_splits["test"]})

        # DATA-07 shortcut auditing
        shortcuts = audit_answer_frequency_shortcuts(bp_splits["train"], bp_splits["test"])
        self.assertAlmostEqual(shortcuts["shortcut_accuracies"]["prior_only_echo"], 12 / 96, places=5)

        # DATA-08 validation-only curriculum experiment set
        val_set = build_validation_curriculum_experiment_set(
            bp_splits, partition_id="val-curriculum-v1", max_tuning_trials=5
        )
        self.assertFalse(val_set["consumes_final_test_budget"])
        self.assertEqual(val_set["allowed_split"], "validation")

        # DATA-13 benchmark exclusion ledger
        ledger = build_benchmark_exclusion_ledger(
            historical_dataset_id="aix-zero-v11-kip-21domain",
            historical_dataset_sha256="a" * 64,
            excluded_cases=[
                {
                    "prompt": "Q:Sam has 5 packs of 5 and buys 2. Total?=",
                    "semantic_identity_hash": semantic_identity_hash(
                        {"domain": "WORD_PROBLEM", "prompt": "Q:Sam has 5 packs of 5 and buys 2. Total?="}
                    ),
                    "reason": "Surface-name variant of Mia train prompt with identical (5, 5, 2) operands",
                }
            ],
            replacement_partition_id="v12-semantic-disjoint-word-problem",
        )
        self.assertTrue(ledger["historical_reports_retain_original_sha256"])

        # TRAIN-01 Bayesian pilot error analysis
        pilot_json = json.loads((ROOT / "research/bayesian-pilot/results.json").read_text(encoding="utf-8"))
        err_analysis = analyze_bayesian_pilot_errors(pilot_json)
        self.assertFalse(err_analysis["used_consumed_test_set_for_hyperparameter_selection"])
        self.assertTrue(err_analysis["tasks"]["BP"]["root_cause_categories"]["below_prior_only_shortcut"])
        self.assertTrue((ROOT / "docs/BAYESIAN-PILOT-ERROR-ANALYSIS.md").is_file())

    def test_wave5_data_05_09_10_11_12_14_15(self):
        ds = build_dataset(seed=42)
        bp_splits = ds["tasks"]["BP"]
        bs_splits = ds["tasks"]["BS"]

        # DATA-05 metamorphic relations
        meta_res = validate_metamorphic_relations(bs_splits["test"][0])
        self.assertTrue(meta_res["all_metamorphic_relations_passed"])
        with self.assertRaisesRegex(ValueError, "assume_conditional_iid=False"):
            validate_metamorphic_relations(bs_splits["test"][0], assume_conditional_iid=False)

        # DATA-09 rational representations round-trip
        for mode in ("reduced_fraction", "num_den_fields", "scaled_integer_bps"):
            enc = encode_rational_representation("3/4", mode=mode)
            dec = decode_rational_representation(enc, mode=mode)
            self.assertEqual(dec, "3/4")
        with self.assertRaisesRegex(ValueError, "cannot be represented exactly in basis points"):
            encode_rational_representation("1/3", mode="scaled_integer_bps")

        # DATA-10 multi-step derivation supervision catches wrong intermediate step even when final matches
        # Q:BS:p=1/2;a=2/3;b=1/3;e=10;P= -> step 1 (e=1) is 2/3, step 2 (e=10) returns to 1/2
        prompt_bs = "Q:BS:p=1/2;a=2/3;b=1/3;e=10;P="
        good_deriv = verify_multistep_bayesian_derivation(prompt_bs, ["2/3", "1/2"], "1/2")
        self.assertEqual(good_deriv["status"], "VALID_DERIVATION_AND_FINAL_ANSWER")
        bad_inter = verify_multistep_bayesian_derivation(prompt_bs, ["1/5", "1/2"], "1/2")
        self.assertEqual(bad_inter["status"], "INTERMEDIATE_STEP_MISMATCH_DESPITE_CORRECT_FINAL")
        self.assertFalse(bad_inter["all_steps_verified"])

        # DATA-11 negative examples are never equal to gold
        negs = generate_interpretable_negative_examples(bp_splits["test"][0])
        self.assertEqual(len(negs), 4)
        self.assertTrue(all(n["negative_answer"] != n["correct_gold"] for n in negs))

        # DATA-12 contributor dataset submission validation
        clean_sub = {
            "dataset_id": "contrib-bp-mini",
            "license_spdx": "Apache-2.0",
            "generator_module_or_script": "aix_zero/bayesian_tasks.py",
            "splits": {
                "train": [bp_splits["train"][0]],
                "validation": [bp_splits["validation"][0]],
                "test": [bp_splits["test"][0]],
            },
        }
        self.assertTrue(validate_contributor_dataset_submission(clean_sub)["accepted_for_queue"])
        bad_sub = dict(clean_sub, license_spdx="PROPRIETARY-RESTRICTED")
        with self.assertRaisesRegex(ValueError, "unsupported or missing license_spdx"):
            validate_contributor_dataset_submission(bad_sub)

        # DATA-14 reproducible data inspection report without withheld test answers
        rep = build_reproducible_data_inspection_report(bp_splits, dataset_id="bp-v1")
        self.assertFalse(rep["contains_withheld_test_answers"])
        self.assertNotIn("top_answers", rep["splits"]["test"])
        self.assertIn("top_answers", rep["splits"]["train"])

        # DATA-15 cross-domain composition pilot
        comp = build_cross_domain_composition_pilot()
        self.assertTrue(comp["independent_oracle_verified"])
        self.assertEqual(len(comp["cases"]), 3)

    def test_wave6_data_16_and_train_03_04_05_06_08_09(self):
        import torch
        from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
        from aix_zero.validation_training_studies import (
            ablate_frozen_backbone_adaptation_scopes,
            compare_answer_only_vs_derivation_supervision,
            compare_curriculum_order_vs_shuffled_control,
            compare_rational_output_representations,
            evaluate_frozen_verifier_head_ranking,
            run_bounded_multiseed_study,
        )

        # DATA-16 M1 dataset readiness audit doc
        self.assertTrue((ROOT / "docs/M1-DATASET-READINESS-AUDIT.md").is_file())

        ds = build_dataset(seed=42)
        bp_splits = ds["tasks"]["BP"]

        # TRAIN-03 answer-only vs derivation supervision on validation
        t03 = compare_answer_only_vs_derivation_supervision(bp_splits, steps=4)
        self.assertFalse(t03["consumes_final_test_budget"])
        self.assertTrue(t03["selected_on_validation_only"])

        # TRAIN-04 rational output representations comparison
        val_sub = bp_splits["validation"][:2]
        t04 = compare_rational_output_representations(
            val_sub,
            {
                "reduced_fraction": [str(val_sub[0]["gold"]), "5/4"],
                "num_den_fields": [encode_rational_representation(str(val_sub[0]["gold"]), "num_den_fields"), "bad"],
            },
        )
        self.assertEqual(t04["representations"]["reduced_fraction"]["exact_value_accuracy"], 0.5)
        self.assertEqual(t04["representations"]["reduced_fraction"]["invalid_probability_rate"], 0.5)

        # TRAIN-05 curriculum order vs shuffled control
        t05 = compare_curriculum_order_vs_shuffled_control(bp_splits, steps=4, seed=42)
        self.assertTrue(t05["identical_example_multiset"])
        self.assertFalse(t05["consumes_final_test_budget"])

        # TRAIN-06 frozen-backbone adaptation scope ablation
        tiny_model = AIXZeroModel(
            AIXZeroConfig(vocab_size=32, d_model=16, n_heads=2, n_kv_heads=2, d_ff=32, max_seq_len=16, use_moe=True, n_experts=2, moe_top_k=1)
        )
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        ab_exp = ablate_frozen_backbone_adaptation_scopes(tiny_model, ids, scope="single_routed_expert", target_expert_slot=0)
        self.assertTrue(ab_exp["frozen_tensors_unchanged"])
        self.assertTrue(ab_exp["attributes_improvement_to_expert_only"])
        ab_head = ablate_frozen_backbone_adaptation_scopes(tiny_model, ids, scope="expert_plus_lm_head", target_expert_slot=0)
        self.assertTrue(ab_head["frozen_tensors_unchanged"])
        self.assertFalse(ab_head["attributes_improvement_to_expert_only"])

        # TRAIN-08 frozen verifier head ranking & calibration
        t08 = evaluate_frozen_verifier_head_ranking(
            [
                {
                    "gold": "1/2",
                    "candidates": [
                        {"prediction": "1/3", "verifier_confidence": 0.2},
                        {"prediction": "1/2", "verifier_confidence": 0.9},
                    ],
                }
            ]
        )
        self.assertEqual(t08["greedy_accuracy"], 0.0)
        self.assertEqual(t08["verifier_selected_accuracy"], 1.0)
        self.assertTrue(t08["verifier_improves_over_greedy"])

        # TRAIN-09 bounded multi-seed study with compute cap and failure tracking
        with self.assertRaisesRegex(ValueError, "exceed declared max_total_steps_cap"):
            run_bounded_multiseed_study(
                seeds=[1, 2, 3], steps_per_seed=50, max_total_steps_cap=100, runner_fn=lambda s: {"validation_score": 0.5}
            )
        t09 = run_bounded_multiseed_study(
            seeds=[42, 43, 44],
            steps_per_seed=10,
            max_total_steps_cap=30,
            runner_fn=lambda s: (_ for _ in ()).throw(RuntimeError("seed failed")) if s == 44 else {"validation_score": 0.2 * (s - 40)},
        )
        self.assertEqual(t09["completed_seed_count"], 2)
        self.assertEqual(t09["failed_seed_count"], 1)

    def test_wave7_train_07_11_13_15_and_expert_07(self):
        import tempfile
        import torch
        from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
        from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
        from aix_zero.validation_training_studies import (
            compare_specialist_initialization_strategies,
            evaluate_completion_and_reasoning_loss_weighting,
            evaluate_experiment_stopping_policy,
            run_fresh_final_specialist_evaluation,
        )

        tiny_model = AIXZeroModel(
            AIXZeroConfig(vocab_size=32, d_model=16, n_heads=2, n_kv_heads=2, d_ff=32, max_seq_len=16, use_moe=True, n_experts=2, moe_top_k=1)
        )
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

        # TRAIN-07 completion & reasoning loss weighting + gold leakage guard
        with self.assertRaisesRegex(ValueError, "gold answer leaked"):
            evaluate_completion_and_reasoning_loss_weighting(
                tiny_model,
                ids,
                prompt_end_idx=3,
                reasoning_end_idx=5,
                inference_prompt_text="Q:BP:p=1/2->H=1?<ans>4/5</ans>",
                gold_answer_text="4/5",
            )
        t07 = evaluate_completion_and_reasoning_loss_weighting(
            tiny_model, ids, prompt_end_idx=3, reasoning_end_idx=5, mask_prompt_tokens=True
        )
        self.assertIn("prompt_loss", t07)
        self.assertIn("reasoning_step_loss", t07)
        self.assertIn("final_answer_loss", t07)
        self.assertGreater(t07["grad_l2_norm"], 0.0)

        # TRAIN-11 specialist initialization strategy comparison
        init_copy = compare_specialist_initialization_strategies(
            tiny_model, ids, strategy="copy_existing_expert", expert_slot=0, steps=2
        )
        init_fresh = compare_specialist_initialization_strategies(
            tiny_model, ids, strategy="fresh_kaiming_uniform_seeded", expert_slot=0, init_seed=99, steps=2
        )
        self.assertNotEqual(
            init_copy["initialization_provenance"]["initial_weight_sha256"],
            init_fresh["initialization_provenance"]["initial_weight_sha256"],
        )
        self.assertEqual(
            init_copy["controlled_protocol"]["trainable_parameter_count"],
            init_fresh["controlled_protocol"]["trainable_parameter_count"],
        )

        # TRAIN-13 fresh final specialist evaluation
        fin_eval = run_fresh_final_specialist_evaluation(
            partition_id="final_reserved_test_v1",
            candidate_scores=[0.1, 0.0, 0.1],
            control_scores={
                "base_model": [0.0, 0.0, 0.0],
                "majority_class": [0.5, 0.5, 0.5],
                "prior_only": [0.125, 0.125, 0.125],
                "untrained_explicit_route": [0.0, 0.0, 0.0],
            },
            min_margin_over_controls=0.05,
        )
        self.assertEqual(fin_eval["promotion_status"], "REJECTED")
        self.assertEqual(fin_eval["strongest_control"], "majority_class")

        # TRAIN-15 experiment stopping and abandonment policy
        with self.assertRaisesRegex(ValueError, "Silent hypothesis reset rejected"):
            evaluate_experiment_stopping_policy(
                hypothesis_id="H-BP-01",
                run_history=[{"run_id": "r1", "outcome": "REJECTED"}],
                hypothesis_reset_attempted=True,
            )
        with self.assertRaisesRegex(ValueError, "maintainer_compute_justification is required"):
            evaluate_experiment_stopping_policy(
                hypothesis_id="H-BP-01",
                run_history=[{"run_id": "r1", "outcome": "REJECTED"}],
                request_additional_compute=True,
            )
        stop_res = evaluate_experiment_stopping_policy(
            hypothesis_id="H-BP-01",
            run_history=[{"run_id": "r1", "outcome": "REJECTED"}, {"run_id": "r2", "outcome": "REJECTED"}],
        )
        self.assertEqual(stop_res["branch_status"], "REJECTED")
        self.assertEqual(stop_res["visible_failed_run_count"], 2)

        # EXPERT-07 bounded fallback policy for failed experts
        with tempfile.TemporaryDirectory() as tmp:
            bank = SparseExpertCartridgeBank(Path(tmp) / "bank")
            bank.export_all_domain_cartridges(tiny_model, {"DOM_0": {}}, {}, domain_expert_map={"DOM_0": 0})
            fb_missing = bank.execute_specialist_with_bounded_fallback(tiny_model, "MISSING_DOM")
            self.assertTrue(fb_missing["fallback_triggered"])
            self.assertEqual(fb_missing["fallback_reason"], "UNAVAILABLE_CARTRIDGE")
            self.assertTrue(fb_missing["base_preserved"])

            fb_loop = bank.execute_specialist_with_bounded_fallback(
                tiny_model, "DOM_0", attempted_chain=["DOM_0"]
            )
            self.assertEqual(fb_loop["fallback_reason"], "RECURSIVE_EXPERT_LOOP_PREVENTED")
            self.assertTrue(fb_loop["base_preserved"])

            fb_bad_out = bank.execute_specialist_with_bounded_fallback(
                tiny_model, "DOM_0", candidate_output="no-answer-tags"
            )
            self.assertEqual(fb_bad_out["fallback_reason"], "INVALID_SPECIALIST_OUTPUT")
            self.assertTrue(fb_bad_out["base_preserved"])

    def test_wave8_expert_06_08_11_15_and_train_16(self):
        import copy
        import tempfile
        import torch
        from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
        from aix_zero.sparse_cartridge_bank import (
            DeviceResidentExpertSlotManager,
            SparseExpertCartridgeBank,
            evaluate_learned_vs_explicit_routing,
        )
        from aix_zero.validation_training_studies import audit_m2_measured_learning_gain_before_scaling

        # TRAIN-16 M2 learning gain audit doc and programmatic gate
        self.assertTrue((ROOT / "docs/M2-LEARNING-GAIN-AUDIT.md").is_file())
        no_go = audit_m2_measured_learning_gain_before_scaling(
            validation_loss_decreased=True,
            held_out_exact_match_gain_over_controls=-0.125,
            independently_reproduced=True,
        )
        self.assertFalse(no_go["unlock_large_volunteer_training"])
        self.assertEqual(no_go["scaling_decision"], "NO_GO_FOR_LARGE_VOLUNTEER_TRAINING")

        # EXPERT-06 learned vs explicit routing comparison
        r_cmp = evaluate_learned_vs_explicit_routing(
            explicit_routing_metrics={
                "task_selection_accuracy": 1.0,
                "answer_accuracy": 0.85,
                "overhead_ms": 0.1,
                "abstention_rate": 0.0,
            },
            learned_routing_metrics={
                "task_selection_accuracy": 0.9,
                "answer_accuracy": 0.70,
                "overhead_ms": 0.4,
                "abstention_rate": 0.05,
            },
        )
        self.assertTrue(r_cmp["learned_router_quality_regression_detected"])
        self.assertFalse(r_cmp["learned_routing_approved"])

        tiny_model = AIXZeroModel(
            AIXZeroConfig(vocab_size=32, d_model=16, n_heads=2, n_kv_heads=2, d_ff=32, max_seq_len=16, use_moe=True, n_experts=2, moe_top_k=1)
        )
        with tempfile.TemporaryDirectory() as tmp:
            bank = SparseExpertCartridgeBank(Path(tmp) / "bank")
            m_good = bank.export_all_domain_cartridges(
                tiny_model, {"DOM_0": {}, "DOM_1": {}}, {}, domain_expert_map={"DOM_0": 0, "DOM_1": 1}
            )
            m_good_copy = copy.deepcopy(m_good)

            # EXPERT-08 compose experts with verified interfaces
            comp_bad = bank.compose_experts_with_verified_interfaces(
                [
                    {"stage_name": "belief_stage", "domain": "DOM_0", "input_contract_valid": True, "output_contract_valid": False},
                    {"stage_name": "decision_stage", "domain": "DOM_1", "input_contract_valid": True, "output_contract_valid": True},
                ]
            )
            self.assertFalse(comp_bad["pipeline_verified"])
            self.assertFalse(comp_bad["final_answer_marked_verified"])
            self.assertEqual(comp_bad["failed_stage"], "belief_stage")

            comp_ok = bank.compose_experts_with_verified_interfaces(
                [
                    {"stage_name": "belief_stage", "domain": "DOM_0", "input_contract_valid": True, "output_contract_valid": True},
                    {"stage_name": "decision_stage", "domain": "DOM_1", "input_contract_valid": True, "output_contract_valid": True},
                ]
            )
            self.assertTrue(comp_ok["pipeline_verified"])
            self.assertTrue(comp_ok["final_answer_marked_verified"])

            # EXPERT-11 bounded device-resident expert slots
            mgr = DeviceResidentExpertSlotManager(tiny_model, bank, max_resident_slots=1)
            s_info = mgr.replace_resident_slot(0, "DOM_0")
            self.assertEqual(s_info["resident_slot_count"], 1)
            self.assertGreater(s_info["total_stored_bank_parameters"], 0)
            with self.assertRaisesRegex(RuntimeError, "Resident slot cap"):
                mgr.replace_resident_slot(1, "DOM_1")

            # EXPERT-15 expert revocation and rollback
            rollback = bank.revoke_and_rollback_expert_version(
                "DOM_0",
                revocation_reason="regression_on_holdout",
                known_good_manifest=m_good_copy,
                active_inflight_sessions=1,
            )
            self.assertTrue(rollback["rollback_reproducible"])
            self.assertEqual(rollback["inflight_request_policy"], "DRAIN_INFLIGHT_SESSIONS_BLOCK_NEW_INSTALLS")


if __name__ == "__main__":
    unittest.main()
