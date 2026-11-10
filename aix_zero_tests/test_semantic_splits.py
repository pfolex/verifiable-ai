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


if __name__ == "__main__":
    unittest.main()
