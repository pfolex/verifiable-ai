"""Unit tests for semantic problem identity and disjoint split verification (`DATA-02` / Issue #43)."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from aix_zero.bayesian_tasks import build_dataset
from aix_zero.knowledge_is_power import build_21_domain_kip_v11_curriculum
from aix_zero.semantic_splits import (
    SPLIT_ACCESS_POLICY,
    build_rotating_test_partition,
    canonical_problem_identity,
    compute_difficulty_factors,
    export_training_split_safe,
    partition_by_semantic_identity,
    semantic_identity_hash,
    verify_disjoint_splits,
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


if __name__ == "__main__":
    unittest.main()
