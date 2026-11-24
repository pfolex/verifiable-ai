"""Unit tests for the Sandboxed Three-Arm SWE Patch Benchmark (`[AIX SWE-01]` / Issue #432)."""
from __future__ import annotations

import difflib
import os
from pathlib import Path
import tempfile
import unittest

from aix_zero.swe_three_arm_benchmark import (
    build_pinned_swe_benchmark_tasks,
    evaluate_patch_in_disposable_git_worktree,
    run_three_arm_swe_patch_comparison,
    validate_unified_diff_security_and_scope,
)


class SWEThreeArmBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = build_pinned_swe_benchmark_tasks()
        self.task1 = self.tasks[0]

    def test_malicious_and_out_of_scope_diffs_rejected_before_worktree_apply(self) -> None:
        allowed = ["pkg/window_math.py"]

        # 1. Path traversal (`..`)
        traversal_diff = (
            "--- a/../../etc/passwd\n"
            "+++ b/../../etc/passwd\n"
            "@@ -1 +1 @@\n"
            "-root\n"
            "+hacked\n"
        )
        r_trav = validate_unified_diff_security_and_scope(traversal_diff, allowed_target_files=allowed)
        self.assertFalse(r_trav["valid"])
        self.assertIn("path_traversal", r_trav["reason"])

        # 2. Editing grader/promotion policy or test file outside allowed_target_files
        policy_diff = (
            "--- a/aix_zero/candidate_policy.py\n"
            "+++ b/aix_zero/candidate_policy.py\n"
            "@@ -1 +1 @@\n"
            "-PASS = False\n"
            "+PASS = True\n"
        )
        r_pol = validate_unified_diff_security_and_scope(policy_diff, allowed_target_files=allowed)
        self.assertFalse(r_pol["valid"])
        self.assertIn("forbidden_protected_file_rejected", r_pol["reason"])

        test_edit_diff = (
            "--- a/tests/test_window.py\n"
            "+++ b/tests/test_window.py\n"
            "@@ -1 +1 @@\n"
            "-self.assertEqual(1, 2)\n"
            "+self.assertEqual(1, 1)\n"
        )
        r_test = validate_unified_diff_security_and_scope(test_edit_diff, allowed_target_files=allowed)
        self.assertFalse(r_test["valid"])
        self.assertIn("file_outside_allowed_target_scope", r_test["reason"])

        # 3. Symlink git mode 120000
        symlink_diff = (
            "diff --git a/pkg/window_math.py b/pkg/window_math.py\n"
            "new file mode 120000\n"
            "--- a/pkg/window_math.py\n"
            "+++ b/pkg/window_math.py\n"
            "@@ -1 +1 @@\n"
            "-def f(): pass\n"
            "+/etc/passwd\n"
        )
        r_sym = validate_unified_diff_security_and_scope(symlink_diff, allowed_target_files=allowed)
        self.assertFalse(r_sym["valid"])
        self.assertIn("symlink", r_sym["reason"])

        # 4. Malformed diff without hunk header
        r_mal = validate_unified_diff_security_and_scope("not a diff", allowed_target_files=allowed)
        self.assertFalse(r_mal["valid"])

    def test_sandbox_timeout_process_cleanup_secret_stripping_and_failing_patch(self) -> None:
        orig_src = self.task1["initial_files"]["pkg/window_math.py"]

        # 1. Patch introducing an infinite hang -> killed cleanly by process-group timeout
        hang_src = (
            "import time\n"
            "def sliding_window_sums(values: list[int], window: int) -> list[int]:\n"
            "    while True:\n"
            "        time.sleep(0.2)\n"
        )
        hang_diff = "".join(
            difflib.unified_diff(
                orig_src.splitlines(keepends=True),
                hang_src.splitlines(keepends=True),
                fromfile="a/pkg/window_math.py",
                tofile="b/pkg/window_math.py",
            )
        )
        res_hang = evaluate_patch_in_disposable_git_worktree(self.task1, hang_diff, timeout_sec=0.8)
        self.assertTrue(res_hang["diff_valid"])
        self.assertTrue(res_hang["patch_applied"])
        self.assertFalse(res_hang["verified_pass"])
        self.assertTrue(res_hang["sandbox_test_report"]["timed_out"])
        self.assertTrue(res_hang["sandbox_test_report"]["process_group_cleaned_up"])

        # 2. Patch attempting to read a host secret environment variable -> stripped by sandbox
        os.environ["AIX_TEST_HOST_SECRET_TOKEN"] = "SUPER_SECRET_DO_NOT_LEAK"
        try:
            secret_probe_src = (
                "import os\n"
                "def sliding_window_sums(values: list[int], window: int) -> list[int]:\n"
                "    if 'AIX_TEST_HOST_SECRET_TOKEN' in os.environ:\n"
                "        raise RuntimeError('HOST_SECRET_LEAKED')\n"
                "    if window <= 0 or window > len(values):\n"
                "        return []\n"
                "    return [sum(values[i : i + window]) for i in range(len(values) - window + 1)]\n"
            )
            secret_diff = "".join(
                difflib.unified_diff(
                    orig_src.splitlines(keepends=True),
                    secret_probe_src.splitlines(keepends=True),
                    fromfile="a/pkg/window_math.py",
                    tofile="b/pkg/window_math.py",
                )
            )
            res_secret = evaluate_patch_in_disposable_git_worktree(self.task1, secret_diff, timeout_sec=5.0)
            self.assertTrue(res_secret["verified_pass"])
        finally:
            os.environ.pop("AIX_TEST_HOST_SECRET_TOKEN", None)

        # 3. Regressing patch that breaks all tests
        bad_src = (
            "def sliding_window_sums(values: list[int], window: int) -> list[int]:\n"
            "    return [-999]\n"
        )
        bad_diff = "".join(
            difflib.unified_diff(
                orig_src.splitlines(keepends=True),
                bad_src.splitlines(keepends=True),
                fromfile="a/pkg/window_math.py",
                tofile="b/pkg/window_math.py",
            )
        )
        res_bad = evaluate_patch_in_disposable_git_worktree(self.task1, bad_diff, timeout_sec=5.0)
        self.assertFalse(res_bad["verified_pass"])
        self.assertTrue(res_bad["regression_detected"])

    def test_three_arm_swe_comparison_and_reviewable_artifact_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rev_dir = Path(tmp) / "reviewable_patches"
            report = run_three_arm_swe_patch_comparison(
                tasks=self.tasks,
                per_arm_token_budget=512,
                total_token_budget=2048,
                review_output_dir=rev_dir,
            )
            self.assertFalse(report["direct_production_self_edit_performed"])
            self.assertFalse(report["reference_fixture_control"]["is_model_written_patch"])
            self.assertEqual(report["reference_fixture_control"]["pass_rate"], 1.0)

            aix_arm = report["arms"]["aix_solo"]
            peer_arm = report["arms"]["peer_solo"]
            collab_arm = report["arms"]["collaborative_aix_plus_peer"]

            # Denominator is always 2 across all 3 arms
            self.assertEqual(aix_arm["offered_tasks_denominator"], 2)
            self.assertEqual(peer_arm["offered_tasks_denominator"], 2)
            self.assertEqual(collab_arm["offered_tasks_denominator"], 2)

            # AIX solo solves Task 1 (1/2), Peer solo solves Task 2 (1/2), Collaborative solves both (2/2)
            self.assertEqual(aix_arm["verified_patches_numerator"], 1)
            self.assertEqual(peer_arm["verified_patches_numerator"], 1)
            self.assertEqual(collab_arm["verified_patches_numerator"], 2)

            # Collaboration combined token and turn usage is explicitly visible
            self.assertEqual(
                collab_arm["resource_usage"]["combined_tokens_used"],
                aix_arm["resource_usage"]["combined_tokens_used"] + peer_arm["resource_usage"]["combined_tokens_used"],
            )
            self.assertGreaterEqual(len(report["reviewable_candidate_patches"]), 4)
            for p_meta in report["reviewable_candidate_patches"]:
                self.assertFalse(p_meta["auto_merged_to_production"])
                self.assertTrue(Path(p_meta["patch_file"]).exists())


if __name__ == "__main__":
    unittest.main()
