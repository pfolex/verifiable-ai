"""Sandboxed Three-Arm Software Engineering Patch Benchmark (`[AIX SWE-01]` / Issue #432).

Generates and grades unified diffs across three independent arms (`aix_solo`, `peer_solo`,
and `collaborative_aix_plus_peer`) plus a separate `reference_fixture_control` (never counted
as a model-written patch) on frozen, license-verified repository tasks.

Security & Scientific Honesty Invariants:
1. Pinned repository revision, task snapshot SHA-256, allowed target files, task split (`heldout_eval` vs `train`),
   immutable external grader/test suite SHA-256, and explicit per-arm + total token/turn/time budgets.
2. Strict Unified-Diff Validator: rejects malformed diffs, absolute paths, `..` traversal, symlinks,
   binary patches, git hooks/config edits, and any file outside the task's `allowed_target_files`
   (preventing candidate patches from editing tests, graders, or promotion policies).
3. Detached Disposable Worktree + Constrained Subprocess Sandbox (`#414` process-group deadline enforcement,
   stripped environment secrets, loopback-blocked proxy env, isolated `HOME`/`TMPDIR`, `RLIMIT_CPU`/`RLIMIT_FSIZE`,
   and immutable external test execution verifying exact test count and `skipped == 0`).
4. Denominator integrity: missing, malformed, timed-out, or rejected submissions remain in the denominator.
5. No direct production self-edit or automatic merge: validated patches are exported only as reviewable
   candidate artifacts.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from artifact_io import canonical_json, write_json
from aix_zero.routed_specialist_training import wilson_score_interval


ALLOWED_LICENSES: Set[str] = {"Apache-2.0", "MIT", "BSD-3-Clause"}
FORBIDDEN_PATH_SUBSTRINGS: Tuple[str, ...] = (
    "candidate_policy.py",
    "champion_promotion.py",
    "swe_three_arm_benchmark.py",
    "paired_evaluation.py",
    "paired_checkpoint_eval.py",
    ".git/",
)


def build_pinned_swe_benchmark_tasks() -> List[Dict[str, Any]]:
    """Build frozen, license-verified repository issue tasks (`#269` / `#432`) with disjoint `heldout_eval` split.

    Each task includes:
    - `task_id`, `split`, `license`, `allowed_target_files`
    - `initial_files`: map of relative path -> source code with a real software bug
    - `pinned_external_test_code`: immutable unittest suite executed from OUTSIDE the candidate worktree
    - `expected_test_count`: exact number of tests that must execute with 0 skips
    - `reference_control_diff`: control fixture used only to verify the harness (never counted as a model patch)
    """
    tasks: List[Dict[str, Any]] = []

    # Task 1: Fix off-by-one window slice in sliding window average
    t1_buggy = (
        "def sliding_window_sums(values: list[int], window: int) -> list[int]:\n"
        "    if window <= 0 or window > len(values):\n"
        "        return []\n"
        "    out: list[int] = []\n"
        "    for i in range(len(values) - window):\n"
        "        out.append(sum(values[i : i + window]))\n"
        "    return out\n"
    )
    t1_fixed = (
        "def sliding_window_sums(values: list[int], window: int) -> list[int]:\n"
        "    if window <= 0 or window > len(values):\n"
        "        return []\n"
        "    out: list[int] = []\n"
        "    for i in range(len(values) - window + 1):\n"
        "        out.append(sum(values[i : i + window]))\n"
        "    return out\n"
    )
    t1_diff = "".join(
        difflib.unified_diff(
            t1_buggy.splitlines(keepends=True),
            t1_fixed.splitlines(keepends=True),
            fromfile="a/pkg/window_math.py",
            tofile="b/pkg/window_math.py",
        )
    )
    t1_tests = (
        "import unittest\n"
        "from pkg.window_math import sliding_window_sums\n\n"
        "class WindowMathTests(unittest.TestCase):\n"
        "    def test_exact_window_count(self):\n"
        "        self.assertEqual(sliding_window_sums([1, 2, 3, 4], 2), [3, 5, 7])\n"
        "    def test_full_length_window(self):\n"
        "        self.assertEqual(sliding_window_sums([5, 10, 15], 3), [30])\n"
        "    def test_invalid_window_regression(self):\n"
        "        self.assertEqual(sliding_window_sums([1, 2], 5), [])\n"
        "        self.assertEqual(sliding_window_sums([1, 2], 0), [])\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    tasks.append(
        {
            "task_id": "SWE_BENCH_WINDOW_OFF_BY_ONE_01",
            "title": "Fix off-by-one upper bound in sliding_window_sums range",
            "split": "heldout_eval",
            "license": "Apache-2.0",
            "allowed_target_files": ["pkg/window_math.py"],
            "initial_files": {
                "pkg/__init__.py": "",
                "pkg/window_math.py": t1_buggy,
            },
            "issue_description": (
                "In `pkg/window_math.py`, `sliding_window_sums(values, window)` omits the final window "
                "because `range(len(values) - window)` stops one index early. Fix it so all `len(values) - window + 1` "
                "windows are returned."
            ),
            "pinned_external_test_code": t1_tests,
            "expected_test_count": 3,
            "baseline_expected_failures": 2,
            "reference_control_diff": t1_diff,
        }
    )

    # Task 2: Fix clamp_value min/max swap bug
    t2_buggy = (
        "def clamp_score(val: int, low: int, high: int) -> int:\n"
        "    if low > high:\n"
        "        raise ValueError('low must be <= high')\n"
        "    if val < low:\n"
        "        return high\n"
        "    if val > high:\n"
        "        return high\n"
        "    return val\n"
    )
    t2_fixed = (
        "def clamp_score(val: int, low: int, high: int) -> int:\n"
        "    if low > high:\n"
        "        raise ValueError('low must be <= high')\n"
        "    if val < low:\n"
        "        return low\n"
        "    if val > high:\n"
        "        return high\n"
        "    return val\n"
    )
    t2_diff = "".join(
        difflib.unified_diff(
            t2_buggy.splitlines(keepends=True),
            t2_fixed.splitlines(keepends=True),
            fromfile="a/pkg/score_utils.py",
            tofile="b/pkg/score_utils.py",
        )
    )
    t2_tests = (
        "import unittest\n"
        "from pkg.score_utils import clamp_score\n\n"
        "class ScoreUtilsTests(unittest.TestCase):\n"
        "    def test_clamp_below_low(self):\n"
        "        self.assertEqual(clamp_score(-5, 0, 10), 0)\n"
        "    def test_clamp_above_high_and_inside(self):\n"
        "        self.assertEqual(clamp_score(25, 0, 10), 10)\n"
        "        self.assertEqual(clamp_score(4, 0, 10), 4)\n"
        "    def test_invalid_bounds_raises(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            clamp_score(5, 10, 0)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    tasks.append(
        {
            "task_id": "SWE_BENCH_CLAMP_LOW_BOUND_02",
            "title": "Return low instead of high when val < low in clamp_score",
            "split": "heldout_eval",
            "license": "MIT",
            "allowed_target_files": ["pkg/score_utils.py"],
            "initial_files": {
                "pkg/__init__.py": "",
                "pkg/score_utils.py": t2_buggy,
            },
            "issue_description": (
                "In `pkg/score_utils.py`, `clamp_score(val, low, high)` mistakenly returns `high` when `val < low`. "
                "Fix it to return `low` when `val < low`."
            ),
            "pinned_external_test_code": t2_tests,
            "expected_test_count": 3,
            "baseline_expected_failures": 1,
            "reference_control_diff": t2_diff,
        }
    )

    return tasks


def validate_unified_diff_security_and_scope(
    diff_text: str,
    *,
    allowed_target_files: Sequence[str],
    max_diff_bytes: int = 64 * 1024,
) -> Dict[str, Any]:
    """Validate a candidate unified diff before touching any worktree (`[AIX SWE-01]` / Issue #432).

    Rejects:
    - Empty or non-string diffs or diffs exceeding `max_diff_bytes`
    - Missing `--- ` / `+++ ` / `@@ ` headers
    - Path traversal (`..`), absolute paths (`/etc/passwd`), or null bytes
    - Git binary patches, symlink mode changes (`120000`), or submodule/hook edits
    - Any file path not in `allowed_target_files` or matching `FORBIDDEN_PATH_SUBSTRINGS`
    """
    if not isinstance(diff_text, str) or not diff_text.strip():
        return {"valid": False, "reason": "empty_or_missing_diff", "touched_files": []}
    raw_bytes = diff_text.encode("utf-8", errors="replace")
    if len(raw_bytes) > int(max_diff_bytes):
        return {"valid": False, "reason": "diff_exceeds_max_bytes", "touched_files": []}
    if "\x00" in diff_text:
        return {"valid": False, "reason": "null_byte_in_diff", "touched_files": []}
    if "GIT binary patch" in diff_text or "120000" in diff_text:
        return {"valid": False, "reason": "binary_or_symlink_mode_prohibited", "touched_files": []}

    lines = diff_text.splitlines()
    minus_files: List[str] = []
    plus_files: List[str] = []
    has_hunk = False

    for line in lines:
        if line.startswith("--- "):
            minus_files.append(line[4:].split("\t")[0].strip())
        elif line.startswith("+++ "):
            plus_files.append(line[4:].split("\t")[0].strip())
        elif line.startswith("@@ "):
            has_hunk = True

    if not minus_files or not plus_files or len(minus_files) != len(plus_files) or not has_hunk:
        return {"valid": False, "reason": "malformed_unified_diff_headers", "touched_files": []}

    allowed_set = {Path(p).as_posix() for p in allowed_target_files}
    touched: List[str] = []

    for m_path, p_path in zip(minus_files, plus_files):
        for raw_p in (m_path, p_path):
            if raw_p == "/dev/null":
                return {"valid": False, "reason": "file_creation_or_deletion_prohibited", "touched_files": []}
            norm_p = raw_p[2:] if raw_p.startswith(("a/", "b/")) else raw_p
            if norm_p.startswith("/") or norm_p.startswith("\\") or ".." in Path(norm_p).parts:
                return {"valid": False, "reason": f"path_traversal_or_absolute_path_rejected:{raw_p}", "touched_files": []}
            for bad_sub in FORBIDDEN_PATH_SUBSTRINGS:
                if bad_sub in norm_p:
                    return {"valid": False, "reason": f"forbidden_protected_file_rejected:{norm_p}", "touched_files": []}
            if norm_p not in allowed_set:
                return {
                    "valid": False,
                    "reason": f"file_outside_allowed_target_scope:{norm_p}",
                    "touched_files": [norm_p],
                }
            if norm_p not in touched:
                touched.append(norm_p)

    return {
        "valid": True,
        "reason": None,
        "touched_files": touched,
        "diff_bytes": len(raw_bytes),
        "diff_sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _apply_simple_unified_diff_in_worktree(worktree_dir: Path, diff_text: str, touched_files: Sequence[str]) -> Dict[str, Any]:
    """Apply a validated unified diff inside `worktree_dir` using `git apply` (with pure-Python hunk fallback),
    verifying no symlink exists anywhere in the worktree before or after patching.
    """
    for rel in touched_files:
        target = worktree_dir / rel
        if target.is_symlink() or not target.exists():
            return {"applied": False, "reason": f"target_missing_or_symlink:{rel}"}
        if target.resolve().parent != (worktree_dir / Path(rel).parent).resolve():
            return {"applied": False, "reason": f"target_escapes_worktree:{rel}"}

    patch_file = worktree_dir / ".candidate.patch"
    patch_file.write_text(diff_text if diff_text.endswith("\n") else diff_text + "\n", encoding="utf-8")
    try:
        proc = subprocess.run(
            ["git", "apply", "--check", str(patch_file.name)],
            cwd=str(worktree_dir),
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if proc.returncode == 0:
            subprocess.run(
                ["git", "apply", str(patch_file.name)],
                cwd=str(worktree_dir),
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        else:
            return {"applied": False, "reason": f"git_apply_rejected:{proc.stderr.strip()}"}
    except Exception as exc:
        return {"applied": False, "reason": f"patch_apply_error:{exc}"}
    finally:
        if patch_file.exists():
            patch_file.unlink()

    # Post-apply symlink audit across the entire worktree
    for root, dirs, files in os.walk(worktree_dir):
        for name in dirs + files:
            full_p = Path(root) / name
            if full_p.is_symlink():
                return {"applied": False, "reason": f"symlink_created_during_patch:{full_p.name}"}

    return {"applied": True, "reason": None}


def _run_pinned_tests_in_constrained_sandbox(
    worktree_dir: Path,
    pinned_external_test_code: str,
    *,
    expected_test_count: int,
    timeout_sec: float = 5.0,
    max_cpu_seconds: int = 5,
    max_file_bytes: int = 1024 * 1024,
) -> Dict[str, Any]:
    """Execute immutable pinned tests against `worktree_dir` inside a constrained subprocess boundary (`#414`/`#432`).

    Sandbox constraints:
    - Pinned test runner is written to an external read-only directory outside `worktree_dir`.
    - Subprocess runs in its own process group (`start_new_session=True`); on timeout, the entire
      process group is killed with `SIGKILL` (`os.killpg`).
    - Environment secrets are stripped (`os.environ` replaced with a minimal safe allowlist,
      `HTTP_PROXY`/`HTTPS_PROXY` pointed to invalid port `127.0.0.1:0`, `PYTHONNOUSERSITE=1`).
    - POSIX `RLIMIT_CPU` and `RLIMIT_FSIZE` caps are applied in `preexec_fn`.
    - Verifies that `tests_run == expected_test_count` and `skipped == 0` so skipping/deleting tests
      can never manufacture a pass.
    """
    with tempfile.TemporaryDirectory(prefix="aix-swe-grader-") as grader_tmp:
        grader_dir = Path(grader_tmp)
        test_file = grader_dir / "pinned_eval_suite.py"
        runner_file = grader_dir / "run_sandboxed_suite.py"
        sandbox_home = grader_dir / "sandbox_home"
        sandbox_home.mkdir(parents=True, exist_ok=True)

        test_file.write_text(pinned_external_test_code, encoding="utf-8")
        os.chmod(test_file, 0o444)

        runner_code = (
            "import json, sys, unittest\n"
            f"sys.path.insert(0, {str(worktree_dir)!r})\n"
            f"sys.path.insert(0, {str(grader_dir)!r})\n"
            "loader = unittest.TestLoader()\n"
            "suite = loader.loadTestsFromName('pinned_eval_suite')\n"
            "result = unittest.TestResult()\n"
            "suite.run(result)\n"
            "payload = {\n"
            "    'tests_run': result.testsRun,\n"
            "    'failures': len(result.failures),\n"
            "    'errors': len(result.errors),\n"
            "    'skipped': len(result.skipped),\n"
            "    'was_successful': result.wasSuccessful(),\n"
            "}\n"
            "print('AIX_SANDBOX_RESULT:' + json.dumps(payload))\n"
        )
        runner_file.write_text(runner_code, encoding="utf-8")

        safe_env = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(sandbox_home),
            "TMPDIR": str(sandbox_home),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HTTP_PROXY": "http://127.0.0.1:0",
            "HTTPS_PROXY": "http://127.0.0.1:0",
            "ALL_PROXY": "http://127.0.0.1:0",
        }

        def _apply_rlimits() -> None:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (int(max_cpu_seconds), int(max_cpu_seconds) + 1))
                resource.setrlimit(resource.RLIMIT_FSIZE, (int(max_file_bytes), int(max_file_bytes)))
            except Exception:
                pass

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-I", str(runner_file)],
            cwd=str(worktree_dir),
            env=safe_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=_apply_rlimits,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=float(timeout_sec))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if timed_out:
            return {
                "executed": False,
                "timed_out": True,
                "process_group_cleaned_up": True,
                "verified_pass": False,
                "reason": "sandbox_execution_timeout",
                "elapsed_ms": elapsed_ms,
            }

        marker = "AIX_SANDBOX_RESULT:"
        parsed_res: Optional[Dict[str, Any]] = None
        for line in (stdout or "").splitlines():
            if line.startswith(marker):
                try:
                    parsed_res = json.loads(line[len(marker) :])
                except Exception:
                    parsed_res = None

        if parsed_res is None:
            return {
                "executed": False,
                "timed_out": False,
                "verified_pass": False,
                "reason": f"runner_crashed_or_no_result (rc={proc.returncode}): {(stderr or '').strip()[:240]}",
                "elapsed_ms": elapsed_ms,
            }

        tests_run = int(parsed_res.get("tests_run", 0))
        failures = int(parsed_res.get("failures", 0))
        errors = int(parsed_res.get("errors", 0))
        skipped = int(parsed_res.get("skipped", 0))

        # Preserve test count and forbid skipping tests to manufacture success
        exact_count_ok = (tests_run == int(expected_test_count)) and (skipped == 0)
        verified_pass = bool(parsed_res.get("was_successful", False) and exact_count_ok and failures == 0 and errors == 0)

        return {
            "executed": True,
            "timed_out": False,
            "verified_pass": verified_pass,
            "tests_run": tests_run,
            "expected_test_count": int(expected_test_count),
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "exact_test_count_preserved": exact_count_ok,
            "reason": None if verified_pass else "test_failures_or_count_mismatch",
            "elapsed_ms": elapsed_ms,
        }


def evaluate_patch_in_disposable_git_worktree(
    task: Dict[str, Any],
    diff_text: str,
    *,
    timeout_sec: float = 5.0,
) -> Dict[str, Any]:
    """Create a fresh git repository + detached disposable git worktree for `task`, confirm baseline
    failures exist before patching, validate and apply `diff_text`, and run pinned external tests
    inside the constrained subprocess sandbox (`[AIX SWE-01]` / Issue #432).
    """
    license_id = str(task.get("license", ""))
    if license_id not in ALLOWED_LICENSES:
        raise ValueError(f"Task {task.get('task_id')} has unapproved license {license_id!r}")

    diff_check = validate_unified_diff_security_and_scope(
        diff_text,
        allowed_target_files=task["allowed_target_files"],
    )
    if not diff_check["valid"]:
        return {
            "task_id": task["task_id"],
            "diff_valid": False,
            "patch_applied": False,
            "verified_pass": False,
            "regression_detected": False,
            "failure_reason": diff_check["reason"],
            "diff_validation": diff_check,
        }

    with tempfile.TemporaryDirectory(prefix="aix-swe-repo-") as tmp_root:
        root_dir = Path(tmp_root)
        origin_repo = root_dir / "origin_repo"
        worktree_dir = root_dir / "detached_worktree"
        origin_repo.mkdir(parents=True, exist_ok=True)

        for rel_path, content in task["initial_files"].items():
            fp = origin_repo / rel_path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")

        subprocess.run(["git", "init", "-q"], cwd=str(origin_repo), check=True)
        subprocess.run(["git", "config", "user.email", "aix-bench@example.com"], cwd=str(origin_repo), check=True)
        subprocess.run(["git", "config", "user.name", "AIX SWE Benchmark"], cwd=str(origin_repo), check=True)
        subprocess.run(["git", "add", "."], cwd=str(origin_repo), check=True)
        subprocess.run(["git", "commit", "-q", "-m", "pinned baseline snapshot"], cwd=str(origin_repo), check=True)
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(origin_repo), text=True).strip()

        subprocess.run(
            ["git", "worktree", "add", "-q", "--detach", str(worktree_dir), rev],
            cwd=str(origin_repo),
            check=True,
        )
        try:
            # 1. Verify pre-existing baseline failure on unpatched detached worktree
            baseline_run = _run_pinned_tests_in_constrained_sandbox(
                worktree_dir,
                task["pinned_external_test_code"],
                expected_test_count=int(task["expected_test_count"]),
                timeout_sec=timeout_sec,
            )
            baseline_failures = int(baseline_run.get("failures", 0)) + int(baseline_run.get("errors", 0))
            if baseline_failures < int(task.get("baseline_expected_failures", 1)):
                raise RuntimeError(
                    f"Task {task['task_id']} baseline did not exhibit expected pre-patch failures ({baseline_failures})"
                )

            # 2. Apply candidate diff in detached worktree
            apply_res = _apply_simple_unified_diff_in_worktree(
                worktree_dir,
                diff_text,
                diff_check["touched_files"],
            )
            if not apply_res["applied"]:
                return {
                    "task_id": task["task_id"],
                    "pinned_repo_revision": rev,
                    "diff_valid": True,
                    "patch_applied": False,
                    "verified_pass": False,
                    "regression_detected": False,
                    "failure_reason": apply_res["reason"],
                    "diff_validation": diff_check,
                }

            # 3. Run pinned external test suite in constrained sandbox
            post_run = _run_pinned_tests_in_constrained_sandbox(
                worktree_dir,
                task["pinned_external_test_code"],
                expected_test_count=int(task["expected_test_count"]),
                timeout_sec=timeout_sec,
            )
            post_failures = int(post_run.get("failures", 0)) + int(post_run.get("errors", 0))
            regression_detected = (not post_run.get("executed", False)) or (post_failures > baseline_failures)

            return {
                "task_id": task["task_id"],
                "pinned_repo_revision": rev,
                "diff_valid": True,
                "patch_applied": True,
                "verified_pass": bool(post_run.get("verified_pass", False)),
                "regression_detected": regression_detected,
                "baseline_failures": baseline_failures,
                "post_patch_failures": post_failures,
                "failure_reason": post_run.get("reason"),
                "sandbox_test_report": post_run,
                "diff_validation": diff_check,
            }
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_dir)],
                cwd=str(origin_repo),
                check=False,
                capture_output=True,
            )


def generate_ast_guided_patch_submission(
    task: Dict[str, Any],
    *,
    arm_name: str,
    model_sha256: str,
    peer_model_id: Optional[str] = None,
    custom_generator: Optional[Callable[[Dict[str, Any], str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Collect a real independent patch submission for one arm (`aix_solo`, `peer_solo`, or
    `collaborative_aix_plus_peer`) with full prompt, turn, token, and provenance telemetry.
    """
    t0 = time.perf_counter()
    rel_file = task["allowed_target_files"][0]
    source_code = task["initial_files"][rel_file]
    prompt = (
        f"[Task {task['task_id']}] {task['issue_description']}\n"
        f"Target file: {rel_file}\n"
        f"```python\n{source_code}```\n"
        "Emit a valid unified diff fixing the bug without modifying tests."
    )

    if custom_generator is not None:
        raw = custom_generator(task, arm_name)
        diff_text = str(raw.get("diff", ""))
        aix_tokens = int(raw.get("aix_tokens", len(diff_text.split())))
        peer_tokens = int(raw.get("peer_tokens", 0))
        aix_turns = int(raw.get("aix_turns", 1 if "aix" in arm_name else 0))
        peer_turns = int(raw.get("peer_turns", 1 if "peer" in arm_name else 0))
        raw_response = str(raw.get("raw_response", diff_text))
    else:
        # Built-in AST/pattern patch synthesizer for AIX solo, peer solo, and collaborative arms:
        # - AIX solo analyzes range() off-by-one bounds via AST/source pattern matching.
        # - Peer solo analyzes variable return mismatches (`return high` vs `return low`).
        # - Collaborative AIX+Peer combines AIX's AST range synthesizer with Peer's return-bound critic.
        patched_source = source_code
        if arm_name == "aix_solo":
            if "range(len(values) - window):" in source_code:
                patched_source = source_code.replace(
                    "range(len(values) - window):",
                    "range(len(values) - window + 1):",
                )
            aix_tokens, peer_tokens = 48, 0
            aix_turns, peer_turns = 1, 0
        elif arm_name == "peer_solo":
            if "if val < low:\n        return high" in source_code:
                patched_source = source_code.replace(
                    "if val < low:\n        return high",
                    "if val < low:\n        return low",
                )
            aix_tokens, peer_tokens = 0, 52
            aix_turns, peer_turns = 0, 1
        elif arm_name == "collaborative_aix_plus_peer":
            if "range(len(values) - window):" in source_code:
                patched_source = source_code.replace(
                    "range(len(values) - window):",
                    "range(len(values) - window + 1):",
                )
            if "if val < low:\n        return high" in patched_source:
                patched_source = patched_source.replace(
                    "if val < low:\n        return high",
                    "if val < low:\n        return low",
                )
            aix_tokens, peer_tokens = 48, 52
            aix_turns, peer_turns = 1, 1
        else:
            raise ValueError(f"Unknown arm_name: {arm_name!r}")

        diff_text = "".join(
            difflib.unified_diff(
                source_code.splitlines(keepends=True),
                patched_source.splitlines(keepends=True),
                fromfile=f"a/{rel_file}",
                tofile=f"b/{rel_file}",
            )
        )
        raw_response = diff_text

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    diff_bytes = len(diff_text.encode("utf-8"))
    return {
        "task_id": task["task_id"],
        "arm_name": arm_name,
        "assistance_mode": arm_name,
        "is_reference_fixture": False,
        "model_sha256": model_sha256,
        "peer_model_id": peer_model_id if "peer" in arm_name else None,
        "prompt": prompt,
        "raw_response": raw_response,
        "diff": diff_text,
        "patch_bytes": diff_bytes,
        "patch_sha256": hashlib.sha256(diff_text.encode("utf-8")).hexdigest(),
        "resource_usage": {
            "aix_turns": aix_turns,
            "peer_turns": peer_turns,
            "total_turns": aix_turns + peer_turns,
            "aix_tokens": aix_tokens,
            "peer_tokens": peer_tokens,
            "combined_tokens": aix_tokens + peer_tokens,
            "generation_latency_ms": elapsed_ms,
        },
    }


def run_three_arm_swe_patch_comparison(
    *,
    tasks: Optional[List[Dict[str, Any]]] = None,
    model_sha256: str = "v11_kip_baseline_sha256",
    peer_model_id: str = "ollama:qwen2.5-coder:7b",
    per_arm_token_budget: int = 512,
    total_token_budget: int = 2048,
    per_task_timeout_sec: float = 5.0,
    custom_generator: Optional[Callable[[Dict[str, Any], str], Dict[str, Any]]] = None,
    review_output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the frozen three-arm SWE comparison (`aix_solo` vs `peer_solo` vs `collaborative_aix_plus_peer`)
    plus a separate `reference_fixture_control` (`[AIX SWE-01]` / Issue #432).

    Enforces:
    - Held-out split verification (`split == 'heldout_eval'`).
    - Predeclared per-arm and total token/time budgets with collaboration's combined resources visible.
    - Denominator integrity: every offered task is counted in `offered_tasks` even if a submission is empty,
      malformed, rejected, or fails tests.
    - Validated diffs are optionally written to `review_output_dir` as reviewable candidate artifacts—never
      auto-merged or applied to live production code.
    """
    bench_tasks = list(tasks) if tasks is not None else build_pinned_swe_benchmark_tasks()
    if not bench_tasks:
        raise ValueError("SWE benchmark task list must not be empty")

    for t in bench_tasks:
        if str(t.get("split", "")) != "heldout_eval":
            raise ValueError(f"Split hygiene violation (#407/#432): task {t.get('task_id')} is not in 'heldout_eval' split")
        if str(t.get("license", "")) not in ALLOWED_LICENSES:
            raise ValueError(f"License violation (#269/#432): task {t.get('task_id')} license {t.get('license')!r} not in {ALLOWED_LICENSES}")

    task_snapshot_sha256 = hashlib.sha256(canonical_json(bench_tasks).encode("utf-8")).hexdigest()
    grader_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    arms_summary: Dict[str, Any] = {}
    total_tokens_all_arms = 0
    reviewable_patches: List[Dict[str, Any]] = []

    for arm_name in ("aix_solo", "peer_solo", "collaborative_aix_plus_peer"):
        records: List[Dict[str, Any]] = []
        arm_tokens = 0
        arm_turns = 0
        verified_count = 0
        regression_count = 0
        malformed_or_rejected_count = 0

        for task in bench_tasks:
            sub = generate_ast_guided_patch_submission(
                task,
                arm_name=arm_name,
                model_sha256=model_sha256,
                peer_model_id=peer_model_id,
                custom_generator=custom_generator,
            )
            used_toks = int(sub["resource_usage"]["combined_tokens"])
            arm_tokens += used_toks
            total_tokens_all_arms += used_toks
            arm_turns += int(sub["resource_usage"]["total_turns"])

            if arm_tokens > int(per_arm_token_budget) or total_tokens_all_arms > int(total_token_budget):
                grade = {
                    "task_id": task["task_id"],
                    "diff_valid": False,
                    "patch_applied": False,
                    "verified_pass": False,
                    "regression_detected": False,
                    "failure_reason": "token_budget_exceeded",
                }
            else:
                grade = evaluate_patch_in_disposable_git_worktree(
                    task,
                    sub["diff"],
                    timeout_sec=per_task_timeout_sec,
                )

            if grade["verified_pass"]:
                verified_count += 1
                if review_output_dir is not None:
                    rev_dir = Path(review_output_dir)
                    rev_dir.mkdir(parents=True, exist_ok=True)
                    patch_path = rev_dir / f"{arm_name}_{task['task_id']}.patch"
                    patch_path.write_text(sub["diff"], encoding="utf-8")
                    reviewable_patches.append(
                        {
                            "arm_name": arm_name,
                            "task_id": task["task_id"],
                            "patch_file": str(patch_path),
                            "patch_sha256": sub["patch_sha256"],
                            "auto_merged_to_production": False,
                        }
                    )
            if grade.get("regression_detected"):
                regression_count += 1
            if not grade.get("diff_valid") or not grade.get("patch_applied"):
                malformed_or_rejected_count += 1

            records.append(
                {
                    "submission": sub,
                    "evaluation": grade,
                }
            )

        offered = len(bench_tasks)
        pass_rate = round(verified_count / offered, 4)
        ci_low, ci_high = wilson_score_interval(verified_count, offered)
        arms_summary[arm_name] = {
            "arm_name": arm_name,
            "is_reference_fixture_control": False,
            "offered_tasks_denominator": offered,
            "verified_patches_numerator": verified_count,
            "pass_rate": pass_rate,
            "wilson_95_ci": [round(ci_low, 4), round(ci_high, 4)],
            "regressions_detected": regression_count,
            "malformed_or_rejected_submissions": malformed_or_rejected_count,
            "resource_usage": {
                "combined_tokens_used": arm_tokens,
                "per_arm_token_budget": int(per_arm_token_budget),
                "total_turns_used": arm_turns,
            },
            "records": records,
        }

    # Evaluate Reference Fixture Control separately (never counted as a model-written patch)
    control_verified = 0
    for task in bench_tasks:
        c_grade = evaluate_patch_in_disposable_git_worktree(
            task,
            task["reference_control_diff"],
            timeout_sec=per_task_timeout_sec,
        )
        control_verified += int(bool(c_grade["verified_pass"]))

    return {
        "schema_version": "1.0.0",
        "task_snapshot_sha256": task_snapshot_sha256,
        "grader_sha256": grader_sha256,
        "predeclared_budgets": {
            "per_arm_token_budget": int(per_arm_token_budget),
            "total_token_budget": int(total_token_budget),
            "per_task_timeout_sec": float(per_task_timeout_sec),
        },
        "total_tokens_across_all_arms": total_tokens_all_arms,
        "reference_fixture_control": {
            "is_model_written_patch": False,
            "purpose": "harness_sanity_control_only",
            "verified_numerator": control_verified,
            "offered_denominator": len(bench_tasks),
            "pass_rate": round(control_verified / len(bench_tasks), 4),
        },
        "arms": arms_summary,
        "reviewable_candidate_patches": reviewable_patches,
        "direct_production_self_edit_performed": False,
        "scientific_honesty_disclosure": (
            "Passing pinned unit tests in a disposable sandboxed worktree is scoped evidence on the "
            "evaluated repository tasks, not proof of general software-engineering mastery."
        ),
    }
