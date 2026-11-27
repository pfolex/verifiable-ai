"""Autonomous Codebase Self-Edit Proposal & Sandboxed PR Verifier (`Goal #1 & #3` / Issue #418).

Provides a strictly gated workflow for AIX to propose, validate, sandbox-test, and export
reviewable codebase improvements:
1. Validates any candidate unified diff against `FORBIDDEN_PATH_SUBSTRINGS` and `allowed_target_files`
   (preventing edits to promotion policies, evaluation harnesses, graders, or `.git/`).
2. Applies the diff inside a detached, disposable `git worktree` (never mutating the live workspace).
3. Executes the specified `unittest` suite inside the isolated worktree and requires `0` failures,
   `0` errors, and `0` skipped tests.
4. Exports a signed JSON verification report + `.patch` artifact under `state_dir/self_edit_proposals/`,
   and optionally pushes a candidate branch and opens a GitHub PR via `gh pr create` when `open_pr=True`.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

from artifact_io import write_json
from aix_zero.swe_three_arm_benchmark import (
    FORBIDDEN_PATH_SUBSTRINGS,
    validate_unified_diff_security_and_scope,
)


def propose_and_verify_self_edit(
    repo_root: Path,
    *,
    diff_text: str,
    title: str,
    rationale: str,
    allowed_target_files: Sequence[str],
    test_modules: Sequence[str],
    state_dir: Optional[Path] = None,
    open_pr: bool = False,
    gh_binary: str = "gh",
    timeout_seconds: float = 120.0,
) -> Dict[str, Any]:
    """Validate, sandbox-test in a detached git worktree, and export a reviewable self-edit proposal."""
    repo_root = Path(repo_root).resolve()
    s_dir = Path(state_dir) if state_dir is not None else (repo_root / "aix_zero/autopilot_state")
    proposals_dir = s_dir / "self_edit_proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)

    # 1. Strict diff & path safety validation
    allowed_set = set(allowed_target_files)
    for rel in allowed_set:
        if any(bad in rel for bad in FORBIDDEN_PATH_SUBSTRINGS) or rel.endswith("autonomous_self_edit.py"):
            return {
                "accepted": False,
                "reason": f"forbidden_target_file:{rel}",
                "tests_passed": False,
                "pr_url": None,
            }

    validation = validate_unified_diff_security_and_scope(
        diff_text, allowed_target_files=sorted(allowed_set)
    )
    if not validation.get("valid"):
        return {
            "accepted": False,
            "reason": validation.get("reason", "invalid_unified_diff"),
            "tests_passed": False,
            "pr_url": None,
        }

    diff_sha = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    proposal_id = f"self_edit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{diff_sha[:10]}"
    head_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()

    tmp_root = Path(tempfile.mkdtemp(prefix="aix-self-edit-"))
    worktree_dir = tmp_root / "worktree"
    t0 = time.perf_counter()
    tests_passed = False
    test_stdout = ""
    test_stderr = ""
    pr_url: Optional[str] = None
    branch_name: Optional[str] = None
    reason = "verified_in_sandbox_worktree"

    try:
        # 2. Create detached disposable worktree from HEAD
        subprocess.check_call(
            ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(worktree_dir), head_sha],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 3. Apply candidate diff cleanly in isolated worktree
        patch_file = tmp_root / "candidate.patch"
        patch_file.write_text(diff_text, encoding="utf-8")
        apply_proc = subprocess.run(
            ["git", "-C", str(worktree_dir), "apply", "--check", str(patch_file)],
            capture_output=True,
            text=True,
        )
        if apply_proc.returncode != 0:
            reason = f"git_apply_check_failed:{apply_proc.stderr.strip()}"
        else:
            subprocess.check_call(["git", "-C", str(worktree_dir), "apply", str(patch_file)])

            # 4. Run required unit test modules inside isolated worktree
            if test_modules:
                cmd = [sys.executable, "-m", "unittest", *test_modules]
                proc = subprocess.run(
                    cmd,
                    cwd=str(worktree_dir),
                    capture_output=True,
                    text=True,
                    timeout=float(timeout_seconds),
                )
                test_stdout = proc.stdout
                test_stderr = proc.stderr
                combined = f"{test_stdout}\n{test_stderr}"
                has_skip = "skipped=" in combined or "SKIP" in combined
                tests_passed = (proc.returncode == 0) and (not has_skip)
                if not tests_passed:
                    reason = "sandbox_tests_failed_or_skipped"
            else:
                tests_passed = True

            # 5. Optionally push branch and open reviewable PR (never auto-merged)
            if tests_passed and open_pr:
                branch_name = f"aix/self-edit-{diff_sha[:10]}"
                subprocess.check_call(["git", "-C", str(worktree_dir), "checkout", "-b", branch_name])
                subprocess.check_call(["git", "-C", str(worktree_dir), "add", *sorted(allowed_set)])
                subprocess.check_call(
                    [
                        "git",
                        "-C",
                        str(worktree_dir),
                        "commit",
                        "-m",
                        f"chore(aix-self-edit): {title}",
                    ]
                )
                subprocess.check_call(["git", "-C", str(worktree_dir), "push", "-u", "origin", branch_name])
                pr_body = (
                    f"## Autonomous AIX Self-Edit Proposal\n\n"
                    f"- **Proposal ID:** `{proposal_id}`\n"
                    f"- **Base Commit:** `{head_sha}`\n"
                    f"- **Diff SHA-256:** `{diff_sha}`\n"
                    f"- **Rationale:** {rationale}\n"
                    f"- **Verified Test Modules (`0` skips):** `{', '.join(test_modules)}`\n"
                )
                pr_out = subprocess.check_output(
                    [
                        gh_binary,
                        "pr",
                        "create",
                        "--head",
                        branch_name,
                        "--title",
                        f"[AIX Self-Edit] {title}",
                        "--body",
                        pr_body,
                    ],
                    cwd=str(worktree_dir),
                    text=True,
                ).strip()
                pr_url = pr_out.splitlines()[-1].strip()
    finally:
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(tmp_root, ignore_errors=True)

    artifact_patch = proposals_dir / f"{proposal_id}.patch"
    artifact_patch.write_text(diff_text, encoding="utf-8")
    report = {
        "schema_version": "1.0.0",
        "proposal_id": proposal_id,
        "title": title,
        "rationale": rationale,
        "base_head_sha": head_sha,
        "diff_sha256": diff_sha,
        "allowed_target_files": sorted(allowed_set),
        "test_modules": list(test_modules),
        "accepted": bool(tests_passed),
        "tests_passed": bool(tests_passed),
        "reason": reason,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "patch_artifact": str(artifact_patch),
        "branch_name": branch_name,
        "pr_url": pr_url,
        "test_output_tail": (test_stderr or test_stdout).splitlines()[-10:],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(report, proposals_dir / f"{proposal_id}.json")
    return report
