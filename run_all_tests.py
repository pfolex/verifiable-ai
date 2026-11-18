#!/usr/bin/env python3
"""Unified ecosystem test runner across `aix_zero_tests`, root `tests`, and all 6 sub-repositories (`SYMBIOSIS-03` / Issue #259)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent
SUBREPOS = ("verisci", "rlvr-forge", "branchbox", "mlx-tree", "steerlens", "everbench")


def run_ecosystem_test_suites(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    suites: List[Dict[str, Any]] = []
    all_passed = True

    # 1. Root suite (aix_zero_tests + tests)
    proc_root = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "aix_zero_tests", "tests"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    ok_root = proc_root.returncode == 0
    all_passed = all_passed and ok_root
    suites.append({"suite": "root_aix_zero_and_contracts", "passed": ok_root, "returncode": proc_root.returncode})

    # 2. Each standalone sub-repository in its own working directory
    for sub in SUBREPOS:
        sub_dir = repo_root / sub
        if not (sub_dir / "tests").exists():
            continue
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=str(sub_dir),
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0
        all_passed = all_passed and ok
        suites.append({"suite": sub, "passed": ok, "returncode": proc.returncode})

    return {
        "all_passed": all_passed,
        "suite_count": len(suites),
        "suites": suites,
    }


if __name__ == "__main__":
    report = run_ecosystem_test_suites()
    for s in report["suites"]:
        mark = "PASS" if s["passed"] else "FAIL"
        print(f"[{mark}] {s['suite']} (rc={s['returncode']})")
    sys.exit(0 if report["all_passed"] else 1)
