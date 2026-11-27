"""Unit tests for the sandboxed autonomous self-edit proposal & PR verifier."""
import difflib
from pathlib import Path
import subprocess
import tempfile
import unittest

from aix_zero.autonomous_self_edit import propose_and_verify_self_edit


class AutonomousSelfEditTests(unittest.TestCase):
    def _init_temp_repo(self, root: Path) -> None:
        subprocess.check_call(["git", "init", str(root)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "-C", str(root), "config", "user.email", "test@example.com"])
        subprocess.check_call(["git", "-C", str(root), "config", "user.name", "AIX Test"])
        (root / "pkg").mkdir(parents=True, exist_ok=True)
        (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (root / "pkg" / "calc.py").write_text("def double(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
        (root / "pkg" / "test_calc.py").write_text(
            "import unittest\n"
            "from pkg.calc import double\n\n"
            "class CalcTests(unittest.TestCase):\n"
            "    def test_double(self):\n"
            "        self.assertEqual(double(3), 6)\n",
            encoding="utf-8",
        )
        subprocess.check_call(["git", "-C", str(root), "add", "."])
        subprocess.check_call(["git", "-C", str(root), "commit", "-m", "initial"], stdout=subprocess.DEVNULL)

    def test_rejects_forbidden_policy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_temp_repo(root)
            res = propose_and_verify_self_edit(
                root,
                diff_text="--- a/aix_zero/candidate_policy.py\n+++ b/aix_zero/candidate_policy.py\n@@ -1 +1 @@\n-a\n+b\n",
                title="forbidden edit",
                rationale="should be blocked",
                allowed_target_files=["aix_zero/candidate_policy.py"],
                test_modules=[],
                state_dir=root / "state",
            )
            self.assertFalse(res["accepted"])
            self.assertIn("forbidden_target_file", res["reason"])

    def test_accepts_verified_patch_in_isolated_worktree_without_mutating_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_temp_repo(root)
            old_src = "def double(x: int) -> int:\n    return x + 1\n"
            new_src = "def double(x: int) -> int:\n    return x * 2\n"
            diff = "".join(
                difflib.unified_diff(
                    old_src.splitlines(keepends=True),
                    new_src.splitlines(keepends=True),
                    fromfile="a/pkg/calc.py",
                    tofile="b/pkg/calc.py",
                )
            )
            res = propose_and_verify_self_edit(
                root,
                diff_text=diff,
                title="fix double implementation",
                rationale="replace x + 1 with x * 2",
                allowed_target_files=["pkg/calc.py"],
                test_modules=["pkg/test_calc.py"],
                state_dir=root / "state",
            )
            self.assertTrue(res["accepted"])
            self.assertTrue(res["tests_passed"])
            # Live repo remains untouched until PR review
            self.assertEqual((root / "pkg" / "calc.py").read_text(encoding="utf-8"), old_src)
            self.assertTrue(Path(res["patch_artifact"]).exists())


if __name__ == "__main__":
    unittest.main()
