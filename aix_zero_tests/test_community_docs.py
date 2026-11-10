"""Verify community governance, conduct, and contributor quickstart documentation (`COMM-01`, `COMM-02`, `COMM-03`)."""
from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CommunityDocsTests(unittest.TestCase):
    def test_code_of_conduct_exists_and_covers_reporting_and_scope(self):
        path = ROOT / "CODE_OF_CONDUCT.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Expected Behavior", text)
        self.assertIn("Unacceptable Behavior", text)
        self.assertIn("Reporting Conduct or Safety Concerns", text)
        self.assertIn("Enforcement Ladder", text)
        self.assertIn("does **not** claim a staffed 24/7 trust-and-safety team", text)

    def test_governance_roster_lists_workstreams_and_opt_in_vacancies(self):
        path = ROOT / "GOVERNANCE.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        for ws in ("OVERALL", "BASE", "EVAL", "ARCH", "DATA", "DIST", "COMM"):
            self.assertIn(f"**`{ws}`**", text)
        self.assertIn("VACANT (Opt-In Open)", text)
        self.assertIn("explicit opt-in consent", text)

    def test_contributing_quickstart_covers_five_offline_paths(self):
        path = ROOT / "CONTRIBUTING.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Zero-Credential Offline Quickstart", text)
        self.assertIn("requirements-aix-zero.txt", text)
        self.assertIn("Path A: Code & Architecture", text)
        self.assertIn("Path B: Evaluation, Verifiers & Statistical Gates", text)
        self.assertIn("Path C: Datasets & Provenance", text)
        self.assertIn("Path D: Independent Reproduction & Documentation", text)
        self.assertIn("Path E: Opt-In Compute Contributions", text)


if __name__ == "__main__":
    unittest.main()
