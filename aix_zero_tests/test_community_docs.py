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
        self.assertIn("Claim Proposal (`<TASK-CODE>`)", text)
        self.assertIn("No Unconsented Assignment or Review Promises", text)

    def test_license_and_provenance_policy_are_present(self):
        lic_path = ROOT / "LICENSE"
        pol_path = ROOT / "docs/LICENSE-AND-PROVENANCE-POLICY.md"
        self.assertTrue(lic_path.is_file())
        self.assertTrue(pol_path.is_file())
        self.assertIn("Apache License", lic_path.read_text(encoding="utf-8"))
        pol_text = pol_path.read_text(encoding="utf-8")
        self.assertIn("Apache-2.0", pol_text)
        self.assertIn("Explicit Exceptions: Datasets, Web Excerpts, and Checkpoints", pol_text)

    def test_wave3_replication_triage_reasoning_website_and_pr_templates(self):
        rep_doc = ROOT / "docs/REPLICATION-AND-TRIAGE.md"
        reason_doc = ROOT / "docs/REASONING-CAPABILITY-SPEC.md"
        web_doc = ROOT / "docs/WEBSITE-INFORMATION-ARCHITECTURE.md"
        pr_default = ROOT / ".github/pull_request_template.md"
        pr_exp = ROOT / ".github/PULL_REQUEST_TEMPLATE/experiment_or_research.md"
        pr_code = ROOT / ".github/PULL_REQUEST_TEMPLATE/code_or_docs.md"

        for p in (rep_doc, reason_doc, web_doc, pr_default, pr_exp, pr_code):
            self.assertTrue(p.is_file(), f"Missing {p}")

        self.assertIn("Independent Reproducer", rep_doc.read_text(encoding="utf-8"))
        self.assertIn("NON_IDENTIFIABLE", reason_doc.read_text(encoding="utf-8"))
        self.assertIn("Distributed coordinator: OFFLINE", web_doc.read_text(encoding="utf-8"))
        self.assertIn("Neural-Only Exact Match", pr_exp.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
