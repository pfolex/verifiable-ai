"""Unit tests for dataset licensing and provenance records (`DATA-01` / Issue #51)."""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

from aix_zero.data_provenance import (
    DEFAULT_CATALOG_PATH,
    assert_public_volunteer_eligible,
    catalog_digest,
    load_provenance_catalog,
    validate_provenance_catalog,
)

ROOT = Path(__file__).resolve().parents[1]


class DataProvenanceTests(unittest.TestCase):
    def test_committed_provenance_catalog_is_valid(self):
        self.assertTrue(DEFAULT_CATALOG_PATH.exists())
        catalog = load_provenance_catalog(DEFAULT_CATALOG_PATH)
        self.assertGreaterEqual(len(catalog["datasets"]), 4)
        ids = {d["dataset_id"] for d in catalog["datasets"]}
        self.assertIn("rlvr-forge-procedural", ids)
        self.assertIn("aix-zero-v11-template-heldout", ids)
        self.assertIn("aix-zero-bayesian-pilot", ids)
        self.assertIn("aix-zero-kip-scholar-council-notes", ids)

    def test_permissive_procedural_datasets_pass_volunteer_gate(self):
        catalog = load_provenance_catalog(DEFAULT_CATALOG_PATH)
        rec = assert_public_volunteer_eligible("aix-zero-v11-template-heldout", catalog)
        self.assertEqual(rec["redistribution_status"], "verified_permissive")
        self.assertTrue(rec["public_volunteer_eligible"])

    def test_restricted_or_unknown_datasets_are_blocked_by_default(self):
        catalog = load_provenance_catalog(DEFAULT_CATALOG_PATH)
        with self.assertRaisesRegex(PermissionError, "blocked from public volunteer jobs"):
            assert_public_volunteer_eligible("aix-zero-kip-scholar-council-notes", catalog)
        with self.assertRaisesRegex(PermissionError, "Unregistered dataset"):
            assert_public_volunteer_eligible("unregistered-dataset", catalog)

    def test_invalid_provenance_combinations_are_rejected(self):
        catalog = load_provenance_catalog(DEFAULT_CATALOG_PATH)

        # Marking an unknown/restricted dataset as public_volunteer_eligible must fail validation
        bad = copy.deepcopy(catalog)
        for d in bad["datasets"]:
            if d["dataset_id"] == "aix-zero-kip-scholar-council-notes":
                d["public_volunteer_eligible"] = True
        bad["catalog_sha256"] = catalog_digest(bad)
        with self.assertRaisesRegex(ValueError, "cannot be public_volunteer_eligible"):
            validate_provenance_catalog(bad)

        # Marking personal data as public_volunteer_eligible must fail validation
        bad_pii = copy.deepcopy(catalog)
        bad_pii["datasets"][0]["contains_personal_data"] = True
        bad_pii["catalog_sha256"] = catalog_digest(bad_pii)
        with self.assertRaisesRegex(ValueError, "contains personal data"):
            validate_provenance_catalog(bad_pii)


if __name__ == "__main__":
    unittest.main()
