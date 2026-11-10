"""Dataset licensing and provenance validation and eligibility gate (`DATA-01` / Issue #51).

Enforces `schemas/dataset-provenance-v1.schema.json` and ensures that datasets with
`redistribution_status` of `restricted` or `unknown`, or datasets containing personal
data or restricted teacher outputs, cannot be marked `public_volunteer_eligible` or
dispatched to public volunteer compute jobs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator

from artifact_io import canonical_json, write_json

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_SCHEMA = ROOT / "schemas/dataset-provenance-v1.schema.json"
DEFAULT_CATALOG_PATH = ROOT / "research/dataset-provenance-catalog.json"


def catalog_digest(catalog: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 over the provenance catalog (excluding `catalog_sha256`)."""
    unsigned = {k: v for k, v in catalog.items() if k != "catalog_sha256"}
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def validate_provenance_record(record: Dict[str, Any]) -> None:
    """Validate cross-field safety invariants on a single dataset provenance record."""
    status = record["redistribution_status"]
    eligible = record["public_volunteer_eligible"]

    if record["contains_personal_data"] and eligible:
        raise ValueError(
            f"Dataset {record['dataset_id']!r} contains personal data and cannot be public_volunteer_eligible"
        )
    if record["depends_on_restricted_teacher_output"] and status == "verified_permissive":
        raise ValueError(
            f"Dataset {record['dataset_id']!r} depends on restricted teacher output and cannot be verified_permissive"
        )
    if status != "verified_permissive" and eligible:
        raise ValueError(
            f"Dataset {record['dataset_id']!r} has redistribution_status={status!r} and cannot be public_volunteer_eligible"
        )
    if status == "verified_permissive" and record["license_spdx"] in ("UNKNOWN", "NONE", "RESTRICTED"):
        raise ValueError(
            f"Dataset {record['dataset_id']!r} is verified_permissive but has invalid license_spdx={record['license_spdx']!r}"
        )


def validate_provenance_catalog(catalog: Dict[str, Any]) -> None:
    """Validate a dataset provenance catalog against the JSON schema and policy invariants."""
    schema = json.loads(PROVENANCE_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(catalog)

    ids = [d["dataset_id"] for d in catalog["datasets"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dataset_id in provenance catalog")

    for record in catalog["datasets"]:
        validate_provenance_record(record)

    expected_sha = catalog_digest(catalog)
    if catalog["catalog_sha256"] != expected_sha:
        raise ValueError(
            f"Provenance catalog SHA-256 mismatch: expected {expected_sha}, got {catalog['catalog_sha256']}"
        )


def assert_public_volunteer_eligible(
    dataset_id: str,
    catalog: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fail closed unless `dataset_id` is registered, `verified_permissive`, and `public_volunteer_eligible`."""
    if catalog is None:
        catalog = load_provenance_catalog()
    else:
        validate_provenance_catalog(catalog)

    by_id = {d["dataset_id"]: d for d in catalog["datasets"]}
    if dataset_id not in by_id:
        raise PermissionError(f"Unregistered dataset {dataset_id!r} is blocked from public volunteer jobs")
    record = by_id[dataset_id]
    if (
        record["redistribution_status"] != "verified_permissive"
        or not record["public_volunteer_eligible"]
        or record["contains_personal_data"]
        or record["depends_on_restricted_teacher_output"]
    ):
        raise PermissionError(
            f"Dataset {dataset_id!r} (status={record['redistribution_status']!r}) is blocked from public volunteer jobs"
        )
    return copy.deepcopy(record)


def make_provenance_catalog(datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
    catalog: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "datasets": copy.deepcopy(datasets),
    }
    catalog["catalog_sha256"] = catalog_digest(catalog)
    validate_provenance_catalog(catalog)
    return catalog


def load_provenance_catalog(path: Path = DEFAULT_CATALOG_PATH) -> Dict[str, Any]:
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_provenance_catalog(catalog)
    return catalog


def save_provenance_catalog(catalog: Dict[str, Any], path: Path = DEFAULT_CATALOG_PATH) -> None:
    validate_provenance_catalog(catalog)
    write_json(catalog, path)
