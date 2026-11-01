"""Versioned, integrity-checked public VeriSci task manifests (EVAL-004)."""
import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from artifact_io import canonical_json, write_json
from ollama_provider import public_task

SCHEMA = Path(__file__).resolve().parent / "schemas/dataset-manifest-v1.schema.json"


def manifest_digest(manifest):
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()


def validate_manifest(manifest):
    schema = json.loads(SCHEMA.read_text())
    Draft202012Validator(schema).validate(manifest)
    ids = [case["sample_id"] for case in manifest["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate manifest sample IDs")
    for case in manifest["cases"]:
        if case["input"] != public_task(case["input"]):
            raise ValueError("manifest contains non-public task fields")
        if case["sample_id"] != case["input"]["task_id"]:
            raise ValueError("manifest sample ID does not match task ID")
    generation = manifest["generation"]
    if manifest["scorer"]["configuration"] != generation:
        raise ValueError("scorer configuration must match manifest generation")
    if len(generation["domains"]) != len(set(generation["domains"])):
        raise ValueError("manifest domains must be unique")
    expected = [(domain, generation["seed"] + offset)
                for domain in generation["domains"] for offset in range(generation["samples_per_domain"])]
    actual = [(case["input"]["domain"], case["sample_id"]) for case in manifest["cases"]]
    expected = [(domain, f"eval-{domain}-d{generation['difficulty']}-s{seed}") for domain, seed in expected]
    if actual != expected or any(case["input"]["difficulty"] != generation["difficulty"] for case in manifest["cases"]):
        raise ValueError("manifest cases do not match generation configuration")
    if manifest_digest(manifest) != manifest["manifest_sha256"]:
        raise ValueError("manifest hash mismatch")


def make_manifest(cases, dataset, scorer, generation):
    manifest = {"schema_version": "1.0.0", "dataset": copy.deepcopy(dataset),
                "scorer": copy.deepcopy(scorer), "generation": copy.deepcopy(generation),
                "cases": [{"sample_id": case["sample_id"], "input": public_task(case["input"])} for case in cases]}
    manifest["manifest_sha256"] = manifest_digest(manifest)
    validate_manifest(manifest)
    return manifest


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    return manifest


def save_manifest(manifest, path):
    validate_manifest(manifest)
    write_json(manifest, path)
