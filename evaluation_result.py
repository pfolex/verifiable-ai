"""Validate the v1 evaluation result schema and cross-sample invariants."""
import argparse
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas/evaluation-result-v1.schema.json"


def validate_evaluation_result(result):
    """Raise ValidationError or ValueError for an invalid result.

    Empty runs have mean score zero. Means tolerate floating-point rounding
    (relative tolerance 1e-9, absolute tolerance 1e-12).
    """
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER).validate(result)
    samples = result["samples"]
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs")
    if result["summary"]["sample_count"] != len(samples):
        raise ValueError("summary sample_count does not match samples")
    scores = [sample["score"] for sample in samples]
    mean = math.fsum(scores) / len(scores) if scores else 0.0
    reported_mean = result["summary"]["mean_score"]
    if not all(math.isfinite(score) for score in scores) or not math.isfinite(reported_mean):
        raise ValueError("scores and mean_score must be finite")
    if not math.isclose(mean, reported_mean, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("summary mean_score does not match samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    for artifact in parser.parse_args().artifacts:
        validate_evaluation_result(json.loads(artifact.read_text()))
        print(f"Valid: {artifact}")
