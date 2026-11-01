"""Run provider submissions through a scorer and persist validated v1 results."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import time
import uuid

from evaluation_result import validate_evaluation_result
from artifact_io import canonical_json, write_json


def run_benchmark(cases, *, dataset, provider, scorer, scorer_metadata, run_id=None, manifest=None):
    """Evaluate every case. Provider/scorer failures count as zero-score samples.

    Invalid adapter metadata, usage, or score values fail the run. They are
    programming/contract errors, rather than model failures to hide as zeroes.
    """
    cases = copy.deepcopy(list(cases))
    ids = [case["sample_id"] for case in cases]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("case sample IDs must be nonempty and unique")
    manifest_hash = hashlib.sha256(canonical_json(cases).encode()).hexdigest()
    if manifest is not None:
        from dataset_manifest import validate_manifest
        validate_manifest(manifest)
        if cases != manifest["cases"] or dataset != manifest["dataset"] or scorer_metadata != manifest["scorer"]:
            raise ValueError("run inputs do not match manifest")
        manifest_hash = manifest["manifest_sha256"]
    dataset = copy.deepcopy(dataset)
    dataset["manifest_sha256"] = manifest_hash
    result = {
        "schema_version": "1.0.0", "run_id": run_id or str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset, "model": copy.deepcopy(provider.model),
        "scorer": copy.deepcopy(scorer_metadata), "samples": [],
        "summary": {"sample_count": 0, "mean_score": 0},
    }
    # Fail bad provenance before invoking a potentially expensive provider.
    validate_evaluation_result(result)
    for case in cases:
        sample = {"sample_id": case["sample_id"], "output": "", "score": 0}
        started = time.perf_counter()
        try:
            response = provider.generate(case["sample_id"], copy.deepcopy(case["input"]))
        except Exception as exc:
            sample["error"] = f"provider: {type(exc).__name__}"
        else:
            sample["output"] = response.output
            sample.update(response.usage())
        sample["latency_ms"] = (time.perf_counter() - started) * 1000
        if "error" not in sample:
            # Validate adapter output/usage before passing it to the scorer.
            probe = dict(result, samples=[sample], summary={"sample_count": 1, "mean_score": 0})
            validate_evaluation_result(probe)
            try:
                sample["score"] = scorer(copy.deepcopy(case["input"]), response.output)
            except Exception as exc:
                sample["error"] = f"scorer: {type(exc).__name__}"
        result["samples"].append(sample)
    scores = [sample["score"] for sample in result["samples"]]
    result["summary"] = {"sample_count": len(scores),
                         "mean_score": math.fsum(scores) / len(scores) if scores else 0}
    # Unknown cost stays unknown; do not silently treat missing costs as free.
    if all("cost_usd" in sample for sample in result["samples"]):
        result["summary"]["total_cost_usd"] = math.fsum(s["cost_usd"] for s in result["samples"])
    validate_evaluation_result(result)
    return result


def save_result(result, path):
    """Validate before touching the destination; atomically replace a complete file."""
    validate_evaluation_result(result)
    write_json(result, path)
