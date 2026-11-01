"""Strict paired comparisons of results bound to one public manifest (EVAL-005)."""
import copy
import json
import math
import statistics
from pathlib import Path

from jsonschema import Draft202012Validator
from artifact_io import canonical_json

from dataset_manifest import validate_manifest
from evaluation_result import validate_evaluation_result


def _known_total(samples, field):
    known = [sample[field] for sample in samples if field in sample]
    total = sum(known) if field.endswith("tokens") else math.fsum(known)
    return {"known_samples": len(known), "known_total": total,
            "total": total if len(known) == len(samples) else None}


def _metrics(samples):
    latency = sorted(sample["latency_ms"] for sample in samples)
    return {"mean_score": math.fsum(sample["score"] for sample in samples) / len(samples),
            "error_count": sum("error" in sample for sample in samples),
            "latency_ms": {"mean": statistics.fmean(latency), "median": statistics.median(latency),
                           "p95": latency[math.ceil(0.95 * len(latency)) - 1]},
            "tokens": {field: _known_total(samples, field) for field in
                       ("input_tokens", "output_tokens", "cached_input_tokens")},
            "cost_usd": _known_total(samples, "cost_usd")}


def _group(pairs, left, right):
    baseline = [left[pair["sample_id"]] for pair in pairs]
    candidate = [right[pair["sample_id"]] for pair in pairs]
    return {"sample_count": len(pairs), "baseline": _metrics(baseline), "candidate": _metrics(candidate),
            "mean_score_delta": math.fsum(pair["delta"] for pair in pairs) / len(pairs),
            "wins": sum(pair["delta"] > 0 for pair in pairs),
            "ties": sum(pair["delta"] == 0 for pair in pairs),
            "losses": sum(pair["delta"] < 0 for pair in pairs)}


def compare_results(baseline, candidate, manifest):
    validate_manifest(manifest)
    expected_dataset = dict(manifest["dataset"], manifest_sha256=manifest["manifest_sha256"])
    expected_ids = {case["sample_id"] for case in manifest["cases"]}
    for result in (baseline, candidate):
        validate_evaluation_result(result)
        if result["dataset"] != expected_dataset or result["scorer"] != manifest["scorer"]:
            raise ValueError("comparison requires identical manifest, dataset and scorer provenance")
        if {sample["sample_id"] for sample in result["samples"]} != expected_ids:
            raise ValueError("result sample IDs do not match manifest")
    left = {sample["sample_id"]: sample for sample in baseline["samples"]}
    right = {sample["sample_id"]: sample for sample in candidate["samples"]}
    pairs = [{"sample_id": case["sample_id"], "domain": case["input"]["domain"],
              "baseline_score": left[case["sample_id"]]["score"],
              "candidate_score": right[case["sample_id"]]["score"],
              "delta": right[case["sample_id"]]["score"] - left[case["sample_id"]]["score"],
              "baseline_error": "error" in left[case["sample_id"]],
              "candidate_error": "error" in right[case["sample_id"]]} for case in manifest["cases"]]
    report = {"schema_version": "1.0.0", "manifest_sha256": manifest["manifest_sha256"],
              "dataset": copy.deepcopy(manifest["dataset"]), "scorer": copy.deepcopy(manifest["scorer"]),
              "baseline": {key: copy.deepcopy(baseline[key]) for key in ("run_id", "model")},
              "candidate": {key: copy.deepcopy(candidate[key]) for key in ("run_id", "model")},
              "pairs": pairs, "overall": _group(pairs, left, right),
              "domains": {domain: _group([pair for pair in pairs if pair["domain"] == domain], left, right)
                          for domain in manifest["generation"]["domains"]}}
    validate_comparison(report)
    return report


def validate_comparison(report):
    canonical_json(report)  # Reject nonfinite numbers everywhere.
    schema = json.loads((Path(__file__).resolve().parent / "schemas/comparison-report-v1.schema.json").read_text())
    Draft202012Validator(schema).validate(report)
    pairs = report["pairs"]
    if len({pair["sample_id"] for pair in pairs}) != len(pairs):
        raise ValueError("duplicate comparison sample IDs")
    if set(report["domains"]) != {pair["domain"] for pair in pairs}:
        raise ValueError("comparison domain mismatch")
    for pair in pairs:
        if not math.isclose(pair["delta"], pair["candidate_score"] - pair["baseline_score"], abs_tol=1e-12):
            raise ValueError("incorrect paired delta")
    for domain, group in [(None, report["overall"]), *report["domains"].items()]:
        selected = [pair for pair in pairs if domain is None or pair["domain"] == domain]
        if group["sample_count"] != len(selected):
            raise ValueError("incorrect comparison count")
        for side in ("baseline", "candidate"):
            if not math.isclose(group[side]["mean_score"], math.fsum(pair[side + "_score"] for pair in selected) / len(selected), abs_tol=1e-12):
                raise ValueError("incorrect comparison mean")
            if group[side]["error_count"] != sum(pair[side + "_error"] for pair in selected):
                raise ValueError("incorrect comparison error count")
        if not math.isclose(group["mean_score_delta"], math.fsum(pair["delta"] for pair in selected) / len(selected), abs_tol=1e-12):
            raise ValueError("incorrect comparison delta")
        for field, condition in (("wins", lambda x: x > 0), ("ties", lambda x: x == 0), ("losses", lambda x: x < 0)):
            if group[field] != sum(condition(pair["delta"]) for pair in selected):
                raise ValueError("incorrect comparison win/tie/loss count")


def markdown_report(report):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines = ["# Evaluation comparison", "",
             f"Baseline: {clean(report['baseline']['model']['model_id'])}", "",
             f"Candidate: {clean(report['candidate']['model']['model_id'])}", "",
             f"Manifest: `{report['manifest_sha256']}`", "",
             "| Domain | N | Baseline | Candidate | Delta | Errors B/C | Mean latency ms B/C |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for domain, group in [("overall", report["overall"]), *report["domains"].items()]:
        b, c = group["baseline"], group["candidate"]
        lines.append(f"| {clean(domain)} | {group['sample_count']} | {b['mean_score']:.6f} | {c['mean_score']:.6f} | "
                     f"{group['mean_score_delta']:+.6f} | {b['error_count']}/{c['error_count']} | "
                     f"{b['latency_ms']['mean']:.2f}/{c['latency_ms']['mean']:.2f} |")
    lines += ["", "## Resource totals", "", "| Metric | Baseline | Candidate |", "| --- | ---: | ---: |"]
    for metric in ("input_tokens", "output_tokens", "cached_input_tokens", "cost_usd"):
        values = []
        for side in ("baseline", "candidate"):
            metrics = report["overall"][side]
            value = metrics["cost_usd"] if metric == "cost_usd" else metrics["tokens"][metric]
            values.append(str(value["total"]) if value["total"] is not None else
                          f"unknown (known subtotal {value['known_total']}, {value['known_samples']} samples)")
        lines.append(f"| {metric} | {values[0]} | {values[1]} |")
    lines += ["", "Scores include zero-score error samples. This descriptive report does not establish statistical improvement.", ""]
    return "\n".join(lines)
