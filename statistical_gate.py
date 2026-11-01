"""Fixed-horizon paired improvement/non-regression gates (EVAL-006)."""
import copy
import hashlib
import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
from artifact_io import canonical_json
from comparison_report import validate_comparison


def validate_policy(policy):
    canonical_json(policy)
    schema = json.loads((Path(__file__).resolve().parent / "schemas/regression-policy-v1.schema.json").read_text())
    Draft202012Validator(schema).validate(policy)


def paired_interval(deltas, *, alpha, comparisons):
    """Simultaneous Hoeffding intervals for independent bounded [-1, 1] pairs."""
    if not deltas or not 0 < alpha < 1 or comparisons < 1:
        raise ValueError("nonempty deltas, valid alpha and comparison count are required")
    if any(not math.isfinite(delta) or not -1 <= delta <= 1 for delta in deltas):
        raise ValueError("paired deltas must be finite and bounded")
    mean = math.fsum(deltas) / len(deltas)
    radius = math.sqrt(2 * math.log(2 * comparisons / alpha) / len(deltas))
    return {"sample_count": len(deltas), "mean_delta": mean,
            "lower": max(-1.0, mean - radius), "upper": min(1.0, mean + radius)}


def assess_comparison(comparison, policy):
    validate_comparison(comparison)
    validate_policy(policy)
    pairs = comparison["pairs"]
    constraints = [("overall", pairs, policy["min_improvement"])]
    constraints += [(domain, [pair for pair in pairs if pair["domain"] == domain], -policy["non_regression_margin"])
                    for domain in comparison["domains"]]
    criteria = {}
    reasons = []
    for name, selected, threshold in constraints:
        interval = paired_interval([pair["delta"] for pair in selected], alpha=policy["alpha"], comparisons=len(constraints))
        decision = "INCONCLUSIVE"
        if policy["independent_samples"] and len(selected) >= policy["min_samples_per_domain"]:
            if interval["lower"] >= threshold:
                decision = "PASS"
            elif interval["upper"] < threshold:
                decision = "FAIL"
        criteria[name] = dict(interval, threshold=threshold, decision=decision)
    error_failure = False
    for name, selected, _ in constraints:
        for side in ("baseline", "candidate"):
            rate = sum(pair[side + "_error"] for pair in selected) / len(selected)
            if rate > policy["max_error_rate"]:
                reasons.append(f"{name}: {side} error rate exceeds policy")
                error_failure = True
    if not policy["independent_samples"]:
        reasons.append("independent sampling assumption has not been declared")
    if any(len(selected) < policy["min_samples_per_domain"] for _, selected, _ in constraints):
        reasons.append("insufficient samples for one or more constraints")
    decisions = [criterion["decision"] for criterion in criteria.values()]
    if error_failure or "FAIL" in decisions:
        decision = "FAIL"
    elif all(value == "PASS" for value in decisions):
        decision = "PASS"
    else:
        decision = "INCONCLUSIVE"
    if "FAIL" in decisions:
        reasons.append("a confidence upper bound is below its required threshold")
    if "INCONCLUSIVE" in decisions and not reasons:
        reasons.append("confidence bounds do not resolve every policy threshold")
    return {"schema_version": "1.0.0", "decision": decision,
            "method": "paired-hoeffding-union-bound", "policy": copy.deepcopy(policy),
            "manifest_sha256": comparison["manifest_sha256"],
            "comparison_sha256": hashlib.sha256(canonical_json(comparison).encode()).hexdigest(),
            "baseline": copy.deepcopy(comparison["baseline"]), "candidate": copy.deepcopy(comparison["candidate"]),
            "simultaneous_constraints": len(constraints), "criteria": criteria, "reasons": reasons}
