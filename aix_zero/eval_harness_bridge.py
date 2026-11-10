"""Bridge connecting AIX-Zero checkpoints and baseline reports to EVAL-001..006 (`EVAL-01`, `EVAL-02`, `EVAL-03`, `EVAL-06`).

Reuses `evaluation_result.py`, `provider_adapters.py`, `benchmark_runner.py`, and
`statistical_gate.py` so AIX-Zero evaluations validate against the repository's
canonical JSON schemas while:
1. Exporting multidimensional scorecards (`schemas/aix-scorecard-v1.schema.json`, `EVAL-02`),
2. Recording explicit `answer_source` (`neural_only`, `arithmetic_repaired`, `exact_solver`,
   `retrieved`, `external_teacher`, `heuristic_control`) and rejecting comparisons that mix
   execution modes or resource budgets (`EVAL-03`),
3. Scoring mathematically equivalent exact rationals and separately flagging invalid
   probabilities, zero denominators, and impossible conditioning events (`EVAL-06`).
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import Draft202012Validator

from artifact_io import canonical_json
from benchmark_runner import run_benchmark
from evaluation_result import validate_evaluation_result
from provider_adapters import ProviderResponse
from statistical_gate import paired_interval, validate_policy

from aix_zero.architecture import AIXZeroModel
from aix_zero.tokenizer import AIXZeroTokenizer
from aix_zero.train_from_scratch import extract_answer

ROOT = Path(__file__).resolve().parents[1]
SCORECARD_SCHEMA = ROOT / "schemas/aix-scorecard-v1.schema.json"

VALID_EXECUTION_MODES = ("neural_only", "symbolic_assisted")
VALID_ANSWER_SOURCES = (
    "neural_only",
    "arithmetic_repaired",
    "exact_solver",
    "retrieved",
    "external_teacher",
    "heuristic_control",
)


def mode_to_answer_source(mode: str) -> str:
    if mode == "neural_only":
        return "neural_only"
    if mode in ("symbolic_assisted", "arithmetic_repaired"):
        return "arithmetic_repaired"
    if mode in VALID_ANSWER_SOURCES:
        return mode
    raise ValueError(f"Unsupported execution mode / answer_source: {mode!r}")


class AIXZeroProvider:
    """Adapter implementing `provider_adapters.Provider` for an AIX-Zero model & tokenizer."""

    def __init__(
        self,
        model: AIXZeroModel,
        tokenizer: AIXZeroTokenizer,
        *,
        checkpoint_sha256: str,
        execution_mode: str = "neural_only",
        num_candidates: int = 2,
        max_new_tokens: int = 40,
        temperature: Optional[float] = None,
    ) -> None:
        if execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(f"execution_mode must be one of {VALID_EXECUTION_MODES}")
        if num_candidates < 1 or max_new_tokens < 1:
            raise ValueError("num_candidates and max_new_tokens must be positive")
        self._model = model
        self._tokenizer = tokenizer
        self.execution_mode = execution_mode
        self.answer_source = mode_to_answer_source(execution_mode)
        self.num_candidates = num_candidates
        self.max_new_tokens = max_new_tokens
        self.temperature = (0.35 if num_candidates > 1 else 0.0) if temperature is None else float(temperature)
        self.inflight_alu = execution_mode == "symbolic_assisted"

        self.model: Dict[str, Any] = {
            "provider": "aix-zero-local",
            "model_id": f"aix-zero-v11-{execution_mode}",
            "revision": checkpoint_sha256,
            "parameters": {
                "execution_mode": execution_mode,
                "answer_source": self.answer_source,
                "inflight_alu": self.inflight_alu,
                "num_candidates": num_candidates,
                "max_new_tokens": max_new_tokens,
                "temperature": self.temperature,
                "physical_parameters": model.count_parameters()["total_parameters"],
                "vocab_size": tokenizer.vocab_size,
                "vocab_sha256": tokenizer.vocab_sha256,
            },
            "adapter_version": "aix-zero-eval-bridge-1.0.0",
        }

    def generate(self, sample_id: str, task: Dict[str, Any]) -> ProviderResponse:
        prompt = task["prompt"]
        prompt_ids = self._tokenizer.encode(prompt, add_bos=True, add_eos=False)
        out_ids, _, _ = self._model.generate_verified(
            prompt_ids,
            eos_id=self._tokenizer.eos_id,
            max_new_tokens=self.max_new_tokens,
            num_candidates=self.num_candidates,
            temperature=self.temperature,
            tokenizer=self._tokenizer,
            inflight_alu=self.inflight_alu,
        )
        trace = self._tokenizer.decode(out_ids, skip_special=True)
        output_tokens = max(0, len(out_ids) - len(prompt_ids))
        return ProviderResponse(
            output=trace,
            input_tokens=len(prompt_ids),
            output_tokens=output_tokens,
            cost_usd=0.0,
        )


def score_aix_trace(task: Dict[str, Any], output: str) -> float:
    """Deterministic exact-match scorer for AIX-Zero `<ans>...</ans>` traces."""
    pred = extract_answer(output)
    return 1.0 if pred == str(task["gold"]) else 0.0


def rational_scorer_metadata(
    *,
    require_probability_range: bool = True,
    require_reduced: bool = False,
) -> Dict[str, Any]:
    """Return provenance metadata for the exact rational and probability-validity scorer (`EVAL-06`)."""
    return {
        "name": "aix-exact-rational-probability-scorer",
        "version": "1.0.0",
        "configuration": {
            "arithmetic": "exact_arbitrary_precision_fraction",
            "numeric_tolerance": 0.0,
            "require_probability_range": require_probability_range,
            "require_reduced": require_reduced,
        },
    }


def score_rational_or_probability(
    output_or_answer: str,
    gold: str,
    *,
    is_probability: bool = True,
    require_reduced: bool = False,
    conditioning_denominator: Optional[Fraction | int] = None,
) -> Dict[str, Any]:
    """Score mathematically equivalent rational answers and separately classify invalid probabilities (`EVAL-06`)."""
    if conditioning_denominator is not None and Fraction(conditioning_denominator) == 0:
        return {
            "score": 0.0,
            "valid_rational": False,
            "valid_probability": False,
            "status": "IMPOSSIBLE_CONDITIONING_EVENT",
            "parsed_fraction": None,
        }

    text = extract_answer(output_or_answer) if "<ans>" in output_or_answer else output_or_answer.strip()
    gold_frac = Fraction(gold.strip())

    # Check explicit division by zero pattern before general regex
    if re.fullmatch(r"\s*\-?\d+\s*/\s*0+\s*", text):
        return {
            "score": 0.0,
            "valid_rational": False,
            "valid_probability": False,
            "status": "ZERO_DENOMINATOR",
            "parsed_fraction": None,
        }

    m = re.fullmatch(r"\s*(\-?\d+)(?:\s*/\s*([1-9]\d*))?\s*", text)
    if not m:
        return {
            "score": 0.0,
            "valid_rational": False,
            "valid_probability": False,
            "status": "MALFORMED_FRACTION",
            "parsed_fraction": None,
        }

    num_s, den_s = m.group(1), m.group(2)
    raw_num = int(num_s)
    raw_den = int(den_s) if den_s is not None else 1
    parsed = Fraction(raw_num, raw_den)
    canonical_str = f"{parsed.numerator}/{parsed.denominator}" if parsed.denominator != 1 else str(parsed.numerator)
    is_reduced = (raw_num == parsed.numerator) and (raw_den == parsed.denominator)

    if is_probability and not (Fraction(0) <= parsed <= Fraction(1)):
        return {
            "score": 0.0,
            "valid_rational": True,
            "valid_probability": False,
            "status": "OUT_OF_RANGE_PROBABILITY",
            "parsed_fraction": canonical_str,
        }

    if require_reduced and not is_reduced:
        return {
            "score": 0.0,
            "valid_rational": True,
            "valid_probability": not is_probability or (Fraction(0) <= parsed <= Fraction(1)),
            "status": "UNREDUCED_FRACTION",
            "parsed_fraction": canonical_str,
        }

    matched = parsed == gold_frac
    return {
        "score": 1.0 if matched else 0.0,
        "valid_rational": True,
        "valid_probability": not is_probability or (Fraction(0) <= parsed <= Fraction(1)),
        "status": "VALID_EXACT_MATCH" if matched else "VALID_WRONG_VALUE",
        "parsed_fraction": canonical_str,
    }


def run_aix_benchmark(
    cases: List[Dict[str, Any]],
    provider: AIXZeroProvider,
    *,
    dataset_name: str = "aix-zero-v11-heldout",
    dataset_revision: str = "v11.0.0",
    split: str = "test",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run an `AIXZeroProvider` through `benchmark_runner.run_benchmark` (`EVAL-003`)."""
    formatted_cases = [
        {
            "sample_id": c.get("sample_id", f"{c['domain']}-{idx:03d}"),
            "input": {
                "domain": c["domain"],
                "prompt": c["prompt"],
                "gold": str(c["gold"]),
            },
        }
        for idx, c in enumerate(cases)
    ]
    dataset = {
        "name": dataset_name,
        "revision": dataset_revision,
        "split": split,
    }
    scorer_metadata = {
        "name": "aix-exact-answer-match",
        "version": "1.0.0",
        "configuration": {"answer_tag": "<ans>...</ans>"},
    }
    return run_benchmark(
        formatted_cases,
        dataset=dataset,
        provider=provider,
        scorer=score_aix_trace,
        scorer_metadata=scorer_metadata,
        run_id=run_id,
    )


def baseline_report_to_eval_results(
    baseline_report: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Convert a committed `aix_zero.benchmark_baseline` report into validated `EVAL-001` results."""
    ckpt_sha = baseline_report["checkpoint_sha256"]
    ds_info = baseline_report["dataset"]
    settings = baseline_report["settings"]
    results: Dict[str, Dict[str, Any]] = {}

    for mode_name, mode_data in baseline_report["modes"].items():
        answer_source = mode_to_answer_source(mode_name)
        samples = []
        for idx, row in enumerate(mode_data["samples"]):
            repeat = row.get("repeat", 0)
            sample_id = f"{row['domain']}-r{repeat}-{idx:03d}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "output": row["trace"],
                    "score": 1.0 if row["correct"] else 0.0,
                    "latency_ms": float(row["latency_ms"]),
                    "cost_usd": 0.0,
                }
            )
        scores = [s["score"] for s in samples]
        eval_result = {
            "schema_version": "1.0.0",
            "run_id": f"v11-baseline-{mode_name}-{ckpt_sha[:12]}",
            "created_at": baseline_report["created_at"],
            "dataset": {
                "name": ds_info["name"],
                "revision": str(ds_info["seed"]),
                "split": "test",
                "manifest_sha256": ds_info["sha256"],
            },
            "model": {
                "provider": "aix-zero-local",
                "model_id": f"aix-zero-v11-{mode_name}",
                "revision": ckpt_sha,
                "parameters": {
                    "execution_mode": mode_name,
                    "answer_source": answer_source,
                    "inflight_alu": mode_name == "symbolic_assisted",
                    "physical_parameters": baseline_report["physical_parameters"],
                    **copy.deepcopy(settings),
                },
                "adapter_version": "aix-zero-eval-bridge-1.0.0",
            },
            "scorer": {
                "name": "aix-exact-answer-match",
                "version": "1.0.0",
                "configuration": {"answer_tag": "<ans>...</ans>"},
            },
            "samples": samples,
            "summary": {
                "sample_count": len(samples),
                "mean_score": math.fsum(scores) / len(scores) if scores else 0.0,
                "total_cost_usd": 0.0,
            },
        }
        validate_evaluation_result(eval_result)
        results[mode_name] = eval_result
    return results


def assert_comparable_conditions(
    baseline_result: Dict[str, Any],
    candidate_result: Dict[str, Any],
    *,
    allow_cross_mode_ablation: bool = False,
) -> None:
    """Reject comparisons that silently mix assisted and unassisted conditions or mismatched budgets (`EVAL-03`)."""
    b_params = baseline_result.get("model", {}).get("parameters", {})
    c_params = candidate_result.get("model", {}).get("parameters", {})

    b_source = b_params.get("answer_source") or mode_to_answer_source(b_params.get("execution_mode", ""))
    c_source = c_params.get("answer_source") or mode_to_answer_source(c_params.get("execution_mode", ""))

    if not allow_cross_mode_ablation and b_source != c_source:
        raise ValueError(
            f"Cannot compare mixed answer_source conditions ({b_source!r} vs {c_source!r}) "
            "without allow_cross_mode_ablation=True"
        )

    for budget_key in ("num_candidates", "max_new_tokens", "device"):
        if budget_key in b_params and budget_key in c_params:
            if b_params[budget_key] != c_params[budget_key]:
                raise ValueError(
                    f"Incomparable sampling/compute budget for {budget_key!r}: "
                    f"{b_params[budget_key]!r} vs {c_params[budget_key]!r}"
                )


def validate_aix_scorecard(scorecard: Dict[str, Any]) -> None:
    """Validate a multidimensional AIX scorecard against `schemas/aix-scorecard-v1.schema.json` (`EVAL-02`)."""
    canonical_json(scorecard)
    schema = json.loads(SCORECARD_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER).validate(scorecard)


def build_multidimensional_scorecard(
    baseline_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Build and validate a multidimensional AIX scorecard (`EVAL-02`, `EVAL-03`) from a baseline report."""
    settings = baseline_report["settings"]
    cfg = baseline_report["config"]
    total_params = int(baseline_report["physical_parameters"])
    # Each routed expert in v11 has 3 * d_model * (d_ff // 2) = 3 * 128 * 128 = 49,152 params.
    # With n_experts=4 and moe_top_k=2, 2 experts are idle per token in sparse mode.
    idle_experts = max(0, int(cfg.get("n_experts", 4)) - int(cfg.get("moe_top_k", 2))) if cfg.get("use_moe") else 0
    expert_params = 3 * int(cfg.get("d_model", 128)) * max(64, int(cfg.get("d_ff", 256)) // 2)
    active_params = total_params - idle_experts * expert_params

    entries = []
    for mode_name, mode_data in baseline_report["modes"].items():
        samples = mode_data["samples"]
        latencies = [float(s["latency_ms"]) for s in samples]
        confidences = [float(s["verifier_confidence"]) for s in samples if "verifier_confidence" in s]
        ponders = [float(s["ponder_depth"]) for s in samples if "ponder_depth" in s]
        invalid_fmt = sum(1 for s in samples if not str(s.get("prediction", "")).strip())

        per_dom = {
            dom: round(d["correct"] / max(1, d["total"]), 6)
            for dom, d in sorted(mode_data["per_domain"].items())
        }
        entries.append(
            {
                "entry_id": f"v11-{mode_name}",
                "model_id": f"aix-zero-v11-{mode_name}",
                "checkpoint_sha256": baseline_report["checkpoint_sha256"],
                "answer_source": mode_to_answer_source(mode_name),
                "capability": {
                    "correct_samples": int(mode_data["correct"]),
                    "total_samples": int(mode_data["total"]),
                    "accuracy_ratio": round(mode_data["correct"] / max(1, mode_data["total"]), 6),
                    "per_domain_accuracy": per_dom,
                },
                "reliability": {
                    "error_samples": 0,
                    "invalid_format_samples": invalid_fmt,
                    "mean_verifier_confidence": round(statistics.fmean(confidences), 6) if confidences else None,
                },
                "latency_ms": {
                    "first_request_ms": float(mode_data["first_request"]["latency_ms"])
                    if "first_request" in mode_data
                    else None,
                    "p50_ms": float(mode_data["p50_response_ms"]),
                    "p95_ms": float(mode_data["p95_response_ms"]),
                    "mean_ms": round(statistics.fmean(latencies), 4),
                },
                "cost_and_energy": {
                    "total_cost_usd": 0.0,
                    "measured_energy_joules": None,
                },
                "resources": {
                    "physical_parameters": total_params,
                    "active_parameters_per_token": active_params,
                    "recurrent_steps": int(cfg.get("n_recurrent_steps", 3)),
                    "mean_ponder_depth": round(statistics.fmean(ponders), 4) if ponders else None,
                    "cached_cartridge_bytes": 0,
                },
            }
        )

    scorecard = {
        "schema_version": "1.0.0",
        "created_at": baseline_report["created_at"],
        "dataset_manifest_sha256": baseline_report["dataset"]["sha256"],
        "budget_contract": {
            "device": str(settings["device"]),
            "threads": int(settings["threads"]),
            "num_candidates": int(settings["num_candidates"]),
            "max_new_tokens": int(settings["max_new_tokens"]),
            "temperature": float(settings["temperature"]),
        },
        "entries": entries,
        "limitations": list(baseline_report["limitations"]),
    }
    validate_aix_scorecard(scorecard)
    return scorecard


def assess_aix_paired_results(
    baseline_result: Dict[str, Any],
    candidate_result: Dict[str, Any],
    domains_by_sample_id: Dict[str, str],
    policy: Dict[str, Any],
    *,
    allow_cross_mode_ablation: bool = False,
) -> Dict[str, Any]:
    """Run simultaneous Hoeffding paired intervals (`EVAL-006`) over two validated AIX-Zero runs."""
    validate_evaluation_result(baseline_result)
    validate_evaluation_result(candidate_result)
    validate_policy(policy)
    assert_comparable_conditions(
        baseline_result,
        candidate_result,
        allow_cross_mode_ablation=allow_cross_mode_ablation,
    )

    if baseline_result["dataset"] != candidate_result["dataset"]:
        raise ValueError("Paired assessment requires identical dataset provenance")

    left = {s["sample_id"]: s for s in baseline_result["samples"]}
    right = {s["sample_id"]: s for s in candidate_result["samples"]}
    if set(left) != set(right) or set(left) != set(domains_by_sample_id):
        raise ValueError("Sample IDs must match across baseline, candidate, and domain mapping")

    pairs = []
    for sample_id in sorted(left):
        b_s = left[sample_id]
        c_s = right[sample_id]
        pairs.append(
            {
                "sample_id": sample_id,
                "domain": domains_by_sample_id[sample_id],
                "baseline_score": b_s["score"],
                "candidate_score": c_s["score"],
                "delta": c_s["score"] - b_s["score"],
                "baseline_error": "error" in b_s,
                "candidate_error": "error" in c_s,
            }
        )

    domains = sorted({p["domain"] for p in pairs})
    constraints: List[Tuple[str, List[Dict[str, Any]], float]] = [
        ("overall", pairs, policy["min_improvement"])
    ]
    constraints.extend(
        (
            dom,
            [p for p in pairs if p["domain"] == dom],
            -policy["non_regression_margin"],
        )
        for dom in domains
    )

    criteria: Dict[str, Any] = {}
    reasons: List[str] = []
    for name, selected, threshold in constraints:
        interval = paired_interval(
            [p["delta"] for p in selected],
            alpha=policy["alpha"],
            comparisons=len(constraints),
        )
        decision = "INCONCLUSIVE"
        if policy["independent_samples"] and len(selected) >= policy["min_samples_per_domain"]:
            if interval["lower"] >= threshold:
                decision = "PASS"
            elif interval["upper"] < threshold:
                decision = "FAIL"
        criteria[name] = dict(interval, threshold=threshold, decision=decision)

    if not policy["independent_samples"]:
        reasons.append("independent sampling assumption has not been declared")
    if any(len(selected) < policy["min_samples_per_domain"] for _, selected, _ in constraints):
        reasons.append("insufficient samples for one or more constraints")

    decisions = [c["decision"] for c in criteria.values()]
    if "FAIL" in decisions:
        overall_decision = "FAIL"
    elif all(d == "PASS" for d in decisions):
        overall_decision = "PASS"
    else:
        overall_decision = "INCONCLUSIVE"

    return {
        "schema_version": "1.0.0",
        "decision": overall_decision,
        "method": "paired-hoeffding-union-bound",
        "policy": copy.deepcopy(policy),
        "simultaneous_constraints": len(constraints),
        "criteria": criteria,
        "reasons": reasons,
        "overall_mean_delta": statistics.fmean(p["delta"] for p in pairs),
    }


def group_clustered_paired_interval(
    pairs: List[Dict[str, Any]],
    *,
    alpha: float = 0.05,
    comparisons: int = 1,
) -> Dict[str, Any]:
    """Compute group-aware paired confidence intervals (`EVAL-05` / Issue #45) by averaging
    paired deltas within each `group_id` first, then applying Hoeffding bounds over the
    independent group means `G` rather than the raw correlated prompt count `N`.
    """
    if not pairs or not (0.0 < alpha < 1.0) or comparisons < 1:
        raise ValueError("Nonempty pairs, valid alpha, and positive comparisons are required")

    by_group: Dict[str, List[float]] = {}
    for p in pairs:
        gid = str(p.get("group_id") or p.get("sample_id"))
        delta = float(p["delta"])
        if not math.isfinite(delta) or not (-1.0 <= delta <= 1.0):
            raise ValueError("Paired deltas must be finite and bounded in [-1, 1]")
        by_group.setdefault(gid, []).append(delta)

    group_means = [statistics.fmean(vals) for vals in by_group.values()]
    raw_interval = paired_interval([float(p["delta"]) for p in pairs], alpha=alpha, comparisons=comparisons)
    group_interval = paired_interval(group_means, alpha=alpha, comparisons=comparisons)

    warnings: List[str] = []
    if len(pairs) > len(by_group):
        warnings.append(
            f"Correlated template/layout prompts detected (prompt_count={len(pairs)} > group_count={len(by_group)}); "
            "using group-clustered confidence radius to avoid inflating sample independence."
        )

    return {
        "prompt_count": len(pairs),
        "group_count": len(by_group),
        "mean_group_delta": group_interval["mean_delta"],
        "group_lower": group_interval["lower"],
        "group_upper": group_interval["upper"],
        "naive_prompt_lower": raw_interval["lower"],
        "naive_prompt_upper": raw_interval["upper"],
        "warnings": warnings,
    }


def evaluate_probability_calibration(
    records: List[Dict[str, Any]],
    *,
    num_bins: int = 5,
) -> Dict[str, Any]:
    """Evaluate probabilistic calibration (`EVAL-07` / Issue #47) separating:
    1. Event-probability predictions (`predicted_prob` vs exact ground-truth `gold_prob` in `[0, 1]`
       and realized binary outcomes `outcome_bit` in `{0, 1}`), and
    2. Model verifier confidence about answer correctness (`verifier_confidence` vs `correct` in `{0, 1}`).
    """
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    if not records:
        raise ValueError("records must be nonempty")

    def _bin_stats(pairs_prob_target: List[Tuple[float, float]]) -> Dict[str, Any]:
        brier = statistics.fmean((p - y) ** 2 for p, y in pairs_prob_target)
        bins = [{"bin_index": i, "lower": i / num_bins, "upper": (i + 1) / num_bins, "count": 0, "mean_pred": 0.0, "mean_target": 0.0} for i in range(num_bins)]
        bucket_lists: List[List[Tuple[float, float]]] = [[] for _ in range(num_bins)]
        for p, y in pairs_prob_target:
            if not (0.0 <= p <= 1.0) or not (0.0 <= y <= 1.0):
                raise ValueError("Probabilities and targets must be in [0, 1]")
            idx = min(num_bins - 1, int(p * num_bins))
            bucket_lists[idx].append((p, y))
        ece = 0.0
        for i, items in enumerate(bucket_lists):
            bins[i]["count"] = len(items)
            if items:
                mp = statistics.fmean(x[0] for x in items)
                mt = statistics.fmean(x[1] for x in items)
                bins[i]["mean_pred"] = round(mp, 6)
                bins[i]["mean_target"] = round(mt, 6)
                ece += (len(items) / len(pairs_prob_target)) * abs(mp - mt)
        return {
            "sample_count": len(pairs_prob_target),
            "brier_score": round(brier, 6),
            "ece": round(ece, 6),
            "reliability_bins": bins,
        }

    event_pairs = [
        (float(Fraction(r["predicted_prob"])), float(Fraction(r["gold_prob"])))
        for r in records
        if "predicted_prob" in r and "gold_prob" in r
    ]
    verifier_pairs = [
        (float(r["verifier_confidence"]), 1.0 if r["correct"] else 0.0)
        for r in records
        if "verifier_confidence" in r and "correct" in r
    ]

    out: Dict[str, Any] = {"num_bins": num_bins}
    if event_pairs:
        out["event_probability_calibration"] = _bin_stats(event_pairs)
    if verifier_pairs:
        out["verifier_confidence_calibration"] = _bin_stats(verifier_pairs)
    return out


def build_latency_and_throughput_report(
    samples: List[Dict[str, Any]],
    *,
    first_request_ms: float,
    warmup_excluded_count: int = 1,
    concurrency: int = 1,
    cache_condition: str = "cold_application_cache_not_os_disk_cold",
) -> Dict[str, Any]:
    """Standardize latency and throughput reporting (`EVAL-08` / Issue #48), separating
    time-to-first-token (`ttft_ms`), full-response latency (`latency_ms`), aggregate throughput,
    and explicit cache condition (`cold_application_cache_not_os_disk_cold` vs `cold_os_disk`).
    """
    if not samples or concurrency < 1 or warmup_excluded_count < 0:
        raise ValueError("Valid samples, concurrency, and warmup count are required")
    if cache_condition not in ("cold_application_cache_not_os_disk_cold", "cold_os_disk", "warm_in_memory"):
        raise ValueError(f"Unsupported cache_condition: {cache_condition!r}")

    full_lat = sorted(float(s["latency_ms"]) for s in samples)
    total_wall_s = sum(full_lat) / 1000.0
    out_tokens = sum(int(s.get("output_tokens", 1)) for s in samples)
    # Estimate or read time-to-first-token (for single-pass encoder + first step)
    ttfts = sorted(
        float(s.get("ttft_ms", float(s["latency_ms"]) / max(1, int(s.get("output_tokens", 1)))))
        for s in samples
    )

    return {
        "cache_condition": cache_condition,
        "concurrency": concurrency,
        "warmup_excluded_count": warmup_excluded_count,
        "first_request_ms": float(first_request_ms),
        "ttft_ms": {
            "p50_ms": round(statistics.median(ttfts), 4),
            "p95_ms": round(ttfts[max(0, math.ceil(0.95 * len(ttfts)) - 1)], 4),
            "mean_ms": round(statistics.fmean(ttfts), 4),
        },
        "full_response_latency_ms": {
            "p50_ms": round(statistics.median(full_lat), 4),
            "p95_ms": round(full_lat[max(0, math.ceil(0.95 * len(full_lat)) - 1)], 4),
            "mean_ms": round(statistics.fmean(full_lat), 4),
        },
        "throughput": {
            "requests_per_second": round(len(samples) / max(1e-9, total_wall_s), 4),
            "output_tokens_per_second": round(out_tokens / max(1e-9, total_wall_s), 4),
        },
    }


def compute_energy_and_useful_work(
    samples: List[Dict[str, Any]],
    *,
    measured_joules: Optional[float] = None,
    estimated_tdp_watts: Optional[float] = None,
    uncertainty_fraction: float = 0.25,
    total_cost_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Account for device runtime vs. verified useful work (`EVAL-10` / Issue #50).

    Never records unmeasured energy or cost as `0.0`; keeps `measured_energy_joules` and
    `total_cost_usd` as `None` when unknown.
    """
    if not samples:
        raise ValueError("samples must be nonempty")
    total_runtime_s = sum(float(s["latency_ms"]) for s in samples) / 1000.0
    verified_samples = [s for s in samples if float(s.get("score", 0.0)) == 1.0 and "error" not in s]
    useful_runtime_s = sum(float(s["latency_ms"]) for s in verified_samples) / 1000.0

    estimate = None
    if measured_joules is None and estimated_tdp_watts is not None:
        if estimated_tdp_watts <= 0 or not (0.0 <= uncertainty_fraction <= 1.0):
            raise ValueError("Invalid TDP watts or uncertainty_fraction")
        est_j = total_runtime_s * estimated_tdp_watts
        estimate = {
            "method": "wall_clock_seconds_times_declared_cpu_tdp_watts",
            "estimated_joules": round(est_j, 4),
            "lower_joules": round(est_j * (1.0 - uncertainty_fraction), 4),
            "upper_joules": round(est_j * (1.0 + uncertainty_fraction), 4),
            "uncertainty_fraction": uncertainty_fraction,
        }

    return {
        "total_device_runtime_seconds": round(total_runtime_s, 6),
        "verified_useful_work_seconds": round(useful_runtime_s, 6),
        "verified_useful_samples": len(verified_samples),
        "total_samples": len(samples),
        "useful_work_efficiency_ratio": round(len(verified_samples) / len(samples), 6),
        "measured_energy_joules": float(measured_joules) if measured_joules is not None else None,
        "estimated_energy": estimate,
        "total_cost_usd": float(total_cost_usd) if total_cost_usd is not None else None,
    }


OPEN_WEIGHT_BASELINE_REGISTRY: List[Dict[str, Any]] = [
    {
        "baseline_id": "qwen2.5-0.5b-instruct-cpu",
        "model_family": "Qwen2.5-0.5B-Instruct",
        "license_spdx": "Apache-2.0",
        "physical_parameters": 494000000,
        "tool_access": "none (neural_only)",
        "sampling_budget": {"device": "cpu", "threads": 1, "temperature": 0.0, "max_new_tokens": 40},
        "purpose": "Smallest permissive open-weight dense transformer baseline for comparing parameter-normalized CPU reasoning accuracy.",
    },
    {
        "baseline_id": "qwen2.5-math-1.5b-instruct-cpu",
        "model_family": "Qwen2.5-Math-1.5B-Instruct",
        "license_spdx": "Apache-2.0",
        "physical_parameters": 1540000000,
        "tool_access": "none (neural_only)",
        "sampling_budget": {"device": "cpu", "threads": 1, "temperature": 0.0, "max_new_tokens": 40},
        "purpose": "Math-specialized permissive open-weight baseline under identical zero-tool prompt and token budgets.",
    },
    {
        "baseline_id": "aix-zero-v11-neural-only",
        "model_family": "AIX-Zero-v11",
        "license_spdx": "Apache-2.0",
        "physical_parameters": 817410,
        "tool_access": "none (neural_only)",
        "sampling_budget": {"device": "cpu", "threads": 1, "temperature": 0.35, "num_candidates": 2, "max_new_tokens": 40},
        "purpose": "Sub-megaparameter (817K) latent-recurrent reference baseline (scores 9/105 neural-only on v11 held-out templates).",
    },
]


def measure_memory_and_cache_budget(
    model: AIXZeroModel,
    *,
    cached_cartridge_bytes: int = 0,
    max_cached_cartridge_bytes: int = 1024 * 1024,
    temporary_activation_bytes: int = 0,
    max_total_tensor_budget_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure and separate process RSS, model parameter bytes, retained LRU cartridge cache bytes,
    and temporary activation/deserialization tensor allocations (`EVAL-09` / Issue #49).
    """
    import resource
    import sys

    if cached_cartridge_bytes < 0 or max_cached_cartridge_bytes < 0 or temporary_activation_bytes < 0:
        raise ValueError("Byte counts must be nonnegative")

    model_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    retained_tensor_bytes = model_param_bytes + model_buffer_bytes + cached_cartridge_bytes
    peak_working_tensor_bytes = retained_tensor_bytes + temporary_activation_bytes

    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports ru_maxrss in bytes; Linux reports in KiB
    rss_bytes = int(rusage.ru_maxrss) if sys.platform == "darwin" else int(rusage.ru_maxrss) * 1024

    cache_within_limit = cached_cartridge_bytes <= max_cached_cartridge_bytes
    total_within_limit = (
        True
        if max_total_tensor_budget_bytes is None
        else peak_working_tensor_bytes <= max_total_tensor_budget_bytes
    )
    status = "WITHIN_BUDGET" if (cache_within_limit and total_within_limit) else "BUDGET_EXCEEDED"

    return {
        "status": status,
        "model_parameter_bytes": int(model_param_bytes),
        "model_buffer_bytes": int(model_buffer_bytes),
        "retained_cartridge_cache_bytes": int(cached_cartridge_bytes),
        "max_cached_cartridge_bytes": int(max_cached_cartridge_bytes),
        "temporary_activation_bytes": int(temporary_activation_bytes),
        "retained_tensor_bytes": int(retained_tensor_bytes),
        "peak_working_tensor_bytes": int(peak_working_tensor_bytes),
        "process_peak_rss_bytes": rss_bytes,
        "cache_within_limit": cache_within_limit,
        "total_within_limit": total_within_limit,
    }


EXTERNAL_CODING_BENCHMARK_COMPATIBILITY: Dict[str, Dict[str, Any]] = {
    "swe-bench-verified": {
        "benchmark_id": "swe-bench-verified",
        "official_harness": "swebench.harness.run_evaluation",
        "license_spdx": "MIT",
        "applicability_to_v11": "NOT_APPLICABLE_AT_817K_VOCAB_134",
        "reason": (
            "AIX-Zero v11 has a 134-token domain-specific vocabulary and 112-token max_seq_len; "
            "it cannot tokenize multi-file repository diffs or execute Dockerized pytest patches."
        ),
        "contamination_risk": "HIGH_FOR_WEB_PRETRAINED_MODELS_LOW_FOR_SYNTHETIC_V11",
        "compute_requirements": "Docker engine + >=16 GB RAM + multi-core CPU for repository test containers",
    },
    "livecodebench-exec": {
        "benchmark_id": "livecodebench-exec",
        "official_harness": "lcb_runner.runner.main",
        "license_spdx": "Apache-2.0",
        "applicability_to_v11": "COMPATIBILITY_FIXTURE_ONLY_ON_TINY_ARITHMETIC_AST",
        "reason": (
            "Full LiveCodeBench requires full Python stdlib tokenization and long-context generation; "
            "only tiny single-line function evaluation (`CODE_TRACE`) is representable in the 134-token v11 vocabulary."
        ),
        "contamination_risk": "TEMPORAL_CUTOFF_REQUIRED",
        "compute_requirements": "Sandboxed Python subprocess execution with strict timeout",
    },
}


def format_external_coding_fixture(
    benchmark_id: str,
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate an external coding benchmark compatibility fixture (`EVAL-12` / Issue #53)
    while explicitly marking `is_full_leaderboard_run = False` so a small harness smoke test
    can never masquerade as an official leaderboard result.
    """
    if benchmark_id not in EXTERNAL_CODING_BENCHMARK_COMPATIBILITY:
        raise ValueError(f"Unsupported external coding benchmark: {benchmark_id!r}")
    if not samples:
        raise ValueError("Compatibility fixture requires at least one sample")

    meta = EXTERNAL_CODING_BENCHMARK_COMPATIBILITY[benchmark_id]
    return {
        "benchmark_id": benchmark_id,
        "official_harness": meta["official_harness"],
        "license_spdx": meta["license_spdx"],
        "is_full_leaderboard_run": False,
        "fixture_only_disclaimer": (
            "Small compatibility fixture only; does NOT represent an official full-benchmark leaderboard score."
        ),
        "sample_count": len(samples),
        "samples": [dict(s) for s in samples],
    }


def verify_independent_reproduction(
    author_report: Dict[str, Any],
    reproducer_report: Dict[str, Any],
    *,
    accuracy_abs_tol: float = 0.0,
) -> Dict[str, Any]:
    """Verify an independent reproduction report against an author report (`EVAL-15` / Issue #61).

    Requires distinct `author_id` and `reproducer_id`, records both environments and signed
    report digests, and blocks release claims (`BLOCKED_BY_DISAGREEMENT`) if scores disagree
    beyond `accuracy_abs_tol` rather than averaging them away.
    """
    author_id = str(author_report.get("author_id", "")).strip()
    reproducer_id = str(reproducer_report.get("reproducer_id", "")).strip()
    if not author_id or not reproducer_id:
        raise ValueError("Both author_id and reproducer_id must be nonempty")
    if author_id == reproducer_id:
        raise ValueError("Independent reproduction requires reproducer_id != author_id")

    if author_report.get("checkpoint_sha256") != reproducer_report.get("checkpoint_sha256"):
        raise ValueError("Reproduction must evaluate the exact same checkpoint_sha256")
    if author_report.get("dataset_sha256") != reproducer_report.get("dataset_sha256"):
        raise ValueError("Reproduction must evaluate the exact same dataset_sha256")
    if author_report.get("answer_source") != reproducer_report.get("answer_source"):
        raise ValueError("Reproduction must use the same answer_source mode")

    a_score = float(author_report["mean_score"])
    r_score = float(reproducer_report["mean_score"])
    diff = abs(a_score - r_score)

    author_digest = hashlib.sha256(canonical_json(author_report).encode("utf-8")).hexdigest()
    reproducer_digest = hashlib.sha256(canonical_json(reproducer_report).encode("utf-8")).hexdigest()

    if diff <= accuracy_abs_tol:
        claim_status = "REPRODUCED_VERIFIED"
        eligible_for_release_claim = True
    else:
        claim_status = "BLOCKED_BY_DISAGREEMENT"
        eligible_for_release_claim = False

    return {
        "claim_status": claim_status,
        "eligible_for_release_claim": eligible_for_release_claim,
        "author_id": author_id,
        "reproducer_id": reproducer_id,
        "author_environment": dict(author_report.get("environment", {})),
        "reproducer_environment": dict(reproducer_report.get("environment", {})),
        "author_report_sha256": author_digest,
        "reproducer_report_sha256": reproducer_digest,
        "author_mean_score": a_score,
        "reproducer_mean_score": r_score,
        "absolute_score_difference": round(diff, 8),
    }


def score_cooperative_game_task(
    prediction: str,
    *,
    v_a: int,
    v_b: int,
    v_ab: int,
    query_type: str = "shapley_player_a",
) -> Dict[str, Any]:
    """Exact verifier for finite 2-player cooperative transferable-utility games (`REASON-10` / Issue #95).

    Strictly differentiates:
    - `grand_coalition_surplus`: $v(\\{A, B\\}) - (v(\\{A\\}) + v(\\{B\\}))$,
    - `shapley_player_a`: $\\phi_A(v) = v(\\{A\\}) + \\frac{1}{2}(v(\\{A, B\\}) - v(\\{A\\}) - v(\\{B\\}))$,
    - `shapley_player_b`: $\\phi_B(v) = v(\\{B\\}) + \\frac{1}{2}(v(\\{A, B\\}) - v(\\{A\\}) - v(\\{B\\}))$,
    - `core_nonempty`: `1` if $v(\\{A, B\\}) \\ge v(\\{A\\}) + v(\\{B\\})$ else `0`.
    """
    surplus = v_ab - (v_a + v_b)
    if query_type == "grand_coalition_surplus":
        gold = str(surplus)
    elif query_type == "shapley_player_a":
        val = Fraction(v_a) + Fraction(surplus, 2)
        gold = f"{val.numerator}/{val.denominator}" if val.denominator != 1 else str(val.numerator)
    elif query_type == "shapley_player_b":
        val = Fraction(v_b) + Fraction(surplus, 2)
        gold = f"{val.numerator}/{val.denominator}" if val.denominator != 1 else str(val.numerator)
    elif query_type == "core_nonempty":
        gold = "1" if surplus >= 0 else "0"
    else:
        raise ValueError(f"Unsupported cooperative game query_type: {query_type!r}")

    scored = score_rational_or_probability(prediction, gold, is_probability=False, require_reduced=True)
    return {
        "query_type": query_type,
        "gold": gold,
        "score": scored["score"],
        "status": scored["status"],
        "parsed_fraction": scored["parsed_fraction"],
    }


ABSTENTION_TOKENS = {"ABSTAIN", "INSUFFICIENT_INFO", "NON_IDENTIFIABLE", "UNKNOWN"}


def score_epistemic_abstention_task(
    prediction: str,
    *,
    is_identifiable: bool,
    gold_answer: Optional[str] = None,
    is_probability: bool = True,
) -> Dict[str, Any]:
    """Score epistemic uncertainty and abstention tasks (`REASON-05` / Issue #107).

    Separates:
    - `USEFUL_ABSTENTION`: model abstains when `is_identifiable is False`,
    - `FALSE_CERTAINTY`: model outputs a concrete numerical claim when `is_identifiable is False`,
    - `UNNECESSARY_ABSTENTION`: model abstains when `is_identifiable is True`,
    - `IDENTIFIABLE_EXACT_MATCH` / `IDENTIFIABLE_WRONG_ANSWER`: model answers when `is_identifiable is True`.
    """
    text = prediction.strip()
    m = re.search(r"<ans>(.*?)</ans>", text, flags=re.DOTALL)
    candidate = m.group(1).strip() if m else text.strip()
    upper = candidate.upper()
    abstained = upper in ABSTENTION_TOKENS

    if not is_identifiable:
        if abstained:
            return {
                "score": 1.0,
                "epistemic_outcome": "USEFUL_ABSTENTION",
                "false_certainty": False,
                "useful_abstention": True,
            }
        return {
            "score": 0.0,
            "epistemic_outcome": "FALSE_CERTAINTY",
            "false_certainty": True,
            "useful_abstention": False,
        }

    if gold_answer is None:
        raise ValueError("gold_answer is required when is_identifiable=True")
    if abstained:
        return {
            "score": 0.0,
            "epistemic_outcome": "UNNECESSARY_ABSTENTION",
            "false_certainty": False,
            "useful_abstention": False,
        }

    rational = score_rational_or_probability(candidate, gold_answer, is_probability=is_probability)
    return {
        "score": rational["score"],
        "epistemic_outcome": "IDENTIFIABLE_EXACT_MATCH" if rational["score"] == 1.0 else "IDENTIFIABLE_WRONG_ANSWER",
        "false_certainty": False,
        "useful_abstention": False,
        "rational_status": rational["status"],
    }


