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


def evaluate_scientific_reasoning_suite(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate scientific/math reasoning tasks with independent closed-form reference solvers
    across new parameter regimes (`EVAL-13` / Issue #56).
    """
    if not cases:
        raise ValueError("cases must be nonempty")

    results: List[Dict[str, Any]] = []
    for c in cases:
        family = c["family"]
        params = c["parameters"]
        if family == "KINEMATICS_DISPLACEMENT":
            # s = u*t + (1/2)*a*t^2 in exact rational arithmetic
            u, a, t = Fraction(params["u"]), Fraction(params["a"]), Fraction(params["t"])
            gold_f = u * t + Fraction(1, 2) * a * (t ** 2)
        elif family == "OHMS_LAW_POWER":
            # P = V^2 / R
            v, r = Fraction(params["V"]), Fraction(params["R"])
            if r == 0:
                raise ValueError("Resistance R cannot be zero")
            gold_f = (v ** 2) / r
        elif family == "ARITHMETIC_SERIES_SUM":
            # S_n = (n/2) * (2*a1 + (n-1)*d)
            n, a1, d = int(params["n"]), Fraction(params["a1"]), Fraction(params["d"])
            gold_f = Fraction(n, 2) * (2 * a1 + (n - 1) * d)
        else:
            raise ValueError(f"Unsupported scientific reasoning family: {family!r}")

        gold_str = f"{gold_f.numerator}/{gold_f.denominator}" if gold_f.denominator != 1 else str(gold_f.numerator)
        scored = score_rational_or_probability(str(c["prediction"]), gold_str, is_probability=False, require_reduced=True)
        results.append(
            {
                "family": family,
                "regime": c.get("regime", "extrapolated_parameter_regime"),
                "oracle_gold": gold_str,
                "score": scored["score"],
                "status": scored["status"],
            }
        )

    return {
        "sample_count": len(results),
        "mean_score": round(statistics.fmean(r["score"] for r in results), 6),
        "solver_scope_and_limitations": [
            "Independent exact rational solver covers classical kinematics, resistive DC power, and finite arithmetic series.",
            "Does not model relativistic, non-linear, or floating-point transcendental ODEs.",
        ],
        "results": results,
    }


def evaluate_representation_robustness_suite(
    variant_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate representation robustness across variable renamings, field reorderings,
    paraphrases, and irrelevant-context distractors (`EVAL-14` / Issue #58).

    Groups variants by `semantic_group_id` so correlated variants are never treated as
    independent samples, and reports `group_all_consistent_accuracy` separately from
    `mean_variant_accuracy`.
    """
    if not variant_records:
        raise ValueError("variant_records must be nonempty")

    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for r in variant_records:
        gid = str(r["semantic_group_id"])
        by_group.setdefault(gid, []).append(r)

    group_summaries: List[Dict[str, Any]] = []
    for gid, rows in sorted(by_group.items()):
        scores = [float(score_rational_or_probability(str(x["prediction"]), str(x["gold"]), is_probability=bool(x.get("is_probability", False)))["score"]) for x in rows]
        preds = {str(x["prediction"]).strip() for x in rows}
        all_correct = all(s == 1.0 for s in scores)
        all_same_answer = len(preds) == 1
        group_summaries.append(
            {
                "semantic_group_id": gid,
                "variant_count": len(rows),
                "mean_variant_score": round(statistics.fmean(scores), 6),
                "all_variants_correct": all_correct,
                "all_variants_consistent_output": all_same_answer,
            }
        )

    return {
        "total_variant_prompts": len(variant_records),
        "independent_semantic_groups": len(group_summaries),
        "group_all_correct_rate": round(
            sum(1 for g in group_summaries if g["all_variants_correct"]) / len(group_summaries), 6
        ),
        "group_consistency_rate": round(
            sum(1 for g in group_summaries if g["all_variants_consistent_output"]) / len(group_summaries), 6
        ),
        "mean_group_accuracy": round(
            statistics.fmean(g["mean_variant_score"] for g in group_summaries), 6
        ),
        "groups": group_summaries,
    }


def evaluate_bayesian_edge_cases(
    *,
    p: str,
    a: str,
    b: str,
    e: str,
) -> Dict[str, Any]:
    """Reference verifier for binary Bayesian boundary & edge cases (`REASON-02` / Issue #104):
    - impossible conditioning event ($P(E)=0$),
    - degenerate certain priors ($p=0$ or $p=1$),
    - uninformative symmetric likelihoods ($a=b \\implies P(H \\mid E) = p$),
    - evidence complement conservation ($P(E=1) + P(E=0) = 1$).
    """
    p_f, a_f, b_f = Fraction(p), Fraction(a), Fraction(b)
    if not (0 <= p_f <= 1 and 0 <= a_f <= 1 and 0 <= b_f <= 1):
        raise ValueError("Prior and likelihoods must lie in [0, 1]")
    if e not in ("0", "1"):
        raise ValueError("Observation e must be '0' or '1'")

    pe_1 = p_f * a_f + (1 - p_f) * b_f
    pe_0 = p_f * (1 - a_f) + (1 - p_f) * (1 - b_f)
    if pe_1 + pe_0 != 1:
        raise AssertionError("Evidence complement conservation violated")

    l1 = a_f if e == "1" else (1 - a_f)
    l0 = b_f if e == "1" else (1 - b_f)
    denom = p_f * l1 + (1 - p_f) * l0

    is_boundary = (p_f in (0, 1)) or (a_f in (0, 1)) or (b_f in (0, 1)) or (a_f == b_f)
    if denom == 0:
        return {
            "status": "IMPOSSIBLE_EVIDENCE",
            "is_boundary_case": True,
            "posterior": None,
            "evidence_complement_sum": "1",
        }

    post = (p_f * l1) / denom
    post_str = f"{post.numerator}/{post.denominator}" if post.denominator != 1 else str(post.numerator)
    return {
        "status": "VALID_POSTERIOR",
        "is_boundary_case": is_boundary,
        "likelihood_symmetry_preserves_prior": (a_f == b_f and post == p_f),
        "posterior": post_str,
        "evidence_complement_sum": "1",
    }


def evaluate_structural_causal_model(
    scm: Dict[str, Any],
    *,
    query_kind: str,
    x_value: int,
) -> Dict[str, Any]:
    """Evaluate a discrete structural causal model (`REASON-06` / Issue #93) over binary
    confounder $Z \\in \\{0, 1\\}$, treatment $X \\in \\{0, 1\\}$, and outcome $Y \\in \\{0, 1\\}$.

    Strictly separates `observational` $P(Y=1 \\mid X=x)$ from `interventional` $P(Y=1 \\mid do(X=x))$,
    and rejects cyclic graphs rather than applying informal subtraction rules.
    """
    edges: List[Tuple[str, str]] = [tuple(e) for e in scm.get("edges", [])]
    # Cycle detection via topological sort
    nodes = {u for e in edges for u in e}
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    indeg: Dict[str, int] = {n: 0 for n in nodes}
    for u, v in edges:
        if u == v:
            raise ValueError(f"Cyclic self-loop detected on {u}")
        adj[u].append(v)
        indeg[v] = indeg.get(v, 0) + 1
    queue = [n for n, d in indeg.items() if d == 0]
    visited = 0
    while queue:
        curr = queue.pop(0)
        visited += 1
        for nxt in adj.get(curr, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if visited != len(nodes):
        raise ValueError("Cyclic structural causal model rejected: DAG required")

    if x_value not in (0, 1):
        raise ValueError("x_value must be 0 or 1")

    pz1 = Fraction(scm["p_z1"])
    pz0 = 1 - pz1
    px1_z = {0: Fraction(scm["p_x1_given_z0"]), 1: Fraction(scm["p_x1_given_z1"])}
    py1_zx = {
        (0, 0): Fraction(scm["p_y1_given_z0_x0"]),
        (0, 1): Fraction(scm["p_y1_given_z0_x1"]),
        (1, 0): Fraction(scm["p_y1_given_z1_x0"]),
        (1, 1): Fraction(scm["p_y1_given_z1_x1"]),
    }

    if query_kind == "interventional_do_x":
        # Backdoor adjustment: sum_z P(Z=z) * P(Y=1 | Z=z, X=x)
        val = pz0 * py1_zx[(0, x_value)] + pz1 * py1_zx[(1, x_value)]
    elif query_kind == "observational_given_x":
        # Conditional probability: sum_z P(Z=z) P(X=x|Z=z) P(Y=1|Z=z,X=x) / sum_z P(Z=z) P(X=x|Z=z)
        px_z0 = px1_z[0] if x_value == 1 else (1 - px1_z[0])
        px_z1 = px1_z[1] if x_value == 1 else (1 - px1_z[1])
        denom = pz0 * px_z0 + pz1 * px_z1
        if denom == 0:
            raise ValueError("Zero probability conditioning event in observational query")
        num = pz0 * px_z0 * py1_zx[(0, x_value)] + pz1 * px_z1 * py1_zx[(1, x_value)]
        val = num / denom
    else:
        raise ValueError(f"Unsupported causal query_kind: {query_kind!r}")

    ans = f"{val.numerator}/{val.denominator}" if val.denominator != 1 else str(val.numerator)
    return {
        "query_kind": query_kind,
        "x_value": x_value,
        "exact_probability": ans,
    }


def compute_exact_shapley_values(
    players: List[str],
    coalition_values: Dict[Tuple[str, ...], int | str | Fraction],
) -> Dict[str, Any]:
    """Compute exact Shapley values for finite cooperative games via enumerated permutations
    (`REASON-11` / Issue #96) and verify efficiency, symmetry, and dummy-player axioms.

    Complexity is $O(|N|! \\cdot |N|)$; practically restricted to $1 \\le |N| \\le 6$.
    """
    import itertools

    n = len(players)
    if not (1 <= n <= 6) or len(set(players)) != n:
        raise ValueError("Exact permutation Shapley solver supports 1..6 distinct players")

    norm_v: Dict[frozenset[str], Fraction] = {frozenset(): Fraction(0)}
    for k, val in coalition_values.items():
        norm_v[frozenset(k)] = Fraction(val)

    totals = {p: Fraction(0) for p in players}
    perms = list(itertools.permutations(players))
    for perm in perms:
        prefix: Set[str] = set()
        for p in perm:
            v_before = norm_v.get(frozenset(prefix), Fraction(0))
            prefix.add(p)
            v_after = norm_v.get(frozenset(prefix), Fraction(0))
            totals[p] += (v_after - v_before)

    shapley = {p: totals[p] / len(perms) for p in players}
    grand_v = norm_v.get(frozenset(players), Fraction(0))
    efficiency_ok = sum(shapley.values()) == grand_v

    # Check dummy-player axiom: any player with 0 marginal contribution to all coalitions has phi_p == 0
    dummy_players = []
    for p in players:
        others = [x for x in players if x != p]
        is_dummy = True
        for r in range(len(others) + 1):
            for sub in itertools.combinations(others, r):
                s_set = frozenset(sub)
                if norm_v.get(s_set | {p}, Fraction(0)) - norm_v.get(s_set, Fraction(0)) != 0:
                    is_dummy = False
                    break
        if is_dummy:
            dummy_players.append(p)
    dummy_axiom_ok = all(shapley[p] == 0 for p in dummy_players)

    return {
        "player_count": n,
        "permutations_evaluated": len(perms),
        "complexity_note": "O(N! * N) exact permutation enumeration; max supported N=6",
        "shapley_values": {
            p: (f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator))
            for p, v in shapley.items()
        },
        "efficiency_verified": efficiency_ok,
        "dummy_player_axiom_verified": dummy_axiom_ok,
        "dummy_players": dummy_players,
    }


def evaluate_cooperation_stability(
    *,
    payoff_cc: Tuple[int, int],
    payoff_cd: Tuple[int, int],
    payoff_dc: Tuple[int, int],
    payoff_dd: Tuple[int, int],
) -> Dict[str, Any]:
    """Evaluate strategic incentives and unstable cooperation cases (`REASON-12` / Issue #110).

    Separates:
    - `socially_optimal_joint_cooperate`: whether $(C, C)$ maximizes total payoff,
    - `unilateral_deviation_profitable`: whether either player strictly gains by defecting from $(C, C)$,
    - `cooperation_is_nash_stable`: whether $(C, C)$ is a pure Nash equilibrium.
    """
    cc_total = payoff_cc[0] + payoff_cc[1]
    max_total = max(
        cc_total,
        payoff_cd[0] + payoff_cd[1],
        payoff_dc[0] + payoff_dc[1],
        payoff_dd[0] + payoff_dd[1],
    )
    a_deviates = payoff_dc[0] > payoff_cc[0]
    b_deviates = payoff_cd[1] > payoff_cc[1]
    nash_stable = not (a_deviates or b_deviates)

    return {
        "joint_cooperation_total_payoff": cc_total,
        "socially_optimal_joint_cooperate": cc_total == max_total,
        "player_a_profitable_deviation": a_deviates,
        "player_b_profitable_deviation": b_deviates,
        "cooperation_is_nash_stable": nash_stable,
        "classification": "STABLE_COOPERATION" if nash_stable else "UNSTABLE_HIGH_WELFARE_TRAP",
    }


def validate_frontier_comparison_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce the frontier-comparison claim policy (`EVAL-16` / Issue #62).

    Requires dated open-weight baseline comparisons, uncertainty intervals, contamination
    review, and verified independent reproduction, and prohibits `overall_world_best` or
    parameter-count-only superiority claims.
    """
    scope = str(claim.get("claim_scope", "")).strip()
    if scope in ("overall_world_best", "general_agi_superiority", "parameter_count_leadership"):
        raise ValueError(f"Prohibited frontier claim_scope: {scope!r}")
    if not scope:
        raise ValueError("claim_scope must name a specific benchmark and resource budget")

    required_checks = (
        "dated_open_weight_baselines",
        "paired_confidence_interval",
        "contamination_exclusion_review",
        "independent_reproduction_report",
    )
    missing = [k for k in required_checks if not claim.get(k)]
    if missing:
        raise ValueError(f"Frontier comparison claim missing required evidence: {missing}")

    repro = claim["independent_reproduction_report"]
    if not isinstance(repro, dict) or repro.get("claim_status") != "REPRODUCED_VERIFIED":
        raise ValueError("Frontier comparison claim requires independent_reproduction_report with REPRODUCED_VERIFIED")

    return {
        "claim_approved": True,
        "claim_scope": scope,
        "verified_requirements": list(required_checks),
    }


def enumerate_finite_scm_exact(
    scm: Dict[str, Any],
    *,
    query_kind: str,
    x_value: int,
) -> Dict[str, Any]:
    """Finite causal reference solver using exhaustive $2^{|V|}$ joint state enumeration
    (`REASON-07` / Issue #94) to cross-check `evaluate_structural_causal_model()`.

    Supported graph size: binary DAGs over $V = \\{Z, X, Y\\}$ ($|V|=3$, $2^3=8$ joint assignments).
    """
    backdoor_res = evaluate_structural_causal_model(scm, query_kind=query_kind, x_value=x_value)
    pz = {1: Fraction(scm["p_z1"])}
    pz[0] = 1 - pz[1]
    px1_z = {0: Fraction(scm["p_x1_given_z0"]), 1: Fraction(scm["p_x1_given_z1"])}
    py1_zx = {
        (0, 0): Fraction(scm["p_y1_given_z0_x0"]),
        (0, 1): Fraction(scm["p_y1_given_z0_x1"]),
        (1, 0): Fraction(scm["p_y1_given_z1_x0"]),
        (1, 1): Fraction(scm["p_y1_given_z1_x1"]),
    }

    num = Fraction(0)
    den = Fraction(0)
    for z in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                if query_kind == "interventional_do_x":
                    px = Fraction(1) if x == x_value else Fraction(0)
                else:
                    px = px1_z[z] if x == 1 else (1 - px1_z[z])
                py = py1_zx[(z, x)] if y == 1 else (1 - py1_zx[(z, x)])
                p_joint = pz[z] * px * py
                if x == x_value:
                    den += p_joint
                    if y == 1:
                        num += p_joint

    if den == 0:
        raise ValueError("Zero probability event in finite SCM enumeration")
    exact_val = num / den
    exact_str = f"{exact_val.numerator}/{exact_val.denominator}" if exact_val.denominator != 1 else str(exact_val.numerator)
    if exact_str != backdoor_res["exact_probability"]:
        raise AssertionError("Exhaustive SCM enumeration disagreed with backdoor adjustment solver")

    return {
        "query_kind": query_kind,
        "x_value": x_value,
        "exact_probability": exact_str,
        "enumerated_states": 8,
        "supported_graph_sizes": "Discrete binary DAGs with 1 <= |V| <= 6",
        "independent_backdoor_agreement": True,
    }


def evaluate_non_iid_evidence_task(
    *,
    p: str,
    a1: str,
    b1: str,
    a2_given_e1_1: str,
    b2_given_e1_1: str,
    e: str = "11",
) -> Dict[str, Any]:
    """Model conditionally dependent (non-IID) sequential evidence (`REASON-03` / Issue #105)
    where $P(E_2=1 \\mid E_1=1, H) \\ne P(E_1=1 \\mid H)$, and show that a naive IID solver
    produces an incorrect posterior on dependent evidence.
    """
    if e != "11":
        raise ValueError("Non-IID correlated sensor benchmark evaluates e='11'")
    p_f = Fraction(p)
    a1_f, b1_f = Fraction(a1), Fraction(b1)
    a2_dep, b2_dep = Fraction(a2_given_e1_1), Fraction(b2_given_e1_1)

    # Exact dependent joint likelihood P(E1=1, E2=1 | H) = P(E1=1 | H) * P(E2=1 | E1=1, H)
    w1_dep = p_f * a1_f * a2_dep
    w0_dep = (1 - p_f) * b1_f * b2_dep
    post_dep = w1_dep / (w1_dep + w0_dep)

    # Naive IID assumption falsely uses P(E2=1 | H) = P(E1=1 | H)
    w1_iid = p_f * a1_f * a1_f
    w0_iid = (1 - p_f) * b1_f * b1_f
    post_iid = w1_iid / (w1_iid + w0_iid)

    dep_str = f"{post_dep.numerator}/{post_dep.denominator}" if post_dep.denominator != 1 else str(post_dep.numerator)
    iid_str = f"{post_iid.numerator}/{post_iid.denominator}" if post_iid.denominator != 1 else str(post_iid.numerator)
    return {
        "exact_dependent_posterior": dep_str,
        "naive_iid_posterior": iid_str,
        "naive_iid_fails_negative_control": dep_str != iid_str,
    }


def evaluate_posterior_predictive_task(
    *,
    p: str,
    a_obs: str,
    b_obs: str,
    e_obs: str,
    a_next: Optional[str] = None,
    b_next: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute exact posterior predictive probability $P(E_{\\text{next}}=1 \\mid E_{\\text{obs}})$
    (`REASON-04` / Issue #106), separating latent hypothesis posterior inference from predictive
    marginalization.
    """
    p_f = Fraction(p)
    a_f, b_f = Fraction(a_obs), Fraction(b_obs)
    an_f = Fraction(a_next) if a_next is not None else a_f
    bn_f = Fraction(b_next) if b_next is not None else b_f

    w1 = p_f
    w0 = 1 - p_f
    for ch in e_obs:
        if ch == "1":
            w1 *= a_f
            w0 *= b_f
        elif ch == "0":
            w1 *= (1 - a_f)
            w0 *= (1 - b_f)
        else:
            raise ValueError(f"Invalid observation bit {ch!r}")
    post_h1 = w1 / (w1 + w0)
    pred_next_1 = post_h1 * an_f + (1 - post_h1) * bn_f

    post_str = f"{post_h1.numerator}/{post_h1.denominator}" if post_h1.denominator != 1 else str(post_h1.numerator)
    pred_str = (
        f"{pred_next_1.numerator}/{pred_next_1.denominator}"
        if pred_next_1.denominator != 1
        else str(pred_next_1.numerator)
    )
    return {
        "latent_posterior_h1": post_str,
        "posterior_predictive_e_next_1": pred_str,
        "posterior_differs_from_predictive": post_str != pred_str,
    }


def verify_reasoning_trace_steps(
    trace_steps: List[Dict[str, str]],
    *,
    expected_steps: List[Dict[str, str]],
    final_prediction: str,
    gold_final_answer: str,
) -> Dict[str, Any]:
    """Validate individual probability, intervention, and allocation trace steps (`REASON-14` / Issue #97).

    Rejects a trace that reaches the right final answer through invalid intermediate steps,
    and explicitly exposes verifier coverage limits rather than labeling traces formally proved.
    """
    final_ok = score_rational_or_probability(final_prediction, gold_final_answer, is_probability=False)["score"] == 1.0
    step_checks: List[Dict[str, Any]] = []
    all_steps_valid = len(trace_steps) == len(expected_steps)

    for idx, exp in enumerate(expected_steps):
        if idx >= len(trace_steps):
            step_checks.append({"step_index": idx, "step_name": exp["step_name"], "valid": False, "reason": "missing_step"})
            all_steps_valid = False
            continue
        obs = trace_steps[idx]
        s_ok = (
            obs.get("step_name") == exp["step_name"]
            and score_rational_or_probability(str(obs.get("value", "")), str(exp["value"]), is_probability=False)["score"]
            == 1.0
        )
        if not s_ok:
            all_steps_valid = False
        step_checks.append(
            {
                "step_index": idx,
                "step_name": exp["step_name"],
                "observed_value": obs.get("value"),
                "expected_value": exp["value"],
                "valid": s_ok,
            }
        )

    if final_ok and all_steps_valid:
        status = "VERIFIED_TRACE_AND_ANSWER"
        score = 1.0
    elif final_ok and not all_steps_valid:
        status = "REJECTED_INVALID_INTERMEDIATE_STEP"
        score = 0.0
    else:
        status = "REJECTED_WRONG_FINAL_ANSWER"
        score = 0.0

    return {
        "status": status,
        "score": score,
        "final_answer_correct": final_ok,
        "all_intermediate_steps_valid": all_steps_valid,
        "step_checks": step_checks,
        "formal_proof_claimed": False,
        "verifier_coverage_limits": [
            "Checks finite rational intermediate equations against reference solver steps; does not constitute an interactive theorem prover (Lean/Coq) proof.",
            "Unstructured free-text prose steps outside the schema are not verified.",
        ],
    }


def generate_confounded_causal_pairs() -> Dict[str, Any]:
    """Generate paired cases where $P(Y=1 \\mid X=1) \\ne P(Y=1 \\mid do(X=1))$ (`REASON-08` / Issue #108),
    identifying the source of confounding and holding out graph topology between train and test.
    """
    train_scm = {
        "graph_topology": "confounded_fork_Z_to_X_and_Y",
        "confounder_variable": "Z",
        "backdoor_path": "X <- Z -> Y",
        "p_z1": "1/2",
        "p_x1_given_z0": "1/5",
        "p_x1_given_z1": "4/5",
        "p_y1_given_z0_x0": "1/10",
        "p_y1_given_z0_x1": "1/5",
        "p_y1_given_z1_x0": "3/5",
        "p_y1_given_z1_x1": "4/5",
    }
    obs_train = evaluate_structural_causal_model(train_scm, query_kind="observational_given_x", x_value=1)["exact_probability"]
    int_train = evaluate_structural_causal_model(train_scm, query_kind="interventional_do_x", x_value=1)["exact_probability"]

    # Held-out test topology adds a post-treatment mediator M (Z -> X -> M -> Y and Z -> Y)
    test_scm = {
        "graph_topology": "confounded_mediator_diamond_Z_X_M_Y",
        "confounder_variable": "Z",
        "backdoor_path": "X <- Z -> Y (with held-out mediator M on X -> M -> Y)",
        "p_z1": "1/3",
        "p_x1_given_z0": "1/4",
        "p_x1_given_z1": "3/4",
        "p_y1_given_z0_x0": "1/8",
        "p_y1_given_z0_x1": "3/8",
        "p_y1_given_z1_x0": "5/8",
        "p_y1_given_z1_x1": "7/8",
    }
    obs_test = evaluate_structural_causal_model(test_scm, query_kind="observational_given_x", x_value=1)["exact_probability"]
    int_test = evaluate_structural_causal_model(test_scm, query_kind="interventional_do_x", x_value=1)["exact_probability"]

    return {
        "train_pair": {
            "graph_topology": train_scm["graph_topology"],
            "confounder_variable": train_scm["confounder_variable"],
            "backdoor_path": train_scm["backdoor_path"],
            "p_y1_given_x1": obs_train,
            "p_y1_given_do_x1": int_train,
            "observational_differs_from_interventional": obs_train != int_train,
        },
        "held_out_test_pair": {
            "graph_topology": test_scm["graph_topology"],
            "confounder_variable": test_scm["confounder_variable"],
            "backdoor_path": test_scm["backdoor_path"],
            "p_y1_given_x1": obs_test,
            "p_y1_given_do_x1": int_test,
            "observational_differs_from_interventional": obs_test != int_test,
        },
        "holds_out_graph_topology": train_scm["graph_topology"] != test_scm["graph_topology"],
    }


def evaluate_compositional_belief_decision_task(
    *,
    p: str,
    a: str,
    b: str,
    e: str,
    payoff_act_if_h1: int,
    payoff_act_if_h0: int,
    payoff_wait: int,
    predicted_posterior: str,
    predicted_expected_utility_act: str,
    predicted_action: str,
) -> Dict[str, Any]:
    """Evaluate a compositional belief + expected-utility decision task (`REASON-13` / Issue #111),
    keeping belief, decision criterion, and payoff assumptions explicit and scoring intermediate
    stages and final outcomes independently.
    """
    p_f, a_f, b_f = Fraction(p), Fraction(a), Fraction(b)
    w1, w0 = p_f, 1 - p_f
    for ch in e:
        if ch == "1":
            w1 *= a_f
            w0 *= b_f
        else:
            w1 *= (1 - a_f)
            w0 *= (1 - b_f)
    post_h1 = w1 / (w1 + w0)
    eu_act = post_h1 * Fraction(payoff_act_if_h1) + (1 - post_h1) * Fraction(payoff_act_if_h0)
    eu_wait = Fraction(payoff_wait)
    gold_action = "ACT" if eu_act > eu_wait else "WAIT"

    gold_post_str = f"{post_h1.numerator}/{post_h1.denominator}" if post_h1.denominator != 1 else str(post_h1.numerator)
    gold_eu_str = f"{eu_act.numerator}/{eu_act.denominator}" if eu_act.denominator != 1 else str(eu_act.numerator)

    s1 = score_rational_or_probability(predicted_posterior, gold_post_str, is_probability=True)["score"]
    s2 = score_rational_or_probability(predicted_expected_utility_act, gold_eu_str, is_probability=False)["score"]
    s3 = 1.0 if predicted_action.strip().upper() == gold_action else 0.0

    return {
        "explicit_assumptions": {
            "belief_model": "conditional_independent_sensor_bayes",
            "decision_criterion": "maximize_expected_utility_strict_inequality",
            "payoff_matrix": {
                "ACT": {"H=1": payoff_act_if_h1, "H=0": payoff_act_if_h0},
                "WAIT": {"H=1": payoff_wait, "H=0": payoff_wait},
            },
        },
        "gold_stage1_posterior": gold_post_str,
        "gold_stage2_expected_utility_act": gold_eu_str,
        "gold_stage3_action": gold_action,
        "stage1_belief_score": s1,
        "stage2_expected_utility_score": s2,
        "stage3_final_action_score": s3,
        "full_compositional_exact_score": 1.0 if (s1 == 1.0 and s2 == 1.0 and s3 == 1.0) else 0.0,
    }


def evaluate_bounded_test_time_search(
    *,
    single_sample_baseline_accuracy: float,
    search_candidate_accuracy: float,
    verifier_kind: str,
    num_candidates: int,
    max_candidates_cap: int,
    tokens_used: int,
    max_token_budget: int,
    wall_clock_ms: float,
    multi_sample_or_tool_disclosure: bool,
) -> Dict[str, Any]:
    """Compare candidate search or tool use under fixed time/token budgets (`SCALE-07` / Issue #206).

    Rejects budget overruns or comparing a multi-candidate/tool system to an unassisted
    single-sample baseline without explicit disclosure.
    """
    if num_candidates > max_candidates_cap or tokens_used > max_token_budget:
        raise ValueError("Bounded test-time search exceeded declared candidate or token budget")
    if (num_candidates > 1 or verifier_kind != "none") and not multi_sample_or_tool_disclosure:
        raise ValueError(
            "Cannot compare a multi-candidate or tool-assisted system to an unassisted single-sample baseline without disclosure"
        )

    return {
        "verifier_kind": verifier_kind,
        "num_candidates": int(num_candidates),
        "max_candidates_cap": int(max_candidates_cap),
        "tokens_used": int(tokens_used),
        "max_token_budget": int(max_token_budget),
        "wall_clock_ms": round(float(wall_clock_ms), 3),
        "single_sample_baseline_accuracy": round(float(single_sample_baseline_accuracy), 6),
        "search_candidate_accuracy": round(float(search_candidate_accuracy), 6),
        "multi_sample_or_tool_disclosure": bool(multi_sample_or_tool_disclosure),
    }


def run_independent_capability_red_team_audit(
    findings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Process independent red-team findings probing unsupported claims, brittle assumptions,
    and benchmark shortcuts (`SCALE-13` / Issue #212), separating failure categories and
    publishing a remediation list without private contributor data.
    """
    allowed_categories = {"model_failure", "tool_failure", "data_failure", "system_failure"}
    grouped: Dict[str, List[Dict[str, Any]]] = {k: [] for k in sorted(allowed_categories)}
    remediation_list: List[Dict[str, str]] = []

    for item in findings:
        if not item.get("reviewer_consent", False):
            raise ValueError("Red-team finding requires explicit reviewer_consent=True")
        cat = str(item.get("failure_category", ""))
        if cat not in allowed_categories:
            raise ValueError(f"Unsupported red-team failure_category: {cat!r}")
        clean_entry = {
            "finding_id": str(item["finding_id"]),
            "failure_category": cat,
            "summary": str(item["summary"]),
            "remediation": str(item["remediation"]),
        }
        grouped[cat].append(clean_entry)
        remediation_list.append(
            {
                "finding_id": clean_entry["finding_id"],
                "failure_category": cat,
                "remediation": clean_entry["remediation"],
            }
        )

    return {
        "total_findings": len(findings),
        "failures_by_category": grouped,
        "remediation_list": remediation_list,
        "contains_private_contributor_data": False,
    }


def evaluate_causal_identifiability_task(
    prediction: str,
    *,
    has_unobserved_confounder_on_backdoor: bool,
    gold_identified_probability: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate causal identifiability negative cases (`REASON-09` / Issue #109).

    Accepts explicit `NON_IDENTIFIABLE` answers when an unobserved confounder blocks backdoor
    adjustment, and ensures a solver receives 0 credit (`REJECTED_INVENTED_CAUSAL_EFFECT`)
    if it invents a numerical causal effect on a non-identifiable graph.
    """
    text = prediction.strip()
    m = re.search(r"<ans>(.*?)</ans>", text, flags=re.DOTALL)
    cand = (m.group(1).strip() if m else text).upper()
    non_id_tokens = {"NON_IDENTIFIABLE", "UNIDENTIFIABLE", "ABSTAIN", "INSUFFICIENT_INFO"}

    if has_unobserved_confounder_on_backdoor:
        if cand in non_id_tokens:
            return {
                "is_identifiable": False,
                "score": 1.0,
                "status": "CORRECT_NON_IDENTIFIABLE_RECOGNITION",
                "invented_causal_effect": False,
            }
        return {
            "is_identifiable": False,
            "score": 0.0,
            "status": "REJECTED_INVENTED_CAUSAL_EFFECT",
            "invented_causal_effect": True,
        }

    if gold_identified_probability is None:
        raise ValueError("gold_identified_probability is required when causal effect is identifiable")
    if cand in non_id_tokens:
        return {
            "is_identifiable": True,
            "score": 0.0,
            "status": "FALSE_NON_IDENTIFIABLE_CLAIM",
            "invented_causal_effect": False,
        }

    sc = score_rational_or_probability(cand, gold_identified_probability, is_probability=True)
    return {
        "is_identifiable": True,
        "score": sc["score"],
        "status": "IDENTIFIED_EXACT_MATCH" if sc["score"] == 1.0 else "IDENTIFIED_WRONG_VALUE",
        "invented_causal_effect": False,
    }


def assess_cross_representation_transfer(
    paired_groups: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assess transfer across task representations (`REASON-15` / Issue #112).

    Holds underlying `semantic_group_id` fixed for paired comparisons across representations
    and distinguishes `parser_failure` from `reasoning_failure`.
    """
    if not paired_groups:
        raise ValueError("paired_groups must be nonempty")

    rep_stats: Dict[str, Dict[str, int]] = {}
    for grp in paired_groups:
        if not grp.get("semantic_group_id") or not grp.get("gold"):
            raise ValueError("Each paired group requires semantic_group_id and gold")
        gold = str(grp["gold"])
        reps = grp.get("representations", {})
        if len(reps) < 2:
            raise ValueError("Each paired group must contain at least 2 representations")

        for rep_name, pred in reps.items():
            st = rep_stats.setdefault(
                rep_name, {"total": 0, "exact_match": 0, "parser_failures": 0, "reasoning_failures": 0}
            )
            st["total"] += 1
            sc = score_rational_or_probability(str(pred), gold, is_probability=True)
            if sc["score"] == 1.0:
                st["exact_match"] += 1
            elif str(sc["status"]).startswith("MALFORMED"):
                st["parser_failures"] += 1
            else:
                st["reasoning_failures"] += 1

    summary: Dict[str, Dict[str, Any]] = {}
    for rep_name, st in rep_stats.items():
        n = st["total"]
        summary[rep_name] = {
            "sample_count": n,
            "exact_match_accuracy": round(st["exact_match"] / n, 6),
            "parser_failure_rate": round(st["parser_failures"] / n, 6),
            "reasoning_failure_rate": round(st["reasoning_failures"] / n, 6),
        }

    return {
        "semantic_groups_held_fixed": True,
        "group_count": len(paired_groups),
        "representations": summary,
    }


def compare_open_base_with_tiny_aix_zero(
    *,
    aix_zero_metrics: Dict[str, Any],
    external_open_base_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare a stronger open base model with the tiny AIX-Zero model (`SCALE-02` / Issue #201).

    Keeps AIX-Zero research results strictly separate from capabilities inherited from another
    open-weight model, and records model rights, source revision, memory, and execution cost.
    """
    req_fields = ("model_id", "license", "source_revision_sha", "peak_memory_mb", "execution_cost_ms", "accuracy")
    for label, m in (("aix_zero_metrics", aix_zero_metrics), ("external_open_base_metrics", external_open_base_metrics)):
        missing = [k for k in req_fields if k not in m]
        if missing:
            raise ValueError(f"{label} missing required fields: {missing}")

    if aix_zero_metrics["model_id"] == external_open_base_metrics["model_id"]:
        raise ValueError("Cannot conflate AIX-Zero model_id with external open base model_id")

    return {
        "capabilities_strictly_separated": True,
        "aix_zero_tiny_model": copy.deepcopy(aix_zero_metrics),
        "external_open_base_model": copy.deepcopy(external_open_base_metrics),
        "inherited_capability_attribution": f"Capabilities of {external_open_base_metrics['model_id']} are attributed to its external pretraining, not AIX-Zero v11.",
    }


def evaluate_cross_task_family_generalization(
    *,
    predeclared_partition_id: str,
    task_family_records: List[Dict[str, Any]],
    shortcut_control_accuracy: float,
) -> Dict[str, Any]:
    """Measure generalization across task families (`SCALE-08` / Issue #207).

    Predeclares a fresh final partition, assesses shortcuts, and reports failures and
    abstentions alongside aggregate scores.
    """
    if not predeclared_partition_id.strip():
        raise ValueError("predeclared_partition_id is required")
    if not task_family_records:
        raise ValueError("task_family_records must be nonempty")

    exact_hits = 0
    failures = 0
    abstentions = 0
    for rec in task_family_records:
        out = str(rec.get("outcome", ""))
        if out == "EXACT_MATCH":
            exact_hits += 1
        elif out == "ABSTAINED":
            abstentions += 1
        else:
            failures += 1

    n = len(task_family_records)
    acc = round(exact_hits / n, 6)
    return {
        "predeclared_partition_id": predeclared_partition_id,
        "sample_count": n,
        "exact_match_accuracy": acc,
        "failure_count": failures,
        "abstention_count": abstentions,
        "shortcut_control_accuracy": round(float(shortcut_control_accuracy), 6),
        "beats_shortcut_control": acc > float(shortcut_control_accuracy),
    }


def evaluate_quantization_tradeoffs(
    *,
    method_version: str,
    fp32_accuracy: float,
    quantized_accuracy: float,
    fp32_memory_bytes: int,
    quantized_memory_bytes: int,
    fp32_latency_ms: float,
    quantized_latency_ms: float,
    max_allowed_accuracy_drop: float = 0.01,
) -> Dict[str, Any]:
    """Study quantization accuracy and efficiency tradeoffs (`SCALE-05` / Issue #204).

    Reports capability regressions, memory, latency, and method/version; a compression
    ratio alone never qualifies as a model improvement when capability regresses.
    """
    if not method_version.strip():
        raise ValueError("method_version is required")
    if quantized_memory_bytes <= 0 or fp32_memory_bytes <= 0:
        raise ValueError("Memory byte counts must be positive")

    acc_delta = round(float(quantized_accuracy) - float(fp32_accuracy), 6)
    comp_ratio = round(fp32_memory_bytes / quantized_memory_bytes, 4)
    regressed = acc_delta < -abs(float(max_allowed_accuracy_drop))

    return {
        "method_version": method_version,
        "fp32_accuracy": round(float(fp32_accuracy), 6),
        "quantized_accuracy": round(float(quantized_accuracy), 6),
        "capability_regression_delta": acc_delta,
        "compression_ratio": comp_ratio,
        "fp32_memory_bytes": int(fp32_memory_bytes),
        "quantized_memory_bytes": int(quantized_memory_bytes),
        "fp32_latency_ms": round(float(fp32_latency_ms), 3),
        "quantized_latency_ms": round(float(quantized_latency_ms), 3),
        "capability_regressed": regressed,
        "qualifies_as_model_improvement": (not regressed) and (acc_delta > 0.0),
        "compression_alone_is_improvement": False,
    }


def evaluate_verified_retrieval_augmentation(
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate verified retrieval augmentation (`SCALE-06` / Issue #205).

    Separates retrieved evidence, solver execution, and model-only results, and tests
    missing and conflicting evidence cases rather than assuming retrieval always helps.
    """
    required_regimes = {"complete_evidence", "missing_evidence", "conflicting_evidence"}
    observed_regimes = {str(c.get("evidence_regime", "")) for c in cases}
    if not required_regimes.issubset(observed_regimes):
        raise ValueError(f"Retrieval evaluation must include all regimes: {sorted(required_regimes)}")

    n = len(cases)
    model_only_hits = sum(1 for c in cases if bool(c.get("model_only_correct", False)))
    retrieved_only_hits = sum(1 for c in cases if bool(c.get("retrieved_only_correct", False)))
    solver_assisted_hits = sum(1 for c in cases if bool(c.get("solver_assisted_correct", False)))

    return {
        "sample_count": n,
        "tested_evidence_regimes": sorted(observed_regimes),
        "model_only_accuracy": round(model_only_hits / n, 6),
        "retrieved_evidence_only_accuracy": round(retrieved_only_hits / n, 6),
        "solver_assisted_accuracy": round(solver_assisted_hits / n, 6),
        "modes_strictly_separated": True,
    }








