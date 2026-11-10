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
