"""Bridge connecting AIX-Zero checkpoints and baseline reports to EVAL-001..006 (`EVAL-01`).

Reuses `evaluation_result.py`, `provider_adapters.py`, `benchmark_runner.py`, and
`statistical_gate.py` so AIX-Zero evaluations validate against the repository's
canonical JSON schemas while strictly separating `neural_only` and `symbolic_assisted`
execution modes.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional, Tuple

from artifact_io import canonical_json
from benchmark_runner import run_benchmark
from evaluation_result import validate_evaluation_result
from provider_adapters import ProviderResponse
from statistical_gate import paired_interval, validate_policy

from aix_zero.architecture import AIXZeroModel
from aix_zero.tokenizer import AIXZeroTokenizer
from aix_zero.train_from_scratch import extract_answer

VALID_EXECUTION_MODES = ("neural_only", "symbolic_assisted")


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
                    "inflight_alu": mode_name == "symbolic_assisted",
                    "physical_parameters": baseline_report["physical_parameters"],
                    ** copy.deepcopy(settings),
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


def assess_aix_paired_results(
    baseline_result: Dict[str, Any],
    candidate_result: Dict[str, Any],
    domains_by_sample_id: Dict[str, str],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """Run simultaneous Hoeffding paired intervals (`EVAL-006`) over two validated AIX-Zero runs."""
    validate_evaluation_result(baseline_result)
    validate_evaluation_result(candidate_result)
    validate_policy(policy)

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
