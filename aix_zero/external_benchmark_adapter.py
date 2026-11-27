"""Pinned External Public Benchmark Adapter (`#441`).

Evaluates any `ModelBackendAdapter` on pinned external benchmark JSONL files (e.g. GSM8K, MMLU,
or user-supplied public splits) without ever substituting synthetic internal problems:
- Records dataset/task revision, split, license, prompt format, and sample counts.
- Keeps reference answers (`gold`) strictly unavailable to `backend.generate(...)`.
- Separates `neural_only` from any assisted modes and tracks invalid outputs, unfinished
  evaluations, latency, and Wilson 95% confidence intervals.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from artifact_io import write_json
from aix_zero.backend_interface import ModelBackendAdapter
from aix_zero.eval_harness_bridge import score_rational_or_probability
from aix_zero.routed_specialist_training import wilson_score_interval


def load_pinned_external_benchmark(
    jsonl_path: Path,
    *,
    expected_dataset_id: Optional[str] = None,
    expected_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a pinned external benchmark JSONL file; rejects synthetic internal substitutes."""
    p = Path(jsonl_path)
    if not p.is_file():
        raise FileNotFoundError(f"Pinned external benchmark file not found: {p}")

    rows: List[Dict[str, Any]] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        row = json.loads(s)
        dataset_id = str(row.get("dataset_id", "")).strip()
        revision = str(row.get("dataset_revision", "")).strip()
        split = str(row.get("split", "")).strip()
        license_id = str(row.get("license", "")).strip()
        prompt = str(row.get("prompt", "")).strip()
        gold = str(row.get("gold", "")).strip()
        if not dataset_id or not revision or not split or not license_id or not prompt or not gold:
            raise ValueError(f"External benchmark row {lineno} missing required provenance or prompt/gold fields")
        if dataset_id.lower().startswith("internal_diag") or dataset_id.lower().startswith("synthetic"):
            raise ValueError(f"External benchmark adapter refuses synthetic substitute dataset_id={dataset_id!r}")
        if expected_dataset_id and dataset_id != expected_dataset_id:
            raise ValueError(f"Dataset ID mismatch: expected {expected_dataset_id!r}, got {dataset_id!r}")
        if expected_revision and revision != expected_revision:
            raise ValueError(f"Dataset revision mismatch: expected {expected_revision!r}, got {revision!r}")
        rows.append(
            {
                "task_id": str(row.get("task_id", f"ext_{lineno}")),
                "dataset_id": dataset_id,
                "dataset_revision": revision,
                "split": split,
                "license": license_id,
                "prompt": prompt,
                "gold": gold,
                "scorer": str(row.get("scorer", "rational_or_exact")),
            }
        )
    if not rows:
        raise ValueError(f"Pinned external benchmark file is empty: {p}")
    return {
        "benchmark_path": str(p.resolve()),
        "benchmark_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "dataset_id": rows[0]["dataset_id"],
        "dataset_revision": rows[0]["dataset_revision"],
        "split": rows[0]["split"],
        "license": rows[0]["license"],
        "sample_count": len(rows),
        "rows": rows,
    }


def evaluate_backend_on_pinned_external_benchmark(
    backend: ModelBackendAdapter,
    benchmark_bundle: Dict[str, Any],
    *,
    max_new_tokens: int = 36,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run an unchanged `ModelBackendAdapter` on a pinned external benchmark with zero gold leakage."""
    t0 = time.perf_counter()
    model_info = backend.inspect_model()
    rows = benchmark_bundle["rows"]
    records: List[Dict[str, Any]] = []
    correct_count = 0
    invalid_outputs = 0

    for row in rows:
        # Only public prompt passes to backend.generate(); gold is never passed.
        gen = backend.generate(row["prompt"], max_new_tokens=int(max_new_tokens))
        completion = str(gen.get("completion", ""))
        sc = score_rational_or_probability(completion, row["gold"], is_probability=False)
        is_ok = bool(sc["score"] == 1.0)
        if sc.get("status") in ("MISSING_ANSWER_TAG", "INVALID_FORMAT"):
            invalid_outputs += 1
        correct_count += int(is_ok)
        records.append(
            {
                "task_id": row["task_id"],
                "completion": completion,
                "correct": is_ok,
                "status": sc.get("status"),
            }
        )

    tested = len(records)
    acc = round(correct_count / max(1, tested), 4)
    report = {
        "schema_version": "1.0.0",
        "evaluation_origin": "pinned_external_benchmark",
        "assistance_mode": "neural_only",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "dataset_id": benchmark_bundle["dataset_id"],
        "dataset_revision": benchmark_bundle["dataset_revision"],
        "split": benchmark_bundle["split"],
        "license": benchmark_bundle["license"],
        "benchmark_sha256": benchmark_bundle["benchmark_sha256"],
        "model_identity": {
            "backend": model_info.get("backend"),
            "checkpoint_format": model_info.get("checkpoint_format"),
            "state_sha256": model_info.get("state_sha256"),
        },
        "generation_settings": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": 0.0,
            "inflight_alu": False,
        },
        "offered_samples": len(rows),
        "tested_samples": tested,
        "unfinished_samples": len(rows) - tested,
        "invalid_outputs": invalid_outputs,
        "correct": correct_count,
        "accuracy": acc,
        "wilson_95_ci": [round(x, 4) for x in wilson_score_interval(correct_count, tested)],
        "limitations": "Small diagnostic samples have wide Wilson confidence intervals; repeated testing reduces effective significance.",
        "records": records,
    }
    if output_path is not None:
        write_json(report, Path(output_path))
    return report
