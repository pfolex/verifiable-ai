"""End-to-End Bring-Your-Own-Model (BYOM) Bounded Experiment Pipeline (`#442` & `#444`).

Connects:
1. Pre-flight backend & model validation (`AIXZeroBackendAdapter` or `HuggingFaceBackendAdapter`).
2. User dataset provenance & split isolation (`load_and_validate_user_jsonl_dataset`).
3. Baseline evaluation on isolated `development` / `retention` splits (fail-closed, no gold leakage).
4. Bounded training step(s) with immutable base preservation.
5. Candidate reload & paired comparison under identical generation settings.
6. Record/reject-only gating by default (promotion requires `allow_promotion=True` AND non-regressed
   `retention` + improved `development` score), plus atomic bundle export (`export_verified_adapter_bundle`).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from artifact_io import write_json
from aix_zero.adapter_export import export_verified_adapter_bundle
from aix_zero.backend_interface import (
    AIXZeroBackendAdapter,
    HuggingFaceBackendAdapter,
    ModelBackendAdapter,
)
from aix_zero.eval_harness_bridge import score_rational_or_probability
from aix_zero.user_dataset_loader import load_and_validate_user_jsonl_dataset


def _extract_gold_from_target(target: str) -> str:
    s = target.strip()
    a_s, a_e = s.find("<ans>"), s.find("</ans>")
    if 0 <= a_s < a_e:
        return s[a_s + len("<ans>") : a_e].strip()
    return s


def _score_split(
    backend: ModelBackendAdapter,
    rows: List[Dict[str, Any]],
    *,
    max_new_tokens: int = 36,
) -> Dict[str, Any]:
    correct = 0
    for row in rows:
        gen = backend.generate(row["prompt_text"], max_new_tokens=int(max_new_tokens))
        gold = _extract_gold_from_target(row["target"])
        sc = score_rational_or_probability(gen["completion"], gold, is_probability=False)
        if sc["score"] == 1.0:
            correct += 1
    total = len(rows)
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / max(1, total), 4) if total else None,
    }


def run_byom_experiment(
    *,
    model_path: Path,
    dataset_path: Path,
    run_dir: Path,
    task_format: str = "instruction_chat",
    steps: int = 4,
    lr: float = 1e-3,
    max_seq_len: int = 112,
    max_new_tokens: int = 36,
    allow_promotion: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute one bounded, reproducible Bring-Your-Own-Model training & evaluation cycle."""
    run_root = Path(run_dir)
    if run_root.exists():
        raise FileExistsError(f"BYOM run_dir already exists; choose a fresh directory: {run_root}")
    m_path = Path(model_path)

    # 1. Pre-flight backend validation
    if m_path.is_dir():
        backend: ModelBackendAdapter = HuggingFaceBackendAdapter(m_path, requested_device=device)
        is_hf = True
    else:
        backend = AIXZeroBackendAdapter(m_path)
        is_hf = False

    base_info = backend.inspect_model()

    # 2. Validate & freeze dataset splits before fitting
    run_root.mkdir(parents=True, exist_ok=False)
    ds = load_and_validate_user_jsonl_dataset(
        dataset_path,
        task_format=task_format,
        tokenizer=getattr(backend, "tokenizer", None),
        max_seq_len=int(max_seq_len),
        manifest_output_path=run_root / "dataset_manifest.json",
    )
    splits = ds["accepted_by_split"]
    train_rows = splits["train"]
    dev_rows = splits["development"]
    ret_rows = splits["retention"]
    if not train_rows:
        raise ValueError("Dataset has zero eligible training rows")

    # 3. Baseline evaluation on unchanged base
    baseline_dev = _score_split(backend, dev_rows, max_new_tokens=max_new_tokens)
    baseline_ret = _score_split(backend, ret_rows, max_new_tokens=max_new_tokens)

    # 4. Bounded training & candidate artifact export via unified backend interface
    train_examples = [{"prompt": r["prompt_text"], "completion": r["target"]} for r in train_rows]
    if is_hf:
        cand_artifact = run_root / "candidate_adapter"
        train_report = backend.train_step(
            train_examples, lr=float(lr), steps=int(steps), output_adapter_dir=cand_artifact
        )
        reloaded: ModelBackendAdapter = HuggingFaceBackendAdapter(m_path, requested_device=device)
        reloaded.reload_artifact(cand_artifact)
    else:
        train_report = backend.train_step(train_examples, lr=float(lr), steps=int(steps))
        cand_artifact = run_root / "candidate_state.pt"
        torch.save(backend.model.state_dict(), cand_artifact)
        reloaded = AIXZeroBackendAdapter(m_path, state_dict_override_path=cand_artifact)
    cand_dev = _score_split(reloaded, dev_rows, max_new_tokens=max_new_tokens)
    cand_ret = _score_split(reloaded, ret_rows, max_new_tokens=max_new_tokens)

    dev_improved = (
        cand_dev["accuracy"] is not None
        and baseline_dev["accuracy"] is not None
        and cand_dev["accuracy"] > baseline_dev["accuracy"]
    )
    ret_held = (
        cand_ret["accuracy"] is None
        or baseline_ret["accuracy"] is None
        or cand_ret["accuracy"] >= baseline_ret["accuracy"]
    )
    # Begin in record/reject-only mode unless allow_promotion=True and both gates pass
    decision = "PROMOTED" if (allow_promotion and dev_improved and ret_held) else "REJECTED_RECORD_ONLY"

    export_manifest = export_verified_adapter_bundle(
        source_path=cand_artifact,
        output_dir=run_root / "exported_bundle",
        base_identity={
            "checkpoint": str(m_path.name),
            "state_sha256": base_info["state_sha256"],
        },
        training_recipe={"steps": int(steps), "lr": float(lr), "max_seq_len": int(max_seq_len)},
        dataset_provenance={
            "dataset_sha256": ds["manifest"]["dataset_sha256"],
            "split_counts": ds["manifest"]["split_counts"],
        },
        evaluation_summary={
            "baseline_dev": baseline_dev,
            "candidate_dev": cand_dev,
            "baseline_retention": baseline_ret,
            "candidate_retention": cand_ret,
            "decision": decision,
        },
        release_label="promoted_champion" if decision == "PROMOTED" else "rejected_candidate_record",
        verify_aix_zero_roundtrip_ckpt=None if is_hf else m_path,
    )

    summary = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "allow_promotion": bool(allow_promotion),
        "base_preserved": True,
        "training_report": train_report,
        "baseline": {"development": baseline_dev, "retention": baseline_ret},
        "candidate": {"development": cand_dev, "retention": cand_ret},
        "export_bundle_sha256": export_manifest["bundle_sha256"],
    }
    write_json(summary, run_root / "byom_summary.json")
    return summary
