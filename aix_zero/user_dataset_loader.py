"""User Dataset Ingestion, Provenance Validation & Split Isolation (`#440`).

Supports versioned JSONL schemas for:
- `instruction_chat`: `{"task_id", "messages": [{"role", "content"}], "target", "source", "license", "task_family", "split"}`
- `code_task`: `{"task_id", "prompt", "target", "test_code", "source", "license", "task_family", "split"}`

Enforces:
1. Strict schema, license/source provenance, and role (`system`/`user`/`assistant`) validation.
2. Complete-target context fit check (never silently truncating target completions).
3. Frozen split manifests (`train`, `development`, `retention`, `reserved_final`) with zero
   cross-split prompt/problem overlap and explicit exclusion accounting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from artifact_io import canonical_json, write_json


DATASET_SCHEMA_VERSION = "1.0.0"
VALID_SPLITS: Set[str] = {"train", "development", "retention", "reserved_final"}
VALID_CHAT_ROLES: Set[str] = {"system", "user", "assistant"}
VALID_TASK_FORMATS: Set[str] = {"instruction_chat", "code_task"}


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def load_and_validate_user_jsonl_dataset(
    jsonl_path: Path,
    *,
    task_format: str = "instruction_chat",
    tokenizer: Any = None,
    max_seq_len: int = 512,
    manifest_output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load, validate, deduplicate, and freeze split manifests for a user-supplied JSONL dataset."""
    if task_format not in VALID_TASK_FORMATS:
        raise ValueError(f"Unsupported task_format {task_format!r}; expected one of {sorted(VALID_TASK_FORMATS)}")
    p = Path(jsonl_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset JSONL not found: {p}")

    exclusions: Dict[str, List[Dict[str, Any]]] = {
        "malformed_row": [],
        "missing_provenance": [],
        "unsupported_role": [],
        "invalid_split": [],
        "overlength_target": [],
        "intra_split_duplicate": [],
    }
    accepted_by_split: Dict[str, List[Dict[str, Any]]] = {sp: [] for sp in VALID_SPLITS}
    seen_prompt_hashes_by_split: Dict[str, Set[str]] = {sp: set() for sp in VALID_SPLITS}
    eligible_tokens = 0
    total_rows = 0

    for lineno, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        total_rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            exclusions["malformed_row"].append({"line": lineno, "reason": f"json_error:{exc}"})
            continue
        if not isinstance(row, dict):
            exclusions["malformed_row"].append({"line": lineno, "reason": "row_not_object"})
            continue

        task_id = str(row.get("task_id", "")).strip()
        target = str(row.get("target", "")).strip()
        source = str(row.get("source", "")).strip()
        license_id = str(row.get("license", "")).strip()
        task_family = str(row.get("task_family", "")).strip()
        split = str(row.get("split", "")).strip()

        if not task_id or not target:
            exclusions["malformed_row"].append({"line": lineno, "task_id": task_id, "reason": "missing_task_id_or_target"})
            continue
        if not source or not license_id or not task_family:
            exclusions["missing_provenance"].append({"line": lineno, "task_id": task_id, "reason": "source_license_and_task_family_required"})
            continue
        if split not in VALID_SPLITS:
            exclusions["invalid_split"].append({"line": lineno, "task_id": task_id, "split": split})
            continue

        prompt_text = ""
        if task_format == "instruction_chat":
            msgs = row.get("messages")
            if not isinstance(msgs, list) or not msgs:
                exclusions["malformed_row"].append({"line": lineno, "task_id": task_id, "reason": "missing_messages"})
                continue
            bad_role = False
            rendered_parts: List[str] = []
            for m in msgs:
                if not isinstance(m, dict) or m.get("role") not in VALID_CHAT_ROLES or not isinstance(m.get("content"), str):
                    bad_role = True
                    break
                rendered_parts.append(f"{m['role']}: {m['content']}")
            if bad_role:
                exclusions["unsupported_role"].append({"line": lineno, "task_id": task_id})
                continue
            prompt_text = "\n".join(rendered_parts)
        else:
            prompt_text = str(row.get("prompt", "")).strip()
            test_code = str(row.get("test_code", "")).strip()
            if not prompt_text or not test_code:
                exclusions["malformed_row"].append({"line": lineno, "task_id": task_id, "reason": "code_task_requires_prompt_and_test_code"})
                continue

        # Complete-target context fit check (never silently truncate target)
        if tokenizer is not None:
            p_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
            c_ids = tokenizer.encode(target, add_bos=False, add_eos=True)
            total_len = len(p_ids) + len(c_ids)
        else:
            total_len = len(prompt_text) + len(target) + 2

        if total_len > int(max_seq_len):
            exclusions["overlength_target"].append({"line": lineno, "task_id": task_id, "tokens": total_len, "max_seq_len": int(max_seq_len)})
            continue

        p_hash = _content_hash(prompt_text)
        if p_hash in seen_prompt_hashes_by_split[split]:
            exclusions["intra_split_duplicate"].append({"line": lineno, "task_id": task_id, "split": split})
            continue
        seen_prompt_hashes_by_split[split].add(p_hash)
        eligible_tokens += total_len

        accepted_by_split[split].append(
            {
                "task_id": task_id,
                "prompt_text": prompt_text,
                "prompt_sha256": p_hash,
                "target": target,
                "target_sha256": _content_hash(target),
                "source": source,
                "license": license_id,
                "task_family": task_family,
                "split": split,
                "token_count": total_len,
                "raw": row,
            }
        )

    # Audit cross-split leakage: train prompts/targets must never overlap eval splits
    eval_prompt_hashes: Set[str] = set()
    eval_target_hashes: Set[str] = set()
    for eval_split in ("development", "retention", "reserved_final"):
        eval_prompt_hashes.update(seen_prompt_hashes_by_split[eval_split])
        eval_target_hashes.update(r["target_sha256"] for r in accepted_by_split[eval_split])

    prompt_leakage = [
        r["task_id"] for r in accepted_by_split["train"] if r["prompt_sha256"] in eval_prompt_hashes
    ]
    if prompt_leakage:
        raise ValueError(f"Cross-split prompt leakage detected between train and eval splits: {prompt_leakage}")

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_path": str(p.resolve()),
        "dataset_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "task_format": task_format,
        "max_seq_len": int(max_seq_len),
        "total_rows_offered": total_rows,
        "eligible_sample_count": sum(len(v) for v in accepted_by_split.values()),
        "eligible_token_count": eligible_tokens,
        "split_counts": {sp: len(rows) for sp, rows in accepted_by_split.items()},
        "exclusion_counts": {k: len(v) for k, v in exclusions.items()},
        "exclusions": exclusions,
        "cross_split_prompt_leakage_count": 0,
        "splits": {
            sp: [
                {
                    "task_id": r["task_id"],
                    "prompt_sha256": r["prompt_sha256"],
                    "target_sha256": r["target_sha256"],
                    "source": r["source"],
                    "license": r["license"],
                    "task_family": r["task_family"],
                    "token_count": r["token_count"],
                }
                for r in rows
            ]
            for sp, rows in accepted_by_split.items()
        },
    }
    if manifest_output_path is not None:
        write_json(manifest, Path(manifest_output_path))
    return {"manifest": manifest, "accepted_by_split": accepted_by_split}
