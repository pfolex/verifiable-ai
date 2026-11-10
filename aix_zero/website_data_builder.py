"""Evidence-based website overview and GitHub-issues-backed roadmap view builder (`WEB-02`, `WEB-03`).

Guarantees:
1. Never describes copied routed expert snapshots (`existing_routed_expert_snapshot`) or
   symbolic-assisted toy scores (`105/105`) as standalone frontier neural intelligence.
2. Links every empirical claim to an immutable repository artifact (`research/baseline/v11-local.json`,
   `research/dispatch/local.json`, `research/bayesian-pilot/results.json`) or explicitly marks
   aspirational items as `PROPOSAL_NOT_YET_VALIDATED`.
3. Separates completed roadmap issue counts from model capability metrics and handles stale or
   unavailable GitHub responses gracefully with explicit timestamps.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def build_project_overview_payload(
    *,
    baseline_path: Path = ROOT / "research/baseline/v11-local.json",
    dispatch_path: Path = ROOT / "research/dispatch/local.json",
    pilot_path: Path = ROOT / "research/bayesian-pilot/results.json",
) -> Dict[str, Any]:
    """Build the evidence-based project overview payload (`WEB-02` / Issue #175)."""
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    dispatch = json.loads(Path(dispatch_path).read_text(encoding="utf-8"))
    pilot = json.loads(Path(pilot_path).read_text(encoding="utf-8"))

    neural = baseline["modes"]["neural_only"]
    assisted = baseline["modes"]["symbolic_assisted"]
    cfg = baseline["config"]
    total_params = int(baseline["physical_parameters"])
    idle_experts = max(0, int(cfg.get("n_experts", 4)) - int(cfg.get("moe_top_k", 2))) if cfg.get("use_moe") else 0
    expert_params = 3 * int(cfg.get("d_model", 128)) * max(64, int(cfg.get("d_ff", 256)) // 2)
    active_sparse_params = total_params - idle_experts * expert_params

    return {
        "schema_version": "1.0.0",
        "project_title": "AIX-Zero / Verifiable-AI",
        "scientific_honesty_notice": (
            "Copied routed-expert snapshots and symbolic-assisted arithmetic repairs are NOT frontier neural "
            "intelligence. Neural-only and symbolic-assisted scores are always reported separately."
        ),
        "measured_capabilities": {
            "active_baseline_checkpoint": baseline["checkpoint"],
            "checkpoint_sha256": baseline["checkpoint_sha256"],
            "dispatch_verified_checkpoint_sha256": dispatch["checkpoint_sha256"],
            "physical_learned_parameters": total_params,
            "active_parameters_per_token_sparse": active_sparse_params,
            "recurrent_passes": int(cfg["n_recurrent_steps"]),
            "neural_only_accuracy": {
                "correct": int(neural["correct"]),
                "total": int(neural["total"]),
                "accuracy_pct": float(neural["accuracy_pct"]),
                "p95_response_ms": float(neural["p95_response_ms"]),
                "evidence_path": "research/baseline/v11-local.json",
            },
            "symbolic_assisted_accuracy": {
                "correct": int(assisted["correct"]),
                "total": int(assisted["total"]),
                "accuracy_pct": float(assisted["accuracy_pct"]),
                "p95_response_ms": float(assisted["p95_response_ms"]),
                "evidence_path": "research/baseline/v11-local.json",
            },
        },
        "rejected_and_negative_experiments": [
            {
                "experiment_id": "bayesian-pilot-bp-bs",
                "status": "REJECTED",
                "bp_neural_only": f"{pilot['tasks']['BP']['candidate']['correct']}/{pilot['tasks']['BP']['candidate']['total']}",
                "bs_neural_only": f"{pilot['tasks']['BS']['candidate']['correct']}/{pilot['tasks']['BS']['candidate']['total']}",
                "evidence_path": "research/bayesian-pilot/results.json",
            }
        ],
        "proposals_not_yet_validated": [
            {
                "proposal_id": "distributed-specialist-cartridge-network",
                "claim_type": "PROPOSAL_NOT_YET_VALIDATED",
                "description": "Federated training of verified domain specialist cartridges across volunteer workers.",
            }
        ],
    }


def build_roadmap_view_payload(
    issues: Optional[List[Dict[str, Any]]],
    *,
    fetched_at_utc: str,
    max_age_seconds: int = 3600,
    now_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a GitHub-issues-backed roadmap view (`WEB-03` / Issue #176).

    Explicitly separates issue management progress from model intelligence metrics and
    marks stale or missing GitHub API responses gracefully.
    """
    t_fetch = datetime.fromisoformat(fetched_at_utc.replace("Z", "+00:00"))
    t_now = (
        datetime.fromisoformat(now_utc.replace("Z", "+00:00"))
        if now_utc is not None
        else datetime.now(timezone.utc)
    )
    age_seconds = max(0.0, (t_now - t_fetch).total_seconds())
    is_stale = (issues is None) or (age_seconds > max_age_seconds)

    if issues is None:
        return {
            "schema_version": "1.0.0",
            "data_status": "UNAVAILABLE_FALLBACK",
            "is_stale": True,
            "fetched_at_utc": fetched_at_utc,
            "age_seconds": round(age_seconds, 2),
            "intelligence_progress_disclaimer": (
                "Completed issue counts measure engineering/specification hygiene only and do NOT represent model capability gains."
            ),
            "workstreams": {},
            "ready_to_claim": [],
        }

    workstreams: Dict[str, Dict[str, int]] = {}
    ready_to_claim: List[Dict[str, Any]] = []

    for iss in issues:
        title = str(iss.get("title", ""))
        m = re.search(r"\[AIX\s+([A-Z]+)-(\d{2})\]", title)
        ws = m.group(1) if m else "OTHER"
        code = f"{m.group(1)}-{m.group(2)}" if m else f"ISSUE-{iss.get('number')}"
        state = str(iss.get("state", "OPEN")).upper()
        labels = [
            (l["name"] if isinstance(l, dict) else str(l))
            for l in iss.get("labels", [])
        ]

        ws_entry = workstreams.setdefault(ws, {"open": 0, "closed": 0, "ready": 0, "blocked": 0})
        if state == "CLOSED":
            ws_entry["closed"] += 1
        else:
            ws_entry["open"] += 1
            if "status:ready" in labels:
                ws_entry["ready"] += 1
                ready_to_claim.append(
                    {
                        "number": iss.get("number"),
                        "code": code,
                        "title": title,
                        "url": iss.get("url") or f"https://github.com/pfolex/verifiable-ai/issues/{iss.get('number')}",
                    }
                )
            elif "status:blocked" in labels:
                ws_entry["blocked"] += 1

    return {
        "schema_version": "1.0.0",
        "data_status": "STALE_CACHED" if is_stale else "LIVE_FRESH",
        "is_stale": is_stale,
        "fetched_at_utc": fetched_at_utc,
        "age_seconds": round(age_seconds, 2),
        "intelligence_progress_disclaimer": (
            "Completed issue counts measure engineering/specification hygiene only and do NOT represent model capability gains."
        ),
        "workstreams": workstreams,
        "ready_to_claim": ready_to_claim,
    }


def build_research_updates_feed(updates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and format dated research updates (`WEB-08` / Issue #181).

    Requires each update to separate `completed_work`, `measurements`, `hypotheses`, and
    `next_experiments`, requires every claimed capability improvement to link to a reproducible
    `evidence_path` (or be marked `UNVERIFIED_HYPOTHESIS`), and preserves visible `corrections`.
    """
    if not updates:
        raise ValueError("updates list must be nonempty")

    validated: List[Dict[str, Any]] = []
    for entry in updates:
        for req in ("update_id", "date_utc", "completed_work", "measurements", "hypotheses", "next_experiments"):
            if req not in entry:
                raise ValueError(f"Update entry missing required field {req!r}")
        for m in entry["measurements"]:
            if m.get("claims_capability_improvement") and not m.get("evidence_path"):
                raise ValueError(
                    f"Measurement in {entry['update_id']!r} claims capability improvement without a reproducible evidence_path"
                )
        validated.append(
            {
                "update_id": str(entry["update_id"]),
                "date_utc": str(entry["date_utc"]),
                "completed_work": list(entry["completed_work"]),
                "measurements": list(entry["measurements"]),
                "hypotheses": [
                    dict(h, verification_status=h.get("verification_status", "UNVERIFIED_HYPOTHESIS"))
                    for h in entry["hypotheses"]
                ],
                "next_experiments": list(entry["next_experiments"]),
                "corrections": list(entry.get("corrections", [])),
            }
        )

    return {
        "schema_version": "1.0.0",
        "update_count": len(validated),
        "updates": validated,
    }


def audit_html_accessibility_and_keyboard_nav(html_text: str) -> Dict[str, Any]:
    """Automated accessibility and keyboard-navigation audit (`WEB-11` / Issue #184)
    supplemented with a documented manual review checklist.
    """
    checks = {
        "has_lang_attribute": bool(re.search(r"<html[^>]*\blang=", html_text, flags=re.IGNORECASE)),
        "has_skip_to_main_link": bool(re.search(r'href="#main', html_text, flags=re.IGNORECASE)),
        "has_main_landmark": "<main" in html_text.lower(),
        "has_nav_landmark": "<nav" in html_text.lower(),
        "has_h1_heading": "<h1" in html_text.lower(),
        "has_visible_focus_indicator_css": ":focus-visible" in html_text,
        "has_prefers_reduced_motion_css": "prefers-reduced-motion" in html_text,
    }
    passed = all(checks.values())
    return {
        "automated_passed": passed,
        "checks": checks,
        "manual_review_checklist": [
            "Tab through all interactive controls in DOM order without keyboard traps.",
            "Verify screen-reader table headers (<th scope='col'>) on benchmark scorecards.",
            "Verify WCAG AA 4.5:1 text contrast on status badges (ACTIVE, REJECTED, OFFLINE).",
        ],
    }

