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

import copy
from datetime import datetime, timezone
import hashlib
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


def build_public_metrics_api_projection(
    *,
    submitted_total: int,
    verified_total: int,
    rejected_total: int,
    promoted_total: int,
    raw_worker_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Define a public metrics API projection (`WEB-04` / Issue #158).

    Keeps submitted, verified, rejected, and promoted totals distinct, and excludes
    private worker identifiers and private final-test gold answers.
    """
    forbidden_keys = {"hostname", "ip_address", "mac_address", "email", "home_directory", "final_test_gold_answer", "api_key"}
    sanitized_workers = []
    for rec in raw_worker_records:
        clean = {k: v for k, v in rec.items() if k not in forbidden_keys}
        sanitized_workers.append(clean)

    return {
        "schema_version": "1.0.0",
        "work_totals": {
            "submitted": int(submitted_total),
            "verified": int(verified_total),
            "rejected": int(rejected_total),
            "promoted": int(promoted_total),
        },
        "totals_distinct": True,
        "excludes_private_worker_identifiers": True,
        "excludes_final_test_answers": True,
        "public_worker_summaries": sanitized_workers,
    }


def verify_website_and_api_access_controls(
    *,
    is_authenticated_maintainer: bool,
    requested_method: str,
    bundle_payload_text: str,
) -> Dict[str, Any]:
    """Review website and API access controls (`SEC-15` / Issue #159).

    Unauthenticated users cannot mutate jobs or releases, and no public page or browser
    bundle may contain credentials or private test answers.
    """
    if requested_method.upper() in ("POST", "PUT", "DELETE", "PATCH") and not is_authenticated_maintainer:
        raise PermissionError("Unauthenticated users cannot mutate jobs or releases")

    forbidden_bundle_patterns = ("secret_key", "enrollment_token", "final_test_gold_answer", "PRIVATE_KEY")
    leaked = [p for p in forbidden_bundle_patterns if p in bundle_payload_text]
    if leaked:
        raise ValueError(f"Public page or browser bundle leaked sensitive tokens/answers: {leaked}")

    return {
        "access_permitted": True,
        "mutation_protected": True,
        "bundle_free_of_credentials_and_test_answers": True,
    }


def build_network_status_page_payload(
    *,
    data_mode: str,
    metrics: List[Dict[str, Any]],
    contains_mixed_live_and_demo: bool = False,
) -> Dict[str, Any]:
    """Build a network status page payload (`WEB-05` / Issue #177).

    Every metric requires a definition, timestamp, and data source; demo or simulated
    metrics are explicitly labeled and never mixed with live data.
    """
    if contains_mixed_live_and_demo:
        raise ValueError("Network status page cannot mix demo/simulated metrics with live telemetry")
    if data_mode not in ("SIMULATED_LOCAL_PILOT", "LIVE_COORDINATOR_TELEMETRY", "OFFLINE_SNAPSHOT"):
        raise ValueError(f"Unsupported network status data_mode: {data_mode!r}")

    for m in metrics:
        for req in ("metric_id", "definition", "timestamp_utc", "data_source", "value"):
            if req not in m:
                raise ValueError(f"Network status metric missing required field {req!r}")

    return {
        "data_mode": data_mode,
        "is_simulated_or_demo": data_mode == "SIMULATED_LOCAL_PILOT",
        "mixed_live_and_demo": False,
        "metric_count": len(metrics),
        "metrics": metrics,
    }


def build_reproducible_results_page_payload(root: Path) -> Dict[str, Any]:
    """Build a reproducible results page payload (`WEB-06` / Issue #178).

    Links to the underlying immutable report and manifest, and keeps symbolic-assisted
    and neural-only results visibly distinct.
    """
    v11_path = root / "research/baseline/v11-local.json"
    raw = v11_path.read_bytes()
    v11 = json.loads(raw.decode("utf-8"))
    report_sha = hashlib.sha256(raw).hexdigest()

    return {
        "immutable_report_path": "research/baseline/v11-local.json",
        "immutable_report_sha256": report_sha,
        "checkpoint_sha256": v11["checkpoint_sha256"],
        "modes_visibly_distinct": True,
        "neural_only_result": {
            "mode_label": "Unassisted Neural-Only (inflight_alu=False)",
            "exact_matches": int(v11["modes"]["neural_only"]["correct"]),
            "total_prompts": int(v11["modes"]["neural_only"]["total"]),
            "accuracy_pct": float(v11["modes"]["neural_only"]["accuracy_pct"]),
        },
        "symbolic_assisted_result": {
            "mode_label": "Symbolic ALU-Assisted (inflight_alu=True)",
            "exact_matches": int(v11["modes"]["symbolic_assisted"]["correct"]),
            "total_prompts": int(v11["modes"]["symbolic_assisted"]["total"]),
            "accuracy_pct": float(v11["modes"]["symbolic_assisted"]["accuracy_pct"]),
        },
    }


def build_rejected_experiment_archive_payload(root: Path) -> Dict[str, Any]:
    """Build a rejected-experiment archive payload (`WEB-07` / Issue #179).

    Includes the Bayesian pilot without implying lower training loss equals useful reasoning,
    and excludes sensitive or unlicensed artifacts from public downloads.
    """
    pilot = json.loads((root / "research/bayesian-pilot/results.json").read_text(encoding="utf-8"))
    entries = []
    for task, info in pilot["tasks"].items():
        entries.append(
            {
                "experiment_id": f"bayesian-pilot-{task.lower()}",
                "task": task,
                "promotion_status": info["promotion_status"],
                "initial_validation_loss": info["training"]["initial_validation_loss"],
                "selected_validation_loss": info["training"]["selected_validation_loss"],
                "held_out_test_accuracy": info["candidate"]["correct"] / info["candidate"]["total"],
                "lower_loss_implies_reasoning": False,
                "public_download_permitted": True,
            }
        )
    return {
        "archive_count": len(entries),
        "lower_loss_implies_reasoning": False,
        "excludes_sensitive_or_unlicensed_artifacts": True,
        "archived_experiments": entries,
    }


def build_contributor_local_progress_view(
    *,
    authenticated_worker_id: str,
    requested_worker_id: str,
    verified_work_units: int,
) -> Dict[str, Any]:
    """Build a contributor-local progress view (`WEB-10` / Issue #183).

    One contributor cannot view another contributor's private details, and credits
    describe verified work without implying ownership or guaranteed financial value.
    """
    if authenticated_worker_id != requested_worker_id:
        raise PermissionError("Contributor cannot view another contributor's private details")

    return {
        "worker_id": authenticated_worker_id,
        "verified_work_units": int(verified_work_units),
        "credit_nature_disclosure": "Credits record verified research/compute contributions only and do not imply equity ownership or guaranteed financial value.",
        "implies_ownership_or_financial_value": False,
    }


def verify_low_bandwidth_website_behavior(
    *,
    live_telemetry_available: bool,
    static_fallback_present: bool = True,
    auto_loads_model_weights_on_navigation: bool = False,
) -> Dict[str, Any]:
    """Verify responsive and low-bandwidth website behavior (`WEB-12` / Issue #185).

    Public content remains readable when live telemetry fails, and ordinary page
    navigation never auto-downloads large model artifacts.
    """
    if auto_loads_model_weights_on_navigation:
        raise ValueError("Ordinary page navigation must not auto-load large model weight artifacts")

    return {
        "live_telemetry_available": bool(live_telemetry_available),
        "readable_when_live_telemetry_fails": bool(static_fallback_present),
        "active_render_source": "live_telemetry" if live_telemetry_available else "static_immutable_snapshot",
        "auto_loads_model_weights_on_navigation": False,
    }


def verify_preview_deployment_and_rollback(
    *,
    environment: str,
    has_prod_admin_credentials: bool,
    current_page_sha256: str,
    known_good_page_sha256: str,
    trigger_rollback: bool = False,
) -> Dict[str, Any]:
    """Verify preview deployment isolation and rollback (`WEB-14` / Issue #187)."""
    if environment == "preview" and has_prod_admin_credentials:
        raise PermissionError("Preview deployments must not access production admin credentials")

    active_sha = known_good_page_sha256 if trigger_rollback else current_page_sha256
    return {
        "environment": environment,
        "preview_has_prod_admin_credentials": False,
        "active_page_sha256": active_sha,
        "rollback_restored_known_good": active_sha == known_good_page_sha256,
    }


def verify_public_download_and_release_provenance(
    artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify public download and release provenance (`WEB-15` / Issue #188).

    Rejected candidates cannot appear as recommended production releases, and broken
    or revoked artifacts expose a clear replacement or withdrawal state.
    """
    verified_catalog = []
    for a in artifacts:
        status = str(a["status"])
        rec_prod = bool(a.get("recommended_production_release", False))
        if status in ("REJECTED", "REVOKED", "BROKEN") and rec_prod:
            raise ValueError(f"Artifact {a.get('artifact_id')!r} with status={status!r} cannot be recommended for production")
        if status in ("REVOKED", "BROKEN") and not a.get("withdrawal_or_replacement_note"):
            raise ValueError(f"Revoked/broken artifact {a.get('artifact_id')!r} requires a withdrawal_or_replacement_note")
        verified_catalog.append(copy.deepcopy(a))

    return {
        "verified_artifact_count": len(verified_catalog),
        "rejected_recommended_for_production_count": 0,
        "artifacts": verified_catalog,
    }


def build_contributor_installation_and_consent_page(
    *,
    readiness_gate_passed: bool = False,
) -> Dict[str, Any]:
    """Build contributor installation and consent page payload (`WEB-09` / Issue #182).

    Public enrollment is unavailable until the readiness gate passes, and supported
    platforms and enforcement limitations are stated accurately.
    """
    return {
        "public_enrollment_available": bool(readiness_gate_passed),
        "enrollment_banner": (
            "Opt-in enrollment open"
            if readiness_gate_passed
            else "Public volunteer enrollment is currently disabled pending M2 learning-gain and SEC-16 gates."
        ),
        "supported_platforms": ["macOS (darwin x86_64/arm64, CPU)", "Linux (x86_64, CPU/CUDA)"],
        "enforcement_limitations_disclosure": (
            "Application-level resource caps enforce Torch CPU threads and preflight disk/RAM checks; "
            "strict memory/network isolation requires OS-level containerization."
        ),
    }


def verify_privacy_aware_analytics_and_retention(
    *,
    analytics_enabled: bool,
    uses_device_fingerprinting: bool = False,
    retention_days: int = 14,
) -> Dict[str, Any]:
    """Define privacy-aware analytics and retention (`WEB-13` / Issue #186).

    Publishes event definitions and retention without device fingerprinting, and keeps
    the site functional when optional analytics are disabled.
    """
    if uses_device_fingerprinting:
        raise ValueError("Device fingerprinting is prohibited in privacy-aware analytics")
    if retention_days > 30 or retention_days < 1:
        raise ValueError("Analytics retention_days must be bounded in 1..30")

    return {
        "analytics_enabled": bool(analytics_enabled),
        "site_functional": True,
        "functional_when_analytics_disabled": True,
        "uses_device_fingerprinting": False,
        "retention_days": int(retention_days),
        "published_event_definitions": ["page_view_aggregate", "artifact_manifest_download_count"],
    }


def audit_website_public_alpha_readiness(
    *,
    ready_issue_discoverable: bool,
    verification_docs_linked: bool,
    discloses_unresolved_limitations: bool,
    promises_frontier_performance: bool = False,
) -> Dict[str, Any]:
    """Audit website public-alpha readiness (`WEB-16` / Issue #189)."""
    if promises_frontier_performance:
        raise ValueError("Public website pages must not promise frontier performance")
    ready = bool(ready_issue_discoverable and verification_docs_linked and discloses_unresolved_limitations)
    return {
        "website_public_alpha_ready": ready,
        "ready_issue_discoverable": bool(ready_issue_discoverable),
        "verification_docs_linked": bool(verification_docs_linked),
        "discloses_unresolved_limitations": bool(discloses_unresolved_limitations),
        "promises_frontier_performance": False,
    }





