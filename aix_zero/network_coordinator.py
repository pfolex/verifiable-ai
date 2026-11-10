"""Volunteer worker capability handshake, threat boundaries, and durable experiment queue
(`WORKER-01` #126, `SEC-01` #127, `NETWORK-02` #134).

Guarantees:
1. Worker capability handshakes never collect PII, private hostnames, MAC addresses, or
   filesystem home paths (`WORKER-01`).
2. Unsupported hardware or disallowed job types receive an explicit `NO_WORK_UNSUPPORTED_CAPABILITY`
   response rather than an unguarded lease.
3. `DurableExperimentQueue` persists job states (`QUEUED`, `LEASED`, `COMPLETED`, `FAILED`)
   atomically on disk so coordinator restarts preserve state and invalid state transitions
   are rejected atomically (`NETWORK-02`).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from artifact_io import write_json
from aix_zero.checkpoint_registry import validate_network_job_contract

PROHIBITED_IDENTITY_FIELDS = {
    "hostname",
    "mac_address",
    "ip_address",
    "username",
    "home_directory",
    "email",
    "serial_number",
}


def validate_worker_capability_handshake(handshake: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a privacy-preserving volunteer worker capability handshake (`WORKER-01` / Issue #126)."""
    leaked_pii = sorted(PROHIBITED_IDENTITY_FIELDS.intersection(handshake.keys()))
    if leaked_pii:
        raise ValueError(f"Worker handshake contains prohibited identity/PII fields: {leaked_pii}")

    required = {
        "worker_ephemeral_id",
        "device_types",
        "available_ram_mb",
        "available_disk_mb",
        "torch_version",
        "safetensors_version",
        "max_lease_seconds",
        "allowed_job_types",
    }
    missing = sorted(required - set(handshake.keys()))
    if missing:
        raise ValueError(f"Worker handshake missing required capability fields: {missing}")

    if int(handshake["available_ram_mb"]) < 64 or int(handshake["available_disk_mb"]) < 32:
        raise ValueError("Worker RAM and disk capabilities must be positive")

    return copy.deepcopy(handshake)


def match_job_to_worker(
    job_contract: Dict[str, Any],
    worker_handshake: Dict[str, Any],
) -> Dict[str, Any]:
    """Match a validated `NETWORK-01` job contract to a worker capability handshake (`WORKER-01`)."""
    validate_network_job_contract(job_contract)
    caps = validate_worker_capability_handshake(worker_handshake)

    req_budget = job_contract["resource_budget"]
    jtype = job_contract["job_type"]

    reasons: List[str] = []
    if jtype not in caps["allowed_job_types"]:
        reasons.append(f"Worker does not permit job_type {jtype!r}")
    if req_budget["device"] not in caps["device_types"]:
        reasons.append(f"Worker does not support device {req_budget['device']!r}")
    if int(caps["available_ram_mb"]) < int(req_budget["max_memory_mb"]):
        reasons.append(
            f"Insufficient worker RAM ({caps['available_ram_mb']} MB < {req_budget['max_memory_mb']} MB)"
        )
    if int(caps["max_lease_seconds"]) < int(req_budget["max_wall_clock_seconds"]):
        reasons.append(
            f"Worker max_lease_seconds ({caps['max_lease_seconds']}s) < job budget ({req_budget['max_wall_clock_seconds']}s)"
        )

    if reasons:
        return {
            "matched": False,
            "response_code": "NO_WORK_UNSUPPORTED_CAPABILITY",
            "job_id": None,
            "reasons": reasons,
        }

    return {
        "matched": True,
        "response_code": "MATCHED_LEASE_ELIGIBLE",
        "job_id": job_contract["job_id"],
        "reasons": [],
    }


THREAT_BOUNDARY_MATRIX: List[Dict[str, str]] = [
    {
        "surface": "Untrusted Job Manifests & Configs",
        "protected_asset": "Worker host filesystem and CPU/memory budget",
        "required_authority_and_control": (
            "Strict JSON Schema validation (`network-job-lease-v1.schema.json`), no arbitrary shell/eval, "
            "and bounded resource limits (`max_steps`, `max_wall_clock_seconds`, `max_memory_mb`)."
        ),
    },
    {
        "surface": "Untrusted Worker Tensor Submissions",
        "protected_asset": "Active baseline checkpoint (`aix_zero_v11_kip.safetensors`) and cartridge catalog",
        "required_authority_and_control": (
            "Safetensors-only serialization (never pickle), SHA-256 verification, finiteness check, "
            "frozen-backbone SHA-256 verification (`EXPERT-02`), and independent statistical gate (`EVAL-15`)."
        ),
    },
    {
        "surface": "Coordinator Queue & Lease State Transitions",
        "protected_asset": "Experiment queue ledger and promotion status",
        "required_authority_and_control": (
            "Atomic state machine (`DurableExperimentQueue`) separating trusted-tester enrollment "
            "from public untrusted worker submissions."
        ),
    },
]


VALID_JOB_TRANSITIONS = {
    "QUEUED": {"LEASED", "FAILED"},
    "LEASED": {"COMPLETED", "FAILED", "QUEUED"},
    "COMPLETED": set(),
    "FAILED": {"QUEUED"},
}


class DurableExperimentQueue:
    """Durable disk-backed experiment job queue (`NETWORK-02` / Issue #134).

    Persists job lifecycle states (`QUEUED`, `LEASED`, `COMPLETED`, `FAILED`) atomically
    so restarting the coordinator preserves state and invalid transitions fail closed.
    """

    def __init__(self, state_path: Path | str) -> None:
        self.state_path = Path(state_path)
        self._state: Dict[str, Any] = {"schema_version": "1.0.0", "jobs": {}}
        if self.state_path.is_file():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self._state, self.state_path)

    def enqueue(self, job_contract: Dict[str, Any]) -> Dict[str, Any]:
        validate_network_job_contract(job_contract)
        jid = job_contract["job_id"]
        if jid in self._state["jobs"]:
            raise ValueError(f"Job {jid!r} already exists in queue")
        entry = {
            "job_id": jid,
            "status": "QUEUED",
            "contract": copy.deepcopy(job_contract),
            "history": [{"to": "QUEUED", "worker_id": None}],
        }
        self._state["jobs"][jid] = entry
        self._persist()
        return copy.deepcopy(entry)

    def transition(
        self,
        job_id: str,
        target_status: str,
        *,
        worker_id: Optional[str] = None,
        result_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if job_id not in self._state["jobs"]:
            raise KeyError(f"Unknown job_id: {job_id!r}")
        entry = self._state["jobs"][job_id]
        curr = entry["status"]
        if target_status not in VALID_JOB_TRANSITIONS.get(curr, set()):
            raise ValueError(f"Invalid job state transition for {job_id}: {curr} -> {target_status}")
        if target_status == "LEASED" and not worker_id:
            raise ValueError("Transition to LEASED requires a nonempty worker_id")

        entry["status"] = target_status
        entry["worker_id"] = worker_id
        if result_summary is not None:
            entry["result_summary"] = copy.deepcopy(result_summary)
        entry["history"].append({"from": curr, "to": target_status, "worker_id": worker_id})
        self._persist()
        return copy.deepcopy(entry)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self._state["jobs"]:
            raise KeyError(f"Unknown job_id: {job_id!r}")
        return copy.deepcopy(self._state["jobs"][job_id])

    def counts_by_status(self) -> Dict[str, int]:
        counts = {"QUEUED": 0, "LEASED": 0, "COMPLETED": 0, "FAILED": 0}
        for j in self._state["jobs"].values():
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        return counts
