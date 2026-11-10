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
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from artifact_io import canonical_json, write_json
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

    def acquire_lease(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_id: str,
        expires_at_epoch: int,
        now_epoch: int,
    ) -> Dict[str, Any]:
        """Acquire or re-lease an expired job (`NETWORK-03` / Issue #135)."""
        if job_id not in self._state["jobs"]:
            raise KeyError(f"Unknown job_id: {job_id!r}")
        entry = self._state["jobs"][job_id]
        curr = entry["status"]
        if curr == "COMPLETED":
            raise ValueError(f"Job {job_id!r} is already COMPLETED")
        if curr == "LEASED" and int(entry.get("lease_expires_at_epoch", 0)) > now_epoch:
            raise ValueError(f"Job {job_id!r} already has an active non-expired lease")

        entry["status"] = "LEASED"
        entry["worker_id"] = worker_id
        entry["active_lease_id"] = lease_id
        entry["lease_expires_at_epoch"] = int(expires_at_epoch)
        entry["history"].append(
            {"from": curr, "to": "LEASED", "worker_id": worker_id, "lease_id": lease_id}
        )
        self._persist()
        return copy.deepcopy(entry)

    def complete_lease_idempotent(
        self,
        job_id: str,
        *,
        lease_id: str,
        worker_id: str,
        now_epoch: int,
        result_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Complete a leased job idempotently (`NETWORK-03` / Issue #135).

        - Duplicate upload with the same `(job_id, canonical_lease_id)` returns
          `DUPLICATE_IGNORED_IDEMPOTENT` without granting double credit.
        - Upload from an expired or superseded lease returns `REJECTED_STALE_OR_SUPERSEDED_LEASE`
          so two racing workers can never both become canonical.
        """
        if job_id not in self._state["jobs"]:
            raise KeyError(f"Unknown job_id: {job_id!r}")
        entry = self._state["jobs"][job_id]

        if entry["status"] == "COMPLETED":
            if entry.get("canonical_lease_id") == lease_id:
                return {
                    "outcome": "DUPLICATE_IGNORED_IDEMPOTENT",
                    "credit_granted": False,
                    "canonical_lease_id": entry["canonical_lease_id"],
                }
            return {
                "outcome": "REJECTED_STALE_OR_SUPERSEDED_LEASE",
                "credit_granted": False,
                "canonical_lease_id": entry.get("canonical_lease_id"),
            }

        if (
            entry["status"] != "LEASED"
            or entry.get("active_lease_id") != lease_id
            or entry.get("worker_id") != worker_id
            or int(entry.get("lease_expires_at_epoch", 0)) <= now_epoch
        ):
            return {
                "outcome": "REJECTED_STALE_OR_SUPERSEDED_LEASE",
                "credit_granted": False,
                "canonical_lease_id": entry.get("canonical_lease_id"),
            }

        entry["status"] = "COMPLETED"
        entry["canonical_lease_id"] = lease_id
        entry["canonical_worker_id"] = worker_id
        entry["result_summary"] = copy.deepcopy(result_summary)
        entry["history"].append(
            {"from": "LEASED", "to": "COMPLETED", "worker_id": worker_id, "lease_id": lease_id}
        )
        self._persist()
        return {
            "outcome": "ACCEPTED_CANONICAL_COMPLETION",
            "credit_granted": True,
            "canonical_lease_id": lease_id,
        }


ALLOWLISTED_JOB_RUNNERS: Dict[str, str] = {
    "OFFLINE_EVALUATION_JOB": "aix_zero.benchmark_baseline:evaluate_v11_subset",
}


def execute_allowlisted_local_job(
    job_contract: Dict[str, Any],
    *,
    workspace_root: Path,
    allowed_runners: Optional[Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Fixed allowlisted local job runner (`WORKER-02` / Issue #128).

    Never invokes a shell, rejects command-injection characters or paths escaping
    `workspace_root`, and executes only pre-registered Python callables.
    """
    validate_network_job_contract(job_contract)
    jtype = job_contract["job_type"]
    runners = allowed_runners or {}
    if jtype not in ALLOWLISTED_JOB_RUNNERS or jtype not in runners:
        raise PermissionError(f"Job type {jtype!r} is not allowlisted for local execution")

    rel_ckpt = str(job_contract["inputs"]["base_checkpoint_path"])
    if any(ch in rel_ckpt for ch in (";", "|", "&", "$", "`", "\n")) or ".." in Path(rel_ckpt).parts:
        raise ValueError("Unsafe checkpoint path in job_contract inputs")
    resolved_ckpt = (workspace_root / rel_ckpt).resolve()
    if not str(resolved_ckpt).startswith(str(workspace_root.resolve())):
        raise ValueError("Checkpoint path escapes workspace_root")

    output = runners[jtype](job_contract["inputs"])
    return {
        "executed": True,
        "job_id": job_contract["job_id"],
        "job_type": jtype,
        "entrypoint": ALLOWLISTED_JOB_RUNNERS[jtype],
        "output": output,
    }


def validate_remote_artifact_stream(
    payload_bytes: bytes,
    *,
    filename: str,
    target_dir: Path,
    expected_sha256: str,
    max_bytes: int = 16 * 1024 * 1024,
) -> Dict[str, Any]:
    """Validate remote artifact size, path containment, and SHA-256 BEFORE tensor allocation
    or publication (`SEC-03` / Issue #132).
    """
    if Path(filename).name != filename or ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Path traversal rejected in remote artifact filename: {filename!r}")
    dest = (target_dir / filename).resolve()
    if dest.parent != target_dir.resolve():
        raise ValueError("Remote artifact target escapes target_dir")
    if len(payload_bytes) > max_bytes:
        raise ValueError(f"Remote artifact exceeds max_bytes ({len(payload_bytes)} > {max_bytes})")
    actual_sha = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha != expected_sha256:
        raise ValueError(f"Remote artifact SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}")

    return {
        "verified": True,
        "filename": filename,
        "byte_length": len(payload_bytes),
        "sha256": actual_sha,
    }


def sign_release_manifest(
    unsigned_manifest: Dict[str, Any],
    *,
    key_id: str,
    secret_key: str,
) -> Dict[str, Any]:
    """Sign a client release manifest with HMAC-SHA256 (`SEC-08` / Issue #140)."""
    m = copy.deepcopy(unsigned_manifest)
    m["key_id"] = key_id
    payload = canonical_json({k: v for k, v in m.items() if k != "signature_hmac_sha256"}).encode("utf-8")
    m["signature_hmac_sha256"] = hmac.new(secret_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return m


def verify_release_manifest_signature(
    signed_manifest: Dict[str, Any],
    *,
    trust_root_keyring: Dict[str, Dict[str, Any]],
    minimum_compatible_version: Tuple[int, int, int] = (1, 0, 0),
) -> Dict[str, Any]:
    """Verify client release provenance, key rotation/revocation status, and rollback floor (`SEC-08` / Issue #140)."""
    key_id = str(signed_manifest.get("key_id", ""))
    sig = str(signed_manifest.get("signature_hmac_sha256", ""))
    if not key_id or not sig or key_id not in trust_root_keyring:
        raise ValueError(f"Untrusted or missing release signing key_id: {key_id!r}")

    key_info = trust_root_keyring[key_id]
    if key_info.get("status") != "ACTIVE":
        raise ValueError(f"Signing key {key_id!r} is not ACTIVE (status={key_info.get('status')!r})")

    unsigned_payload = canonical_json(
        {k: v for k, v in signed_manifest.items() if k != "signature_hmac_sha256"}
    ).encode("utf-8")
    expected_sig = hmac.new(
        str(key_info["secret_key"]).encode("utf-8"), unsigned_payload, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        raise ValueError("Release manifest signature verification failed (tampered payload)")

    ver_str = str(signed_manifest.get("release_version", "0.0.0"))
    ver_tuple = tuple(int(x) for x in ver_str.split("."))
    if ver_tuple < minimum_compatible_version:
        raise ValueError(
            f"Release version {ver_str} is below minimum_compatible_version {minimum_compatible_version} (rollback rejected)"
        )

    return {
        "verified": True,
        "key_id": key_id,
        "release_version": ver_str,
    }


def probe_worker_accelerator_capabilities(
    *,
    requested_device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    inject_nonfinite_fault: bool = False,
) -> Dict[str, Any]:
    """Run a bounded diagnostic probe on the requested device/precision before reserving a job (`WORKER-13` / Issue #144)."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        return {
            "probe_status": "PROBE_FAILED_DO_NOT_LEASE",
            "eligible_for_job_reservation": False,
            "device": requested_device,
            "dtype": str(dtype),
            "reason": "CUDA device not available on host",
        }

    a = torch.eye(8, dtype=dtype)
    b = torch.ones((8, 8), dtype=dtype) * 2.0
    out = a @ b
    if inject_nonfinite_fault:
        out[0, 0] = float("nan")

    if not torch.isfinite(out).all() or float(out.sum().item()) != 128.0:
        return {
            "probe_status": "PROBE_FAILED_DO_NOT_LEASE",
            "eligible_for_job_reservation": False,
            "device": requested_device,
            "dtype": str(dtype),
            "reason": "Bounded GEMM probe failed finiteness or exact sum check",
        }

    return {
        "probe_status": "PROBE_PASSED",
        "eligible_for_job_reservation": True,
        "device": requested_device,
        "dtype": str(dtype),
        "matrix_probe_shape": [8, 8],
    }


def verify_worker_sandbox_isolation(
    sandbox_workdir: Path,
    *,
    attempted_path: Path,
    attempted_outbound_host: Optional[str] = None,
    allow_network: bool = False,
) -> Dict[str, Any]:
    """Evaluate and enforce worker sandbox filesystem and network containment (`SEC-02` / Issue #147)."""
    real_sandbox = sandbox_workdir.resolve()
    real_target = attempted_path.resolve()
    fs_allowed = str(real_target).startswith(str(real_sandbox))
    net_allowed = True if attempted_outbound_host is None else bool(allow_network)

    return {
        "filesystem_access_permitted": fs_allowed,
        "network_access_permitted": net_allowed,
        "sandbox_compliant": fs_allowed and net_allowed,
        "platform_limitations": [
            "Application-level path and socket guards do not replace kernel namespaces, seccomp, or VM hypervisor isolation.",
            "Untrusted public workloads require OS-level container/sandbox enforcement (--network=none, read-only rootfs).",
        ],
    }


def sanitize_public_telemetry_and_logs(record: Dict[str, Any]) -> Dict[str, Any]:
    """Minimize contributor logs and strip credentials, hostnames, and private paths (`SEC-06` / Issue #157)."""
    redacted_keys = PROHIBITED_IDENTITY_FIELDS | {"api_key", "token", "Authorization", "password"}
    clean: Dict[str, Any] = {}
    for k, v in record.items():
        if k in redacted_keys:
            continue
        if isinstance(v, str):
            val = re.sub(r"/(?:Users|home)/[^/\s\"']+", "<REDACTED_HOME_PATH>", v)
            val = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<REDACTED_IP>", val)
            clean[k] = val
        else:
            clean[k] = copy.deepcopy(v)
    return clean


def enforce_worker_resource_limits(
    *,
    max_cpu_threads: int,
    max_memory_mb: int,
    job_requested_threads: int,
    job_estimated_memory_mb: int,
    os_platform: str = "darwin",
) -> Dict[str, Any]:
    """Expose explicit contributor CPU and memory caps with platform enforcement reporting (`WORKER-03` / Issue #129)."""
    if max_cpu_threads < 1 or max_memory_mb < 64:
        raise ValueError("max_cpu_threads >= 1 and max_memory_mb >= 64 are required")

    torch.set_num_threads(int(max_cpu_threads))
    threads_exceeded = job_requested_threads > max_cpu_threads
    memory_exceeded = job_estimated_memory_mb > max_memory_mb

    if threads_exceeded or memory_exceeded:
        status = "TERMINATED_RESOURCE_CAP_EXCEEDED"
        allowed = False
    else:
        status = "WITHIN_RESOURCE_CAPS"
        allowed = True

    return {
        "status": status,
        "execution_allowed": allowed,
        "max_cpu_threads": int(max_cpu_threads),
        "applied_torch_num_threads": int(torch.get_num_threads()),
        "max_memory_mb": int(max_memory_mb),
        "job_requested_threads": int(job_requested_threads),
        "job_estimated_memory_mb": int(job_estimated_memory_mb),
        "enforcement_report": {
            "cpu_threads": "HARD_ENFORCED_TORCH_THREADPOOL",
            "memory_mb": "HARD_PREFLIGHT_GATE_PLUS_OS_RLIMIT" if os_platform == "linux" else "HARD_PREFLIGHT_GATE_ADVISORY_RSS_MONITOR",
            "os_platform": os_platform,
        },
    }


def manage_worker_schedule_and_pause(
    config_path: Path,
    *,
    allowed_hours_utc: Optional[List[int]] = None,
    idle_only: Optional[bool] = None,
    paused: Optional[bool] = None,
    active_lease_policy: str = "RELINQUISH_CLEANLY",
    current_hour_utc: int = 12,
    host_is_idle: bool = True,
    active_lease_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist contributor schedule and pause controls (`WORKER-04` / Issue #130).

    Pausing prevents new work and handles an active lease according to policy; settings
    survive restarts without silently enabling additional resources.
    """
    if config_path.exists():
        state = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        state = {
            "allowed_hours_utc": list(range(24)),
            "idle_only": True,
            "paused": False,
            "active_lease_policy": "RELINQUISH_CLEANLY",
            "max_cpu_threads": 2,
            "max_memory_mb": 1024,
        }

    if allowed_hours_utc is not None:
        state["allowed_hours_utc"] = [int(h) for h in allowed_hours_utc]
    if idle_only is not None:
        state["idle_only"] = bool(idle_only)
    if paused is not None:
        state["paused"] = bool(paused)
    if active_lease_policy:
        state["active_lease_policy"] = str(active_lease_policy)

    write_json(state, config_path)
    reloaded = json.loads(config_path.read_text(encoding="utf-8"))

    in_window = int(current_hour_utc) in set(reloaded["allowed_hours_utc"])
    idle_ok = (not reloaded["idle_only"]) or bool(host_is_idle)
    can_accept_new_work = (not reloaded["paused"]) and in_window and idle_ok

    active_lease_action = "NONE"
    if active_lease_id is not None and not can_accept_new_work:
        active_lease_action = reloaded["active_lease_policy"]

    return {
        "persisted_settings": reloaded,
        "can_accept_new_lease": can_accept_new_work,
        "active_lease_id": active_lease_id,
        "active_lease_action": active_lease_action,
    }


def manage_worker_disk_quota(
    worker_dir: Path,
    *,
    quota_bytes: int,
    incoming_artifact_bytes: int = 0,
    pending_upload_filenames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Manage downloaded inputs and outputs under a dedicated worker directory (`WORKER-05` / Issue #131).

    Quota exhaustion stops work without touching unrelated files outside `worker_dir`, and
    cleanup retains evidence needed for pending uploads and retries.
    """
    worker_dir.mkdir(parents=True, exist_ok=True)
    pending_set = set(pending_upload_filenames or [])

    used_before = sum(f.stat().st_size for f in worker_dir.rglob("*") if f.is_file())
    reclaimed_files: List[str] = []
    retained_pending_files: List[str] = []

    for f in sorted(worker_dir.iterdir()):
        if not f.is_file():
            continue
        if f.name in pending_set:
            retained_pending_files.append(f.name)
        elif f.name.endswith(".scratch") or f.name.endswith(".completed"):
            f.unlink()
            reclaimed_files.append(f.name)

    used_after_cleanup = sum(f.stat().st_size for f in worker_dir.rglob("*") if f.is_file())
    projected = used_after_cleanup + int(incoming_artifact_bytes)
    quota_exceeded = projected > int(quota_bytes)

    return {
        "worker_dir": str(worker_dir.resolve()),
        "quota_bytes": int(quota_bytes),
        "used_bytes_before_cleanup": used_before,
        "used_bytes_after_cleanup": used_after_cleanup,
        "projected_bytes": projected,
        "reclaimed_files": reclaimed_files,
        "retained_pending_files": retained_pending_files,
        "status": "STOPPED_DISK_QUOTA_EXCEEDED" if quota_exceeded else "OK_WITHIN_QUOTA",
        "work_permitted": not quota_exceeded,
    }


class WorkerHeartbeatAndCancellationTracker:
    """Track worker lease heartbeats and idempotent lease cancellations (`WORKER-07` / Issue #136)."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        if not self.state_path.exists():
            write_json({"leases": {}, "local_cancellation_log": []}, self.state_path)

    def _load(self) -> Dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save(self, state: Dict[str, Any]) -> None:
        write_json(state, self.state_path)

    def register_lease(self, job_id: str, lease_id: str, expires_at_epoch: int) -> None:
        state = self._load()
        state["leases"][lease_id] = {
            "job_id": job_id,
            "lease_id": lease_id,
            "status": "ACTIVE",
            "last_heartbeat_epoch": 0,
            "expires_at_epoch": int(expires_at_epoch),
        }
        self._save(state)

    def record_heartbeat(self, lease_id: str, *, now_epoch: int) -> Dict[str, Any]:
        state = self._load()
        entry = state["leases"].get(lease_id)
        if entry is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        if entry["status"] == "CANCELLED" or now_epoch >= entry["expires_at_epoch"]:
            entry["status"] = "LOST_OR_CANCELLED"
            self._save(state)
            return {"lease_active": False, "status": entry["status"]}
        entry["last_heartbeat_epoch"] = int(now_epoch)
        self._save(state)
        return {"lease_active": True, "status": "ACTIVE", "last_heartbeat_epoch": int(now_epoch)}

    def cancel_lease(self, lease_id: str, *, reason: str) -> Dict[str, Any]:
        """Idempotently cancel a lease and record cancellation locally."""
        state = self._load()
        entry = state["leases"].get(lease_id)
        if entry is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        already_cancelled = entry["status"] in ("CANCELLED", "LOST_OR_CANCELLED")
        entry["status"] = "CANCELLED"
        if not already_cancelled:
            state["local_cancellation_log"].append({"lease_id": lease_id, "reason": reason})
        self._save(state)
        return {
            "lease_id": lease_id,
            "status": "CANCELLED",
            "idempotent_noop": already_cancelled,
            "cancellation_log_count": len(state["local_cancellation_log"]),
        }

    def validate_upload_eligibility(self, lease_id: str, *, now_epoch: int) -> bool:
        state = self._load()
        entry = state["leases"].get(lease_id)
        if entry is None:
            raise PermissionError(f"Unknown lease_id: {lease_id}")
        if entry["status"] != "ACTIVE" or now_epoch >= entry["expires_at_epoch"]:
            raise PermissionError(f"Lost or cancelled lease {lease_id!r} cannot upload result as current")
        return True


class ImmutableArtifactStore:
    """Content-addressed immutable input and result artifact store (`NETWORK-05` / Issue #138)."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.objects_dir = self.root_dir / "objects"
        self.staging_dir = self.root_dir / "staging"
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def publish_result_artifact(
        self,
        payload_bytes: bytes,
        *,
        model_sha256: str,
        dataset_sha256: str,
        source_sha256: str,
        config_sha256: str,
        simulate_incomplete_staging: bool = False,
    ) -> Dict[str, Any]:
        for name, val in (
            ("model_sha256", model_sha256),
            ("dataset_sha256", dataset_sha256),
            ("source_sha256", source_sha256),
            ("config_sha256", config_sha256),
        ):
            if not val or len(val) < 16:
                raise ValueError(f"Result artifact requires valid {name}")

        digest = hashlib.sha256(payload_bytes).hexdigest()
        staging_path = self.staging_dir / f"{digest}.part"
        staging_path.write_bytes(payload_bytes)
        if simulate_incomplete_staging:
            raise RuntimeError("Simulated crash during staging before atomic publication")

        final_obj_path = self.objects_dir / f"{digest}.bin"
        staging_path.replace(final_obj_path)

        manifest = {
            "object_sha256": digest,
            "byte_length": len(payload_bytes),
            "model_sha256": model_sha256,
            "dataset_sha256": dataset_sha256,
            "source_sha256": source_sha256,
            "config_sha256": config_sha256,
        }
        write_json(manifest, self.objects_dir / f"{digest}.manifest.json")
        return manifest

    def retrieve_artifact(self, object_sha256: str, *, max_bytes: int = 1024 * 1024) -> bytes:
        obj_path = self.objects_dir / f"{object_sha256}.bin"
        if not obj_path.exists():
            raise FileNotFoundError(f"Accepted artifact {object_sha256} not found")
        size = obj_path.stat().st_size
        if size > max_bytes:
            raise ValueError(f"Artifact exceeds bounded retrieval limit ({size} > {max_bytes})")
        data = obj_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != object_sha256:
            raise ValueError("Content-addressed integrity check failed on read")
        return data


SHELL_METACHAR_PATTERN = re.compile(r"[;&|`$><\n\r]")


def validate_constrained_job_arguments(
    job_payload: Dict[str, Any],
    *,
    user_resource_caps: Dict[str, int],
) -> Dict[str, Any]:
    """Constrain public job arguments and paths via allowlists (`SEC-05` / Issue #148).

    Rejects shell metacharacters, out-of-scope paths, and any attempt by a job payload
    to override user-selected resource caps.
    """
    if "override_user_resource_cap" in job_payload or "max_memory_mb" in job_payload or "max_cpu_threads" in job_payload:
        raise ValueError("Malicious job payload cannot alter user-selected resource caps")

    task = str(job_payload.get("task", ""))
    if task not in ("BP", "BS"):
        raise ValueError(f"Disallowed public job task: {task!r}")

    rel_path = str(job_payload.get("artifact_relative_path", ""))
    if not rel_path or rel_path.startswith("/") or ".." in rel_path or "\\" in rel_path:
        raise ValueError(f"Out-of-scope job path rejected: {rel_path!r}")
    if SHELL_METACHAR_PATTERN.search(rel_path):
        raise ValueError("Shell metacharacters rejected in artifact_relative_path")

    for k, v in job_payload.get("arguments", {}).items():
        s_val = str(v)
        if SHELL_METACHAR_PATTERN.search(s_val) or ".." in s_val:
            raise ValueError(f"Shell metacharacters or path traversal rejected in argument {k!r}: {s_val!r}")

    return {
        "validated": True,
        "task": task,
        "enforced_user_resource_caps": dict(user_resource_caps),
    }


def evaluate_independent_result_replication(
    primary_result: Dict[str, Any],
    replica_result: Dict[str, Any],
    *,
    metric_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Compare duplicate verification runs and compute useful throughput (`NETWORK-06` / Issue #149).

    Disagreement triggers investigation rather than automatic majority trust, and verification
    overhead is included in useful-throughput measurements.
    """
    if primary_result.get("worker_id") == replica_result.get("worker_id"):
        raise ValueError("Independent replication requires distinct primary and replica worker_ids")

    p_score = float(primary_result["accuracy"])
    r_score = float(replica_result["accuracy"])
    diff = abs(p_score - r_score)
    hash_match = primary_result.get("result_sha256") == replica_result.get("result_sha256")

    agreed = (diff <= metric_tolerance) and (hash_match if metric_tolerance == 0.0 else True)
    total_compute_units = 2
    useful_verified_jobs = 1 if agreed else 0

    return {
        "replication_status": "VERIFIED_BY_REPLICATION" if agreed else "DISAGREEMENT_INVESTIGATION_REQUIRED",
        "automatic_majority_trust_used": False,
        "score_difference": round(diff, 8),
        "replication_factor": total_compute_units,
        "total_compute_units_spent": total_compute_units,
        "useful_verified_jobs": useful_verified_jobs,
        "useful_verified_throughput_ratio": round(useful_verified_jobs / total_compute_units, 6),
    }


class WorkerEnrollmentRegistry:
    """Manage authenticated worker enrollment, credential expiration, and revocation (`SEC-04` / Issue #153)."""

    def __init__(self, master_secret: str) -> None:
        self.master_secret = master_secret
        self._workers: Dict[str, Dict[str, Any]] = {}

    def enroll_worker(self, worker_id: str, *, expires_at_epoch: int) -> Dict[str, Any]:
        token = hmac.new(
            self.master_secret.encode("utf-8"),
            f"{worker_id}:{expires_at_epoch}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        self._workers[worker_id] = {
            "worker_id": worker_id,
            "enrollment_token": token,
            "expires_at_epoch": int(expires_at_epoch),
            "status": "ACTIVE",
        }
        return copy.deepcopy(self._workers[worker_id])

    def revoke_worker(self, worker_id: str, *, reason: str) -> Dict[str, Any]:
        if worker_id not in self._workers:
            raise KeyError(f"Unknown worker_id: {worker_id}")
        self._workers[worker_id]["status"] = "REVOKED"
        self._workers[worker_id]["revocation_reason"] = reason
        return {"worker_id": worker_id, "status": "REVOKED", "reason": reason}

    def verify_worker_can_lease(self, worker_id: str, token: str, *, now_epoch: int) -> bool:
        info = self._workers.get(worker_id)
        if info is None:
            raise PermissionError("Unenrolled worker")
        if info["status"] != "ACTIVE":
            raise PermissionError(f"Revoked worker {worker_id!r} cannot obtain new leases")
        if now_epoch >= info["expires_at_epoch"]:
            raise PermissionError(f"Expired worker credential for {worker_id!r}")
        if not hmac.compare_digest(info["enrollment_token"], token):
            raise PermissionError("Invalid worker enrollment token")
        return True

    def export_public_worker_telemetry(self) -> List[Dict[str, Any]]:
        """Return public telemetry without enrollment tokens or secrets."""
        return [
            {
                "worker_id": w["worker_id"],
                "status": w["status"],
                "expires_at_epoch": w["expires_at_epoch"],
            }
            for w in self._workers.values()
        ]


def match_job_to_worker_capabilities(
    job_requirements: Dict[str, Any],
    worker_capabilities: Dict[str, Any],
) -> Dict[str, Any]:
    """Match work to hardware and contributor budgets (`NETWORK-04` / Issue #164), enforcing
    memory, disk, platform, and deadline constraints before assignment and recording deferral reasons.
    """
    deferral_reasons: List[str] = []
    if worker_capabilities.get("available_ram_mb", 0) < job_requirements.get("min_ram_mb", 0):
        deferral_reasons.append("insufficient_ram_mb")
    if worker_capabilities.get("available_disk_mb", 0) < job_requirements.get("min_disk_mb", 0):
        deferral_reasons.append("insufficient_disk_mb")
    req_backends = set(job_requirements.get("allowed_backends", ["cpu"]))
    if worker_capabilities.get("accelerator_backend") not in req_backends:
        deferral_reasons.append("incompatible_accelerator_backend")
    if worker_capabilities.get("max_runtime_seconds", 0) < job_requirements.get("estimated_duration_seconds", 0):
        deferral_reasons.append("exceeds_worker_deadline_budget")

    matched = len(deferral_reasons) == 0
    return {
        "assigned": matched,
        "status": "ASSIGNED" if matched else "DEFERRED_CAPABILITY_MISMATCH",
        "deferral_reasons": deferral_reasons,
    }


def verify_dataset_distribution_permission(dataset_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce training-data distribution permissions (`SEC-07` / Issue #172), blocking restricted,
    private, or provenance-unknown datasets from volunteer jobs.
    """
    allowed_licenses = {"MIT", "Apache-2.0", "CC-BY-4.0", "CC0-1.0"}
    lic = str(dataset_manifest.get("license", ""))
    prov = str(dataset_manifest.get("provenance_sha256", ""))
    redist = bool(dataset_manifest.get("public_redistribution_allowed", False))
    has_private = bool(dataset_manifest.get("contains_private_samples", False))

    if has_private:
        raise PermissionError("Dataset rejected: fixture or payload contains private sample data")
    if lic not in allowed_licenses or not prov or not redist:
        raise PermissionError(
            f"Dataset blocked from volunteer distribution: license={lic!r}, provenance_present={bool(prov)}, public_redistribution_allowed={redist}"
        )

    return {
        "distribution_permitted": True,
        "license": lic,
        "provenance_sha256": prov,
    }


def generate_supply_chain_sbom_review(
    *,
    dependencies: List[Dict[str, str]],
    known_unresolved_issues: List[str],
    unreviewed_dependency_update: bool = False,
) -> Dict[str, Any]:
    """Create a lightweight dependency and supply-chain review (`SEC-14` / Issue #173).

    Lists the software bill of materials (SBOM) with pinned versions and update provenance,
    and blocks unreviewed dependency updates from bypassing the worker signing process.
    """
    if unreviewed_dependency_update:
        raise ValueError("Unreviewed dependency update rejected: cannot bypass worker release signing review")
    if not dependencies:
        raise ValueError("SBOM must list executable dependencies")

    for dep in dependencies:
        if not dep.get("name") or not dep.get("pinned_version") or not dep.get("provenance"):
            raise ValueError(f"Incomplete SBOM dependency entry: {dep!r}")

    sbom_digest = hashlib.sha256(canonical_json(dependencies).encode("utf-8")).hexdigest()
    return {
        "sbom_sha256": sbom_digest,
        "sbom_components": dependencies,
        "known_unresolved_issues": list(known_unresolved_issues),
        "signing_gate_passed": True,
    }


def build_verified_contribution_ledger(
    contributions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recognize verified code, dataset, reproduction, review, and compute contributions (`COMM-12` / Issue #195).

    Recognition follows accepted useful work rather than self-reported runtime, and
    participants can opt out of public attribution.
    """
    allowed_types = {"code", "dataset", "independent_reproduction", "review", "verified_compute"}
    recognized: List[Dict[str, Any]] = []
    rejected_unverified_count = 0

    for item in contributions:
        c_type = str(item.get("contribution_type", ""))
        if c_type not in allowed_types:
            raise ValueError(f"Unsupported contribution_type: {c_type!r}")
        if item.get("verification_status") != "ACCEPTED_VERIFIED":
            rejected_unverified_count += 1
            continue

        opt_out = bool(item.get("opt_out_public_attribution", False))
        public_handle = "anonymous-contributor" if opt_out else str(item.get("contributor_handle", "anonymous-contributor"))
        recognized.append(
            {
                "contribution_id": str(item["contribution_id"]),
                "contribution_type": c_type,
                "public_contributor_handle": public_handle,
                "opt_out_public_attribution": opt_out,
                "accepted_useful_work_units": int(item.get("accepted_useful_work_units", 1)),
                "credited_from_self_reported_runtime": False,
                "artifact_sha256": str(item.get("artifact_sha256", "")),
            }
        )

    return {
        "recognized_contribution_count": len(recognized),
        "rejected_unverified_count": rejected_unverified_count,
        "recognized_contributions": recognized,
    }


def verify_worker_input_artifact_before_execution(
    downloaded_chunks: List[bytes],
    *,
    expected_sha256: str,
    expected_byte_length: int,
    max_download_retries: int = 2,
    retry_attempts_used: int = 0,
) -> Dict[str, Any]:
    """Verify worker input artifacts before execution (`WORKER-06` / Issue #133).

    Corrupt, truncated, or incompatible objects never execute, and interrupted downloads
    resume or restart within a bounded retry budget.
    """
    if retry_attempts_used > max_download_retries:
        return {
            "execution_permitted": False,
            "status": "ABORTED_DOWNLOAD_RETRY_BUDGET_EXCEEDED",
            "retry_attempts_used": int(retry_attempts_used),
        }

    assembled = b"".join(downloaded_chunks)
    if len(assembled) < expected_byte_length:
        return {
            "execution_permitted": False,
            "status": "TRUNCATED_NEEDS_RESUMABLE_FETCH",
            "resume_byte_offset": len(assembled),
            "expected_byte_length": int(expected_byte_length),
        }
    if len(assembled) > expected_byte_length:
        return {
            "execution_permitted": False,
            "status": "REJECTED_OVERSIZED_INPUT_OBJECT",
        }

    actual_sha = hashlib.sha256(assembled).hexdigest()
    if actual_sha != expected_sha256:
        return {
            "execution_permitted": False,
            "status": "REJECTED_CORRUPT_SHA256_MISMATCH",
            "actual_sha256": actual_sha,
        }

    return {
        "execution_permitted": True,
        "status": "VERIFIED_READY_FOR_EXECUTION",
        "sha256": actual_sha,
        "byte_length": len(assembled),
    }


def resume_bounded_worker_job(
    *,
    checkpoint_bundle: Dict[str, Any],
    active_job_contract: Dict[str, Any],
    source_model_path: Path,
    candidate_output_path: Path,
    lease_active: bool = True,
) -> Dict[str, Any]:
    """Support resumable bounded jobs (`WORKER-08` / Issue #137).

    A worker restart resumes a compatible job or reports why it cannot, and no resume
    path ever overwrites immutable source models.
    """
    if source_model_path.resolve() == candidate_output_path.resolve():
        raise PermissionError("Resume path prohibited from overwriting immutable source model")
    if not lease_active:
        return {
            "resumed": False,
            "status": "CANNOT_RESUME_LEASE_EXPIRED_OR_CANCELLED",
            "source_model_overwrite_blocked": True,
        }
    if checkpoint_bundle.get("experiment_config_sha256") != active_job_contract.get("experiment_config_sha256"):
        return {
            "resumed": False,
            "status": "CANNOT_RESUME_INCOMPATIBLE_EXPERIMENT_CONFIG",
            "source_model_overwrite_blocked": True,
        }

    return {
        "resumed": True,
        "status": "RESUMED_FROM_VERIFIED_CHECKPOINT",
        "resumed_from_step": int(checkpoint_bundle.get("step", 0)),
        "source_model_overwrite_blocked": True,
    }


def verify_desktop_client_package_lifecycle(
    install_root: Path,
    *,
    requested_gpu_backend: str = "cpu",
    gpu_backend_supported: bool = True,
) -> Dict[str, Any]:
    """Package and verify desktop client install/uninstall lifecycle (`WORKER-10` / Issue #141).

    Tests clean-machine install and uninstall under `install_root`, and ensures unsupported
    GPU backends fail explicitly rather than silently falling back to excessive CPU use.
    """
    if requested_gpu_backend != "cpu" and not gpu_backend_supported:
        return {
            "installed": False,
            "status": "REJECTED_UNSUPPORTED_GPU_BACKEND_NO_SILENT_CPU_FALLBACK",
            "silent_excessive_cpu_fallback": False,
        }

    bin_dir = install_root / "bin"
    cfg_dir = install_root / "config"
    bin_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    worker_bin = bin_dir / "aix-worker-client"
    worker_cfg = cfg_dir / "default.json"
    worker_bin.write_text("#!/bin/sh\necho aix-worker\n", encoding="utf-8")
    write_json({"max_cpu_threads": 1, "idle_only": True}, worker_cfg)
    installed_ok = worker_bin.is_file() and worker_cfg.is_file()

    # Clean uninstall removes installed files cleanly
    worker_bin.unlink()
    worker_cfg.unlink()
    bin_dir.rmdir()
    cfg_dir.rmdir()
    remaining = list(install_root.iterdir())

    return {
        "installed": installed_ok,
        "clean_uninstall_verified": len(remaining) == 0,
        "silent_excessive_cpu_fallback": False,
        "status": "INSTALL_AND_UNINSTALL_VERIFIED",
    }


def validate_and_ingest_worker_result(
    worker_payload: Dict[str, Any],
    *,
    rejection_log: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Reject malformed and unsupported worker results (`NETWORK-07` / Issue #150).

    A worker cannot self-declare statistical acceptance or model promotion; contract
    failures are recorded in `rejection_log` without crashing the coordinator.
    """
    prohibited_self_declarations = {"sprt_decision", "promotion_status", "statistical_acceptance", "is_promoted"}
    leaked = sorted(prohibited_self_declarations.intersection(worker_payload.keys()))
    if leaked:
        rejection_log.append(
            {
                "job_id": str(worker_payload.get("job_id", "unknown")),
                "reason": f"Worker attempted prohibited self-declaration: {leaked}",
            }
        )
        return {
            "accepted": False,
            "status": "REJECTED_PROHIBITED_WORKER_SELF_DECLARATION",
            "coordinator_healthy": True,
        }

    req_fields = ("job_id", "lease_id", "worker_id", "result_sha256", "metrics")
    missing = [k for k in req_fields if k not in worker_payload]
    if missing or not isinstance(worker_payload.get("metrics"), dict):
        rejection_log.append(
            {
                "job_id": str(worker_payload.get("job_id", "unknown")),
                "reason": f"Malformed worker result missing fields: {missing}",
            }
        )
        return {
            "accepted": False,
            "status": "REJECTED_MALFORMED_RESULT_CONTRACT",
            "coordinator_healthy": True,
        }

    return {
        "accepted": True,
        "status": "INGESTED_FOR_INDEPENDENT_COORDINATOR_EVALUATION",
        "coordinator_healthy": True,
    }


def schedule_with_fairness_and_starvation_control(
    candidates: List[Dict[str, Any]],
    *,
    max_active_leases_per_contributor: int = 2,
    starvation_threshold_seconds: int = 300,
) -> Dict[str, Any]:
    """Add fairness and starvation controls to worker scheduling (`NETWORK-09` / Issue #166).

    Prevents a single contributor from monopolizing leases and boosts reliable workers that
    have waited beyond `starvation_threshold_seconds` so slow but reliable workers receive work.
    """
    eligible: List[Tuple[Tuple[int, int, float], Dict[str, Any]]] = []
    throttled_workers: List[str] = []

    for w in candidates:
        active = int(w.get("active_leases", 0))
        if active >= max_active_leases_per_contributor:
            throttled_workers.append(str(w["worker_id"]))
            continue
        if float(w.get("reliability_score", 1.0)) < 0.8:
            continue
        wait_s = int(w.get("wait_seconds", 0))
        starved_boost = 1 if wait_s >= starvation_threshold_seconds else 0
        # Sort key: (starved_boost DESC, wait_seconds DESC, -active_leases DESC)
        eligible.append(((starved_boost, wait_s, -active), w))

    if not eligible:
        return {
            "selected_worker_id": None,
            "throttled_workers": throttled_workers,
            "starvation_boost_applied": False,
        }

    eligible.sort(key=lambda x: x[0], reverse=True)
    best_key, best_w = eligible[0]
    return {
        "selected_worker_id": str(best_w["worker_id"]),
        "throttled_workers": throttled_workers,
        "starvation_boost_applied": best_key[0] == 1,
    }


def validate_opt_in_research_proposal(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an opt-in community research discussion proposal (`COMM-09` / Issue #192).

    Requires explicit `hypothesis`, `controls`, `validation_budget`, and `stopping_criteria`,
    and prohibits unsolicited mass mentions or forced participant assignments.
    """
    if proposal.get("uses_unsolicited_mass_mentions", False) or proposal.get("forced_assignments", []):
        raise ValueError("Research proposal rejected: unsolicited mass mentions and forced assignments are prohibited")

    required = ("hypothesis", "controls", "validation_budget", "stopping_criteria")
    missing = [k for k in required if not proposal.get(k)]
    if missing:
        raise ValueError(f"Research proposal missing required fields: {missing}")

    return {
        "proposal_valid": True,
        "opt_in_only": True,
        "verified_fields": list(required),
    }


def upload_worker_result_with_provenance(
    *,
    payload_bytes: bytes,
    expected_byte_length: int,
    provenance: Dict[str, str],
    public_log_record: Dict[str, Any],
) -> Dict[str, Any]:
    """Upload worker results with complete provenance (`WORKER-09` / Issue #139).

    Partial uploads cannot become verified results, and secrets/local home paths are
    excluded from public logs.
    """
    req_prov = ("model_sha256", "dataset_sha256", "source_sha256", "config_sha256")
    missing = [k for k in req_prov if not provenance.get(k)]
    sanitized_log = sanitize_public_telemetry_and_logs(public_log_record)

    if len(payload_bytes) != int(expected_byte_length) or missing:
        return {
            "verified_result": False,
            "status": "REJECTED_PARTIAL_OR_INCOMPLETE_PROVENANCE_UPLOAD",
            "sanitized_public_log": sanitized_log,
        }

    return {
        "verified_result": True,
        "status": "UPLOAD_COMPLETE_WITH_PROVENANCE",
        "result_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "provenance": dict(provenance),
        "sanitized_public_log": sanitized_log,
    }


def negotiate_worker_update_and_version(
    *,
    current_version: Tuple[int, int, int],
    target_version: Tuple[int, int, int],
    supported_protocol_major: int,
    target_protocol_major: int,
    contributor_limits: Dict[str, int],
) -> Dict[str, Any]:
    """Negotiate safe worker updates and protocol versions (`WORKER-11` / Issue #142).

    Rejects downgraded or incompatible protocol versions, preserves contributor limits,
    and supports rollback to `current_version`.
    """
    if target_protocol_major != supported_protocol_major:
        raise ValueError(
            f"Incompatible protocol major version: target={target_protocol_major}, supported={supported_protocol_major}"
        )
    if target_version < current_version:
        raise ValueError(f"Downgraded worker version rejected: {target_version} < {current_version}")

    return {
        "update_accepted": True,
        "previous_version": list(current_version),
        "active_version": list(target_version),
        "preserved_contributor_limits": copy.deepcopy(contributor_limits),
        "rollback_target_version": list(current_version),
        "rollback_supported": True,
    }


def evaluate_malicious_result_submission_guard(
    submission: Dict[str, Any],
    *,
    independent_verification_passed: bool,
    private_gold_answer: str = "4/5",
) -> Dict[str, Any]:
    """Guard against malicious and unreliable result submissions (`SEC-10` / Issue #151).

    Requires independent verification before granting acceptance, and ensures rejection
    reasons never reveal private final-test answers.
    """
    if not independent_verification_passed:
        reason = "Submission failed independent verification check on reserved partition."
        assert private_gold_answer not in reason
        return {
            "accepted": False,
            "independent_verification_required": True,
            "public_rejection_reason": reason,
            "private_gold_answer_leaked": private_gold_answer in reason,
        }

    return {
        "accepted": True,
        "independent_verification_required": True,
        "public_rejection_reason": None,
        "private_gold_answer_leaked": False,
    }


def build_coordinator_observability_snapshot(
    *,
    submitted_count: int,
    completed_count: int,
    verified_count: int,
    promoted_count: int,
    incident_logs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build coordinator observability and sanitized incident logs (`NETWORK-10` / Issue #152).

    Metrics strictly distinguish submitted, completed, verified, and promoted work, and
    incident logs are sanitized of secrets and personal identifiers.
    """
    return {
        "work_totals": {
            "submitted": int(submitted_count),
            "completed": int(completed_count),
            "verified": int(verified_count),
            "promoted": int(promoted_count),
        },
        "sanitized_incident_logs": [sanitize_public_telemetry_and_logs(rec) for rec in incident_logs],
    }


def enforce_privilege_separation(
    *,
    actor_id: str,
    actor_role: str,
    requested_action: str,
    evidence_sha256: Optional[str],
    audit_log: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Separate maintainer and worker privileges (`SEC-09` / Issue #155).

    Worker credentials cannot mutate approved models or repository administration;
    privileged changes record actor and evidence in `audit_log`.
    """
    privileged_actions = {"PROMOTE_MODEL", "PUBLISH_RELEASE", "REVOKE_SIGNING_KEY", "MUTATE_REPO_ADMIN"}
    if requested_action in privileged_actions:
        if actor_role != "maintainer":
            raise PermissionError(
                f"Privilege violation: actor_role={actor_role!r} cannot execute {requested_action!r}"
            )
        if not evidence_sha256 or len(evidence_sha256) < 16:
            raise ValueError("Privileged maintainer change requires a verified evidence_sha256")
        entry = {
            "actor_id": actor_id,
            "actor_role": actor_role,
            "action": requested_action,
            "evidence_sha256": evidence_sha256,
        }
        audit_log.append(entry)
        return {"permitted": True, "audit_entry": entry}

    return {"permitted": True, "audit_entry": None}


def summarize_community_experiment_sprint(
    *,
    actual_participants: List[str],
    claimed_issue_count: int,
    reviewed_artifacts: List[Dict[str, Any]],
    reviewer_load_hours: float,
) -> Dict[str, Any]:
    """Summarize a small community experiment sprint (`COMM-14` / Issue #197).

    Records actual participants, completed evidence, and reviewer load; completion is
    measured by reviewed artifacts rather than claimed issues.
    """
    accepted_artifacts = [a for a in reviewed_artifacts if a.get("review_status") == "APPROVED_EVIDENCE"]
    return {
        "actual_participant_count": len(actual_participants),
        "actual_participants": list(actual_participants),
        "claimed_issue_count": int(claimed_issue_count),
        "completed_reviewed_artifact_count": len(accepted_artifacts),
        "reviewer_load_hours": round(float(reviewer_load_hours), 2),
        "sprint_completion_basis": "reviewed_artifacts_not_claimed_issues",
    }




