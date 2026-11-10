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

