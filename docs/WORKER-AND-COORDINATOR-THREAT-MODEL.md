# Worker and Coordinator Threat Boundaries (`SEC-01` / Issue #127)

This specification defines the trust boundaries, protected assets, required authority per state transition, and deployment tiers for the `verifiable-ai` distributed experiment network (`SEC-01`).

---

## 1. Untrusted Surfaces vs. Protected Assets

| Untrusted Surface | Threat Vector | Protected Asset | Enforced Control |
| :--- | :--- | :--- | :--- |
| **1. Job Manifests & Configs** | Malicious or malformed job attempting unbounded resource consumption or arbitrary code execution on a volunteer worker. | Volunteer worker CPU, memory, disk, and host OS. | Validated against [`schemas/network-job-lease-v1.schema.json`](../schemas/network-job-lease-v1.schema.json) (`validate_network_job_contract`); fixed declarative task runners (`OFFLINE_EVALUATION_JOB`, `BOUNDED_SPECIALIST_TRAINING_JOB`) with zero `eval`/`exec` or shell hooks. |
| **2. Worker-Submitted Checkpoints / Shards** | Poisoned weights, `NaN`/`Inf` injection, deserialization attacks, or fabricated benchmark metrics. | Active baseline (`aix_zero_v11_kip.safetensors`), cartridge bank, and published `research/` ledger. | `safetensors`-only weight loading (no Python `pickle`), SHA-256 verification, finiteness check, `base_checkpoint_sha256` match (`EXPERT-02`), and independent local re-evaluation (`EVAL-15`). |
| **3. Worker Handshake & Telemetry** | Fingerprint/PII leakage or spoofed hardware claims. | Contributor privacy and coordinator queue integrity. | `validate_worker_capability_handshake()` in [`aix_zero/network_coordinator.py`](../aix_zero/network_coordinator.py) rejects any payload containing `hostname`, `ip_address`, `mac_address`, `username`, or `home_directory`. |
| **4. Public Website & Status Endpoints** | Stale cache poisoning or misrepresentation of offline simulation as a live network. | Public scientific integrity. | Explicit `is_stale` timestamps (`WEB-03`) and offline coordinator banner (`WEB-01`). |

---

## 2. Authority Required for State Transitions

| Transition | Required Authority | Verification Gate |
| :--- | :--- | :--- |
| `QUEUED` $\rightarrow$ `LEASED` | Coordinator matchmaker + Worker capability handshake (`WORKER-01`) | `match_job_to_worker()` == `MATCHED_LEASE_ELIGIBLE` |
| `LEASED` $\rightarrow$ `COMPLETED` (Candidate Pending Audit) | Leased Worker submission | Valid `evaluation_result` JSON + finite `.safetensors` shard matching `base_checkpoint_sha256` |
| `PENDING_AUDIT` $\rightarrow$ `ELIGIBLE_NOT_ACTIVATED` | Independent Reproducer + Coordinator Verifier (`EVAL-15`) | `verify_independent_reproduction()` == `REPRODUCED_VERIFIED` and `empirical_gate.decision == "PASS"` |
| `ELIGIBLE_NOT_ACTIVATED` $\rightarrow$ `APPROVED_ACTIVE` | Workstream Maintainer (`OVERALL` / `ARCH` / `EVAL`) | Explicit human maintainer sign-off and update to `research/checkpoint-inventory.json` |

---

## 3. Deployment Tiers: Trusted-Tester vs. Public Enrollment

1. **Phase 1 — Local & Trusted-Tester Cohort (`M2` Three-Computer Network):**
   - Only pre-registered trusted-tester worker keys may lease `BOUNDED_SPECIALIST_TRAINING_JOB` tasks.
   - All returned shards are re-evaluated end-to-end on the coordinator before any status update.
2. **Phase 2 — Open Opt-In Public Enrollment (`M4`):**
   - Unauthenticated or newly enrolled volunteer workers are restricted to deterministic `OFFLINE_EVALUATION_JOB` replication tasks until quorum agreement is established.
