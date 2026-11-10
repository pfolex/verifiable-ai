# Distributed-Compute Alpha Readiness Audit (`NETWORK-16` / Issue #171)

## Audit Summary

- **Three-Node Local Coordination Pilot:** `VERIFIED_INFRASTRUCTURE` (`run_three_node_coordination_experiment()`, `run_three_computer_volunteer_client_pilot()`)
- **Broad Public Volunteer Enrollment:** `BLOCKED` (`public_enrollment_blocked: true`)
- **Programmatic Gate:** [`audit_distributed_compute_alpha_readiness()`](../aix_zero/network_coordinator.py)

## Three-Node Pilot Findings & Overhead

1. **Replication & Coordinator Overhead:**
   - With `2x` independent replication (`NETWORK-06`), `8%` coordinator scheduling/manifest verification overhead, and `5%` retry/churn allowance, a 3-node cohort yields `~1.218x` net verified useful throughput over a single machine (`12.18` vs `10.0` verified jobs/sec), **not** `3.0x`.
2. **Why Broad Public Enrollment Remains Blocked:**
   - Although lease idempotency (`NETWORK-03`), artifact SHA-256 verification (`SEC-03`), resource caps (`WORKER-03`), and pause/schedule controls (`WORKER-04`) pass verification, the `M2` learning-gain gate (`TRAIN-16`) remains `NO_GO` because single-expert frozen-backbone training on `BP`/`BS` achieved `0.0%` held-out test accuracy. We do not consume public volunteer electricity/compute without a reproduced held-out learning recipe.
