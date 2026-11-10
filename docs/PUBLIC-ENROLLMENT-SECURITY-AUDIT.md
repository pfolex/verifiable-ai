# Public Enrollment Security & Readiness Audit (`SEC-16` / Issue #160)

## Audit Decision

- **Broad Public Volunteer Enrollment Status:** `DISABLED_LOCKED` (`broad_volunteer_signup_enabled: false`)
- **Programmatic Gate:** [`audit_public_enrollment_readiness()`](../aix_zero/network_coordinator.py)

## Requirement Checklist

| Requirement ID | Control Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| `SEC-02 (#147)` | Worker sandbox isolation & platform limitations documented | `PASSED_WITH_OS_CONTAINER_REQUIREMENT` | App-level path/socket guards must be paired with OS container isolation for untrusted workloads. |
| `SEC-05 (#148)` | Constrained job arguments & path allowlists | `PASSED` | Shell metacharacters and resource-cap overrides rejected. |
| `SEC-10 (#151)` | Malicious submission guard & private test answer protection | `PASSED` | Independent verification required; rejection logs never leak gold answers. |
| `SEC-11 (#154)` | Public endpoint rate, storage, and queue quotas | `PASSED` | Rate/storage/queue guards tested. |
| `SEC-13 (#156)` | Incident response & credential/artifact revocation drill | `PASSED` | Tabletop revocation and quarantine drill verified. |
| `SEC-15 (#159)` | Website and API unauthenticated mutation protection | `PASSED` | Unauthenticated mutations blocked; bundles scanned for secrets. |
| `TRAIN-16 (#103)` | Measured held-out learning gain before volunteer scaling | `FAILED_NO_GO_GATE` | Single-expert Bayesian pilot was `REJECTED` (`0/96` held-out test accuracy); large volunteer compute remains locked. |

## Policy

Broad public volunteer signup remains **disabled** while any security gate or the `M2` measured learning-gain gate (`TRAIN-16`) remains in `NO_GO` status.
