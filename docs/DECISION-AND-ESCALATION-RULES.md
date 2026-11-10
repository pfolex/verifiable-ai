# Project Decision and Escalation Rules (`COMM-13` / Issue #196)

## Governance Structure & Honest Scope

- **Current Governance Reality:** `pfolex/verifiable-ai` is maintained by its repository owner and consenting open-source reviewers. We do **not** claim a fictional multi-institution governing board or foundation (`fictional_governing_board_claimed: false`).
- **Unresolved Roles & Coverage Gaps:** Independent security auditing and secondary release-signing custody remain explicit open contributor roles rather than assumed background guarantees.

## Decision & Escalation Protocol

1. **Evidence-Linked Decisions:**
   - Every architectural, dataset, benchmark, or model-promotion decision must be recorded via [`record_project_governance_decision()`](../aix_zero/network_coordinator.py) with an immutable `evidence_sha256` and the list of `consenting_maintainers`.
2. **Scientific Integrity Escalation:**
   - Any contributor may open a discrepancy or negative-result report when a published claim conflicts with [`research/baseline/v11-local.json`](../research/baseline/v11-local.json), [`research/dispatch/local.json`](../research/dispatch/local.json), or [`research/bayesian-pilot/results.json`](../research/bayesian-pilot/results.json).
   - Disputed capability claims default to **unapproved/experimental** until an independent reproduction passes [`verify_independent_reproduction()`](../aix_zero/eval_harness_bridge.py).

## Maintainer Succession & Bus-Factor Plan (`COMM-15` / Issue #198)

- **Least-Privilege & Explicit Consent:** New maintainer or reviewer responsibilities require explicit opt-in consent and scoped permissions (`build_maintainer_succession_and_bus_factor_plan()`).
- **Workstream Coverage & Honest Gaps:**
  - `evaluation_harness`: Primary (`pfolex`), Backup (`open-reproducer-role` / automated CI verification).
  - `release_signing`: Primary (`pfolex`), Backup (`HONEST_COVERAGE_GAP_SINGLE_MAINTAINER` — documented key rotation & revocation procedure in `SEC-08`).

