## Experiment or Research Capsule Summary

- **Linked Roadmap Issue(s):** Closes #
- **Experiment ID & Config Schema:** Validated against `schemas/experiment-config-v1.schema.json`
- **Outcome Verdict:** [ ] `ACCEPTED` | [ ] `REJECTED` | [ ] `INCONCLUSIVE` | [ ] `REPLICATED`

## Metrics & Evidence Separation

- **Neural-Only Exact Match (No Symbolic Repair):**
- **Symbolic-Assisted Exact Match:**
- **Target Domain Delta vs. `v11` Baseline:**
- **Non-Target Anchor Retention Delta:**
- **Physical Learned Parameters vs. Active Per-Token Parameters:**

## Reproducibility Capsule

- **Checkpoint SHA-256:**
- **Dataset SHA-256:**
- **Seed(s) & Hardware Profile:**
- **Exact Offline Verification Command:**
  ```bash
  python -m aix_zero.benchmark_baseline
  ```

## Failure Modes, Limitations & Provenance

- **Observed Regressions or Failure Modes:**
- **Data / Model Rights Confirmation:** All datasets and checkpoints referenced are registered in `research/dataset-provenance-catalog.json` and `research/checkpoint-inventory.json`.
