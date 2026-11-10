# M2 Measured Learning Gain Audit (`TRAIN-16` / Issue #103)

## Decision Summary

- **Milestone Gate Status:** `NO_GO_FOR_LARGE_VOLUNTEER_TRAINING` (Single-expert frozen-backbone training recipe remains `REJECTED`; large volunteer training is **not** unlocked from validation loss reduction alone)
- **Programmatic Gate Function:** [`audit_m2_measured_learning_gain_before_scaling()`](../aix_zero/validation_training_studies.py)

## Empirical Evidence Reviewed

1. **Frozen-v11 Single-Expert Bayesian Pilot ([`research/bayesian-pilot/results.json`](../research/bayesian-pilot/results.json)):**
   - **BP (`Bayes Posterior`):** Validation token loss decreased (`9.146 -> 0.564`), and training exact match reached `14.1%` (`48/340`), but held-out test exact match (`0/96 = 0.0%`) failed to beat the `prior_only` shortcut control (`12/96 = 12.5%`) or majority baseline (`3/96 = 3.12%`). Decision: **`REJECTED`**.
   - **BS (`Sequential Bayes`):** Validation token loss decreased (`9.373 -> 0.506`), and training exact match reached `21.2%` (`72/340`), but held-out test exact match (`0/96 = 0.0%`) failed to beat the majority baseline (`3/96 = 3.12%`). Decision: **`REJECTED`**.
2. **Independent Reproduction Requirement (`EVAL-15` / `#61`):**
   - [`verify_independent_reproduction()`](../aix_zero/eval_harness_bridge.py) requires a second environment to reproduce any positive held-out gain within declared tolerances before a capability claim or volunteer scaling campaign can be approved.

## Scientific Honesty Policy

- Decreasing cross-entropy loss or training-split memorization **never** unlocks large-scale volunteer compute.
- Any candidate recipe that fails held-out semantic-group generalization remains recorded as `REJECTED` in the repository and public artifacts.
