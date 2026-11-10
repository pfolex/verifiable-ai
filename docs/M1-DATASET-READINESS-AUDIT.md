# Milestone `M1` Dataset Readiness Audit (`DATA-16` / Issue #75)

This audit reviews dataset rights, semantic leakage controls, independent reference oracles, shortcut auditing, and curriculum selection rules across all datasets in `pfolex/verifiable-ai` (`DATA-01` through `DATA-15`).

---

## 1. Corpus-by-Corpus Readiness & Risk Ledger

| Dataset ID | Provenance & SPDX License | Semantic Split Disjointness (`DATA-02`) | Independent Reference Oracle (`DATA-04`) | Shortcut Audit (`DATA-07`) | Documented Risks & Exclusions (`DATA-13`) | Promotion Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`aix-zero-bayesian-pilot-v1`** (`BP` & `BS`) | Deterministic rational generator (`aix_zero/bayesian_tasks.py`); `Apache-2.0` | **100% Disjoint** (`85` train, `16` val, `24` test `(p,a,b)` triples per task) | Verified via `_independent_float_bayes_oracle()` (`mismatch_count == 0`) | `prior_only` scores `12/96` (`12.5%`) on `BP` and `24/144` (`16.67%`) on `BS` | Narrow binary hypothesis space $H \in \{0, 1\}$; conditional IID assumed in `BS`. | **APPROVED_FOR_M1** |
| **`aix-zero-v11-kip-21domain`** | Deterministic 21-domain generator (`aix_zero/knowledge_is_power.py`); `Apache-2.0` | **20 / 21 domains disjoint**; 5 `WORD_PROBLEM` prompts had surface-name (`Sam`/`Mia`) operand overlap | Verified via `InFlightSymbolicVerifier` and `evaluate_scientific_reasoning_suite()` | Constant majority `<5%` per domain | Legacy `WORD_PROBLEM` overlap logged in `build_benchmark_exclusion_ledger()`; new splits must use `partition_by_semantic_identity()`. | **HISTORICAL_BASELINE_LOCKED** |
| **`compose-bp-expected-utility-v1`** (`DATA-15`) | Two-primitive composition generator (`build_cross_domain_composition_pilot()`); `Apache-2.0` | Held-out evaluation-only composition family | Verified against independent float posterior $\times$ linear utility oracle | No constant or prior-copy shortcut matches linear expected utility | Small pilot size; requires scaling before multi-expert curriculum training. | **APPROVED_PILOT_EVAL** |
| **Legacy `v5`/`v11` Web Text Snippets** (Wikipedia / arXiv) | External HTTP fetches during historical `v5`/`v11` distillation; mixed `CC-BY-SA-4.0` / arXiv licenses | Unindexed transient web fetches (not committed as raw text) | No exact symbolic oracle for open-ended text snippets | N/A | **Blocked from new training queues** under `DATA-12` until immutable snapshot hashes and license-compatible filtering are provided. | **EXCLUDED_FROM_NEW_TRAINING** |

---

## 2. Enforced `M1` Data Governance Rules

1. **Zero Unreviewed Data Promotion:** No dataset may enter a training or benchmark queue solely because it improves a score; every submission must pass `validate_contributor_dataset_submission()` (`DATA-12`).
2. **Validation-Only Curriculum & Representation Selection (`DATA-08`, `DATA-09`, `DATA-10`):** Representation choices (`reduced_fraction` vs `num_den_fields` vs `scaled_integer_bps`) and multi-step derivation supervision are selected strictly on the `validation` split (`consumes_final_test_budget == False`).
3. **Verdict:** **`M1_DATASET_FOUNDATION_AUDITED`** — All active synthetic reasoning corpora have verified licenses, independent reference oracles, and explicit shortcut baselines, and legacy web/surface-overlap risks are isolated and versioned.
