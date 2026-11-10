# Community Progress Templates, Recruitment, and Beginner Contribution Paths (`COMM-08`, `COMM-10`, `COMM-11`)

This document provides the standardized templates and onboarding paths for `verifiable-ai` (`COMM-08` Issue `#180`, `COMM-10` Issue `#193`, and `COMM-11` Issue `#194`).

---

## 1. Measured Progress-Update Template (`COMM-08` / Issue #180)

Every periodic project update must use this structure, cite immutable repository evidence + execution mode (`neural_only` vs `symbolic_assisted`), and use the explicit zero-activity state when no new empirical measurements occurred.

```markdown
# AIX-Zero Measured Progress Update — [YYYY-MM-DD]

## 1. Empirical Capability & Evaluation Changes
- **Active Baseline Checkpoint:** `aix_zero_v11_kip.safetensors` (`817,410` physical learned parameters; `719,106` active per-token in sparse top-2 mode)
- **Neural-Only Exact Match (`inflight_alu=False`):** `9 / 105` (`8.57%`) — Evidence: [`research/baseline/v11-local.json`](../research/baseline/v11-local.json)
- **Symbolic-Assisted Exact Match (`inflight_alu=True`):** `105 / 105` (`100.0%`) — Evidence: [`research/baseline/v11-local.json`](../research/baseline/v11-local.json)
- **New Validated Model Capability Gains This Period:** `NONE` (Do not invent capability gains from completed documentation or infrastructure issues.)

## 2. Negative / Inconclusive Experiment Ledger
- **Frozen-`v11` Single-Expert Bayesian Pilot (`BP` / `BS`):** `REJECTED` (`3/96` neural-only on `BP`, `1/144` neural-only on `BS`; below `prior_only` shortcut controls `10/96` and `24/144`) — Evidence: [`research/bayesian-pilot/results.json`](../research/bayesian-pilot/results.json)

## 3. Infrastructure, Schemas & Blockers
- **Completed Engineering / Governance Tasks:** [List commit SHAs and closed issue numbers]
- **Current Blockers:** [List technical or statistical blockers]
- **Contributor Activity:** [Report actual merged PRs or state `No external contributor PRs merged this period`]
```

---

## 2. Evidence-Based Recruitment Announcement (`COMM-10` / Issue #193)

> **Outreach Governance Rule:** Maintainers must explicitly choose target destinations and approve any external posting before this announcement is shared outside the repository.

```markdown
### Invitation: Contribute to `verifiable-ai` (Sub-Megaparameter Neural-Symbolic Verification & Sparse Cartridge Research)

We are building open, reproducible infrastructure for **AIX-Zero**, an `Apache-2.0` research codebase exploring how small latent-recurrent neural models interact with deterministic symbolic verifiers and modular SwiGLU expert cartridges.

**Current Measured Status (No Hype):**
- **Physical Scale:** `817,410` physical learned parameters (`aix_zero_v11_kip.safetensors`), with `719,106` active parameters per token under sparse top-2 routing.
- **Measured Accuracy Separation:** On our 21-domain held-out template suite (`105` prompts), the `v11` checkpoint scores **`9 / 105` (`8.57%`) neural-only** and **`105 / 105` (`100.0%`) when paired with `InFlightSymbolicVerifier`**.
- **Published Negative Results:** Our first frozen-backbone single-expert Bayesian pilot (`49,152` trainable parameters) scored `3 / 96` on `BP` and `1 / 144` on `BS` (`REJECTED`, below the `prior_only` shortcut). We publish failed experiments openly so contributors can help solve genuine generalization bottlenecks.

**Skills Sought (CPU-Friendly, Zero Paid API Keys Required):**
1. **Evaluation & Statistical Rigor:** Paired Hoeffding gates, semantic duplicate detection, and shortcut auditing (`aix_zero/eval_harness_bridge.py`, `aix_zero/semantic_splits.py`).
2. **Independent Replication:** Running offline CPU benchmarks (`python -m aix_zero.benchmark_baseline`) and submitting signed replication reports (`docs/REPLICATION-AND-TRIAGE.md`).
3. **Curriculum & Representation Design:** Designing validation-only ablations to overcome frozen-backbone pointer-copy shortcuts (`docs/BAYESIAN-PILOT-ERROR-ANALYSIS.md`).

**Commitments & Expectations:**
- Opt-in claiming on issues labeled `status:ready` in `https://github.com/pfolex/verifiable-ai/issues`.
- No datacenter GPUs required for `M0`–`M2` tasks (`pytest -q aix_zero_tests tests` runs on CPU in <35 seconds).
```

---

## 3. Three Well-Scoped Beginner Contribution Paths (`COMM-11` / Issue #194)

> **Guardrail:** Never label blocked research breakthroughs or open-ended model training milestones as `good first issue`. Beginner paths must be self-contained, deterministic, and verifiable offline on a laptop CPU.

| Beginner Path | Concrete Example Task | Expected Deliverable | Review & Verification Route |
| :--- | :--- | :--- | :--- |
| **Path 1: Documentation & Provenance Auditing** | Audit a committed dataset or checkpoint entry in `research/dataset-provenance-catalog.json` or `research/checkpoint-inventory.json` to ensure all fields and license tags match disk artifacts. | A small PR updating markdown docs or catalog descriptions without modifying `.safetensors` weights. | Automated check via `pytest -q aix_zero_tests/test_data_provenance.py aix_zero_tests/test_checkpoint_registry.py`. |
| **Path 2: Deterministic Validation & Property-Test Fixtures** | Add edge-case unit tests for `score_rational_or_probability()`, `canonical_problem_identity()`, or `cross_check_generator_with_independent_oracle()` (e.g., testing negative rationals, unreduced fractions, or injected generator typos). | New unit test cases in `aix_zero_tests/test_eval_harness_bridge.py` or `aix_zero_tests/test_semantic_splits.py`. | Automated check via `pytest -q aix_zero_tests/test_eval_harness_bridge.py aix_zero_tests/test_semantic_splits.py`. |
| **Path 3: Offline Independent Reproduction Report** | Run `python -m aix_zero.benchmark_baseline` and `python -m aix_zero.verify_sparse_dispatch` on your local CPU and compare against `research/baseline/v11-local.json` within `docs/HARDWARE-AND-NUMERICAL-TOLERANCES.md`. | A signed Replication Report comment or JSON fixture verified by `verify_independent_reproduction()`. | Maintainer review against the checklist in [`docs/REPLICATION-AND-TRIAGE.md`](REPLICATION-AND-TRIAGE.md). |
