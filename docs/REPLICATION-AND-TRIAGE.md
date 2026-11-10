# Independent Replication Roles and Bounded Triage Process (`COMM-05`, `COMM-07`)

This policy governs how `verifiable-ai` manages contributor issue queues (`COMM-05`, Issue `#101`) and separates experiment authorship from independent verification (`COMM-07`, Issue `#60`).

---

## 1. Bounded Triage and Ready-Queue Governance (`COMM-05`)

To keep the roadmap actionable and prevent contributors from claiming blocked or underspecified issues:

1. **Dependency Verification Before `status:ready`:**
   - An issue may only carry `status:ready` after every prerequisite listed under its `## Dependencies` section is closed with reproducible evidence on `main`.
   - Issues with open dependencies must remain `status:blocked`.
2. **Bounded Ready-Queue Limits:**
   - Each workstream (`area:baseline`, `area:evaluation`, `area:data`, `area:training`, `area:experts`, `area:reasoning`, `area:network`, `area:serving`, `area:safety`, `area:community`, `area:website`, `area:governance`) maintains a bounded active claim queue (target maximum: **3 concurrently `status:in-progress` issues per workstream** unless executed in a coordinated dependency wave).
3. **Stale Claim Release (14-Day Policy):**
   - If an issue labeled `status:in-progress` receives no linked commits, PR updates, or reproducibility notes for **14 calendar days**, maintainers may return it to `status:ready` so other contributors are not blocked.
4. **Closing Issues on Negative or Inconclusive Results:**
   - Research and training issues do **not** require positive metric improvements to close.
   - An issue is completed when the hypothesis is tested under the locked benchmark protocol and the outcome (`ACCEPTED`, `REJECTED`, or `INCONCLUSIVE`) is recorded in versioned `research/` artifacts alongside its failure modes (for example, `research/bayesian-pilot/results.json`).

---

## 2. Independent Replication Roles (`COMM-07`)

Non-hardware-rich contributors can provide high-value scientific verification without training models from scratch by serving as **Independent Reproducers**.

### Role Separation
| Role | Responsibilities | Prohibited Actions |
| :--- | :--- | :--- |
| **Experiment Author** | Implements code/training run, produces initial `research/` JSON evidence, records exact commit SHA, seed, and checkpoint SHA-256. | May not mark their own experimental claim as independently replicated. |
| **Independent Reproducer** | Runs locked offline evaluation commands (`python -m aix_zero.benchmark_baseline`, `python -m aix_zero.verify_sparse_dispatch`, `pytest`) on an independent environment and submits a Replication Report. | Does not alter model weights, prompt templates, or evaluation splits during replication. |

### Replication Report Checklist
Every independent replication comment or PR must record:
- **Target Issue / Claim ID** and **Git Commit SHA** tested.
- **Operating System, CPU/Accelerator, and Python/PyTorch versions** (`python -V`, `torch.__version__`).
- **Checkpoint SHA-256** verified against `research/checkpoint-inventory.json`.
- **Exact Command Executed** and whether observed neural-only (`9/105`) and symbolic-assisted (`105/105`) scores match within the tolerances defined in [`docs/HARDWARE-AND-NUMERICAL-TOLERANCES.md`](HARDWARE-AND-NUMERICAL-TOLERANCES.md).
- **Verdict:** `REPLICATED`, `DEVIATION_WITHIN_TOLERANCE`, or `REPLICATION_FAILED` (with full stderr/stdout log).
