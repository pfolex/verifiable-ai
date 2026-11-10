# Website Information Architecture and Audience Specification (`WEB-01`)

This document specifies the public information architecture, target audiences, and truthful data-state rules for the `verifiable-ai` public website (`WEB-01`, Issue `#174`).

---

## 1. Target Audiences and Primary User Journeys

| Audience Segment | Primary Need | Entry Page & Key Action |
| :--- | :--- | :--- |
| **1. Prospective Contributors (Code, Data, Triage)** | Find unblocked `good first issue` or `status:ready` tasks and understand the local verification workflow. | `/contribute` $\rightarrow$ Filter ready issues & run offline verification (`pytest`). |
| **2. Independent Researchers & Auditors** | Inspect verified benchmarks, parameter accounting (`817,410` physical vs. `719,106` active per-token), and negative/inconclusive research results. | `/results` $\rightarrow$ Inspect `research/baseline/v11-local.json`, `research/dispatch/local.json`, and `research/bayesian-pilot/results.json`. |
| **3. Model & Verifier Users** | Run the local neural-symbolic REPL or verification harness on their own machine and understand capability boundaries. | `/` (Landing) $\rightarrow$ Quickstart commands classified as `LOCAL_OFFLINE` in [`aix_zero/checkpoint_registry.py`](../aix_zero/checkpoint_registry.py). |

---

## 2. Sitemap and Required Page Contracts

| Route / Section | Purpose | Source of Truth Artifact(s) | Unavailable-Data / Truthfulness State |
| :--- | :--- | :--- | :--- |
| **`/` (Landing Page)** | Present the project thesis (verifiable neural-symbolic AI + sparse cartridge experts) with exact current scale and accuracy separation. | [`README.md`](../README.md), [`research/checkpoint-inventory.json`](../research/checkpoint-inventory.json) | Must display **`817,410` learned parameters** (`9/105` neural-only, `105/105` symbolic-assisted) and never imply a `1T` model is trained or deployed. |
| **`/contribute` (Contributor Hub)** | Onboard code, benchmark, data, and replication contributors without requiring datacenter GPUs. | [`CONTRIBUTING.md`](../CONTRIBUTING.md), [`docs/REPLICATION-AND-TRIAGE.md`](REPLICATION-AND-TRIAGE.md) | If no live coordinator queue is running, direct contributors to GitHub issues labeled `status:ready` and offline replication tasks. |
| **`/roadmap` (Milestone & Dependency Explorer)** | Show `M0`–`M5` milestones, dependency waves, and completion status. | [`docs/AIX-1T-MASTER-ROADMAP.md`](AIX-1T-MASTER-ROADMAP.md), GitHub Issues (`#10`) | Clearly distinguish completed infrastructure/specs from open research milestones (`M2`–`M5`). |
| **`/results` (Verified Evidence & Negative Results Ledger)** | Render machine-verified scorecards, neural vs. symbolic accuracy breakdowns, and negative/rejected experiments. | [`research/baseline/v11-local.json`](../research/baseline/v11-local.json), [`research/dispatch/local.json`](../research/dispatch/local.json), [`research/bayesian-pilot/results.json`](../research/bayesian-pilot/results.json) | Must render `REJECTED` / `INCONCLUSIVE` outcomes alongside accepted baselines; never hide failed specialist training runs. |
| **`/network` (Distributed Worker & Telemetry Status)** | Explain the planned volunteer/federated specialist training architecture and show live or offline coordinator status. | [`docs/AIX-ZERO-EXPERT-STORAGE.md`](AIX-ZERO-EXPERT-STORAGE.md) | **Explicit Offline Banner Required:** While distributed worker coordination (`M4`) is in specification/simulation phase, display `"Distributed coordinator: OFFLINE (Local simulation only)"` rather than simulated live counters. |
| **`/updates` (Research Changelog)** | Chronological log of baseline locks, schema updates, and experimental capsules. | Git commit history and `research/` READMEs | Every update entry must link to a commit SHA and reproducible command. |
