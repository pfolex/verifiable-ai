# AIX Governance and Opt-In Maintainer Roster (`COMM-01` / Issue #59)

## 1. Core Principles

The repository owner (`@pfolex`) remains the decision maker until additional maintainers explicitly agree to documented responsibilities in writing via a public comment or pull request. No external board, advisory group, or guaranteed review capacity is assumed to exist.

- **Voluntary Participation:** Contributors propose and implement scoped work voluntarily. No person is listed as a coordinator, reviewer, or maintainer without their explicit opt-in consent.
- **Visible Vacancies:** Unfilled workstream coordinator and reviewer seats remain explicitly marked `VACANT (Opt-In Open)` rather than hidden or implied to be staffed.
- **Evidence-Gated Promotion:** Independent reproducers confirm headline claims using their own recorded environment. Release or specialist promotion requires a consenting maintainer and evidence linked to an immutable artifact.
- **Least Privilege:** Compute donors and workers cannot approve their own results, alter release catalogs, or receive administrative repository access merely for donating compute.

---

## 2. Opt-In Maintainer, Coordinator, and Reviewer Roster

All roles below require explicit opt-in via a comment on [Issue #10 (`[AIX ROADMAP]`)](https://github.com/pfolex/verifiable-ai/issues/10) or a pull request updating this table. Stepping down at any time requires only a comment or PR removing your handle.

| Workstream Code | Workstream Scope | Coordinator (Triage & Dependency Tracking) | Primary Reviewer(s) | Independent Reproducer(s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`OVERALL`** | Repository Ownership, Final Release Signoff & Conduct Escalation | `@pfolex` (Active) | `@pfolex` | *VACANT (Opt-In Open)* | **Active (Single-Maintainer)** |
| **`BASE`** | Baseline Integrity, Checkpoints, Tokenizer Identity & CPU Reproducibility (`BASE-01`–`BASE-04`) | `@pfolex` (Interim) | `@pfolex` | *VACANT (Opt-In Open)* | **Seeking Secondary Reviewer & Reproducer** |
| **`EVAL`** | Evaluation Schemas, Verifiers, Statistical Gates & Benchmarks (`EVAL-01`–`EVAL-16`) | `@pfolex` (Interim) | `@pfolex` | *VACANT (Opt-In Open)* | **Seeking Coordinator & Reviewers** |
| **`ARCH`** | Latent-Recurrent Core, Sparse MoRE Dispatch, Cartridge Bank & Specialists (`ARCH-01`–`ARCH-15`) | *VACANT (Opt-In Open)* | `@pfolex` | *VACANT (Opt-In Open)* | **Coordinator Seat Vacant** |
| **`DATA`** | Dataset Provenance, Licensing, Curriculum Generators & Leakage Audits (`DATA-01`–`DATA-12`) | *VACANT (Opt-In Open)* | `@pfolex` | *VACANT (Opt-In Open)* | **Coordinator Seat Vacant** |
| **`DIST`** | Volunteer Compute Worker, Coordinator API, Quorums & Sandboxing (`DIST-01`–`DIST-11`) | *VACANT (Opt-In Open)* | `@pfolex` | *VACANT (Opt-In Open)* | **Coordinator Seat Vacant** |
| **`COMM`** | Governance, Contributor Onboarding, Reproducibility & Public Reporting (`COMM-01`–`COMM-11`) | `@pfolex` (Interim) | `@pfolex` | *VACANT (Opt-In Open)* | **Seeking Coordinator & Reviewers** |

---

## 3. Role Responsibilities & Time Expectations

1. **Contributor (Ad-Hoc):**
   - Claims one `status:ready` issue at a time by commenting with scope and validation plan.
   - No ongoing commitment beyond the claimed task.
2. **Independent Reproducer (Opt-In per Artifact):**
   - Runs the offline reproduction commands in [`CONTRIBUTING.md`](./CONTRIBUTING.md) and [`docs/AIX-ZERO-EXPERT-STORAGE.md`](./docs/AIX-ZERO-EXPERT-STORAGE.md) on their own machine and posts the resulting JSON report hash and hardware summary.
3. **Reviewer (Opt-In by Workstream):**
   - Reviews pull requests within their declared area of competence, checking tests, scientific honesty rules, and schema/digest invariants.
   - May decline or defer any review; no response-time SLA is implied.
4. **Workstream Coordinator (Opt-In by Workstream):**
   - Helps triage new issues in their workstream, updates `status:blocked` -> `status:ready` labels when prerequisites close, and flags dependency conflicts.
5. **Maintainer (Opt-In + Owner Confirmation):**
   - Merges reviewed PRs that pass CI and acceptance criteria, enforces [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md), and signs off on artifact releases (never self-approving their own trained model promotions).

---

## 4. Research Proposals and Decision Records

Research proposals state a hypothesis, controls, validation budget, stop rule, rights/provenance, and measurable outcome. Negative results (`REJECTED` / `INCONCLUSIVE`) are valid first-class outputs. Priorities change through a public decision record explaining the evidence and affected dependencies. A large backlog is not a staffing commitment or a guaranteed delivery schedule.

Live coordination and issue index: https://github.com/pfolex/verifiable-ai/issues/10
