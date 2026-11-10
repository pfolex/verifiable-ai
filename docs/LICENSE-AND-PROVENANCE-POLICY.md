# Software Licensing, Checkpoint Rights & Contribution Terms (`COMM-16` / Issue #199)

## 1. Repository Software License (`Apache-2.0`)

With the repository owner's confirmation to complete the `M0` foundation issues, the first-party source code, evaluation schemas, verifiers, and procedural dataset generators in `pfolex/verifiable-ai` are licensed under the **Apache License, Version 2.0** (`SPDX-License-Identifier: Apache-2.0`), published in [`LICENSE`](../LICENSE).

### Inbound=Outbound Contribution Terms
Under Section 5 of the Apache License 2.0 (`Submission of Contributions`), any contribution intentionally submitted for inclusion in this repository (via pull request, issue patch, or commit) is licensed under `Apache-2.0` without additional terms or conditions, unless explicitly marked `"Not a Contribution"`. Contributors must have the legal right to submit their code or data and must not submit third-party code or datasets with incompatible or unknown licenses.

---

## 2. Explicit Exceptions: Datasets, Web Excerpts, and Checkpoints

Public availability in a Git repository does **not** automatically make every file `Apache-2.0`. The following artifacts have distinct provenance and redistribution rules tracked in [`research/dataset-provenance-catalog.json`](../research/dataset-provenance-catalog.json) (`DATA-01`) and [`research/checkpoint-inventory.json`](../research/checkpoint-inventory.json) (`BASE-05`):

| Artifact Category | Path / Pattern | SPDX / License Status | Redistribution & Volunteer-Compute Eligibility |
| :--- | :--- | :--- | :--- |
| **First-Party Source Code & Schemas** | `*.py`, `schemas/*.json`, `aix_zero/*.py`, `tests/`, `aix_zero_tests/` | `Apache-2.0` (`verified_permissive`) | Eligible for redistribution and public volunteer execution. |
| **First-Party Procedural Tasks** | `rlvr-forge-procedural`, `aix-zero-v11-template-heldout`, `aix-zero-bayesian-pilot` (`research/bayesian-pilot/dataset.json`) | `Apache-2.0` (`verified_permissive`) | Eligible for redistribution and public volunteer execution (`DATA-01`). |
| **Codex Scholar Notes & Web Excerpts** | `aix_zero/knowledge_codex/knowledge_index.json`, `WEB_RESEARCH_LOG.md`, `entries/*.md` | `CC-BY-SA-4.0 AND OTHER` (`restricted`) | Contains Wikipedia summaries (`CC-BY-SA-4.0`), arXiv metadata, and local Ollama (`aix-8b:latest`, `aix3:latest`) teacher outputs. **Blocked** from public volunteer training jobs by default (`DATA-01`). |
| **Baseline v1–v5 & v8 Checkpoints** | `aix_zero/checkpoints/aix_zero_v1..v4.safetensors`, `aix_zero_v8.safetensors` | `Apache-2.0` (`verified_permissive_first_party`) | Trained 100% from scratch on first-party procedural arithmetic/algebra/calculus/code templates. |
| **Symbiont & KIP Checkpoints (`v6`, `v7`, `v9`, `v10`, `v11`)** | `aix_zero/checkpoints/aix_zero_v6..v11_kip.safetensors` | `Apache-2.0` weights with `mixed_or_teacher_assisted_history` | Numeric curriculum problems (`build_21_domain_kip_v11_curriculum`) are 100% first-party procedural code, **but** `v6`/`v7` symbiont runs and `v9`–`v11` codex sessions queried local Ollama teachers (`aix-8b:latest`, `aix3:latest`). Recorded with explicit provenance warnings in [`research/checkpoint-inventory.json`](../research/checkpoint-inventory.json). |
| **Rejected Bayesian Specialist Candidates** | `research/bayesian-pilot/candidates/*/*.safetensors` | `Apache-2.0` (`rejected_research_artifact`) | Trained on first-party `research/bayesian-pilot/dataset.json` starting from frozen `v11`; marked `REJECTED` (`promotion_status: "REJECTED"`). |

---

## 3. Third-Party Dependencies

Runtime and test dependencies (`torch`, `numpy`, `safetensors`, `jsonschema`, `transformers`, `peft`, `accelerate`, `huggingface-hub`) are governed by their respective upstream permissive licenses (BSD-3-Clause, Apache-2.0, MIT) and are installed via `pip` rather than vendored into this repository.
