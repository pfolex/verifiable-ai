# Milestone `M0` Foundation Readiness Audit (`BASE-16` / Issue #39)

This audit reviews all `M0` (Foundation, Provenance, Evaluation Lock, and Governance) criteria in `pfolex/verifiable-ai` to verify that baseline runs reproduce offline, provenance is hashed and versioned, and public claims match empirical evidence.

---

## 1. Criterion-by-Criterion Readiness Matrix

| Roadmap Code | Requirement | Implementation & Evidence Artifact(s) | Verification Test Suite | Status |
| :--- | :--- | :--- | :--- | :--- |
| **`BASE-01` (`#24`)** | Remove absolute host paths from committed artifacts | `research/baseline/v11-local.json`, `research/dispatch/local.json`, `research/bayesian-pilot/*.json` | `aix_zero_tests/test_published_evidence.py` | **PASS** (`f67335d`) |
| **`BASE-02` (`#25`)** | Lock deterministic Python/PyTorch/NumPy/Safetensors dependencies | [`requirements-aix-zero.txt`](../requirements-aix-zero.txt), [`requirements-test.txt`](../requirements-test.txt) | `aix_zero_tests/test_environment_lock.py` | **PASS** (`527e7f0`) |
| **`BASE-03` (`#26`)** | Version ordered tokenizer vocabulary identity (`vocab_sha256`, `prefix_sha256`) | [`aix_zero/tokenizer.py`](../aix_zero/tokenizer.py) | `aix_zero_tests/test_tokenizer_identity.py` | **PASS** (`527e7f0`) |
| **`BASE-04` (`#27`)** | Align public parameter, latency, and neural vs. symbolic accuracy claims | [`README.md`](../README.md), [`docs/AIX-1T-MASTER-ROADMAP.md`](AIX-1T-MASTER-ROADMAP.md), [`docs/AIX-ZERO-EXPERT-STORAGE.md`](AIX-ZERO-EXPERT-STORAGE.md) | `aix_zero_tests/test_published_evidence.py` | **PASS** (`f67335d`) |
| **`BASE-05` (`#28`)** | Index every committed `.safetensors` artifact with SHA-256 and provenance status | [`research/checkpoint-inventory.json`](../research/checkpoint-inventory.json), [`schemas/checkpoint-inventory-v1.schema.json`](../schemas/checkpoint-inventory-v1.schema.json) | `aix_zero_tests/test_checkpoint_registry.py` | **PASS** (`2314ee3`) |
| **`BASE-06` (`#29`)** | Side-effect-free strict checkpoint loader (no silent dir creation or random resize) | `load_checkpoint_strict()` in [`aix_zero/checkpoint_registry.py`](../aix_zero/checkpoint_registry.py) | `aix_zero_tests/test_checkpoint_registry.py` | **PASS** (`2314ee3`) |
| **`BASE-07` (`#30`)** | Separate stable offline release commands from experimental research commands | `CLI_COMMAND_CLASSIFICATION` in [`aix_zero/checkpoint_registry.py`](../aix_zero/checkpoint_registry.py) | `aix_zero_tests/test_checkpoint_registry.py` | **PASS** (`5e5c6f9`) |
| **`BASE-08` (`#31`)** | Machine-readable experiment configuration contract with canonical SHA-256 | [`schemas/experiment-config-v1.schema.json`](../schemas/experiment-config-v1.schema.json), `validate_experiment_config()` | `aix_zero_tests/test_checkpoint_registry.py` | **PASS** (`5e5c6f9`) |
| **`BASE-09` (`#32`)** | Interrupted experiment recovery verifying config, data, optimizer, and RNG state | `save_training_recovery_bundle()`, `resume_training_from_bundle()` in [`aix_zero/checkpoint_registry.py`](../aix_zero/checkpoint_registry.py) | `aix_zero_tests/test_checkpoint_registry.py` | **PASS** |
| **`BASE-10` (`#33`)** | Structured failure and partial-run records (`is_validated_checkpoint = False`) | `build_experiment_failure_record()` in [`aix_zero/checkpoint_registry.py`](../aix_zero/checkpoint_registry.py) | `aix_zero_tests/test_checkpoint_registry.py` | **PASS** |
| **`BASE-11` (`#34`)** | Document cross-hardware numerical tolerances (`rtol=1e-5, atol=1e-6`) | [`docs/HARDWARE-AND-NUMERICAL-TOLERANCES.md`](HARDWARE-AND-NUMERICAL-TOLERANCES.md) | `aix_zero_tests/test_checkpoint_registry.py` | **PASS** (`2314ee3`) |
| **`BASE-12` (`#35`)** | Clean-checkout CI smoke test running both `tests/` and `aix_zero_tests/` | [`.github/workflows/fishtest-ci.yml`](../.github/workflows/fishtest-ci.yml) | `aix_zero_tests/test_checkpoint_registry.py` | **PASS** (`2314ee3`) |
| **`BASE-13` (`#36`)** | Protect source & indexed baseline checkpoints from accidental overwrite | `save_checkpoint_protected()` in [`aix_zero/checkpoint_registry.py`](../aix_zero/checkpoint_registry.py) | `aix_zero_tests/test_checkpoint_registry.py` | **PASS** (`5e5c6f9`) |
| **`BASE-14` (`#37`)** | Reproducible failure capsule for the rejected Bayesian single-expert pilot | [`research/bayesian-pilot/README.md`](../research/bayesian-pilot/README.md), [`research/bayesian-pilot/results.json`](../research/bayesian-pilot/results.json) | `aix_zero_tests/test_bayesian_pilot.py`, `test_published_evidence.py` | **PASS** (`5e5c6f9`) |
| **`EVAL-01..06` (`#40`–`#46`)** | Unified evaluation bridge, multidimensional scorecard, paired Hoeffding gates, and exact rational scorer | [`aix_zero/eval_harness_bridge.py`](../aix_zero/eval_harness_bridge.py), [`schemas/aix-scorecard-v1.schema.json`](../schemas/aix-scorecard-v1.schema.json) | `aix_zero_tests/test_eval_harness_bridge.py` | **PASS** (`2314ee3`, `5e5c6f9`) |
| **`DATA-01..02` (`#51`, `#43`)** | Dataset provenance catalog & semantic split deduplication | [`research/dataset-provenance-catalog.json`](../research/dataset-provenance-catalog.json), [`aix_zero/semantic_splits.py`](../aix_zero/semantic_splits.py) | `aix_zero_tests/test_data_provenance.py`, `test_semantic_splits.py` | **PASS** (`527e7f0`, `2314ee3`) |
| **`COMM-01..04, 16`** | Code of conduct, governance roster, offline contributor quickstart, and `Apache-2.0` licensing | [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md), [`GOVERNANCE.md`](../GOVERNANCE.md), [`CONTRIBUTING.md`](../CONTRIBUTING.md), [`LICENSE`](../LICENSE) | `aix_zero_tests/test_community_docs.py` | **PASS** (`527e7f0`, `2314ee3`) |

---

## 2. Documented Unresolved Baseline Limitations & Historical Gaps

Advancing beyond `M0` requires keeping the following empirical limitations visible in all reports:

1. **Historical Checkpoint Provenance Gaps (`v1`–`v11`):**
   - Checkpoints `aix_zero_v1_symbiotic.safetensors` through `aix_zero_v11_kip.safetensors` were trained before `BASE-08` experiment-config hashing and included local Ollama teacher distillation (`qwen2.5:0.5b`) or Wikipedia/arXiv text ingestion (`v5`, `v11`) without immutable raw HTTP response snapshots. These gaps are permanently recorded in [`research/checkpoint-inventory.json`](../research/checkpoint-inventory.json).
2. **Legacy `v11` `WORD_PROBLEM` Surface-Name Overlap:**
   - Semantic split auditing (`DATA-02`, `DATA-13`) identified 5 held-out `WORD_PROBLEM` prompts in `v11` (`Sam has 5 packs of 5 and buys 2`) whose operand triple `(5, 5, 2)` also appeared in `v11` training under a different name (`Mia has 5 packs of 5 and buys 2`). All other 20 `v11` domains are strictly disjoint, and `partition_by_semantic_identity()` prevents this overlap in all future splits.
3. **Neural-Only vs. Symbolic-Assisted Capability Gap:**
   - The frozen `817,410`-parameter `v11` baseline achieves `9/105` (`8.57%`) neural-only exact match and `105/105` (`100.0%`) only when paired with `InFlightSymbolicVerifier`.
4. **Rejected Single-Expert Bayesian Pilot (`BASE-14`, `TRAIN-01`):**
   - Training a single `49,152`-parameter expert inside the frozen `v11` backbone scored `3/96` (`3.125%`) on `BP` and `1/144` (`0.694%`) on `BS`, failing its promotion screen (`REJECTED`) and underperforming the `prior_only` shortcut (`10.42%` / `16.67%`).

---

## 3. Readiness Verdict & Reproducible Audit Command

- **Verdict:** **`M0_FOUNDATION_READY`** — All baseline, provenance, evaluation, and governance locks are enforced by automated tests, and all historical limitations and negative results are explicitly documented.
- **Offline Verification Command:**
  ```bash
  pytest -q aix_zero_tests tests
  ```
