# Verifiable AI Suite ("Fishtest for LLMs")

[![Fishtest Parameter PR Pipeline](https://github.com/pfolex/verifiable-ai/actions/workflows/fishtest-ci.yml/badge.svg)](https://github.com/pfolex/verifiable-ai/actions/workflows/fishtest-ci.yml)

A modular ecosystem of **6 standalone, Unix-style repositories** that connect **ground-truth scientific verifiers**, **synthetic RLVR environments**, **copy-on-write agent sandboxes**, **Apple Silicon prefix-tree inference**, **mechanistic interpretability steering**, and **Stockfish-style statistical parameter merging**.

Together, they solve a foundational problem in open-source AI: **allowing anyone on consumer hardware (like a Mac) to train, steer, statistically verify, and submit a Pull Request of model parameters (`ΔW`) across 4 rigorous scientific domains.**

---

## Architecture: How the 6 Repositories Interlock

```
+-----------------------------------------------------------------------------------+
| 6. verisci          Universal Ground-Truth Scientific & Formal Cartridges         |
|                     [Logic/SMT]  [Silicon/RTL]  [Systems/Concurrency]  [Matter]   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. rlvr-forge       Seed-Locked Procedural Task Generator & GRPO Environment      |
|                     Infinite curriculum tasks + deterministic verifiable rewards  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. branchbox        Deterministic Copy-on-Write Execution DAG                     |
|                     Sub-ms fork(), step(), rollback() & contrastive pair mining   |
+--------------------+-----------------------------------------+--------------------+
                     |                                         |
                     v                                         v
+-----------------------------------------+   +-------------------------------------+
| 4. mlx-tree        Apple Silicon Tree   |   | 5. steerlens   Mech-Interp, SAEs &  |
|                    Radix KV-Cache &     |<->|                Contrastive Steering |
|                    Local GRPO LoRA ΔW   |   |                Vectors (CAA)        |
+--------------------+--------------------+   +------------------+------------------+
                     |                                           |
                     +---------------------+---------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
| 1. everbench        Contamination-Proof Benchmark, SPRT Gate & TIES/DARE Merger   |
|                     Verifies ΔElo on unseen seeds & merges winning parameter PRs  |
+-----------------------------------------------------------------------------------+
```

---

## Live Repository Index

| Repository | Pillar | Description |
| :--- | :--- | :--- |
| [**`pfolex/verisci`**](https://github.com/pfolex/verisci) | **Pillar 6: Science & Formal Bridges** | Deterministic state-transition and reward verifiers for **Formal Logic (`logic`)**, **Digital Logic & RTL (`silicon`)**, **Distributed Systems (`systems`)**, and **Molecular Chemistry (`matter`)**. |
| [**`pfolex/rlvr-forge`**](https://github.com/pfolex/rlvr-forge) | **Pillar 2: Synthetic Env & RLVR** | Cryptographically seeded procedural task generator (`TaskForge`), Gymnasium-style `RLVREnv`, and Group Relative Policy Optimization (`GRPO`) advantage normalizer. |
| [**`pfolex/branchbox`**](https://github.com/pfolex/branchbox) | **Pillar 3: Agent Sandbox DAG** | Copy-on-Write execution state DAG with SHA-256 deduplication, `fork()`, `step()`, `rollback()`, PUCT tree search, and automatic contrastive trajectory pair extraction. |
| [**`pfolex/mlx-tree`**](https://github.com/pfolex/mlx-tree) | **Pillar 4: Tree Inference & Local Trainer** | Radix-tree prefix-shared KV cache (`RadixKVCache`) and local `GRPOLoRATrainer` that produces portable `ParameterDelta` weight updates on Apple Silicon. |
| [**`pfolex/steerlens`**](https://github.com/pfolex/steerlens) | **Pillar 5: Mech-Interp & Steering** | Top-K Sparse Autoencoders (`SparseAutoencoder`), Contrastive Activation Addition (`compute_caa_vector`), and parameter-folded steering vectors from tree rollouts. |
| [**`pfolex/everbench`**](https://github.com/pfolex/everbench) | **Pillar 1: Eval, SPRT & Weight Merger** | Unseen HMAC seed generator, Stockfish-style Sequential Probability Ratio Test (`SPRTGate`) for non-regression validation, and `TIESMerger` / `DAREMerger` for fusing contributor weight PRs. |

---

## Quickstart: Clone & Run the Full "Parameter Pull Request" Loop

```bash
git clone https://github.com/pfolex/verifiable-ai.git
cd verifiable-ai
for repo in verisci rlvr-forge branchbox mlx-tree steerlens everbench; do
  git clone "https://github.com/pfolex/${repo}.git"
done

python3 run_fishtest_loop.py
```

## Versioned evaluation results (EVAL-001)

Evaluation artifacts use `schemas/evaluation-result-v1.schema.json`. Validate
schema fields and semantic invariants before accepting or saving an artifact:

```python
from evaluation_result import validate_evaluation_result
validate_evaluation_result(result)
```

For existing JSON artifacts and contract tests:

```bash
python -m pip install -r requirements-test.txt
python evaluation_result.py examples/evaluation-result-v1.json
python -m unittest discover -s tests -v
```

Validation rejects duplicate sample IDs, mismatched sample counts, and incorrect
mean scores. Empty runs must report a zero mean. Mean comparisons allow relative
rounding error of `1e-9` and absolute error of `1e-12`. CI requires JSON Schema
Draft 2020-12 validation, including date-time formats; missing dependencies fail
the contract suite instead of skipping it.

## Benchmark runner (EVAL-002)

Run actual provider submissions through VeriSci and save a validated v1 artifact:

```bash
python -m pip install -r requirements-test.txt
python run_benchmark.py --provider reference --output benchmark-results/reference.json
python evaluation_result.py benchmark-results/reference.json
```

This requires the `verisci` and `rlvr-forge` checkouts from the quickstart. The
default evaluates two seeded tasks in each of the four domains. The reference
provider is an oracle baseline with access to reference solutions, not a trained
model measurement. The replay provider evaluates recorded model responses without
API calls. See [the EVAL-002 specification](docs/EVAL-002.md) for the recording
format, provenance, error behavior, and extension interface. CI tests both adapters,
runs the reference benchmark, and uploads its validated JSON result.

## Live model evaluation (EVAL-003)

Evaluate an installed Ollama model using the same seeded tasks and result contract:

```bash
python run_benchmark.py --provider ollama --model aix3:latest \
  --domains logic --samples-per-domain 1 --difficulty 1 \
  --max-output-tokens 2048 --timeout 240 --output benchmark-results/live.json
python evaluation_result.py benchmark-results/live.json
```

Start Ollama and install the chosen model before running. The adapter sends only
public task fields and verifier inputs; reference solutions stay local to the
scorer. It records the installed model digest, generation settings, actual token
usage when available, and measured latency. CI uses a local HTTP fixture for all
four domains and failure paths, so it needs neither model downloads nor credentials.
See [EVAL-003](docs/EVAL-003.md) for configuration and limitations.

## Reproducible datasets (EVAL-004)

Export a public task manifest and rerun its exact tasks:

```bash
python run_benchmark.py --manifest-only --export-manifest tasks.json
python run_benchmark.py --manifest tasks.json --output benchmark-results/reference.json
```

Manifests exclude reference solutions and bind task contents, generation settings
and source revisions with a verified hash. See [EVAL-004](docs/EVAL-004.md) and
the [remaining evaluation roadmap](docs/EVALUATION-ROADMAP.md).

## Compare models (EVAL-005)

Run baseline and candidate against one saved manifest, then compare them:

```bash
python compare_benchmarks.py baseline.json candidate.json --manifest tasks.json \
  --output comparison.json --markdown comparison.md
```

Reports include overall and per-domain scores, errors, usage, latency and costs.
Mismatched provenance is rejected, and unknown usage remains explicit. See
[EVAL-005](docs/EVAL-005.md).

## Statistical gates (EVAL-006)

```bash
python gate_benchmarks.py baseline.json candidate.json --manifest tasks.json \
  --policy examples/regression-policy-v1.json --output gate.json
```

Gates require overall improvement and non-regression in every domain, using
simultaneous paired confidence bounds. Exit codes are 0 PASS, 1 FAIL, 2
INCONCLUSIVE and 3 invalid data/output failure. The example policy deliberately
does not assert independent sampling; small smoke runs must remain inconclusive.
See [EVAL-006](docs/EVAL-006.md) before using a gate for a release decision.

## Hosted providers and costs (EVAL-007)

With `OPENAI_API_KEY` or `GROQ_API_KEY` set in the environment:

```bash
python run_benchmark.py --manifest tasks.json --provider openai --model "$MODEL_ID" \
  --pricing rates.json --output benchmark-results/hosted.json
```

OpenAI Responses and Groq Chat adapters send public inputs, retain reported usage,
and estimate token costs from an explicit versioned price card. The example cards
contain synthetic rates, not current prices. Cache-write usage uses result schema
1.1.0; existing 1.0.0 artifacts remain supported. See [EVAL-007](docs/EVAL-007.md)
for authentication, pricing, error handling and verification limits.
