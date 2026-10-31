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
the contract suite instead of skipping it. Provider adapters and benchmark runner
integration are follow-up work.
