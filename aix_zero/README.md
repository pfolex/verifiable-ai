# AIX-Zero (`v1 -> v10`) & *Knowledge Is Power*

**`AIX-Zero`** is a 100% from-scratch neuro-symbolic reasoning model initialized from $\mathcal{N}(0, 0.02^2)$ random Gaussian noise in pure PyTorch (`816,642` physical parameters / `1,774,852` effective virtual parameters with 3-step recurrent ACT looping) paired with **`Knowledge Is Power`**—an autonomous Multi-LLM Symbiotic Council (`aix-8b:latest` $\leftrightarrow$ `aix3:latest`) that researches new domains, verifies multi-step derivations in a deterministic Python sandbox, distills them into `AIX-Zero`'s standalone `.safetensors` weights in RAM, and self-documents every learned skill in the **[Knowledge Is Power Codex](knowledge_codex/KNOWLEDGE_IS_POWER_CODEX.md)**.

---

## Key Highlights (`AIX-Zero v10-KIP`)

- **100% From-Scratch Architecture (`aix_zero/architecture.py`):**
  - `RMSNorm` + `RotaryEmbedding (RoPE)` + `GroupedQueryAttention (GQA)`
  - `PromptPointerAttention` ($\tau = 0.35$) for exact operand copying from the prompt
  - `RecurrentReasoningBlock` with 3-step Velocity-Aware Adaptive Computation Time (ACT) ponder gate
  - `MixtureOfReasoningExperts (MoRE)` with 1 Shared Expert + 4 Top-2 Routed Specialist Experts
  - `DifferentiableMemoryBank` (8 persistent working-memory slots)
  - `ValueVerificationHead` + $K$-Step In-Flight Symbolic ALU (`evaluate_inflight_subexpr`)
  - Dynamic lifelong vocabulary expansion (`resize_token_embeddings`, `131` hybrid word + atomic-digit tokens)
- **`90/90 (100.0%)` Held-Out Benchmark Accuracy Across All 18 Domains:**
  - `ARITH_CHAIN`, `ALGEBRA_SOLVE`, `MODULAR_NT`, `LOGIC_DEDUCT`, `WORD_PROBLEM`, `CALCULUS_DERIV`, `SYSTEM_2X2`, `CODE_TRACE`, `QUADRATIC_EVAL`, `PHYSICS_KINEMATICS`, `GEOMETRY_AREA`, `NUMBER_SEQ`, `MULTI_STEP_3` (3-step chains), `ELECTRICAL_POWER`, `FINANCE_INTEREST`, `THERMO_FIRST_LAW`, `COMBINATORICS_GRID`, and `PROBABILITY_EV` (3-step decision chains).
- **Self-Documented *Knowledge Is Power* Codex (`aix_zero/knowledge_codex/`):**
  - [Master Encyclopedia (`KNOWLEDGE_IS_POWER_CODEX.md`)](knowledge_codex/KNOWLEDGE_IS_POWER_CODEX.md)
  - [Symbiotic Multi-AI Dialogues (`SYMBIOTIC_DIALOGUES.md`)](knowledge_codex/SYMBIOTIC_DIALOGUES.md)
  - [Machine-Readable Knowledge Graph (`knowledge_index.json`)](knowledge_codex/knowledge_index.json)
  - [18 Dedicated Domain Monographs (`entries/*.md`)](knowledge_codex/entries/)

---

## Quickstart Commands

```bash
# 1. Launch the interactive AIX-Zero REPL & Live Teacher (with CodexRetriever lookup):
PYTHONPATH=. python aix_zero/interactive_repl.py
PYTHONPATH=. python aix_zero/interactive_repl.py --demo

# 2. Run the "Knowledge Is Power" v10 Multi-LLM Discovery, Consolidation & Self-Documentation Engine:
PYTHONPATH=. python aix_zero/knowledge_is_power.py --cycles 8

# 3. Run the Time-Budgeted Continuous Self-Training Daemon (e.g., 10 minutes):
PYTHONPATH=. python aix_zero/continuous_time_trainer.py --duration-minutes 10

# 4. Run AIX-Lang Tier-0 Formal Verification with `solve_with aix_zero:v8`:
PYTHONPATH=. python aix_lang.py aix_zero_v8_demo.aix
```
