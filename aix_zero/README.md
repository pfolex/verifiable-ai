# AIX-Zero (`v1 -> v11`) & *Knowledge Is Power* (Web-Connected)

**`AIX-Zero`** is a 100% from-scratch neuro-symbolic reasoning model initialized from $\mathcal{N}(0, 0.02^2)$ random Gaussian noise in pure PyTorch (`817,026` physical parameters / `1,775,236` effective virtual parameters with 3-step recurrent ACT looping) paired with **`Knowledge Is Power v11`**—a **Web-Connected** (`Wikipedia REST API` + `arXiv Open Research API` via `WebKnowledgeFetcher`) autonomous Multi-LLM Symbiotic Council (`aix-8b:latest` $\leftrightarrow$ `aix3:latest`) that researches new domains from live scientific literature, verifies multi-step derivations in a deterministic Python sandbox, distills them into `AIX-Zero`'s standalone `.safetensors` weights in RAM, and self-documents every learned skill in the **[Knowledge Is Power Codex](knowledge_codex/KNOWLEDGE_IS_POWER_CODEX.md)**.

---

## Key Highlights (`AIX-Zero v11-KIP`)

- **100% From-Scratch Architecture (`aix_zero/architecture.py`):**
  - `RMSNorm` + `RotaryEmbedding (RoPE)` + `GroupedQueryAttention (GQA)`
  - `PromptPointerAttention` ($\tau = 0.35$) for exact operand copying from the prompt
  - `RecurrentReasoningBlock` with 3-step Velocity-Aware Adaptive Computation Time (ACT) ponder gate
  - `MixtureOfReasoningExperts (MoRE)` with 1 Shared Expert + 4 Top-2 Routed Specialist Experts
  - `DifferentiableMemoryBank` (8 persistent working-memory slots)
  - `ValueVerificationHead` + $K$-Step In-Flight Symbolic ALU (`evaluate_inflight_subexpr`) supporting 2-step, 3-step, and 4-step chains
  - Dynamic lifelong vocabulary expansion (`resize_token_embeddings`, `134` hybrid word + atomic-digit tokens)
- **`105/105 (100.0%)` Held-Out Benchmark Accuracy Across All 21 Domains:**
  - `ARITH_CHAIN`, `ALGEBRA_SOLVE`, `MODULAR_NT`, `LOGIC_DEDUCT`, `WORD_PROBLEM`, `CALCULUS_DERIV`, `SYSTEM_2X2`, `CODE_TRACE`, `QUADRATIC_EVAL`, `PHYSICS_KINEMATICS`, `GEOMETRY_AREA`, `NUMBER_SEQ`, `MULTI_STEP_3` (3-step chains), `ELECTRICAL_POWER`, `FINANCE_INTEREST`, `THERMO_FIRST_LAW`, `COMBINATORICS_GRID`, `PROBABILITY_EV` (3-step decision chains), `ELECTRO_THERMAL` (cross-domain Joule heating + thermodynamics), `KINETIC_WORK` (cross-domain mechanical work-energy + friction), and `MULTI_STEP_4` (4-step compositional chains).
- **Self-Documented *Knowledge Is Power* Codex (`aix_zero/knowledge_codex/`):**
  - [Master Encyclopedia (`KNOWLEDGE_IS_POWER_CODEX.md`)](knowledge_codex/KNOWLEDGE_IS_POWER_CODEX.md)
  - [Live Internet Research Log (`WEB_RESEARCH_LOG.md`)](knowledge_codex/WEB_RESEARCH_LOG.md)
  - [Symbiotic Multi-AI Dialogues (`SYMBIOTIC_DIALOGUES.md`)](knowledge_codex/SYMBIOTIC_DIALOGUES.md)
  - [Machine-Readable Knowledge Graph (`knowledge_index.json`)](knowledge_codex/knowledge_index.json)
  - [21 Dedicated Domain Monographs (`entries/*.md`)](knowledge_codex/entries/)

---

## Quickstart Commands

```bash
# 1. Launch the interactive AIX-Zero REPL & Live Teacher (with CodexRetriever + `!web <topic>` lookup):
PYTHONPATH=. python aix_zero/interactive_repl.py
PYTHONPATH=. python aix_zero/interactive_repl.py --demo

# 2. Run the "Knowledge Is Power" v11 Web-Connected Multi-LLM Discovery & Self-Documentation Engine:
PYTHONPATH=. python aix_zero/knowledge_is_power.py --cycles 8
PYTHONPATH=. python aix_zero/knowledge_is_power.py --topic "Quantum harmonic oscillator"

# 3. Run the Time-Budgeted Continuous Self-Training Daemon (e.g., 10 minutes):
PYTHONPATH=. python aix_zero/continuous_time_trainer.py --duration-minutes 10

# 4. Run AIX-Lang Tier-0 Formal Verification with `solve_with aix_zero:v8`:
PYTHONPATH=. python aix_lang.py aix_zero_v8_demo.aix
```
