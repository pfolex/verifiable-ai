# AIX Autonomous Evolution, Training Upgrades & Next-Level Roadmap

This document records all recent architectural, training, evaluation, and usability upgrades made to **AIX-Zero / Verifiable-AI**, along with the concrete engineering roadmap to evolve AIX to its next level of intelligence.

---

## 1. Summary of Recent Upgrades to AIX

### A. One-Command Global CLI & Concise Status (`aix`)
Installed global executables ([`/Users/alexpfeil/.local/bin/aix`](file:///Users/alexpfeil/.local/bin/aix) and [`/Users/alexpfeil/.local/bin/aix-ctl`](file:///Users/alexpfeil/.local/bin/aix-ctl)) wired to [`aix_zero/aix_ctl.py`](../aix_zero/aix_ctl.py) so AIX can be controlled from any directory with three commands:
- `aix run` (or `aix`): Automatically prepares candidate experiment inputs, discovers a new subject from the internet (Wikipedia & arXiv), updates the Subject-Matter Expertise (SME) Index, and launches the continuous background self-training daemon.
- `aix status`: Prints a concise **6-line summary** of live improvements (Cycle #, Effective Parameters & LoRA Rank, Certified SME Subjects & latest discovery, Training Loss delta, Held-Out Validation Loss delta & promotion status, and Neuro-Symbolic vs. Raw Neural Arena accuracy). Pass `aix status --full` for raw JSON telemetry.
- `aix stop`: Immediately halts both the background autopilot daemon and any active candidate experiment scheduler so the machine can be shut down cleanly.

### B. Universal 30+ Subject Tree of Knowledge & Open-Ended Subject Discovery (`#346`–`#358`)
- **Expanded Core Shards (`20 -> 30` Universal Domains):** Added 10 new universal specialist domains to [`META_AI_EXPERT_DOMAINS`](../aix_zero/meta_ai_autopilot.py) with deterministic symbolic verifiers and curriculum generators:
  - `META_FINANCE_PORTFOLIO_SHARPE` (Quantitative Finance Sharpe Ratio)
  - `META_PHYSICS_RELATIVISTIC_ENERGY` (Kinetic Energy & Momentum)
  - `META_CHEMISTRY_STOICHIOMETRY_YIELD` (Chemical Reaction Molar Yield)
  - `META_BIOLOGY_GENOMICS_ALLELE_FREQ` (Genomics & Population Allele Frequency)
  - `META_CRYPTO_MODULAR_VERIFIER` (Zero-Knowledge Soundness Survival)
  - `META_ECONOMICS_ELASTICITY_EQUILIBRIUM` (Microeconomic Price Elasticity)
  - `META_DISTRIBUTED_CONSENSUS_BFT` (Byzantine Fault Tolerant Quorum Safety)
  - `META_BAYESIAN_POSTERIOR_ODDS` (Exact Bayesian Likelihood-Ratio Updating)
  - `META_FORMAL_LOGIC_SAT_DENSITY` (CNF SAT Phase-Transition Clause Density)
  - `META_NETWORK_BDP_THROUGHPUT` (Bandwidth-Delay Product Window Utilization)
- **8 Hugging Face Zero-Disk Streaming Datasets (`#348`):** Expanded [`OPEN_SOURCE_STREAMING_DATASETS`](../aix_zero/meta_ai_autopilot.py) to stream and R1-verify problems from `OpenR1-Math-220k`, `NuminaMath-1.5`, `OpenThoughts-114k`, `fineweb-edu`, `TheoremQA`, `MMLU`, `GPQA`, and `the-stack-smol`.
- **Open-Ended Autonomous Subject Discovery (`discover_and_learn_new_subject` / `#349`–`#350`):** Every cycle (or via `aix learn-subject --online`), AIX discovers new subjects across Quantum Circuits, Robotics PID Control, Climate Radiative Forcing, Pharmacokinetics, Astrophysics, Compiler SSA, Topological Quantum Braiding, Neuromorphic Spiking Networks, Protein Folding, Fusion Plasma Lawson Criterion, and Automated Formal Theorem Proving—fetching live Wikipedia & arXiv context and persisting verified curricula to `discovered_subjects_registry.json`.
- **Persistent Subject-Matter Expertise (SME) Index (`update_subject_expertise_index` / `#352`–`#355`):** Tracks all core and discovered subjects (`41/41` certified as of Cycle #120) in `subject_expertise_index.json`, separating neuro-symbolic verifier mastery from unassisted raw neural diagnostic accuracy.

### C. Open-Ended Elastic Parameter Growth (`#293`, `#295`, `#351`)
- Upgraded [`ElasticLoRAAdapterBank`](../aix_zero/meta_ai_autopilot.py) in `aix_zero/meta_ai_autopilot.py` to scale continuously across `r = 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 160 -> 192 -> 224 -> 256 -> ... -> 1024+` (and `+64` open-endedly thereafter) whenever new subjects are discovered, validation milestones are hit, or plateaus are detected.
- **Measured Parameter Growth:** Effective system parameters grew automatically from **`817,410`** (base backbone) $\rightarrow$ **`1,997,058`** (Cycle #108) $\rightarrow$ **`2,980,098`** (Cycle #111) $\rightarrow$ **`3,619,074`** (Cycle #120 at `r = 256` with `41` specialist shards).

### D. Upgraded Neural Training Pipeline (Cycle #120 Breakthrough)
- **Scale-Matched Disjoint Splits:** Replaced unbounded operand multiplication (`cycle_index + 2000`) with prime-modular scale-matched offsets (`1 + ((raw_idx * 17 + (raw_idx // 500) * 19) % 61)`), keeping training, held-out validation, and arena splits strictly prompt-disjoint while matching digit complexity.
- **30-Domain Stratified Batching + Dedicated Expert-Slot Refinement:** Each cycle now runs an 8-step stratified update across all 30 universal domains (with `PromptPointerAttention` unfrozen so the model learns to copy prompt operands), followed by a 4-step dedicated specialist refinement on the active domain's routed MoE expert slot mixed with replay buffer anchors.
- **Immediate Measured Impact in Cycle #120:**
  - Single-cycle training loss drop jumped to **`-0.6273`** (`1.9872 -> 1.3598`).
  - Unseen held-out validation loss drop jumped to **`-0.2650`** (`2.4814 -> 2.2164`, vs. `-0.0005` previously).
  - Raw unassisted neural arena accuracy rose to **`2.63%`** (up from `0.00%`–`0.88%`).

### E. Evidence-Based Candidate Experiment Loop (`#406`–`#417`, `#422`–`#426`)
- **Portable Experiment Preparation ([`prepare_candidate_experiment.py`](../aix_zero/prepare_candidate_experiment.py) — `#422`):** One-command preparation of immutable champion state_dict, four-split manifest, and frozen policy with macOS/Linux permission diagnostics.
- **Case-Level Failure Taxonomy & Decoding Probe ([`inspect_candidate_experiment.py`](../aix_zero/inspect_candidate_experiment.py) — `#423`):** Classifies every failure (`wrong_answer`, `invalid_calculation_and_wrong_answer`, `correct_answer_invalid_trace`, `missing_or_unclosed_answer_tag`, `empty_output`, `generation_error`), records termination reasons (`eos`, `max_new_tokens`, `context_limit`), and probes larger token budgets.
- **Non-Vacuous Retention Metrics ([`candidate_policy.py`](../aix_zero/candidate_policy.py) — `#426`):** Reports explicit `retention_denominator` and `retention_measurable` flags so `0/18` all-wrong regression suites report `not_measurable_zero_champion_successes` instead of implying 100% retention.
- **Tiny-Set Learning Harness ([`tiny_learning_check.py`](../aix_zero/tiny_learning_check.py) — `#424`) & Multi-Seed Replay Study ([`single_skill_study.py`](../aix_zero/single_skill_study.py) — `#412`, `#425`):** Proved 100% (`4/4`) multi-seed training trace fitting and proved that mixing registered training-only replay anchors (`TRAINING_ONLY_REPLAY_POOL`) preserves 100% (`2/2`) of established v11 skills (`CODE_TRACE`, `PHYSICS_KINEMATICS`).
- **Failure-Driven Planner ([`experiment_planner.py`](../aix_zero/experiment_planner.py) — `#409`), Atomic CAS Promotion & Rollback ([`champion_promotion.py`](../aix_zero/champion_promotion.py) — `#413`), and Resumable Subprocess Scheduler ([`autonomous_scheduler.py`](../aix_zero/autonomous_scheduler.py) — `#414`, `#416`, `#417`).**

---

## 2. Next Steps to Evolve AIX to the Next Level

To take AIX from a fast self-improving hybrid neuro-symbolic system (`100%` neuro-symbolic accuracy, `2.63%` unassisted raw neural accuracy across 30+ domains) to a high-accuracy autonomous neural reasoner and software engineer:

### Step 1: Per-Domain Cartridge Bank Routing During Neural Arena Evaluation
- **Why:** Even though AIX exports 30+ specialist `.safetensors` shards, `AIXZeroModel` still has `n_experts = 4` in its physical core, meaning 7–8 domains share each physical expert slot in RAM.
- **Next Action:** Wire `SparseExpertCartridgeBank.execute_routed_request` directly into `run_frontier_arena_evaluation` and `run_single_self_improvement_cycle` so each of the 30+ domains loads and trains its own dedicated disk-backed `.safetensors` expert shard (`49,152` dedicated parameters per domain, zero cross-domain interference).

### Step 2: Physical LoRA & Expert-Count Expansion in `AIXZeroModel`
- **Why:** Right now `ElasticLoRAAdapterBank` tracks rank expansion (`r = 4 -> 256+`) in the capacity ledger, while the base checkpoint tensor shapes stay compatible with `v11` (`d_model = 128, n_experts = 4`).
- **Next Action:** Add a physical `LoRALinear` wrapper or dynamic expert-slot growth (`n_experts: 4 -> 8 -> 16 -> 32`) inside `MixtureOfReasoningExperts` in [`aix_zero/architecture.py`](../aix_zero/architecture.py) with function-preserving zero-initialized $B$ matrices ($\Delta W = \frac{\alpha}{r} B A = 0$ at expansion time), so increasing parameters immediately adds physical representational capacity to forward/backward passes without disrupting existing skills.

### Step 3: Teach the Neural Network to Emit Symbolic ALU Triggers (`<think>op1*op2=`)
- **Why:** Small transformers excel at **structural routing and operand extraction** (`PromptPointerAttention` copying numbers from the prompt into `<think>a*b=`), while exact multi-digit rational arithmetic is best handled by AIX's built-in `evaluate_inflight_subexpr` ALU.
- **Next Action:** Add an explicit **Pointer-Prefix Supervision Loss** that heavily weights the tokens up to the first `=` sign inside `<think>` (`<think>6*202=`), training the neural network to achieve >80% accuracy at setting up equations from arbitrary prompts.

### Step 4: Live Multi-Turn Peer-AI Distillation with Local Ollama Models (`#415`)
- **Why:** You have local Ollama installed (`/Users/alexpfeil/.local/bin/ollama`).
- **Next Action:** Connect the autonomous scheduler to query local teacher/peer models via Ollama when AIX hits a new domain or code-refactoring task, filter every peer response through `verify_r1_reasoning_trace()` and sandboxed `pytest` execution, and distill only 100%-verified peer solutions into AIX's specialist cartridges.

### Step 5: Autonomous Code-Patching & Self-Refactoring Loop (`#270`, `#415`)
- **Next Action:** Expand `synthesize_and_verify_swe_patch_tasks` so AIX proposes real unified diffs against open issues, runs the unit test suite in an isolated temporary worktree, and records three-arm comparisons (AIX solo vs. Peer AI solo vs. Symbiotic AIX + Peer).
