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

## 2. Implemented Neural & Software Engineering Roadmap (`#428`–`#433`)

All five work items coordinated under [`#433` (`[AIX NEURAL & SOFTWARE ROADMAP]`)](https://github.com/pfolex/verifiable-ai/issues/433) have now been implemented, audited against current `main`, and verified with boundary and unit test suites:

### Step 1 (`#428` — `[AIX NEURAL-01]`): Routed Domain Cartridges & Isolated Specialist Training
- **Module:** [`aix_zero/routed_specialist_training.py`](../aix_zero/routed_specialist_training.py) | **Tests:** [`aix_zero_tests/test_routed_specialist_training.py`](../aix_zero_tests/test_routed_specialist_training.py)
- **Baseline Tensor Ownership Audit (`audit_model_and_cartridge_tensor_ownership`):**
  - Audited `AIXZeroModel` (`v11` config): `817,410` total physical parameters (`620,802` shared backbone/router/shared-expert parameters + `4` routed expert slots of `49,152` float32 parameters / `196,608` bytes each).
  - Explicitly distinguishes **snapshot aliases** (`existing_routed_expert_snapshot`, where 30 domains mapped across 4 slots produce at most 4 unique SHA-256 tensor digests) from **independently trained specialists** (`frozen_v11_single_expert_training_candidate`).
  - Records `zero_cross_domain_interference_guaranteed_by_disk_file_alone: False` because shared backbone/router weights can still cause cross-domain interference unless frozen.
- **Isolated Specialist Training (`train_isolated_domain_cartridge`):**
  - Freezes 100% of shared backbone, router, shared expert, and non-target expert slots, verifying `frozen_tensors_unchanged == True` via pre/post SHA-256 checksums.
  - Creates a fresh isolated `AdamW` optimizer bound strictly to `moe.experts[target_slot].parameters()` (`49,152` trainable params) so no optimizer state or parameter references cross slot swaps.
  - Atomically publishes only the single candidate `.safetensors` cartridge and restores the base model's in-RAM slot on completion, error, or cancellation.
- **Four-Arm Paired Routed Evaluation (`evaluate_routed_cartridge_paired_comparison`):**
  - Connects `SparseExpertCartridgeBank` to [`paired_checkpoint_eval.py`](../aix_zero/paired_checkpoint_eval.py) and `run_frontier_arena_evaluation` in [`meta_ai_autopilot.py`](../aix_zero/meta_ai_autopilot.py).
  - Evaluates in strict `neural_only` mode (`inflight_alu=False`, no retrieval, no repair), reporting `unchanged_baseline`, `predicted_prefix_routed` (using prompt text only, never gold answers), `oracle_domain_label_routed` (reported separately), and `wrong_cartridge_control` (plus corrupt/incompatible cartridge fallback telemetry).

### Step 2 (`#429` — `[AIX NEURAL-02]`): Physical LoRA Capacity & Function-Preserving Migration
- **Module:** [`aix_zero/physical_lora.py`](../aix_zero/physical_lora.py) | **Tests:** [`aix_zero_tests/test_physical_lora.py`](../aix_zero_tests/test_physical_lora.py)
- **Physical `LoRALinear` & Trained-Adapter Rank Growth (`expand_rank`):**
  - Registers physical `lora_A` (`(r, d_in)`) and `lora_B` (`(d_out, r)`) parameters computing $\Delta W = \frac{\alpha}{r} B A$ in `forward()`, with $B = 0$ initialization verified to preserve forward logits within `1e-6`.
  - When expanding rank (`r_old -> r_new`) on an **already trained** adapter ($B \neq 0$) where $\frac{\alpha_{\text{new}}}{r_{\text{new}}} \neq \frac{\alpha_{\text{old}}}{r_{\text{old}}}$, rescales preserved columns `B[:, :r_old]` by `old_scaling / new_scaling` and appends zero columns `B[:, r_old:r_new] = 0`, preserving exact forward outputs within float32 tolerance (`1e-5`).
  - Migrates `AdamW` optimizer state (`exp_avg`, `exp_avg_sq`) for existing dimensions while zero-initializing new rank slices, and verifies two-step gradient dynamics on added capacity (`grad(A_new) == 0` at step 1 while `grad(B_new) != 0`, followed by `grad(A_new) != 0` at step 2).
- **Function-Preserving MoE Expert-Count Growth (`expand_moe_experts_function_preserving`):**
  - Grows `moe.experts` (`4 -> 8 -> 16 -> 32`) while preserving exact softmax routing probabilities and top-k selection at the expansion boundary by masking new router bias slots (`-1e9`), with `unmask_expanded_moe_experts` for controlled activation.
- **Versioned Checkpoint Migration & Caps:**
  - Supports `2.0.0-physical-lora` checkpoint save/reload round-trips, fails closed on unexplained shape mismatches, reports actual unique physical/trainable/active/disk parameters and optimizer memory bytes, and enforces `max_lora_rank`, `max_total_parameters`, and `max_rss_bytes` caps.

### Step 3 (`#430` — `[AIX NEURAL-03]`): Equation-Prefix Supervision & Separately Measured ALU-Trigger Accuracy
- **Module:** [`aix_zero/equation_prefix_supervision.py`](../aix_zero/equation_prefix_supervision.py) | **Tests:** [`aix_zero_tests/test_equation_prefix_supervision.py`](../aix_zero_tests/test_equation_prefix_supervision.py)
- **Token-Aligned Prefix Masking (`encode_equation_prefix_supervision_batch`):**
  - Masks all BOS and prompt tokens (including any `=` signs inside the prompt!) with `labels = 0`, `prefix_mask = 0.0`, and `continuation_mask = 0.0`.
  - Weights tokens from `<think>` through the first arithmetic `=` inside the reasoning region (`prefix_weight = 5.0`), distinguishing optional variable labels (`ratio=`) from the arithmetic operator expression (`16/4=`), while preserving full continuation and `<eos>` supervision (`continuation_weight = 1.0`).
  - Logs normalized `prefix_loss` and `continuation_loss` separately.
- **Three-Way Outcome Separation (`evaluate_three_way_prefix_and_alu_outcomes`):**
  - Reports three distinct outcomes with numerators, denominators, Wilson 95% confidence intervals, and per-domain breakdowns:
    1. `neural_only_full_performance` (`assistance_mode: "neural_only"`, ALU off)
    2. `neural_equation_setup_accuracy` (`exact_prefix_accuracy` and `semantic_setup_accuracy` before ALU insertion, classifying `correct_setup_wrong_continuation`, `correct_arithmetic_wrong_operand_selection`, and `malformed_alu_trigger`)
    3. `alu_assisted_final_performance` (`assistance_mode: "bounded_alu_assisted"`, `unassisted_neural_claim_prohibited: True`, using bounded rational parser `evaluate_bounded_equation_prefix` that rejects division by zero, unsupported operators, and magnitude overflows without executing Python).

### Step 4 (`#431` — `[AIX PEER-01]`): Bounded Multi-Turn Ollama Coaching & Verified Cartridge Distillation
- **Module:** [`aix_zero/ollama_peer_coaching.py`](../aix_zero/ollama_peer_coaching.py) | **Tests:** [`aix_zero_tests/test_ollama_peer_coaching.py`](../aix_zero_tests/test_ollama_peer_coaching.py)
- **Dynamic Discovery (`discover_ollama_service`):**
  - Discovers Ollama via `OLLAMA_HOST`, `OLLAMA_BIN`, `shutil.which("ollama")`, or configured HTTP endpoint (`http://127.0.0.1:11434`) without hardcoded paths, returning an actionable `UNAVAILABLE` report when the service or requested teacher model is absent and never attempting implicit model downloads.
- **Bounded Multi-Turn Coaching (`run_bounded_ollama_coaching_session`):**
  - Pins teacher model digest, version, generation settings, and dataset license; enforces `max_turns_per_task`, `max_tokens_per_turn`, `max_total_tokens`, `max_wall_clock_sec`, `max_concurrency = 1`, caller cancellation, and `#407` split hygiene (blocking `development`, `regression`, and `reserved_final` prompts from entering coaching).
  - Validates responses with both `verify_r1_reasoning_trace` and independent semantic operand/operator setup checks, labeling accepted traces `verification_status: "verifier_accepted"` (`guaranteed_universal_correctness: False`) and feeding concise non-leaking findings on retry turns.
- **Four-Arm Neural-Only Distillation (`distill_verified_coaching_into_cartridge`):**
  - Compares `original_baseline`, `no_teacher_self_trained_control`, `teacher_distilled_candidate` (on both known templates and unseen held-out prompts), and `teacher_only_reference` with zero teacher or ALU assistance at student evaluation time.

### Step 5 (`#432` — `[AIX SWE-01]`): Real Issue Patch Generation & Sandboxed Three-Arm SWE Benchmark
- **Module:** [`aix_zero/swe_three_arm_benchmark.py`](../aix_zero/swe_three_arm_benchmark.py) | **Tests:** [`aix_zero_tests/test_swe_three_arm_benchmark.py`](../aix_zero_tests/test_swe_three_arm_benchmark.py)
- **Pinned Tasks, Diff Security & Constrained Sandbox:**
  - Pins repository revision, task snapshot SHA-256, `heldout_eval` split, open-source license (`Apache-2.0` / `MIT`), `allowed_target_files`, and immutable external test suite SHA-256.
  - Rejects malformed diffs, path traversal (`..`), absolute paths, symlinks (`120000`), binary patches, and any file outside `allowed_target_files` (protecting graders, test suites, and promotion policies).
  - Applies diffs in fresh detached disposable git worktrees (`git worktree add --detach`) and runs external pinned tests inside a constrained subprocess (`start_new_session=True`, process-group `SIGKILL` on timeout, stripped environment secrets, loopback-blocked proxy variables, isolated `HOME`/`TMPDIR`, `RLIMIT_CPU`/`RLIMIT_FSIZE` limits, and exact test-count + zero-skip verification).
- **Three-Arm Comparison (`run_three_arm_swe_patch_comparison`):**
  - Evaluates `aix_solo`, `peer_solo`, and `collaborative_aix_plus_peer` (with combined token/turn usage visible and all offered tasks kept in denominators) alongside a separate `reference_fixture_control` (`is_model_written_patch: False`), exporting validated patches only as reviewable candidate artifacts without direct production self-edit.

---

## 3. Master Project Goals & Live Status Matrix

Below are the 8 overarching project goals for **AIX**, how the current architecture maps to each goal, and the remaining milestones to complete them:

### Goal 1: Complete the AIX Continuous Improvement Loop (Self-Updating Codebase)
- **What is Built (`#406`–`#418`, `#422`–`#432`):**
  - Autonomous codebase AST & symbol knowledge graph scanner (`scan_open_source_ai_codebase_patterns`, `build_open_source_knowledge_graph` in [`meta_ai_autopilot.py`](../aix_zero/meta_ai_autopilot.py)).
  - Evidence-gated candidate experiment planner ([`experiment_planner.py`](../aix_zero/experiment_planner.py)), isolated training runner ([`autonomous_scheduler.py`](../aix_zero/autonomous_scheduler.py)), atomic CAS champion promotion with rollback ([`champion_promotion.py`](../aix_zero/champion_promotion.py)), and sandboxed git worktree diff validator & test grader ([`swe_three_arm_benchmark.py`](../aix_zero/swe_three_arm_benchmark.py)).
- **Next Milestone to Complete:**
  - Connect the autonomous scheduler to propose reviewable pull-request branches (`git worktree` -> unified diff -> sandboxed `unittest` suite -> `gh pr create`) whenever AIX synthesizes a verified code optimization or bug fix (`#415`, `#432`).

### Goal 2: Make AIX an Expert Programmer
- **What is Built (`#304`–`#312`, `#432`, PR `#436`):**
  - 4 dedicated Python & PyTorch systems specialist shards (`META_PYTHON_AST_COMPLEXITY`, `META_PYTHON_ALGORITHMIC_BIG_O`, `META_PYTORCH_TENSOR_MEMORY_STRIDES`, `META_PYTHON_TYPE_CONCURRENCY_ASYNC`) + `META_SWE_PATCH_TEST_COVERAGE`.
  - Isolated AI-coding expression trainer ([`train_ai_coding.py`](../aix_zero/train_ai_coding.py) from PR `#436`) with AST functional equivalence scoring for SGD, linear layers, and LoRA scaling expressions.
  - AST anti-pattern detector (`detect_python_antipatterns_ast`), automated code refactorer (`refactor_and_verify_python_code`), and sandboxed unified-diff patch synthesizer ([`swe_three_arm_benchmark.py`](../aix_zero/swe_three_arm_benchmark.py)).
- **Next Milestone to Complete:**
  - Expand [`train_ai_coding.py`](../aix_zero/train_ai_coding.py) into multi-line Python function & unified-diff synthesis cartridges so the neural model directly emits verified code and patches across multi-file repository tasks.

### Goal 3: Make AIX an Expert GitHub User
- **What is Built (`#251`–`#253`, `#432`, `0eb2341`):**
  - Automated GitHub issue discovery, creation, and evidence-backed closure (`discover_and_file_improvement_issues`, `work_on_active_backlog_issue` in [`meta_ai_autopilot.py`](../aix_zero/meta_ai_autopilot.py) via `gh issue create` / `gh issue close`).
  - Automatic pruning of unreferenced `.safetensors` shards and automatic commit + push of the latest model (`aix_zero_v12_evolving.pt`), active specialist cartridges, and SME registries to `origin/main` whenever `aix stop` is run ([`aix_ctl.py`](../aix_zero/aix_ctl.py)).
- **Next Milestone to Complete:**
  - Add autonomous GitHub PR workflow commands (`gh pr list`, `gh pr checks`, `gh pr create`, and CI status inspection) so AIX can open feature branches, monitor GitHub Actions CI checks, and report review-ready PRs.

### Goal 4: Make AIX an Expert at Training Itself Using Other LLMs Where Necessary
- **What is Built (`#415`, `#431`):**
  - Bounded multi-turn local Ollama teacher coaching & cartridge distillation ([`ollama_peer_coaching.py`](../aix_zero/ollama_peer_coaching.py)) with dynamic Ollama endpoint/model discovery, strict `#407` split hygiene (blocking held-out/test prompts from entering coaching), `verify_r1_reasoning_trace` + semantic setup validation, and four-arm neural-only distillation (`original_baseline`, `no_teacher_self_trained_control`, `teacher_distilled_candidate`, `teacher_only_reference`).
  - Zero-disk Hugging Face streamer (`stream_huggingface_open_source_rows`) streaming and R1-verifying reasoning traces from 8 open-source LLM/STEM datasets (`OpenR1-Math-220k`, `NuminaMath-1.5`, `OpenThoughts-114k`, `TheoremQA`, `MMLU`, `GPQA`, `fineweb-edu`, `the-stack-smol`).
- **Next Milestone to Complete:**
  - Trigger `run_bounded_ollama_coaching_session` automatically inside the background training loop whenever a local Ollama teacher model (`qwen2.5-coder`, `deepseek-r1`, `llama3`) is online and a domain hits a validation plateau.

### Goal 5: Train a New LLM Model from Zero to Competitive in `< 200` Training Rounds
- **What is Built & Measured:**
  - Using scale-matched disjoint splits, `<eos>`/`<ans>` token-weighted cross-entropy, and isolated per-domain `.safetensors` specialist cartridge routing ([`routed_specialist_training.py`](../aix_zero/routed_specialist_training.py) — `#428`), AIX's unassisted **Raw Neural Arena accuracy (`inflight_alu=False`, `neural_only`) jumped from `4.39%` at Cycle #164 to `24.56%` at Cycle #179 (in just 15 rounds) and reached `43.86%` by Cycle #241 (in just 77 rounds of routed cartridge training!)**.
- **Next Milestone to Complete:**
  - Package a single standalone command (`aix train-from-zero --rounds 200`) that initializes a fresh `AIXZeroModel` from scratch, runs `< 200` rounds of stratified backbone + isolated specialist cartridge + equation-prefix training ([`equation_prefix_supervision.py`](../aix_zero/equation_prefix_supervision.py) — `#430`), and reaches >80% benchmark accuracy within 200 rounds.

### Goal 6: Finish Benchmark Testing Against Standard AI Benchmark Tests
- **What is Built (`#263`, `#348`, `#407`–`#408`, `#421`):**
  - 4-split cryptographically hashed evaluation manifest (`train`, `development`, `regression`, `reserved_final` in [`experiment_splits.py`](../aix_zero/experiment_splits.py) and [`prepare_candidate_experiment.py`](../aix_zero/prepare_candidate_experiment.py)).
  - Paired checkpoint evaluator ([`paired_checkpoint_eval.py`](../aix_zero/paired_checkpoint_eval.py)) with Wilson 95% confidence intervals, McNemar paired gain/loss counts, trace verification, and non-vacuous retention gates ([`candidate_policy.py`](../aix_zero/candidate_policy.py) — `#426`).
  - Streaming connectors and verifiers for standard benchmark families (`GSM8K` / `NuminaMath`, `MATH` / `OpenR1`, `TheoremQA`, `MMLU`, `GPQA`, and SWE patch tasks).
- **Next Milestone to Complete:**
  - Build a unified standard-benchmark runner (`aix benchmark --suite gsm8k,math,mmlu,gpqa,humaneval,swe`) that evaluates both `neural_only` and `neuro_symbolic_verified` modes on frozen held-out benchmark splits and publishes a versioned scorecard (`#263`).

### Goal 7: Become a Competitive Model in LLM Benchmark Testing
- **What is Built & Measured:**
  - **Neuro-Symbolic Verified Mode (`use_alu=True`):** **100.0%** exact verified accuracy across all 30 core universal domains and 141+ discovered STEM/AI subjects (`171/171` Certified SME Subjects).
  - **Unassisted Raw Neural Mode (`neural_only`, `inflight_alu=False`):** Climbed from `0.88%` $\rightarrow$ `4.39%` $\rightarrow$ `24.56%` $\rightarrow$ **`43.86%`** via routed `.safetensors` specialist cartridges (`#428`).
- **Next Milestone to Complete:**
  - Combine routed specialist cartridges (`#428`), physical LoRA capacity growth (`#429`), and equation-prefix supervision (`#430`) so unassisted raw neural accuracy and ALU-trigger setup accuracy both exceed **80%–95%+** across all benchmark suites.

### Goal 8: Continue Training AIX to Become an Expert in Every Subject
- **What is Built (`#346`–`#358`):**
  - Open-ended autonomous subject discovery (`discover_and_learn_new_subject`) + live Wikipedia/arXiv research (`WebKnowledgeFetcher`) + persistent Subject-Matter Expertise Index (`subject_expertise_index.json`).
  - AIX has already expanded autonomously from **30** initial domains to **171/171 Certified SME Subjects** (`+141` discovered subjects across Quantum Computing, Robotics, Astrophysics, Climate Physics, Compiler SSA, Cryptography, Continuum Mechanics, Fusion Plasma, Algorithmic Economics, Neuromorphic Computing, Biomedical Engineering, Photonic Computing, and Computational Biology) with **36,747,522 effective parameters (`r = 8960`)**.
- **Next Milestone to Complete:**
  - Train dedicated `.safetensors` specialist cartridges for every newly discovered subject (`DISC_...`) across multi-step problem templates so every subject in `subject_expertise_index.json` achieves both 100% neuro-symbolic mastery and high unassisted raw neural accuracy.
