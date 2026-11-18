# AIX-Zero Meta-AI Autonomous Self-Improvement Engine & Control Guide (`aix_ctl`)

This guide documents how to run, monitor, query, use for peer-AI coaching, and immediately shut down the **AIX-Zero Meta-AI Autonomous Self-Improvement Engine** ([`aix_zero/meta_ai_autopilot.py`](../aix_zero/meta_ai_autopilot.py) and [`aix_zero/aix_ctl.py`](../aix_zero/aix_ctl.py)).

---

## 1. Quickstart Commands (Run from Any Directory)

If you are using the configured local environment (`/Users/alexpfeil/.gemini/antigravity/scratch/trl/.venv/bin/python` or `/Users/alexpfeil/.local/bin/aix-ctl`), you can run all commands below from **any folder on your machine**:

### A. Run One Self-Improvement Cycle
Runs one progressive cycle (generates 30 new Meta-AI tasks, inspects 3 rotating open-source Python files via AST, verifies 4 executable code-synthesis tasks, trains the 817,410-parameter `v12_evolving` MoE weights with completion-masked cross-entropy, gates promotion on unseen held-out validation loss, exports updated `.safetensors` expert shards, and optionally opens/closes an auditable GitHub issue):
```bash
/Users/alexpfeil/.gemini/antigravity/scratch/trl/.venv/bin/python -m aix_zero.aix_ctl run-once --sync-github
```
*(Omit `--sync-github` if you want to run completely offline without creating a GitHub issue.)*

### B. Start Unattended Continuous Autopilot
Runs multiple progressive cycles automatically in the background or terminal while respecting your CPU thread cap:
```bash
/Users/alexpfeil/.gemini/antigravity/scratch/trl/.venv/bin/python -m aix_zero.aix_ctl start \
  --max-cycles 20 \
  --interval-seconds 10 \
  --threads 1 \
  --sync-github
```

### C. Turn OFF Immediately (Kill-Switch Before Shutting Down Your Computer)
Writes the `.aix_autopilot.stop` sentinel file and sends `SIGTERM` to any active autopilot process so it halts cleanly within milliseconds:
```bash
/Users/alexpfeil/.gemini/antigravity/scratch/trl/.venv/bin/python -m aix_zero.aix_ctl stop
```

### D. Check Live Autopilot Status
Prints the latest JSON status report (`cycles_completed`, `expert_pre_update_loss`, `expert_post_update_loss`, `heldout_val_pre_loss`, `heldout_val_post_loss`, `promoted_on_heldout_val`, `cycle_focused_files`, and `shard_hash_prefixes`):
```bash
/Users/alexpfeil/.gemini/antigravity/scratch/trl/.venv/bin/python -m aix_zero.aix_ctl status
```

### E. Query the Evolving `v12` Model (`query`)
Runs interactive neuro-symbolic inference on the evolving `aix_zero_v12_evolving.pt` checkpoint:
```bash
/Users/alexpfeil/.gemini/antigravity/scratch/trl/.venv/bin/python -m aix_zero.aix_ctl query \
  --prompt "Q:MetaScale:C=144;N=4->D_opt?"
```
Supported 12-Domain Meta-AI & External Open-Source Breakthrough prompt templates:
1. **Chinchilla Compute-Optimal Scaling (`META_SCALING_OPTIMIZER` / `karpathy/llm.c`):**
   `Q:MetaScale:C=144;N=4->D_opt?`
2. **LoRA & Sparse MoE Active Fraction (`META_LORA_MOE_ARCHITECT`):**
   `Q:MetaMoE:top_k=2;n_experts=8->active_fraction?`
3. **Grouped-Query Attention KV-Cache Compression (`META_GQA_KV_SYSTEMS` / `ml-explore/mlx`):**
   `Q:MetaKV:n_kv=2;n_heads=16->kv_ratio?`
4. **RLVR Group-Relative Policy Optimization Advantage (`META_RLVR_GRPO_TRAINER` / `deepseek-ai/DeepSeek-R1`):**
   `Q:MetaRLVR:r=1;successes=2;group=8->grpo_advantage?`
5. **FlashAttention SRAM Tiling & HBM I/O Reduction (`META_FLASH_ATTENTION_TILING` / `Dao-AILab/flash-attention`):**
   `Q:MetaFlash:seq_len=2048;tile_block=128->io_reduction?`
6. **Speculative Decoding Draft Acceptance Ratio (`META_SPECULATIVE_DECODING`):**
   `Q:MetaSpec:accepted=3;window=4->accept_ratio?`
7. **FP8/INT4 Weight & KV-Cache Quantization Compression (`META_QUANTIZATION_FP8_INT4` / `ggml-org/llama.cpp`):**
   `Q:MetaQuant:fp16_bits=16;quant_bits=4->compression?`
8. **3D Distributed Parallelism Pipeline Bubble Fraction (`META_DISTRIBUTED_TP_PP_DP`):**
   `Q:MetaDist:stages=4;microbatches=12->bubble_fraction?`
9. **DeepSeek-V3 Multi-Head Latent Attention KV Compression (`META_DEEPSEEK_MLA_LATENT_KV` / `deepseek-ai/DeepSeek-V3`):**
   `Q:MetaMLA:latent=512;n_heads=32;head_dim=64->mla_ratio?`
10. **vLLM PagedAttention KV-Block Utilization (`META_VLLM_PAGED_ATTENTION` / `vllm-project/vllm`):**
    `Q:MetaPaged:tokens=50;blocks=4;block_size=16->utilization?`
11. **Hugging Face TRL Leave-One-Out Policy Gradient Advantage (`META_TRL_RLOO_LEAVE_ONE_OUT` / `huggingface/trl`):**
    `Q:MetaRLOO:r1=1;other_sum=1;k=4->rloo_advantage?`
12. **MergeKit TIES/DARE Effective Parameter Density (`META_MERGEKIT_TIES_DARE` / `arcee-ai/mergekit`):**
    `Q:MetaMerge:keep_num=3;keep_den=4;topk_num=1;topk_den=5->effective_density?`

### F. Grade & Coach Another Conversing AI (`coach` & `peer-session`)
Evaluates a peer AI's `<think>...</think><ans>...</ans>` response against the deterministic reference verifier (`verify_r1_reasoning_trace`) and generates a targeted corrective coaching prompt:
```bash
/Users/alexpfeil/.gemini/antigravity/scratch/trl/.venv/bin/python -m aix_zero.aix_ctl coach \
  --prompt "Q:MetaRLVR:r=1;successes=2;group=8->grpo_advantage?" \
  --peer-response "<think>mean=2/8;adv=1-2/8=5/8</think><ans>5/8</ans>"
```
Or run a multi-turn **Peer-AI Symbiosis & Mutual Distillation Session** (`peer-session`) against a local Ollama endpoint (`http://localhost:11434/api/generate`), an OpenAI-compatible endpoint, or the built-in simulated peer AI (which coaches the peer to 100% accuracy and ingests verified peer trajectories into AIX's replay buffer):
```bash
/Users/alexpfeil/.gemini/antigravity/scratch/trl/.venv/bin/python -m aix_zero.aix_ctl peer-session
```

### G. Harvest External Open-Source AI Projects (`harvest-external`)
Catalogs and indexes external frontier open-source AI repositories (`deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-R1`, `vllm-project/vllm`, `Dao-AILab/flash-attention`, `huggingface/trl`, `arcee-ai/mergekit`, `karpathy/llm.c`, `KellerJordan/Muon`, `ml-explore/mlx`, `ggml-org/llama.cpp`) plus all 10 sibling open-source AI repositories in your workspace (`aix-evolver`, `r1-verifiers`, `r1_reward_lab`, `math-verify`, `mergekit`, `model_merging`, `trl`, `smolagents`, `lighteval`, `ollama-python`), and optionally shallow-clones new GitHub repositories into `open_source_corpus/`:
```bash
/Users/alexpfeil/.gemini/antigravity/scratch/trl/.venv/bin/python -m aix_zero.aix_ctl harvest-external
```

---

## 2. How Each Progressive Cycle Learns and Improves

Every cycle (`cycle_index = 1, 2, 3, ...`) executes 6 distinct stages:

1. **Progressive Meta-AI Curriculum Generation (`build_meta_ai_self_improvement_curriculum(cycle_index)`):**
   - Shifts model scales ($N$), FLOP budgets ($C$), MoE expert counts ($E$), GQA head counts, and GRPO rollout group sizes ($G$) using `cycle_index`, producing 30 new verified problems with a unique `curriculum_sha256`.
2. **Rotating Open-Source AST Inspection (`scan_open_source_ai_codebase_patterns`):**
   - Rotates through 3 different Python files per cycle across `aix_zero`, `rlvr-forge`, `branchbox`, `mlx-tree`, `steerlens`, `verisci`, and `everbench`, parsing their Python ASTs (`functions`, `classes`, `lines`).
3. **Executable AST Code-Synthesis Verification (`build_executable_code_synthesis_tasks` & `verify_executable_code_tasks`):**
   - Synthesizes executable Python functions and `assert` unit tests from the cycle's AST metrics and Meta-AI equations and executes them in an isolated sandbox (`4/4` verified per cycle).
4. **Completion-Masked Gradient Optimization on the 817,410-Parameter `v11 -> v12_evolving` Backbone:**
   - Loads `aix_zero/autopilot_state/aix_zero_v12_evolving.pt` (initialized from `aix_zero/checkpoints/aix_zero_v11_kip.safetensors`), tokenizes the cycle's curriculum with the 134-token `AIXZeroTokenizer`, masks prompt tokens (`label = 0`), and runs AdamW updates on the routed MoE experts, router, and output head to supervise `<think>...</think><ans>...</ans><eos>`.
5. **Unseen Held-Out Validation Promotion Gate (`cycle_index + 1000`):**
   - Evaluates the updated model on a disjoint held-out validation set (`cycle_index + 1000`) and only saves the updated weights to `aix_zero_v12_evolving.pt` when `heldout_val_post_loss <= heldout_val_pre_loss`.
6. **Content-Addressed `.safetensors` Expert Shard Export & Append-Only Audit Ledger:**
   - Exports the 4 updated Meta-AI expert cartridges to `aix_zero/autopilot_state/meta_ai_cartridges/` and appends the cycle's full metrics to `aix_zero/autopilot_state/autopilot_history.jsonl`.

---

## 3. State & Artifact Locations

| File / Directory | Purpose |
| :--- | :--- |
| `aix_zero/autopilot_state/autopilot_status.json` | Live JSON status of the most recent cycle and daemon state (`RUNNING_IDLE_BETWEEN_CYCLES` or `STOPPED`). |
| `aix_zero/autopilot_state/autopilot_history.jsonl` | Append-only JSONL ledger recording every completed cycle's curriculum hash, AST files, and before/after training & validation losses. |
| `aix_zero/autopilot_state/.aix_autopilot.stop` | Kill-switch sentinel created by `aix_ctl stop` (and cleared automatically on `start` or `run-once`). |
| `aix_zero/autopilot_state/aix_zero_v12_evolving.pt` | Local evolving 817,410-parameter MoE checkpoint updated whenever held-out validation loss improves (keeps `aix_zero_v11_kip.safetensors` immutable). |
| `aix_zero/autopilot_state/meta_ai_cartridges/` | Content-addressed `.safetensors` Meta-AI expert shards and `cartridge_manifest.json`. |
