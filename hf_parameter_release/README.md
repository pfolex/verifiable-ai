---
library_name: peft
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
tags:
  - rlvr
  - grpo
  - lora
  - ties-merging
  - mechanistic-interpretability
  - activation-steering
  - formal-verification
  - verifiable-ai
license: apache-2.0
---

# DeepSeek-R1-Distill-Qwen-1.5B-Verifiable-Sci-LoRA

A **mathematically verified** Rank-16 LoRA & Contrastive Activation Addition (CAA) parameter update for [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), trained and statistically gated using the [Verifiable AI Suite](https://github.com/pfolex/verifiable-ai).

## How These Parameters Were Trained & Verified
1. **DARE-TIES Spectral Initialization:** Initialized from singular subspaces of the DARE-rescaled parameter delta between `Qwen2.5-Coder-1.5B-Instruct` and `DeepSeek-R1-Distill-Qwen-1.5B`.
2. **Decentralized Domain-Specialist RLVR (`rlvr-forge` + `verisci`):** Trained 4 independent rank-4 specialist adapters on ground-truth verified solutions vs. failed rollouts across `logic`, `silicon`, `systems`, and `matter`.
3. **Orthogonal TIES-Merging & Mechanistic Steering (`everbench` + `steerlens`):** Fused the 4 specialist adapters via TIES-Merging so zero domains suffer sequential catastrophic forgetting, paired with 1536-dim Contrastive Activation Addition steering vectors.
4. **Stockfish SPRT Verification (`everbench`):** Verified on 70 held-out HMAC-SHA256 procedural seeds per domain (`huggingface-release-epoch-2026-10-28`):
   - **Logic (`logic`):** W/L/D = 70/0/0 (`ACCEPT_H1`, Margin Gain = `+22.6804`)
   - **Silicon (`silicon`):** W/L/D = 70/0/0 (`ACCEPT_H1`, Margin Gain = `+17.3208`)
   - **Systems (`systems`):** W/L/D = 70/0/0 (`ACCEPT_H1`, Margin Gain = `+23.0268`)
   - **Matter (`matter`):** W/L/D = 70/0/0 (`ACCEPT_H1`, Margin Gain = `+18.8766`)
