# AIX Bring-Your-Own-Model (BYOM) Workflow Guide (`#437`–`#444`)

This guide documents the reproducible end-to-end workflow for inspecting a model, validating a user-supplied JSONL dataset, running bounded training/evaluation, and exporting a checksummed release bundle without mutating the base checkpoint.

## 1. Reference Environment & Supported Platform Matrix
- **Reference Tested Environment:** macOS (Apple Silicon / CPU), Python 3.11, `torch`, `transformers`, `peft`, `safetensors`.
- **Supported HF Architectures (Tested Matrix):**
  - `qwen2`: `TESTED_SUPPORTED` (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
  - `llama`, `mistral`: `COMPATIBLE_EXPERIMENTAL` (requires local safetensors + native chat template)
- **Unsupported for Training:** `.gguf` / Ollama blobs (inference-only; export to local HF safetensors first). `trust_remote_code` is never enabled.

## 2. Step-by-Step Commands

### Inspect Model Compatibility & Memory Estimates
```bash
aix inspect-model --path aix_zero/checkpoints/aix_zero_v11_kip.safetensors
aix inspect-model --path /path/to/local-hf-or-mergekit-dir --device cpu
```

### Run Bounded BYOM Experiment (Record/Reject-Only by Default)
```bash
aix byom \
  --model aix_zero/checkpoints/aix_zero_v11_kip.safetensors \
  --dataset /path/to/dataset.jsonl \
  --run-dir ./byom-run-01 \
  --steps 10 \
  --lr 1e-3
```
For full Hugging Face causal-LM LoRA runs, use `train_aix.py prepare`, `train_aix.py train`, and `train_aix.py report`.

## 3. Interpreting Rejected Results & Limitations
- **`REJECTED_RECORD_ONLY`** is the expected safe default when `allow_promotion=False` or when candidate accuracy does not beat baseline on `development` without regressing `retention`.
- **Base Preservation:** `aix_zero_v11_kip.safetensors` and user HF base directories are never modified in place.
- **Second-Operator Reproduction Gate:** Marked **PENDING** until an independent operator on a clean environment runs the workflow and records source, model, dataset, and environment SHA-256 hashes.
