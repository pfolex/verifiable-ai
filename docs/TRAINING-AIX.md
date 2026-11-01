# Train and measure your own AIX candidate

The self-service workflow trains an adapter through the full Hugging Face causal
language model, then evaluates actual generated submissions with VeriSci. It does
not use the older scripts' projected hidden-state/token-margin proxies as evidence
of model quality. It is a supervised reference-answer LoRA pilot, not a GRPO/RL
implementation and not a guarantee of improvement.

## First run

From this repository, with the `verisci` and `rlvr-forge` checkouts from the README:

```bash
./scripts/aix.sh setup
./scripts/aix.sh doctor
./scripts/aix.sh run --run-dir training-runs/pilot-001 --steps 20
```

`setup` creates a separate `.venv-training` environment and installs
`requirements-training.txt`. With uv installed it uses Python 3.12; otherwise it
uses `python3 -m venv` (Python 3.9–3.12 is required by this pinned stack). This avoids
changing the existing model-merging environment, where the installed Transformers
5.x and PyTorch 2.2 imports were incompatible. `AIX_TRAINING_ENV` can select another
training environment directory.

For the existing Transformers 5-style AIX export, the loader explicitly reads its
separate `chat_template.jinja` and maps the nested default RoPE theta into the
compatible 4.x model config in memory. Unsupported or conflicting RoPE settings
are rejected, not replaced with a library default. Base files remain unchanged.

The base defaults to `../model_merging/merged_r1_aix3_verifiable`, matching the
existing local AIX checkpoint. On another computer, specify it explicitly:

```bash
./scripts/aix.sh doctor --base-model /path/to/aix-hf-checkpoint
./scripts/aix.sh run --base-model /path/to/aix-hf-checkpoint \
  --run-dir training-runs/pilot-001 --steps 20
```

`AIX_BASE_MODEL` is another way to set that directory. It must be a local, complete
AIX/Qwen2 Hugging Face **safetensors** checkpoint, with matching config and tokenizer;
an Ollama model name or GGUF file is not a trainable input. No model is downloaded,
registered, merged into the baseline, or pushed to a hosting service by this tool.
The existing `aix3` Ollama tag is untouched. Both sides are evaluated through the
same full-precision Transformers backend instead of comparing different runtimes
or quantizations.

`doctor` imports the training stack, reports CUDA/MPS availability, checks token IDs
and checkpoint embedding shape, and reports weight size without loading all model
weights. On the inspected machine, the checkpoint is 3.55 GB and only CPU is exposed.
CPU training uses float32 weights plus activations and optimizer memory; training
and generation can be slow on that machine. Start with a small run. Do not infer a
runtime estimate from the tiny CI model.

## What `run` does

1. Create a new experiment directory, fingerprint the base checkpoint/config/tokenizer,
   and save the training recipe, generation settings and workflow source hashes.
2. Prepare verified reference-answer training examples and a public evaluation
   manifest. Different training/evaluation seeds and splits are used. Exact public
   problem contents are hashed without IDs; duplicated training examples and any
   exact train/evaluation problem overlap are excluded.
3. Generate baseline answers on the held-out manifest before training.
4. Train LoRA on the selected final attention layers with causal-LM cross-entropy
   on assistant answer/EOS tokens only. Prompt labels are masked. All original
   parameters are frozen; no baseline files are rewritten. Training visits domains
   round-robin with seeded per-domain shuffles.
5. Save only the adapter, reload the unchanged base plus adapter, and generate
   candidate answers on the identical manifest and generation configuration.
6. Write comparison JSON/Markdown and the statistical gate report. The command can
   complete even when the model is worse or the gate is FAIL/INCONCLUSIVE; completion
   is not promotion. Review scores and error counts in `comparison.md`.

Every model stage runs in its own process so the full checkpoint can be released
between baseline, training and candidate evaluation. There are no two full models
kept in memory for a side-by-side comparison. Experiment directories must be fresh
and separate from the base; existing measurements/adapters are not overwritten.
Checkpoint, dataset, source, adapter and experiment hashes guard accidental reuse
of changed inputs. These are integrity records, not signatures or execution
attestations.

## Settings

Defaults: 20 optimizer updates, rank 8, final 2 attention layers (`q_proj`, `v_proj`),
learning rate 0.0001, batch size 1, at most 1024 training tokens, and 1024 generated
tokens per evaluation case. Training requests up to 16 unique problems per domain,
using difficulties 1–10; evaluation uses two seeds per domain at difficulty 3.

```bash
./scripts/aix.sh run --run-dir training-runs/pilot-002 \
  --steps 40 --rank 8 --last-n-layers 2 --learning-rate 0.0001 \
  --train-samples-per-domain 16 --eval-samples-per-domain 2 \
  --max-seq-length 1024 --max-new-tokens 1024 --gradient-checkpointing
```

Use `--domains logic silicon` for a narrower experiment. `--steps` must cover every
requested domain at least once. `--device auto|cpu|mps|cuda` chooses hardware; auto
prefers CUDA, then MPS, then CPU. CUDA uses bfloat16 where supported and otherwise
float32; CPU/MPS use float32. CUDA/MPS support depends on the pinned PyTorch build
and hardware; the end-to-end integration test runs on CPU.

Complete examples that do not fit `--max-seq-length` are skipped, not truncated.
Training fails if this leaves a domain with no examples. Increase the length in a
new experiment if needed. Some procedural domains have only a few distinct public
problems; the actual prepared/eligible/visited counts and overlength exclusions
are saved rather than claiming the requested count was achieved. Steps are optimizer
updates, not complete epochs over every prepared example.

The prompt uses the checkpoint's native chat template. If it opens an R1 `<think>`
span, the workflow closes it in the prefix identically for training and evaluation,
so the supervised target and measured continuation are JSON submissions. It does
not recover an answer from hidden reasoning or silently remove non-JSON explanations.
Greedy generation uses the tokenizer's actual EOS ID, rather than trusting a
possibly different generation default in the merged model config.

## Outputs and recovery

```
training-runs/pilot-001/
  experiment.json         # immutable recipe, base/data/code identities and audit
  train.jsonl             # private training inputs and verified target answers
  eval-manifest.json      # public held-out task manifest, no reference answers
  baseline.json           # actual pre-training generation results
  adapter/                # PEFT LoRA weights/config, separate from the base
  training-report.json    # finite losses, actual visited examples, versions/hashes
  candidate.json          # actual adapter generation results
  comparison.json
  comparison.md
  gate.json
```

Run individual stages if you want to inspect data before spending training time:

```bash
./scripts/aix.sh prepare --run-dir training-runs/manual-001
./scripts/aix.sh evaluate --run-dir training-runs/manual-001 --which baseline
./scripts/aix.sh train --run-dir training-runs/manual-001
./scripts/aix.sh evaluate --run-dir training-runs/manual-001 --which candidate
./scripts/aix.sh report --run-dir training-runs/manual-001
```

If generation is interrupted before its result file is saved, repeat that evaluate
stage. If training is interrupted before adapter save, repeat `train` to restart its
seeded updates from scratch. This version does not resume optimizer checkpoints.
A completed or partially saved adapter requires a fresh run directory. Never
manually splice adapters, base checkpoints or recipes between experiments.

Model-generation/scoring errors are retained as zero-score samples and do not
prevent the pilot from reaching its report. Invalid data, missing dependencies,
nonfinite loss/gradients, changed identities and subprocess failures stop execution
with a nonzero exit code. The default policy rejects runs with infrastructure/model
format errors and otherwise leaves undersized/undeclared sampling inconclusive.

## Interpreting improvement

A lower training loss is not evidence of better generated solutions. Compare actual
held-out VeriSci scores, errors and per-domain changes. Small pilots are descriptive.
Different seeds can repeat identical circuits, schedules or molecules, and exact
content exclusion does not rule out related or isomorphic problems. The data audit
reports evaluation uniqueness. The default gate does not assert independent samples;
preparation rejects an independent-sampling declaration when the manifest contains
known exact duplicates. More seeds from a tiny template family are not a substitute
for a sufficiently diverse benchmark. See [EVAL-006](EVAL-006.md) for statistical
requirements; this script never automatically promotes a candidate.

Implementation follows [PEFT's LoRA workflow](https://huggingface.co/docs/peft/quicktour)
and [Transformers local model loading](https://huggingface.co/docs/transformers/main_classes/model).
Tests use a tiny randomly initialized Qwen2 checkpoint: they verify real forward/backward
updates, nonzero saved LoRA weights, reload/generation, masked prompt loss, unchanged
base bytes, result/report validity and refusal to overwrite or reuse changed inputs.
No full-size AIX training run or improvement claim is part of the script's validation.
