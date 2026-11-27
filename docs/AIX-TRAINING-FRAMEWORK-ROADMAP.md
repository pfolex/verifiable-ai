# AIX training framework: direction, status and delivery roadmap

Planning baseline: November 26, 2026. Audited source: `7a45f8ffe6002f6fedd61da1537b15bef3f30aed`.
Tracking epic: [#437](https://github.com/pfolex/verifiable-ai/issues/437).

## Mission and first product outcome

AIX is intended to become a downloadable framework that lets users improve models
they create or obtain, including compatible Mergekit exports. World-leading
intelligence is the research ambition. The product commitment is reproducible
training and honest evidence of capability change, not inevitable improvement.

AIX-Zero remains a small custom research backend. The framework must also support
compatible pretrained/merged Hugging Face models without forcing them into the
AIX-Zero architecture or tokenizer.

First release gate: a fresh user installs AIX, validates their compatible model,
registers licensed data, generates a baseline, trains an isolated adapter, reloads
and evaluates it, and exports a usable artifact plus reproducible report. The
original model stays unchanged. A correctly rejected candidate is a valid result.

## Audited status

| Component | Existing foundation | Remaining limitation |
| --- | --- | --- |
| Custom AIX-Zero | Small model, routed cartridges, symbolic assistance and candidate experiments | Internal diagnostic gains are not external competitive evidence |
| HF training | train_aix.py and aix_training_runtime.py perform actual LoRA training and before/after generation | Current path centers on Qwen2-style local safetensors, fixed adapter targets and a narrow procedural curriculum |
| Autonomous control | Planner, registry, worker deadlines and promotion modules exist | Default scheduler invokes the custom candidate path, not a unified HF service |
| Coding | PR #436 merged; expression-training and functional scoring exist | Initial preserved-model pilot remained 0/12 development correct |
| Distribution | Public source and cycle-189 checkpoint bundle | Onboarding/source-build PRs #273/#288 remain open at audit time |
| CI | Latest audited main pipeline passed | Tested mechanics do not demonstrate intelligence gain |

See [the current HF training guide](TRAINING-AIX.md),
[candidate experiments](CANDIDATE-EXPERIMENTS.md) and
[checkpoint publication](CHECKPOINT-PUBLICATION.md). These are separate workflows,
not a claim that the proposed unified interface already exists.

## Evaluation corrections required first

In the audited standard_benchmarks.py, names containing GSM8K, MMLU and GPQA group
AIX-generated curriculum items; they do not load those actual benchmark datasets.
Rename these internal diagnostics and version old report mappings.

The default swe_three_arm_benchmark.py patch generator performs programmed source
replacements and assigns fixed token counts. These are heuristic controls, not
evidence that AIX or an Ollama peer generated a patch. Actual model attribution
requires captured generation calls and measured usage.

Keep unassisted, cartridge-routed, ALU-assisted and peer-assisted results separate.
Subject labels, effective-parameter ledgers and lower loss do not establish
expertise. Physical/trainable parameters and genuinely held-out task outcomes
must be measured independently.

A disposable worktree, resource limits and proxy environment settings do not alone
isolate untrusted code. Unattended generated-code tests require an actual constrained
execution boundary with filesystem, network and resource controls.

## Work packages

| Issue | Deliverable | Sequence |
| --- | --- | --- |
| [#438](https://github.com/pfolex/verifiable-ai/issues/438) | Unify installation, experiment configuration and backend interfaces | Foundation |
| [#439](https://github.com/pfolex/verifiable-ai/issues/439) | Validate user-supplied and Mergekit models with a tested compatibility matrix | After backend contract |
| [#440](https://github.com/pfolex/verifiable-ai/issues/440) | Add user datasets with provenance, code-task formats and split isolation | Alongside import |
| [#441](https://github.com/pfolex/verifiable-ai/issues/441) | Correct benchmark labels and integrate genuine external evaluations | P0 honesty correction; external adapter after import/data |
| [#442](https://github.com/pfolex/verifiable-ai/issues/442) | Extend bounded autonomous experiments to user-supplied HF models | After import, data and evaluation |
| [#443](https://github.com/pfolex/verifiable-ai/issues/443) | Export adapters and optional merged models with reproducible lineage | After artifact and model contracts |
| [#444](https://github.com/pfolex/verifiable-ai/issues/444) | Ship and independently reproduce the first bring-your-own-model workflow | Final integration and second-operator reproduction |

Existing #406–#417 and #418 own the shared autonomy foundations. Extend their
implementation and link remaining acceptance gaps instead of creating competing
ledgers, planners or promotion engines. Coordinate #271/#287 and the open
onboarding/source-build PRs.

## Supported-model policy

Begin with one integration-tested Qwen2-compatible family. A Mergekit origin alone
does not guarantee architecture compatibility. Validate weights/index shards,
config, tokenizer, chat template, context, RoPE and adapter targets before training.
Add families only with actual train/save/reload/generation tests.

Record a merge recipe and parent lineage when provided; explicitly mark unknown
lineage. GGUF and Ollama tags are not trainable inputs for the current HF path.
Do not enable remote model code or silently download weights to make an input work.

CPU/MPS/CUDA support and memory estimates must be documented per tested environment.
Training memory includes weights, activations, gradients and optimizer state;
weight file size alone is insufficient. Large-model GPU, quantized and distributed
training are later, separately validated extensions.

## Learning and release evidence

1. Freeze model/data identities, training recipe, baseline, budget and success rule.
2. Keep training, development, retention and reserved final evaluations separate.
3. Train only an isolated candidate; record full outcomes, including failures.
4. Compare actual generated answers under identical inference/tool settings.
5. Require sufficient retention and improvement evidence before controlled promotion.
6. Export adapter/base lineage and reload it independently; preserve rollback.
7. Obtain a second operator's clean-machine reproduction before declaring the release gate complete.

Repeated adaptive development comparisons are not independent final tests.
External benchmark claims require actual pinned datasets, scorers and comparable
baselines. No frontier claim follows from passing this engineering backlog.

## Subsequent research

Broaden verified programming tasks from expressions to functions and real patches,
evaluate teacher-assisted versus AIX-only generation, and explore physical capacity
growth only when bounded experiments justify it. Use existing tools for training
and evaluation where suitable rather than rebuilding every component.

No fixed timeline, guaranteed accuracy, paid compute, public execution service or
automatic production self-edit is implied by this roadmap.
