# Recorded offline candidate experiments

Run one bounded CPU fine-tuning attempt, compare its candidate against the exact
starting checkpoint on development and regression questions, and record the result.
This command does not start the daemon, push to GitHub, or replace the live model.
A passing diagnostic screen remains `evaluated` for review; it is not a claim of
statistical significance or independent final-test performance.

From the repository root, using the Python environment with the AIX dependencies:

```sh
"$PYTHON" -m aix_zero.run_candidate_experiment \
  --champion /path/to/preserved-backbone.pt \
  --manifest /path/to/split-manifest.json \
  --policy /path/to/policy.json \
  --registry ./candidate-experiments \
  --steps 5 --learning-rate 0.001 \
  --max-training-seconds 120 --max-new-tokens 36 --seed 42
```

`--champion` is a v11-compatible **torch state_dict**, loaded with
`weights_only=True` and strict key/shape checking. It is not a safetensors file.
The checked-in v11 safetensors checkpoint supplies model architecture and tokenizer.
Its digest, current Python source hashes, git revision, software versions, seed,
hardware, training settings and screening policy are retained with the attempt.

Create the manifest with `aix_zero.experiment_splits.build_split_manifest` using
`lambda row: row['prompt']` as identity. Supply exactly four splits: `train`,
`development`, `regression`, `final`. The first three must be nonempty.
Rows require `prompt` and `gold`; training rows also require `completion` containing
a trace accepted by the existing trusted trace verifier. All training targets must
fit the model context completely. Do not train on benchmark answers. Prompt identity
checks do not detect paraphrases or shared templates: the dataset producer must
establish semantic separation. The final split is not evaluated by this command.

Example policy JSON (choose and freeze thresholds before training):

```json
{"min_cases": 20, "min_net_gain": 2, "max_regression_losses": 0, "max_trace_losses": 0}
```

The trainer handles 1–256 examples, 1–1000 steps, one CPU thread, AdamW, gradient
clipping and finite-loss/weight checks. It trains the existing expert/router/head/
decoder/memory parameter groups, without adding parameters. Its wall-time budget is
cooperative between operations; it cannot interrupt a blocked operation. Callbacks
run in-process and are trusted application code, not a security sandbox.

Each attempt has immutable metadata and lifecycle events, candidate checkpoint,
training metrics, and paired evaluation evidence. The registry keeps digest-named
artifacts and verifies them before use. Training failure marks the attempt failed;
no-gain or damaging candidates are rejected. Invalid/incomplete evaluation blocks
eligibility. Read errors from the command and inspect the recorded attempt before
retrying; a retry creates a new attempt rather than silently resuming training.

Still required for an autonomous service: failure-driven curriculum planning,
process isolation and hard resource deadlines, locking/resume scheduling,
independent statistical confirmation, atomic champion promotion/rollback, and
end-to-end acceptance tests.

## Initial integration trial

A local Mac CPU trial used a preserved cycle-98 backbone with two training examples,
two development examples and two regression examples, two AdamW steps at 0.0004,
seed 42, one CPU thread and greedy neural-only generation (36 tokens).
No independent final split was evaluated. These tiny, template-derived diagnostic
sets are a functional smoke test, not a capability benchmark.

Both training examples fit without truncation. Training loss fell from 1.3585255
to 1.0575835; 413,440 parameters were trainable and there were 68 supervised tokens
per step. Training took approximately 6.67 seconds. Development net gain was zero,
with zero regression losses and zero trace losses. The candidate was rejected for
insufficient development gain. No live model was promoted.

The tested local subset passed 15 unittest methods, covering registry, split,
paired evaluation, runner and real-training boundaries. The Mac disconnected during
subsequent publication preparation, so the later repeat trial was not confirmed.

## Portable input preparation (`#422`)

Instead of hand-editing `/path/to/...` placeholders or writing into protected OS folders
(such as macOS `Documents/`), prepare a validated, self-contained experiment bundle in a
repo-relative or temporary directory:

```sh
"$PYTHON" -m aix_zero.prepare_candidate_experiment \
  --output-dir ./candidate-experiment-bundle \
  --profile single_skill_tiny \
  --steps 15 --learning-rate 0.002 \
  --max-training-seconds 120 --max-new-tokens 36 --seed 42
```

This command verifies the source checkpoint, writes `champion.pt`, `split_manifest.json`,
`policy.json`, and `bundle.json` atomically without overwriting existing files, and prints
the exact runnable `python -m aix_zero.run_candidate_experiment ...` launch command.

## Case-level rejection diagnostics and non-vacuous retention (`#423`, `#426`)

Inspect any recorded attempt and optionally probe a larger decoding budget without
mutating the registry or champion:

```sh
"$PYTHON" -m aix_zero.inspect_candidate_experiment \
  --registry ./candidate-experiments \
  --attempt d17d7da8d3cb40b9ab46f8c419edd4e6 \
  --probe-max-new-tokens 64 \
  --probe-split development
```

### Diagnosis of rejected attempt `d17d7da8d3cb40b9ab46f8c419edd4e6`

- **Identities & Training:** Trained from champion `424f2d46aab127eb0afb4480cbc79810e0b174b1185341862ac6d5be2921d91b`
  on 20 multi-domain cycle-101 examples for 5 AdamW steps (`lr=0.001`, seed `42`, `586` supervised tokens/step).
  Training loss decreased from `1.7130` to `1.0852`, producing candidate `237ab0fef03d06487044859445f3c4c3cfe06cb7865d9d0eeef75516520627ae`.
- **Exact Development Loss (`development_net_gain = -1`, `trace_losses = 1`):**
  Both the answer loss and the trace-validity loss occurred on the same `META_TRL_RLOO_LEAVE_ONE_OUT`
  question (`Q:MetaRLOO:r1=1;other_sum=1;k=4->rloo_advantage?`, gold `2/3`):
  - Champion: `<think>base=1/3;adv=1-1/3=2/3</think><ans>2/3</ans><eos>` (`answer_correct=True`, `trace_valid=True`, terminated by `eos`).
  - Candidate: `<think>ba==1/5=4;=1;p4</ans><eos>` (`answer_correct=False`, `trace_valid=False`, `failure_kind=missing_or_unclosed_answer_tag`, terminated by `eos`).
  Because the candidate emitted `<eos>` early after corrupting the `<think>` equation and omitting `<ans>`,
  this regression was caused by representation drift from multi-domain fine-tuning, **not** by the 36-token generation cap.
- **Why `0/18` Regression Losses Is Non-Informative (`#426`):**
  On the 18 cycle-3100 regression items, both champion and candidate scored `0/18` (`both_wrong = 18`,
  `champion_correct = 0`, `candidate_correct = 0`, `retained_champion_correct = 0`, `losses = 0`,
  `retention_denominator = 0`). While `losses = 0` is arithmetically true, a zero denominator cannot measure
  skill retention. `compute_retention_metrics` now reports `retention_measurable = False`, `retention_rate = null`,
  `zero_losses_non_informative = True`, and `retention_status = "not_measurable_zero_champion_successes"`.
  When a policy specifies `min_champion_regression_correct >= 1`, screening requires non-zero baseline coverage
  and includes a fixed champion-success retention subset (`CODE_TRACE`, `PHYSICS_KINEMATICS`).

## Tiny-set learning and generalization check (`#424`)

Run the predeclared tiny-set fitting and generalization harness across multiple seeds:

```sh
"$PYTHON" -m aix_zero.tiny_learning_check \
  --seeds 42 43 \
  --steps 15 --learning-rate 0.002 \
  --max-training-seconds 120 --max-new-tokens 36
```

This verifies tokenizer round-trip, BOS/EOS framing, shifted completion labels, nonzero
gradients, and frozen-backbone invariance, then compares exact-match answer and trace
validity on the tiny single-skill training set (`META_QUANTIZATION_FP8_INT4`), a prompt-disjoint
same-skill development set, and the fixed retention suite.

## Predeclared single-skill recipe study and training-only replay (`#425`, `#412`)

Run the predeclared multi-seed comparison between target-only training and replay-balanced
training across at least three seeds (`42`, `43`, `44`):

```sh
"$PYTHON" -m aix_zero.single_skill_study \
  --seeds 42 43 44 \
  --max-training-seconds 120 --max-new-tokens 36
```

Training replay anchors (`TRAINING_ONLY_REPLAY_POOL`) are registered inside the `train`
split of the manifest and verified through `eligible_training_rows` so held-out evaluation
prompts or answers can never enter replay.

## Failure-driven planning, resumable subprocess scheduler, and atomic CAS promotion (`#409`, `#413`, `#414`, `#416`, `#417`)

The autonomous improvement loop connects failure-driven planning (`aix_zero.experiment_planner`),
subprocess-isolated candidate execution with hard wall-clock deadlines and single-worker
locking (`aix_zero.autonomous_scheduler`), and atomic compare-and-swap champion promotion
with post-switch smoke rollback (`aix_zero.champion_promotion`):

```sh
# 1. Prepare portable inputs in a repo-relative folder
"$PYTHON" -m aix_zero.prepare_candidate_experiment \
  --output-dir ./candidate-experiment-bundle

# 2. Run a bounded record-and-reject autonomous loop (1 to N attempts)
"$PYTHON" -m aix_zero.autonomous_scheduler \
  --registry ./candidate-experiment-bundle/registry \
  run \
  --champion ./candidate-experiment-bundle/champion.pt \
  --manifest ./candidate-experiment-bundle/split_manifest.json \
  --policy ./candidate-experiment-bundle/policy.json \
  --max-cycles 2 --max-attempt-seconds 180

# 3. Inspect live operator status and per-attempt evidence
"$PYTHON" -m aix_zero.autonomous_scheduler \
  --registry ./candidate-experiment-bundle/registry \
  status

# 4. Stop an active scheduler or recover interrupted attempts
"$PYTHON" -m aix_zero.autonomous_scheduler \
  --registry ./candidate-experiment-bundle/registry \
  stop --reason "operator_pause"

"$PYTHON" -m aix_zero.autonomous_scheduler \
  --registry ./candidate-experiment-bundle/registry \
  recover
```




## Opt-in physical LoRA candidate training

The standalone physical-LoRA module is now connected to the recorded experiment
trainer and paired evaluator. Add `--lora-rank 16` to the experiment command:

```sh
"$PYTHON" -m aix_zero.run_candidate_experiment \
  --champion ./candidate-inputs/champion.pt \
  --manifest ./candidate-inputs/manifest.json \
  --policy ./candidate-inputs/policy.json \
  --registry ./candidate-experiments \
  --steps 5 --learning-rate 0.001 --max-training-seconds 120 \
  --max-new-tokens 36 --seed 42 \
  --lora-rank 16 --lora-alpha 16 --max-parameters 2000000
```

Prepare the input bundle first with the preparation command above. This option
adds real A/B tensors to the module's configured attention/shared-expert targets,
freezes the original weights, and trains only the adapters. It does not alter the
live daemon, its ledger rank or the champion. There is no automatic promotion.

Training evidence records actual starting/added/total physical parameters,
adapter/trainable counts, optimizer memory and the logit difference at adapter
insertion. New B matrices are zero; insertion must preserve the initial sampled
logits within tolerance. Training then changes the adapters, so held-out and
retention evaluation remains mandatory. Parameter growth is not a learning claim.

Candidate bytes use a versioned physical-LoRA payload with exact target names,
architecture and tokenizer identity; the same evaluator accepts legacy backbones
and these expanded candidates. A physical candidate may be used as the next
experiment's champion. A larger requested rank preserves the existing adapter's
effective delta before fresh-optimizer training, including when alpha/r changes.
Shrinking ranks and same-rank alpha changes are rejected.

Experiment ranks are limited to 1–128 and a physical parameter cap is enforced.
The adapter path currently supports the v11 backbone configuration; expert-count
growth has a separate migration contract and is intentionally rejected here.
Standalone physical checkpoints lacking tokenizer identity are not silently
accepted as experiment inputs. Do not use the legacy daemon to load or promote
a versioned physical-LoRA payload until it has an explicit compatible loader.
