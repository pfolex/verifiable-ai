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
