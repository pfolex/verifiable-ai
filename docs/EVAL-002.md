# EVAL-002: provider-to-result benchmark integration

There is no separate issue specification for EVAL-002. This task implements the
EVAL-001 follow-up: connect provider adapters and a benchmark runner to the v1
evaluation result contract.

## Acceptance criteria

- A common provider interface returns output and optional measured usage, without
  assigning scores. Local reference and recorded-response adapters implement it.
- The runner evaluates actual submissions using a separate scorer, computes sample
  counts and means, and tracks provider/model, parameters, dataset revision,
  manifest hash, scorer revision, latency, optional tokens and optional cost.
- Provider and scorer exceptions produce zero-score error samples and do not drop
  samples. Invalid contract data fails the run. Unknown usage is omitted; total
  cost is reported only when every sample has a known cost.
- Every result is validated before return and before saving. Writes are atomic;
  invalid results and write failures preserve an existing destination.
- A CLI exercises all four VeriSci domains with seed-controlled evaluation tasks.
  CI runs regression tests and uploads a validated benchmark artifact.

## Run a benchmark

```bash
python -m pip install -r requirements-test.txt
python run_benchmark.py --provider reference --seed 2026 --difficulty 3 \
  --samples-per-domain 2 --output benchmark-results/reference.json
```

Use `--domains logic silicon` to select domains. Samples use the `eval` split and
IDs such as `eval-logic-d3-s2026`. Seeds advance by one within each domain. The
dataset revision is the RLVR-Forge Git commit; the scorer version is the VeriSci
Git commit. The runner rejects dirty dependency source trees. A SHA-256 digest of
canonical JSON cases (ordered sample IDs and complete task inputs) identifies the
exact manifest. Seed, difficulty, domains and count are stored in scorer
configuration so the manifest can be regenerated from those source revisions.
Run IDs and UTC timestamps identify executions; latency is measured in milliseconds
using a monotonic clock. The original six-repository training showcase is unchanged.

The reference solver uses privileged ground truth. Its scores test the evaluation
plumbing and verifiers; they do not demonstrate learned model performance.

## Replay provider responses

Create a JSON recording with the original model identity and a response for each
generated sample ID. For a one-sample logic run its shape is:

```json
{
  "model": {
    "provider": "your-provider",
    "model_id": "your-model",
    "revision": "your-model-revision",
    "parameters": {"temperature": 0}
  },
  "responses": {
    "eval-logic-d3-s2026": {
      "output": "{\"proof_steps\": []}",
      "input_tokens": 100,
      "output_tokens": 10,
      "cost_usd": 0.001
    }
  }
}
```

```bash
python run_benchmark.py --provider replay --recording recording.json \
  --domains logic --samples-per-domain 1 --output benchmark-results/replay.json
```

The illustrative empty proof is not a correct solution. Output must be a JSON
submission accepted by the domain cartridge. Recorded IDs must match the selected
manifest exactly; missing or extra IDs are a configuration error. Replay preserves
recorded token/cost values, model identity and parameters. Its `latency_ms` measures
replay lookup time, not the original provider request. The adapter version identifies
replay explicitly. Omit unknown usage instead of estimating it.

## Error and persistence behavior

Provider exceptions and malformed submissions are retained as samples with score
zero. Errors identify the stage and exception class without persisting exception
messages, which can contain secrets. The CLI saves a valid partial-failure result
and exits 1 when any sample has an error; valid low scores alone do not fail the CLI.
Configuration and contract errors fail without writing a result. Output and known
cost are retained when scoring fails. The saved output itself may contain provider
content; choose an appropriate artifact destination.

`save_result` validates schema and invariants, rejects nonfinite JSON numbers, writes
a temporary file in the destination directory, flushes it, then atomically replaces
the destination. It cleans temporary files if replacement fails.

## Extension interface and scope

Implement `Provider.generate(sample_id, task) -> ProviderResponse` plus a `model`
metadata dictionary to add another adapter. Supply a scorer callable and versioned
metadata to `run_benchmark`. The caller must not expose reference solutions to an
untrusted model: the local procedural CLI includes privileged task metadata for the
reference solver and offline replay, and is not a live hosted-provider harness.
Hosted API adapters, retries, parallel execution, and trained-model inference are
future work; no credentials, paid calls, or external SDKs are needed here.

## Verification

```bash
python -m unittest discover -s tests -v
python run_benchmark.py --output benchmark-results/reference.json
python evaluation_result.py benchmark-results/reference.json
```

Tests cover mixed correct/incorrect scores, usage/cost accounting, manifest
stability, duplicate IDs, provider/scorer failures, invalid adapter data, unknown
usage, empty runs, overwrite protection, failed-write cleanup, CLI configuration
errors, and reference/replay evaluation across every domain.
