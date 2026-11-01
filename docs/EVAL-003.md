# EVAL-003: live model evaluation through Ollama

EVAL-003 extends the EVAL-002 runner with real model inference. Ollama is the first
live provider because the repository's AIX scripts already use its local chat API.

## Acceptance criteria

- A live adapter implements the existing provider interface and CLI without
  changing the v1 result schema or reference/replay behavior.
- Outbound requests include only public task fields and the verifier's public
  specification; generator metadata, reference solutions and unknown fields are
  excluded. The scorer retains the complete task locally.
- Requests have explicit generation limits, finite timeouts and no automatic
  retries. Setup checks that the model is installed; it never downloads a model.
- Results record the installed model digest, generation settings, token counters
  when supplied, and measured latency. Unknown costs and usage remain omitted.
- Service failures produce redacted zero-score samples through the existing
  runner. Setup/configuration errors fail before writing an artifact.
- Offline HTTP integration tests cover every domain, public-input isolation,
  usage, model identity, invalid responses, timeouts and server failures.
- Verify a local model request, the full regression suite and GitHub CI before
  merging the feature branch into main.

## Run

Use the dependency checkouts and Python dependencies from the README. Start Ollama
and make the desired model available separately, then run:

```bash
python run_benchmark.py --provider ollama --model aix3:latest \
  --domains logic --difficulty 1 --samples-per-domain 1 \
  --max-output-tokens 2048 --timeout 240 --output benchmark-results/live.json
python evaluation_result.py benchmark-results/live.json
```

Choose an installed model with `--model`; there is no implicit model selection.
The default origin is `http://localhost:11434`; `--ollama-url` accepts an HTTP(S)
origin without a path, query or embedded credentials. This adapter has no hosted
authentication integration. Provider-specific options are rejected for reference
and replay runs, and `--recording` is only accepted for replay.

Default inference settings are temperature 0, `num_predict` 2048 and a 120-second
socket timeout. `--temperature`, `--max-output-tokens` and `--timeout` override
them. `--thinking auto|on|off` selects the model default or explicitly enables or
disables thinking when supported by the chosen model. An explicit selection is
recorded in model parameters and sent as the top-level API `think` field.
The benchmark seed also sets the inference seed. No generation retries or
sample dropping occur. A low score alone does not fail the CLI; provider or scorer
errors save a validated result and exit 1. Setup failures exit 2 without writing.

## Public inputs and scoring

The adapter builds a fresh projection with task ID, domain, difficulty, prompt and
an explicit per-domain list of public specification keys. It never serializes the
input task wholesale. All metadata, top-level unknown fields and unknown spec keys
are excluded, including the reference solution. Domain instructions describe the
submission shape without supplying an answer. Inputs come from the trusted local
TaskForge generator; this projection is not a sanitizer for arbitrary content
embedded inside an allowed public prompt or specification value.

The scorer receives the original complete task. Only the returned `message.content`
is evaluated; thinking text and explanation are not substituted for missing output.
Empty or malformed JSON is retained, with known usage, as a scorer-error sample.
Partial credit is the existing VeriSci reward, not a binary correctness claim.

## API mapping and provenance

The adapter uses [GET /api/tags](https://docs.ollama.com/api/tags) to resolve an
installed model name and digest. An untagged name resolves to `:latest`. After
each generation it checks that the installed digest still matches, rejecting
results if a tag moved. This catches observable local tag changes; it is not an
attestation of remote server execution or protection against a tag changing and
changing back between checks.

Requests use [POST /api/chat](https://docs.ollama.com/api/chat), `stream: false`,
JSON output format, system instructions and the projected user task. Returned
prompt, generated and cached-prompt token counts map to the v1 token fields when
present. Token counts must be nonnegative integers. The adapter requires a complete
response from the resolved model name. Latency measures the request path, including
the model-digest check; model inference settings and the adapter version are saved.
Cost is omitted because the API supplies no dollar cost for local compute.

## Verification and limits

```bash
python -m unittest discover -s tests -v
```

The test HTTP server uses held-out fixture responses to exercise all four domains
through the actual CLI and network transport. Captured requests are checked for
reference leakage; the fixture's perfect score is a plumbing check, not a model
benchmark. Tests also cover mutable model tags, malformed responses, usage counters,
timeout forwarding, unavailable models, HTTP errors, and configuration rejection.
CI runs these tests without starting Ollama or downloading models.

Live smoke results are operational checks on a small selected sample, not a
representative model quality assessment. Thinking models may exhaust an output
budget before emitting a valid JSON submission; increase the limit if needed.
Larger benchmark studies, hosted providers, authentication, pricing, retry policy,
parallel execution and dataset publication remain outside this task.

### Local smoke evidence

The committed [live smoke artifact](../examples/evaluation-result-v1-ollama-smoke.json)
records an actual `aix3:latest` request for `eval-logic-d1-s2026`, with a 2048-token
limit and 240-second timeout. It returned a JSON submission, 209 input tokens,
1926 generated tokens and 208 cached prompt tokens. VeriSci assigned **0.3 partial
credit**, with no provider or scorer exception. The submission is not a correct
proof; this establishes that live inference, usage capture, scoring and persistence
work together, not that the model solved the task. The artifact includes the model
digest and exact dataset/scorer source revisions.

An earlier 256-token run exhausted its budget with no final content. Its saved
result correctly retained the 256-token usage, assigned score zero and reported a
scorer JSON error. A first longer request hit its 60-second timeout and produced
a redacted provider-error sample. These failures were observed during the smoke
check; the successful request used the larger explicit limits above.
A separate 512-token request with `--thinking off` also returned no final content
on this AIX model. Do not assume that disabling thinking guarantees a submission
within a small token limit.
