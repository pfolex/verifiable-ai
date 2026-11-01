# EVAL-007: authenticated hosted providers and token cost tracking

Acceptance: two hosted API adapters using the existing public-input boundary;
authentication via environment variables; explicit model selection; bounded,
non-streaming requests with no automatic retry; normalized usage and explicit
versioned pricing; charged refusal/truncation preservation; redacted failures;
offline transport/authentication tests; documented limits; CI before merge.

## Run a hosted model

Set `OPENAI_API_KEY` or `GROQ_API_KEY` securely in the process environment. API keys
are not CLI arguments, persisted model parameters or log messages. Export a public
manifest first, choose a compatible model explicitly, and optionally prepare a
price card for that exact provider/model:

```bash
python run_benchmark.py --manifest-only --export-manifest tasks.json
python run_benchmark.py --manifest tasks.json --provider openai --model "$MODEL_ID" \
  --pricing rates.json --max-output-tokens 2048 --timeout 120 --output openai-result.json
python run_benchmark.py --manifest tasks.json --provider groq --model "$GROQ_MODEL_ID" \
  --pricing groq-rates.json --max-output-tokens 2048 --timeout 120 --output groq-result.json
```

Keys and model availability are account-specific. The adapter does not select or
pull models. Use a model that supports the selected API's JSON output format and
any explicitly chosen settings. Without `--temperature`, hosted requests omit it
rather than assuming every reasoning model supports temperature 0. The manifest
seed controls task generation; it is not claimed as a hosted inference seed.
Ollama's `--thinking` and `--ollama-url` are rejected for hosted providers.

## API contracts

OpenAI uses `/v1/responses` with bearer authentication, non-streaming JSON output
and `store: false`. Text is collected from assistant message output-text blocks,
not assumed to occupy the first output item; reasoning is not scored as an answer.
See the [official text generation guide](https://developers.openai.com/api/docs/guides/text),
[JSON mode guide](https://developers.openai.com/api/docs/guides/structured-outputs),
and [Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses).

Groq uses its [Chat Completions endpoint](https://console.groq.com/docs/api-reference),
bearer authentication, `max_completion_tokens` and JSON object response format.
The adapter requests no tools, conversations, background jobs or external files.
Both adapters send only the same explicit public task/specification projection
used by Ollama. Generator metadata and reference solutions remain local.

The timeout is a finite positive socket timeout; the output limit must be a
positive integer. There is one generation request per sample and no automatic
retry, including for authentication failures, rate limits, timeouts and server
errors. A timeout may still have incurred a provider charge; when trustworthy usage
is unavailable the result's cost remains unknown, not zero.

`--api-url` overrides the origin for a compatible gateway or test server. Only
HTTPS is accepted outside loopback HTTP fixtures, and URLs cannot contain paths,
queries or credentials. HTTP redirects are refused so bearer headers are never
forwarded through a redirect. API errors persist only an exception class. Model
output is still user data; ordinary output text is not automatically anonymized.

The API-reported served model is recorded as `model.revision` at the end of a run.
It is a reported model identity, not a weight hash or remote execution attestation.
An observed change in served model identity within a run is rejected. Request
model identity, endpoint, settings, adapter version and complete configured price
card are retained; secrets are not.

## Usage versions and cost estimates

Both APIs normalize total input, generated output and cached-read token counters.
OpenAI also reports cache-write counters in its current
[prompt caching API](https://developers.openai.com/api/docs/guides/prompt-caching).
When cache-write usage is present, results use schema version **1.1.0** and the
`cache_write_input_tokens` field. Ordinary results remain 1.0.0. Both contracts
are validated; writes are not silently added to the frozen 1.0.0 schema.
Input token totals include normal, cached-read and cache-write input tokens;
read plus write counters cannot exceed the total. Comparison input totals include
all three, while the existing cached-token comparison subtotal denotes read hits.
The raw 1.1 result retains the separate write count for detailed auditing.

A price card follows `schemas/token-pricing-v1.schema.json`: pricing version,
rate-card ID, provider, exact request model, USD currency, and per-million rates
for ordinary input, cached reads, cache writes and output. Null cache rates mean
unknown. `examples/pricing-*-example.json` are **synthetic examples, not current
provider prices**. Copy the structure and supply the correct model and rates for
your account and service tier. Prices are never fetched or silently updated.

The estimate uses decimal arithmetic before converting the final USD value:

```
(normal_input * input_rate + cached_reads * cached_rate
 + cache_writes * write_rate + output * output_rate) / 1_000_000
normal_input = input_total - cached_reads - cache_writes
```

If a required usage count or applicable rate is missing, cost is omitted. A
missing cache counter only becomes immaterial when its configured rate equals
ordinary input; that permits an estimate without inventing a counter. A known
zero cache count does not require a cache price. Price-card identity mismatches,
negative/nonfinite prices and impossible counters are rejected. These are
configured token-rate estimates, not billing receipts: taxes, account discounts,
tools, audio/images, storage and other fees are outside this text-only adapter.
[Groq caching documentation](https://console.groq.com/docs/prompt-caching) describes
its cached-read usage field. Check current tariffs before preparing a card.

Refused and truncated responses retain known token usage, partial text where
available and estimated cost, but receive a zero score and explicit provider error.
The CLI saves the artifact and exits 1. Malformed submissions similarly retain
usage through scorer-error handling. Complete cost summaries require a known cost
for every sample and are validated against the sample sum. Unknown charges remain
explicit in EVAL-005 comparison reports.

## Verification

The suite authenticates to local HTTP fixtures for both APIs, runs all four domains
through the real CLI and verifies result validity, cost totals and no private-task
leakage. It also tests missing credentials, key redaction, redirects, timeouts,
401/429/5xx responses without retries, unknown usage, model drift, refusals,
truncation, cache writes, pricing mismatches and preservation of existing output.
No OpenAI or Groq credentials were available during implementation. No paid hosted
call or account-specific availability claim is part of this verification; the
network contracts and authentication are verified against fixtures using official
API documentation. CI requires these offline tests along with all existing suites,
manifest reruns, comparisons and the deliberately inconclusive smoke gate.
