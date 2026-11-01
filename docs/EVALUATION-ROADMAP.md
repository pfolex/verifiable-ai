# Evaluation milestones

EVAL-001–003 established the result contract, runner, reference/replay adapters,
and live Ollama inference. EVAL-004–007 are implemented and merged:

- **EVAL-004:** versioned public dataset manifests, integrity validation, exact
  saved-task reruns, source revision checks and manifest-bound result artifacts.
- **EVAL-005:** strict paired baseline/candidate comparison on one manifest and
  scorer, with overall and per-domain scores, failures, token usage, latency and
  known/unknown costs; export machine-readable and human-readable reports.
- **EVAL-006:** explicit statistical improvement/non-regression policy, conservative
  paired confidence bounds with multiple-domain control, PASS/FAIL/INCONCLUSIVE
  decisions, minimum sample sizes and automation-friendly exit codes. Small smoke
  sets must not be presented as statistical evidence of improvement.
- **EVAL-007:** authenticated hosted provider adapters, public-input isolation,
  bounded requests, normalized token usage, explicit versioned price inputs,
  accurate cached/uncached token costing and unknown-cost behavior. CI uses HTTP
  fixtures rather than spending credentials or money.

Hosted providers in EVAL-007 are OpenAI Responses and Groq Chat. Cache-write usage
is explicitly versioned as result 1.1.0, retaining support for result 1.0.0.

Each milestone requires regression tests, usage/limitation documentation, a feature
branch PR and passing CI before merge. No real hosted call is required without
available credentials; fixtures verify authentication and transport behavior.

Implementation history: [EVAL-004 PR #4](https://github.com/pfolex/verifiable-ai/pull/4),
[EVAL-005 PR #5](https://github.com/pfolex/verifiable-ai/pull/5),
[EVAL-006 PR #6](https://github.com/pfolex/verifiable-ai/pull/6), and
[EVAL-007 PR #7](https://github.com/pfolex/verifiable-ai/pull/7).
