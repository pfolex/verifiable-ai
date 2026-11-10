# Multidimensional AIX Scorecard, Execution-Mode Separation & Rational Scoring (`EVAL-02`, `EVAL-03`, `EVAL-06`)

## 1. No Single Composite "Smartest-AI" Score (`EVAL-02` / Issue #41)

AIX (`verifiable-ai`) prohibits collapsing model evaluations into a single weighted composite index or claiming a general `"smartest AI"` score from narrow benchmarks. Every evaluation scorecard ([`schemas/aix-scorecard-v1.schema.json`](../schemas/aix-scorecard-v1.schema.json)) keeps the following **five orthogonal measurement dimensions** visible with explicit physical units:

| Dimension | Metrics & Units | Purpose |
| :--- | :--- | :--- |
| **1. Capability** | `correct_samples` (count), `total_samples` (count), `accuracy_ratio` (`[0, 1]`), `per_domain_accuracy` (`[0, 1]` per domain) | Measures exact verified task accuracy on held-out problem identities without averaging away domain failures. |
| **2. Reliability** | `error_samples` (count), `invalid_format_samples` (count), `mean_verifier_confidence` (`[0, 1]` or `null`) | Tracks runtime exceptions, malformed/unparseable answers (e.g., division by zero or probabilities outside `[0, 1]`), and internal verifier confidence. |
| **3. Latency** | `first_request_ms` (`ms`), `p50_ms` (`ms`), `p95_ms` (`ms`), `mean_ms` (`ms`) | Reports full end-to-end wall-clock generation + verification time in milliseconds, separating the first request from warm median/p95 latencies. |
| **4. Cost & Energy** | `total_cost_usd` (`USD` or `null`), `measured_energy_joules` (`J` or `null`) | Records billed API cost and hardware energy counters when available; unmeasured values remain explicitly `null` rather than silently reported as `0`. |
| **5. Resource Budget** | `physical_parameters` (count), `active_parameters_per_token` (count), `recurrent_steps` (count), `mean_ponder_depth` (steps), `cached_cartridge_bytes` (`bytes`) | Distinguishes physical stored parameters (`817,410` in `v11`), active per-token parameters under sparse top-k dispatch (`719,106` when 2 of 4 routed experts execute), and recurrent passes (`3`). |

---

## 2. Comparable Budget & Execution-Mode Separation (`EVAL-03` / Issue #42)

Every evaluation result and scorecard entry must declare an explicit `answer_source` from the controlled vocabulary:
- `neural_only`: Pure neural forward pass (`inflight_alu=False`), no external tools, no operand snapping, no arithmetic repair.
- `arithmetic_repaired`: Neural generation with in-flight symbolic ALU interception (`inflight_alu=True` in `evaluate_inflight_subexpr`), which repairs operands and computes arithmetic after `=`.
- `exact_solver`: Deterministic symbolic/rational solver (`solve_prompt` in `bayesian_tasks.py` or `ReferenceProvider` in `provider_adapters.py`).
- `retrieved`: Inference augmented by external codex or document lookup (`CodexRetriever`).
- `external_teacher`: Outputs produced or filtered by an external model (`aix-8b:latest`, `aix3:latest`, etc.).
- `heuristic_control`: Non-learned baselines (`majority`, `prior_only`, `constant`).

### Strict Comparison Gate (`assert_comparable_conditions`)
`assert_comparable_conditions(baseline, candidate)` in [`aix_zero/eval_harness_bridge.py`](../aix_zero/eval_harness_bridge.py) raises a `ValueError` if a model-improvement comparison:
1. Mixes different `answer_source` / `execution_mode` conditions (e.g., comparing `neural_only` baseline against `arithmetic_repaired` candidate without an explicit tool-ablation flag), or
2. Uses mismatched sampling/compute budgets (`num_candidates`, `max_new_tokens`, `device`).

---

## 3. Exact Rational & Probability-Validity Scoring (`EVAL-06` / Issue #46)

Implemented in [`aix_zero/eval_harness_bridge.py`](../aix_zero/eval_harness_bridge.py) (`score_rational_or_probability`), the exact rational scorer:
1. **Recognizes Mathematically Equivalent Rationals:** Parses integer and fraction outputs (`a/b`) using exact arbitrary-precision `fractions.Fraction`, so `"2/4"` and `"1/2"` receive identical exact score `1.0` when `require_reduced=False`, or flags unreduced fractions when `require_reduced=True`.
2. **Separately Flags Invalid Probabilities & Malformed Outputs:** Distinguishes:
   - `VALID_EXACT_MATCH` (`score = 1.0`),
   - `VALID_WRONG_VALUE` (`score = 0.0`, valid rational in `[0, 1]` for probability tasks but unequal to gold),
   - `OUT_OF_RANGE_PROBABILITY` (`score = 0.0`, rational `< 0` or `> 1` on probability tasks),
   - `ZERO_DENOMINATOR` (`score = 0.0`, e.g., `"1/0"`),
   - `IMPOSSIBLE_CONDITIONING_EVENT` (`score = 0.0`, conditioning on a zero-probability event),
   - `MALFORMED_FRACTION` (`score = 0.0`, empty or non-rational string).
3. **Stores Scorer Identity & Tolerances in Provenance:** Exports `rational_scorer_metadata()` directly into `evaluation-result-v1` `scorer` provenance.
