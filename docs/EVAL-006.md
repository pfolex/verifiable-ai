# EVAL-006: statistical improvement and regression gates

Acceptance: paired, bounded-score confidence intervals; explicit versioned policy;
minimum sample sizes; overall improvement and every-domain non-regression;
multiple-comparison control; conservative inconclusive behavior; error policy;
automation-friendly exit codes; tests for all decisions and hidden regressions.

```bash
python gate_benchmarks.py baseline.json candidate.json --manifest tasks.json \
  --policy examples/regression-policy-v1.json --output gate.json
```

The CLI reconstructs the comparison from authoritative result artifacts and the
validated manifest, rather than trusting an edited comparison report. Exit codes:
0 PASS, 1 FAIL, 2 INCONCLUSIVE, 3 invalid data or output failure. Argument syntax/path
errors also exit 2 through argparse; inspect stderr and the output report to
separate configuration mistakes from a valid inconclusive assessment. Invalid
input does not overwrite an existing gate artifact.

## Policy and decisions

The policy schema is `schemas/regression-policy-v1.schema.json`. It declares family
error probability `alpha`, minimum samples per domain, required overall mean score
improvement, allowed per-domain regression margin, maximum error rate and whether
the independent-sampling assumption has been declared. Both baseline and candidate
must meet the error-rate ceiling overall and in every domain. Infrastructure errors
must not create an apparent improvement by depressing the baseline.

The committed policy deliberately leaves `independent_samples` false. Use a
predeclared fixed sample plan and justify independent task pairs before setting it
true. Software cannot establish independence or representativeness from a hash.
Even an oracle smoke result must not silently become a statistical PASS.

A criterion passes when its confidence lower bound is at least its threshold,
fails when its upper bound is below it, and otherwise remains inconclusive. Overall
threshold is `min_improvement`; each domain threshold is negative
`non_regression_margin`. All criteria must pass for PASS. Any statistically resolved
failure or error-rate violation yields FAIL. Missing sampling assumptions or too few
samples prevent a statistical conclusion; error policy violations can still fail.

## Bounds and assumptions

For paired score differences X = candidate - baseline in [-1,1], Hoeffding's
bounded-variable inequality gives P(|mean(X)-E(X)| >= e) <= 2 exp(-n e^2/2).
Using m = 1 + number of domains and a union bound, the interval radius is
`sqrt(2 * log(2*m/alpha) / n)` for each criterion, clipped to [-1,1]. Thus coverage
is simultaneous across the overall and domain means without assuming those
criteria are independent. See [Hoeffding (1963), Probability Inequalities for Sums
of Bounded Random Variables](https://doi.org/10.1080/01621459.1963.10500830).

The assumption is independence of evaluated task pairs, not independence of the two
models within a pair. These distribution-free intervals are intentionally
conservative and can require many samples. They do not certify a biased, duplicated,
contaminated or correlated benchmark. Predeclare the model pair, manifest, policy
and sample count; repeatedly checking until a pass invalidates this fixed-horizon
coverage. There is no automatic stopping, adaptive sample collection or correction
across multiple candidate selection attempts. The existing EverBench SPRT showcase
is unchanged and is not substituted for these paired partial-score intervals.

Each report preserves model/run identities, manifest and comparison hashes, the
complete policy, all confidence bounds/thresholds and reasons for the decision.
Tests verify the formula, multiple-domain widening, large synthetic gains/losses,
hidden domain regressions, small samples, undecided equality, undeclared sampling,
invalid policies and CLI exit codes. CI requires the tiny reference/rerun smoke
comparison to remain INCONCLUSIVE rather than falsely declaring improvement.
