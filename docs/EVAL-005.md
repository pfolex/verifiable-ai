# EVAL-005: baseline/candidate comparisons

Acceptance: pair all samples from results bound to one exact manifest, dataset and
scorer; report overall and per-domain scores, failures, tokens, latency and costs;
export versioned JSON plus readable Markdown; reject invalid or mismatched inputs.

```bash
python compare_benchmarks.py baseline.json candidate.json --manifest tasks.json \
  --output comparison.json --markdown comparison.md
```

Run both models with `run_benchmark.py --manifest tasks.json` first. Comparison
aligns by sample ID, not file order. IDs must cover the complete manifest exactly,
and dataset provenance and the entire scorer identity/configuration must match.
Legacy hashes cannot be silently substituted for full-manifest hashes. The report
includes run/model identities and one paired score/error record per sample.

Scores include error samples at their recorded score (the runner records zero).
The output includes per-domain and overall win/tie/loss counts, paired mean score
deltas, error counts, mean/median latency and nearest-rank p95. Token and cost
totals distinguish complete totals from known subtotals with coverage counts;
unknown usage is never presented as free or as zero tokens. Aggregations are
recomputed from samples rather than trusting summary totals.

`schemas/comparison-report-v1.schema.json` defines the JSON format. Validation also
checks pair uniqueness, score deltas, domain coverage, counts, means, errors and
win/tie/loss totals. Input failures occur before report writes; input/output path
collisions are rejected. JSON output is atomic. Markdown is a convenience report.
These are descriptive comparisons, not evidence that a measured gain generalizes.
Statistical decisions are provided separately by EVAL-006.

Tests exercise cross-domain metrics, reordered results, failures, unknown costs,
provenance/ID mismatches, JSON/Markdown CLI output and preservation on invalid input.
CI compares a reference run against its saved-manifest rerun and uploads reports.
