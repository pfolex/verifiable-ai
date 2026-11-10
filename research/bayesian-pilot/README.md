# Rejected Bayesian specialist pilot

This capsule records a local 300-step training run for each of two 49,152-parameter experts while freezing the original v11 base. These candidates are research results and are not activated by default.

| Task | Candidate exact answers | Prior-only control | Decision |
| --- | --- | --- | --- |
| One observation (BP) | 3/96 | 12/96 | REJECTED |
| Observation sequences (BS) | 1/144 | 26/144 | REJECTED |

The base and untrained forced-route controls scored zero on these held-out tasks. A separately evaluated exact rational solver scored 100%; that is not evidence of learned neural reasoning. Both restoration audits preserved all 105 original-domain generation traces: 9/105 neural-only and 105/105 assisted answers remained correct.

See [results.json](results.json), [dataset.json](dataset.json), and the candidate manifests under [candidates/](candidates/). Manifests bind the source, data, base checkpoint, exported weights, routing policy, and evaluation protocol. Candidate shards are isolated artifacts, not replacements for the original checkpoint.

Reproduce from the repository root with the dependencies in `requirements-aix-zero.txt`:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m aix_zero.train_bayesian_pilot --output-dir benchmark-results/bayesian-pilot --steps 300
```

This is a local reproduction, not an independent reproduction. The published test examples have been consumed; further model selection requires fresh final test data. Source hashes describe this publication stage; earlier baseline and dispatch capsules record their own stage-specific sources. See [the protocol](../../docs/AIX-ZERO-BAYESIAN-PILOT.md) for task splits, controls, and promotion gates.
