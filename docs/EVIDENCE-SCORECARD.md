# AIX evidence scorecard

Reviewed against repository revision `4f25de2f2e53136197ed5ee2d68b57207bfbc7d1` on November 18, 2026. The table summarizes historical committed artifacts. Current reproduction runs should retain their own identities and hardware measurements. Refresh the review date after checking new artifacts; this page has no live telemetry connection.

| Experiment | Result | Evidence and interpretation |
|---|---|---|
| v11 neural-only arithmetic | 9/105 correct, 8.57% | [Baseline JSON](../research/baseline/v11-local.json), generated November 8, 2026. Same-template held-out arithmetic. |
| v11 with arithmetic helper | 105/105 correct, 100% | Same baseline report. The helper repairs arithmetic, including operands and operators. |
| Frozen Bayesian probability specialist BP | 3/96 correct, rejected | [Pilot results](../research/bayesian-pilot/results.json). Repeated layouts share probability groups. |
| Frozen sequential Bayesian specialist BS | 1/144 correct, rejected | Same pilot report. Inspect controls and gate reasons before interpreting a candidate's score. |
| Two-cartridge storage demo | Run locally to check installation and corruption rejection | `python -m aix_zero.demo_cartridges`. Exported snapshots duplicate existing expert weights. |
| AIX/peer/collaborative programming comparison | Pending actual model submissions and evaluation | [Issue #270](https://github.com/pfolex/verifiable-ai/issues/270). |

## Baseline settings and cost

The committed v11 has 817,410 physical parameters and a 134-token vocabulary. Its checkpoint SHA-256 is `3af801152922cd27c40c8e349bdb68b6a10efb87445716c3262b85daf5cb44cb`.

The baseline used CPU, one PyTorch thread, seed 42, two candidates per request, temperature 0.35, and a maximum of 40 generated tokens per candidate. The JSON binds the dataset and source files by hash and records per-case traces, settings, versions, and hardware. The recorded machine was an Intel MacBook Air with 16 GB RAM.

| Mode | Median full response | p95 full response |
|---|---|---|
| Neural-only | 453.05 ms | 884.03 ms |
| Arithmetic helper | 235.59 ms | 363.70 ms |

These timings come from that machine and run. Peak memory and dollar cost are unavailable in this baseline artifact. Repeated templates and layouts affect sample independence; the table reports raw counts without a statistical superiority claim. The Bayesian pilot includes empirical promotion screens and restoration audits in its linked report.

## Reproduce and inspect

Follow [GETTING-STARTED.md](GETTING-STARTED.md) for the locked CPU environment, pinned BranchBox dependency, demo, and baseline command. The generated report contains the experiment's own provenance and response distributions. The pilot protocol and published artifacts are described in [AIX-ZERO-BAYESIAN-PILOT.md](AIX-ZERO-BAYESIAN-PILOT.md).

New claims should link immutable reports with their source and checkpoint identities. Record peer models, exact solvers, and other assistance in the execution settings. Training loss, verified reference answers, and exported shard counts each describe a different measurement; report task success from actual generated answers or executed patches. Include failed and inconclusive experiments in updates.
