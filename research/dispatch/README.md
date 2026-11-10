# Clean publication dispatch study

`local.json` is the clean dispatch-branch reproduction. Source hashes identify this stage, before the separate pilot change. All expert weights remain resident.

| Mode | Dense correct | Sparse correct | Dense median | Sparse median |
| --- | ---: | ---: | ---: | ---: |
| neural_only | 9/105 | 9/105 | 483.61 ms | 487.38 ms |
| symbolic_assisted | 105/105 | 105/105 | 203.08 ms | 215.09 ms |

Both comparisons produced identical full traces for all 105 cases. Per-case seeds are paired between dense/sparse and differ from the aggregate storage-baseline RNG stream. The benchmark is narrow template-family arithmetic, not broad capability evidence.

Dense remains the default: these full-response results show no benefit from sparse dispatch for the current four-expert model. Synthetic 16/64-expert kernel experiments show larger gains in some workloads but are not trained model extensions or full-model speedup claims.

The JSON also includes fresh-process RSS and independently measured application-cache hits, misses, eviction, and installation. Files were OS-cached; these are not cold-disk results. Memory and latency limits are stated in `docs/AIX-ZERO-SPARSE-DISPATCH.md`.
