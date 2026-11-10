# M3 Expert-Library Release Readiness Audit (`EXPERT-16` / Issue #124)

## Audit Decision

- **Infrastructure Mechanics Status:** `VERIFIED` (Content-addressed `.safetensors` packaging, SHA-256 verification, duplicate-key rejection, temporary session rollback, bounded LRU hot-cache, and revocation/rollback tested)
- **Production Specialist Claim Decision:** `NO_GO_FOR_PRODUCTION_SPECIALIST_CLAIM_INFRASTRUCTURE_ONLY`
- **Programmatic Gate:** [`audit_m3_expert_library_release_readiness()`](../aix_zero/sparse_cartridge_bank.py)

## Evidence Summary

1. **Snapshot Cartridges vs. Promoted Specialists:**
   - The 21 cartridges in `aix_zero/cartridges/cartridge_manifest.json` carry `"provenance": "existing_routed_expert_snapshot"` (copies of existing `v11` routed experts `0..7` with `32,768` parameters each).
   - The frozen-backbone single-expert Bayesian training candidates (`BP`, `BS`) in [`research/bayesian-pilot/results.json`](../research/bayesian-pilot/results.json) achieved `0.0%` held-out exact match and remain `REJECTED` (`promoted_specialist_count = 0`).
2. **Dispatch & Latency Measurements ([`research/dispatch/local.json`](../research/dispatch/local.json)):**
   - Python-level sparse dispatch reduces standalone MoE microkernel time (`0.600ms -> 0.449ms`) while full-model forward time is `6.373ms` dense vs `6.949ms` sparse.
   - We explicitly prohibit advertising unlimited expert scaling or constant latency from file-count arithmetic.
