# AIX-Zero measured research baseline

AIX-Zero is a small experimental PyTorch neuro-symbolic model. The v11 checkpoint contains **817,410 physical parameters**, 134 vocabulary entries, grouped-query attention, three recurrent passes, a shared expert and four routed experts, prompt-pointer attention, learned memory slots, and a learned verification head. Recurrent computation does not create additional learned weights.

On the existing 105 template-family held-out cases, the local baseline scored **9/105 neural-only** and **105/105 with the in-flight arithmetic helper**. That helper can repair operands/operators and calculate results. These scores are not broad reasoning or frontier-model benchmarks. Full-response timing is hardware-dependent; no <25 ms guarantee is established.

The research baseline adds integrity-checked export/reload of existing routed expert snapshots, bounded CPU tensor caching, and explicit installation. Snapshots are not independently trained specialists and do not add learned capacity. Default MoE execution still computes every expert; automatic routing/offloading is not implemented here.

See [setup, measurements, storage contracts, and reproduction](../docs/AIX-ZERO-EXPERT-STORAGE.md). Source checkpoint files are not changed by the offline baseline command. Legacy training, web/council, and generated-codex programs remain separate experimental tools; generated claims are not evaluation evidence.

Opt-in selected-token dispatch and dense-reference comparisons are documented in [the dispatch study](../docs/AIX-ZERO-SPARSE-DISPATCH.md). All expert weights remain resident. Dense remains the default because the current four-expert model showed no full-response benefit.
