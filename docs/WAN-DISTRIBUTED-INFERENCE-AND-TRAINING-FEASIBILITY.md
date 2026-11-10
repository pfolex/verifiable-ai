# WAN Distributed Inference and Decentralized Training Feasibility Study (`NETWORK-14` #168, `NETWORK-15` #170)

## 1. WAN Distributed Inference Feasibility (`NETWORK-14` / Issue #168)

- **Programmatic Model:** [`evaluate_wan_distributed_inference_feasibility()`](../aix_zero/network_coordinator.py)
- **Key Finding:** Because `AIX-Zero v11` executes `3` recurrent passes per generated token (`n_recurrent_steps = 3`), splitting a single forward pass across WAN nodes introduces `>= 3` network round trips per token. At a modest `40ms` WAN RTT and `40` generated tokens, network latency alone adds `>= 4,800ms` per prompt compared to `~14ms` local CPU execution.
- **Scientific Honesty Rule:** Aggregate pooled memory across volunteer machines **never** implies interactive low-latency inference (`infers_interactive_speed_from_pooled_memory: false`). Interactive inference must run locally on a single device with local expert shards.

## 2. Decentralized Training Feasibility & Threat Model (`NETWORK-15` / Issue #170)

- **Programmatic Model:** [`evaluate_decentralized_training_feasibility()`](../aix_zero/network_coordinator.py)
- **Comparison Under Fixed Budget:**
  1. **Independent Bounded Jobs (`SPECIALIST_TRAINING_JOB` / `OFFLINE_EVALUATION_JOB`):** Each worker runs a self-contained seed/domain job and uploads a `.safetensors` candidate (`~131 KB`) + JSON report at completion. Communication overhead is `< 2%`, and candidates are verified on a reserved held-out partition.
  2. **Synchronous WAN All-Reduce Prototype:** Requires per-step gradient synchronization across heterogeneous peers, exposing the run to stragglers, stale gradients, and untrusted gradient poisoning (`SEC-12`).
- **Go / No-Go Decision:** Synchronous WAN gradient all-reduce is `NO_GO`. Only asynchronous, self-contained evaluation and single-expert candidate jobs are permitted, and large-scale volunteer training remains locked until `TRAIN-16` (`M2` held-out capability gain) passes.
