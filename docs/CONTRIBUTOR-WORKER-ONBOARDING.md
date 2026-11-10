# Contributor Worker Onboarding and Troubleshooting Guide (`WORKER-15` / Issue #146)

## Voluntary Opt-In Principle

- Participating in `verifiable-ai` evaluation or specialist-training jobs is **100% voluntary and opt-in**.
- You can complete a full local evaluation on your own machine without registering a public account or waiting for maintainer intervention.

## Step-by-Step Self-Contained Local Evaluation

Run the deterministic `v11` baseline evaluation and test suite on CPU (`threads=1`):

```bash
python -m pytest -q aix_zero_tests/test_published_evidence.py aix_zero_tests/test_eval_harness_bridge.py
```

## Enforced vs. Advisory Resource Limits (`WORKER-03` / `SEC-02`)

| Control | Enforcement Level | Platform Details |
| :--- | :--- | :--- |
| **CPU Threads (`max_cpu_threads`)** | `HARD_ENFORCED_TORCH_THREADPOOL` | Calls `torch.set_num_threads()` and rejects jobs requesting more threads. |
| **Memory Cap (`max_memory_mb`)** | `HARD_PREFLIGHT_GATE` + OS-dependent | Preflight check blocks oversized jobs; Linux supports `RLIMIT_AS`, whereas macOS RSS monitoring is advisory unless run inside a container. |
| **Disk Quota (`quota_bytes`)** | `HARD_PREFLIGHT_GATE` | Confined to dedicated `worker_dir`; never touches files outside `worker_dir`. |
| **Pause / Stop (`paused=True`)** | `IMMEDIATE_LOCAL_CONTROL` | Prevents new leases and relinquishes or aborts active work immediately. |

## Troubleshooting Common Issues

1. **`PROBE_FAILED_DO_NOT_LEASE`:** Accelerator GEMM finiteness probe failed or requested GPU backend is unavailable; run with `--device cpu`.
2. **`STOPPED_DISK_QUOTA_EXCEEDED`:** Dedicated worker directory reached `quota_bytes`; completed `.scratch` files are reclaimed automatically while pending uploads are preserved.
3. **`REJECTED_CORRUPT_SHA256_MISMATCH`:** Downloaded input shard failed content-addressed SHA-256 verification before execution.
