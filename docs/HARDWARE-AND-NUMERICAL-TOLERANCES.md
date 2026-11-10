# Supported Hardware, Execution Environments & Numerical Tolerances (`BASE-11` / Issue #34)

## 1. Tested CPU Reference Baseline vs. Experimental Accelerators

AIX (`verifiable-ai`) defines its canonical verification and reproducibility baseline on **single-threaded FP32 CPU execution** (`torch.float32`, `torch.set_num_threads(1)`).

| Execution Tier | Hardware / Backend | Status | Support & Verification Scope |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Canonical Reference)** | `x86_64` and `arm64` (`Apple Silicon`) CPU, Python 3.11 / 3.12, `torch==2.2.2`, `dtype=torch.float32`, `threads=1` | **Supported & CI-Tested** | Used by `.github/workflows/fishtest-ci.yml` (`ubuntu-latest`) and local macOS baselines (`research/baseline/v11-local.json`, `research/dispatch/local.json`). |
| **Tier 2 (Experimental Local Accelerators)** | Apple `mps` / MLX (`mlx-tree`), NVIDIA `cuda` (`float16` / `bfloat16`) | **Experimental Only** | Not required for any contributor workflow or CI gate. Numerics may diverge from CPU `float32` due to fused-multiply-add (FMA), reduced-precision accumulation, or non-deterministic atomic reductions. |
| **Unsupported Configurations** | Unpinned PyTorch `<2.0`, `float16` on CPU without native kernel support, multi-node asynchronous gradient updates without quorum verification | **Unsupported** | Excluded from baseline reproduction claims. |

---

## 2. Dtype-Specific Numerical Equivalence Tolerances

Cross-platform CPU BLAS libraries (e.g., Intel MKL on `x86_64` vs. Apple Accelerate on `arm64` vs. OpenBLAS on Linux runners) can reorder floating-point reductions at the last bit of precision (`~1e-7` in `float32`). Therefore, AIX **never promises bitwise logit equivalence across different CPU architectures or operating systems**. Instead, equivalence checks enforce explicit dtype-specific tolerances:

| Comparison Type | Dtype | Relative Tolerance (`rtol`) | Absolute Tolerance (`atol`) | Where Enforced |
| :--- | :--- | :--- | :--- | :--- |
| **Checkpoint & Shard Storage Round-Trip** | `torch.float32` (same host, save -> load) | `0.0` (exact tensor equality) | `0.0` (`torch.equal`) | [`aix_zero_tests/test_cartridge_bank.py`](../aix_zero_tests/test_cartridge_bank.py) |
| **Dense vs. Sparse Expert Dispatch (Forward & Gradients)** | `torch.float32` (same host & seed) | `1e-5` | `1e-6` | [`aix_zero_tests/test_sparse_dispatch.py`](../aix_zero_tests/test_sparse_dispatch.py) |
| **Evaluation Summary Mean Score & Cost Aggregation** | IEEE-754 `float64` (Python `float`) | `1e-9` | `1e-12` | [`evaluation_result.py`](../evaluation_result.py), [`comparison_report.py`](../comparison_report.py) |
| **Cross-Platform Forward Logits (`x86_64` vs. `arm64` / Linux vs. macOS)** | `torch.float32` | `1e-4` | `1e-5` | [`aix_zero/checkpoint_registry.py`](../aix_zero/checkpoint_registry.py) |
| **Experimental Reduced Precision (`bfloat16` / `float16`)** | `torch.bfloat16` / `torch.float16` | `1e-2` | `1e-2` | Experimental accelerator studies only (`SCALE-05`) |

---

## 3. Greedy Decoding and Argmax Tie-Breaking Caveat

Even when `float32` logits agree within `atol=1e-5` across two different CPU architectures (for example, an Intel `x86_64` laptop vs. an Ubuntu GitHub Actions runner), a near-tied `argmax` at a single decoding step can flip one token and change downstream autoregressive tokens. Consequently:
- Every baseline report (`research/baseline/v11-local.json`) records the exact `hardware` dictionary (`platform`, `machine`, `processor`, CPU model) and `versions` (`python`, `torch`, `numpy`, `safetensors`).
- Exact trace reproduction is guaranteed on the **same platform/BLAS stack** with `threads=1` and fixed seed (`42`), while cross-platform reproductions compare verified task accuracy and confidence intervals rather than assuming bitwise-identical raw logit strings.
