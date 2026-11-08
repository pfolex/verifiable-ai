# Selected-token expert dispatch (stage 3)

The model now offers an opt-in sparse execution path while retaining the original
dense reference as the default. Select the execution policy at runtime:

```python
from pathlib import Path
from aix_zero.benchmark_baseline import load_v11

model, tokenizer = load_v11(Path('aix_zero/checkpoints/aix_zero_v11_kip.safetensors'))
model.set_expert_dispatch('sparse')
# Run normal model.forward(...) or generate_verified(...).
model.set_expert_dispatch('dense')
```

This changes execution only. Checkpoint keys, parameter shapes, expert weights,
router probabilities, top-k selection, shared expert, balancing loss, and model
configuration stay compatible. Every expert remains resident in RAM. There is no
automatic disk routing, device migration, specialist training, or v12 reasoning
addition in this stage.

## How it works

The sparse branch flattens batch/sequence positions, gathers the tokens assigned
to each expert, evaluates that expert only for those tokens, then scatters outputs
into their original top-k positions. The weighted sum follows the same top-k order
as the reference. An expert with no assignments is not called. Runtime work still
includes scanning the expert list; this is not a constant-time million-expert router.
Smaller matrix multiplications and gather/scatter overhead can outweigh saved
computation at small expert counts.

Training retains the dense implementation's optimizer behavior for idle experts:
their gradients are explicit zero tensors instead of missing gradients, preserving
AdamW decay/state updates. This allocates gradients for idle experts and is not a
sparse optimizer or training-memory optimization. CPU floating-point kernels can
produce small rounding differences because matrix batch shapes change. Tests use
explicit tolerances rather than promising bitwise equality.

## Reproduce experiments

From the repository root, with the Python 3.11 environment described in
`AIX-ZERO-EXPERT-STORAGE.md`:

```sh
.venv-training/bin/python -m unittest discover -s aix_zero_tests -v
.venv-training/bin/python -m aix_zero.benchmark_dispatch \
  --output benchmark-results/stage3-dispatch.json
```

The benchmark covers:

- Paired, alternating-order CPU F32 kernel timings with 4, 16, and 64 resident
  experts, top-2 routing, and inputs of (batch, sequence) = (1,1), (1,32), (4,112).
  Each mode receives three excluded warmups and 15 timed samples by default.
- Fresh-process peak RSS measurements for each mode/expert count using (4,112).
  RSS includes interpreter/framework overhead and all resident expert parameters;
  differences are approximate, not an allocator-level peak-memory guarantee.
- Snapshot loading with empty application caches, verified cache hits, repeated
  eviction over three experts in a two-entry cache, and installation with eviction.
  Hash validation reads the file even on tensor-cache hits. Files are OS-cached;
  application-cache misses are not cold-disk measurements. Stored files and
  manifests are created in temporary directories and removed after each experiment.
- Paired v11 generation over all 105 existing held-out template-family cases,
  both neural-only and symbolically assisted. Each case uses a common seed in both
  dispatch modes, two candidates, and a 40-token cap. Order alternates between
  modes. Per-case seeds differ from the earlier aggregate baseline's RNG stream;
  compare dense versus sparse within this paired run. All case timings, including
  the first case, are retained in these response distributions.

The 16/64-expert models are synthetic performance fixtures, not trained capacity
extensions. Results do not establish broader reasoning ability, automatic disk
paging throughput, or an inference-latency guarantee. Reports include checkpoint
and source hashes, hardware, settings, numerical differences, per-case traces,
load-cache hit/miss counts, and bounded-cache usage.

Dedicated tests check float64 output/input/parameter gradient equivalence across
expert counts and top-k choices, shared-expert on/off, noncontiguous inputs,
selected-token execution counts, AdamW updates for idle experts, and real v11
forward/greedy-generation equivalence. The existing cartridge tests also run under
the same CI test-discovery step.
