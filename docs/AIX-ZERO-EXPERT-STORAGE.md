# Offline v11 baseline and expert snapshot storage

## Reproduce (Locked Minimal CPU Research Environment — BASE-02)

Use Python 3.11 (or 3.12) on CPU and install the pinned research dependencies in an isolated environment. No GPU, web search, council call, or paid API credential is required:

```sh
python3.11 -m venv .venv-training
.venv-training/bin/python -m pip install -r requirements-aix-zero.txt
# Import-only dependency used by the existing curriculum module:
git clone https://github.com/pfolex/branchbox.git branchbox
git -C branchbox checkout 49873a0662ad13de7943619bf8fd822d3d3458a9
# Run offline unit tests and environment lock checks:
.venv-training/bin/python -m unittest discover -s aix_zero_tests -v
.venv-training/bin/python -m unittest discover -s tests -v
# Run the full offline v11 baseline and cartridge export:
.venv-training/bin/python -m aix_zero.benchmark_baseline \
  --output benchmark-results/v11-baseline.json \
  --export-dir benchmark-results/v11-cartridges
```

Pinned core dependency versions in `requirements-aix-zero.txt`:
- `torch==2.2.2`
- `numpy==1.26.4`
- `safetensors==0.8.0`
- `jsonschema[format]>=4.18,<5`

The existing curriculum imports BranchBox (`49873a0662ad13de7943619bf8fd822d3d3458a9`); the offline baseline itself makes no web, council, or hosted-provider calls. Existing checkpoint files are not overwritten. The GitHub Actions research job (`aix-zero-research` in `.github/workflows/fishtest-ci.yml`) runs the storage contracts and full offline baseline on a clean runner and uploads its own report.

The benchmark strictly loads v11 without randomly expanding embeddings. It records checkpoint/configuration/source identity, runtime versions, hardware, seed, dataset digest, per-case predictions, and per-domain results. Public checkpoint paths are repository-relative or a filename, not a contributor's home directory.

Default settings are CPU, one PyTorch thread, seed 42, two candidates, at most 40 generated tokens per candidate, and 105 held-out combinations from the existing template families. Neural-only and symbolic-assisted modes are separate. The helper can repair operands/operators as well as calculate arithmetic, so assisted accuracy is not neural-only reasoning evidence.

One excluded warmup follows the separately reported first request in each mode. Timing is full-response generation plus final verification. First application requests are not guaranteed OS-cache-cold. Repeated cases are not new independent accuracy samples. CPU timings on one Intel Mac do not establish a latency guarantee.

`research/baseline/v11-local.json` records the clean publication-branch run. Historical local runs scored 9/105 neural-only and 105/105 assisted, with varying response times. The report pins the actual settings and source used for its measurements.

## Snapshot interface

```python
from pathlib import Path
from aix_zero.benchmark_baseline import load_v11
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank

model, tokenizer = load_v11(Path('aix_zero/checkpoints/aix_zero_v11_kip.safetensors'))
bank = SparseExpertCartridgeBank(Path('benchmark-results/v11-cartridges'),
                                max_cached_bytes=1024 * 1024,
                                max_cached_experts=2)
mapping = {f'SOURCE_EXPERT_{i}': i for i in range(model.config.n_experts)}
manifest = bank.export_all_domain_cartridges(
    model, {name: {} for name in mapping}, {}, domain_expert_map=mapping)
bank.install_cartridge(model, 'SOURCE_EXPERT_0')
```

Each v11 routed expert has 49,152 parameters (196,608 raw F32 bytes). Export preserves its exact tensors. There is no shared-expert blending, generated routing key, or claim of independent domain specialization. Source-slot assignments must be explicit. Shard accuracy is unknown and the statistical gate is `NOT_EVALUATED`; optional base-model accuracy is a separate field.

The version-1 manifest records complete SHA-256 digests, tensor keys/shapes/dtypes, model configuration, sizes, and source slots. Content-addressed shards are atomically saved before the manifest is published. Failed publication leaves the old manifest and its referenced files usable. Unreferenced old shards are retained; garbage collection and concurrent writers are unsupported.

Loading verifies path containment, size, full hash (including cache hits), tensor schema, and finite values. Installation also checks full configuration and target tensor keys/shapes/dtypes, stages device transfers before copying, and preserves Parameter objects. The LRU cache bounds both retained CPU tensor bytes and entry count. A shard too large for the cache can be served without retention if it satisfies the separate serialized-size limit.

`page_in_cartridge()` loads/cache-checks without changing model weights. `install_cartridge()` explicitly copies into an existing slot. Normal requests do not page or select experts automatically. The model must be in eval mode, and callers must serialize installation with inference/training: the bank lock does not protect arbitrary external model access.

## Boundaries

Cache budgets exclude resident model weights, deserialization buffers, transfer staging, framework allocations, and OS caches. Hashes detect corruption relative to a manifest but do not authenticate an untrusted manifest. Matching configuration is not proof of a trained specialist's base-weight identity. This release is experimental storage of existing snapshots, not a production registry or automatic offloader.

The REPL exposes `!expert <domain> [slot]` and cache status for its default `aix_zero/checkpoints/cartridges` directory. Importing the REPL does not create that directory. A vocabulary-expanded model requires configuration-compatible snapshots. `!save` retains its legacy behavior of writing the active checkpoint; the offline baseline never invokes it. Use a disposable checkpoint for REPL teaching/swap experiments.

Opt-in sparse execution is documented in `AIX-ZERO-SPARSE-DISPATCH.md`; it changes computation, not expert weight residency. The independently trained Bayesian pilot is a separate follow-up change. The unreviewed local v12 arithmetic prototype, extra tokenizer tokens, generated codex claims, and unrelated execution report are excluded from this baseline publication.
