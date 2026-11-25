# Try AIX on CPU

AIX is an experimental toolkit for evaluating models and researching verifiable learning. This walkthrough uses the committed 817,410-parameter v11 checkpoint and snapshots of two existing experts. It exercises disk loading, restores the base output, and rejects a corrupted shard.

Use Python 3.11 or 3.12 on Linux or macOS with the pinned CPU research dependencies. CI exercises Python 3.11 on Linux. The historical baseline ran on an Intel MacBook Air with 16 GB RAM. Minimum memory and disk requirements have yet to be measured. Allow space for the Python environment and PyTorch installation. Installation requires network access. The demo runs offline afterward.

```sh
git clone https://github.com/pfolex/verifiable-ai.git
cd verifiable-ai
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-aix-zero.txt
.venv/bin/python scripts/aix_doctor.py
.venv/bin/python -m aix_zero.demo_cartridges
```

Expected output includes both domain names, their slots, a rejected corrupted cartridge, and a JSON summary with `base_restored: true`. Temporary shards are removed when the demo exits. Learned specialist gains require a separate evaluation.

## Reproduce the published arithmetic baseline

The full curriculum imports BranchBox. Use the same pinned revision as the research CI.

```sh
git clone https://github.com/pfolex/branchbox.git branchbox
git -C branchbox checkout 49873a0662ad13de7943619bf8fd822d3d3458a9
.venv/bin/python -m unittest discover -s aix_zero_tests -v
.venv/bin/python -m aix_zero.benchmark_baseline \
  --output benchmark-results/my-v11-baseline.json
```

Run these commands from the repository root, where `aix_zero/` and `requirements-aix-zero.txt` are visible. Compare the output with [the evidence scorecard](EVIDENCE-SCORECARD.md) and retain the generated hardware, dataset, source, and checkpoint identities.

## Resolve setup errors

| Symptom | Action |
|---|---|
| Python cannot import `aix_zero` | Change to the cloned repository root and use that checkout's virtual environment. |
| Python cannot import `branchbox` | Clone BranchBox inside this checkout and check out the revision above. |
| Missing research dependencies | Install `requirements-aix-zero.txt` with the same Python used to run the command. |
| Missing checkpoint | Restore the committed file under `aix_zero/checkpoints/` and rerun the doctor. |
| Permission denied under Documents | Move or clone the repository to a directory accessible to your terminal. |
| Ollama connection fails | The demo and baseline work offline. Live peer evaluation requires a reachable service and installed model, as described in [EVAL-003](EVAL-003.md). |

For the four-domain evaluation suite, follow the dependency instructions in the root README. For a first contribution, use [CONTRIBUTING.md](../CONTRIBUTING.md) and the tasks linked from the [sponsorship roadmap](SPONSORSHIP-ROADMAP.md).
