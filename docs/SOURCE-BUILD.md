# Build an AIX application from source

This build produces a portable Python application archive containing v11 inference source, the trained checkpoint, tokenizer, license, dependency declarations, and an integrity manifest. Use Python 3.11 with the pinned research dependencies. Platform dependencies such as PyTorch are installed in the destination environment.

From a checkout containing the build scripts, select a committed revision containing both scripts and run the builder. Local edits are excluded because the build reads Git objects.

```sh
python scripts/build_aix.py --revision HEAD --output /tmp/aix-v11.zip
mkdir /tmp/aix-v11
python -m zipfile -e /tmp/aix-v11.zip /tmp/aix-v11
cd /tmp/aix-v11
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-aix-zero.txt
.venv/bin/python aix.py --prompt 'Q:Word:2+3?'
```

Choose unused output paths. The builder refuses to overwrite an existing archive. The runtime verifies file hashes, loads the checkpoint through the strict loader, checks finite logits, and runs greedy neural generation offline. A successful smoke run can still produce an incorrect answer; the build validates execution and artifact integrity.

The build manifest records the source commit and SHA-256 for each packaged file. Two builds of the same revision produce identical archives using fixed ZIP metadata and uncompressed entries. Publish the archive digest through a trusted release channel. Integrity checks compare against the supplied manifest; they do not authenticate its publisher.

The v11 vocabulary supports a narrow set of characters and task forms. Unsupported characters and prompts exceeding the context budget fail explicitly. Use the existing evaluation harness for capability measurements.

The next milestones are platform-specific executables with bundled dependencies, a CI build matrix, and isolated candidate builds from AIX-generated patches with the same held-out benchmarks as the current release. Work is tracked in issue #287 and the programming comparison in #270.
