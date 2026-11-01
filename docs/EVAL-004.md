# EVAL-004: reproducible public dataset manifests

Acceptance: export schema-versioned public task manifests; hash their entire
identity, generation configuration and ordered cases; reject corrupt/invalid
manifests; rerun saved tasks; bind results to the manifest; keep reference solutions
private; verify round trips and failure paths locally and in CI.

```bash
python run_benchmark.py --manifest-only --export-manifest tasks.json \
  --seed 2026 --difficulty 1 --samples-per-domain 1
python run_benchmark.py --manifest tasks.json --provider reference --output baseline.json
python run_benchmark.py --manifest tasks.json --provider ollama --model aix3:latest \
  --output candidate.json --timeout 240
```

`--export-manifest` can also accompany a normal generated run. Saved manifests
cannot be combined with generation overrides, and input/output paths must differ.
The manifest schema is `schemas/dataset-manifest-v1.schema.json`. The digest is
SHA-256 of canonical JSON (sorted keys, compact separators, ASCII escaping),
excluding only the root `manifest_sha256` field. Array order matters. Every task
has an explicit public specification; private metadata and unknown fields are
rejected. A committed four-domain example is in `examples/dataset-manifest-v1.json`.

Reruns use the saved public tasks directly. The installed VeriSci revision must
match the saved scorer revision. Reference runs additionally require the original
RLVR-Forge revision and regenerate private solutions, verifying that the regenerated
public manifest exactly matches the saved one before evaluating. No solutions are
published or sent to live providers. Hashes detect modification, not authorship;
an actor who edits a manifest can also compute a new hash.

New CLI results store the complete manifest hash in `dataset.manifest_sha256`.
Older EVAL-001–003 artifacts remain valid, but their legacy case-list hashes cannot
be used as evidence of matching a new full manifest. Comparisons must fail closed
on mismatched provenance rather than guessing that two old/new hashes are equivalent.
The generic Python runner retains its old case-list hashing when no manifest is
provided. With a manifest, it checks cases, dataset and scorer identity before any
provider call. Manifest and result writes validate before atomic replacement.

Tests cover stable hashes, export/rerun equality across all domains, tampering,
duplicate IDs, hidden fields, invalid specs, input mismatches, source revision
mismatches, path collisions and preservation of existing files on failure. CI
exports a manifest, reruns it and uploads both public manifest and validated results.
