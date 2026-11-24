# Pause, verify and publish an AIX runtime checkpoint

This command packages the currently selected daemon model and its runtime
dependencies as a versioned GitHub Release artifact. It does not commit changing
weights to Git, change an independently evaluated champion, or certify intelligence.

## Commands

From your repository root, with the AIX environment activated:

~~~sh
python -m aix_zero.aix_ctl checkpoint
~~~

This requests a pause at the end of the current cycle, waits for its writer lock,
creates and verifies a bundle under ./checkpoint-releases, and leaves AIX paused.
The existing immediate stop command keeps its shutdown behavior.

To create, publish, verify by downloading, and resume the previously running daemon:

~~~sh
python -m aix_zero.aix_ctl checkpoint \
  --publish --repo pfolex/verifiable-ai --resume
~~~

If your aix alias forwards to aix_zero.aix_ctl, the short form is:

~~~sh
aix checkpoint --publish --repo pfolex/verifiable-ai --resume
~~~

Publication requires an installed authenticated GitHub CLI with permission to
create releases in the explicitly selected repository. Use --gh /accessible/path/gh
if gh is not on PATH. No downloads, release creation or training happens on import.

A daemon started before this update cannot honor the boundary protocol. The
command refuses to snapshot it while it is alive. Stop that older daemon manually,
update and restart the code, then use the new checkpoint command. A stopped legacy
snapshot is labeled legacy_stopped_boundary_unconfirmed; its exact cycle boundary
cannot be reconstructed from a saved status flag alone.

## What is preserved

- The selected aix_zero_v12_evolving.pt backbone, including embedded physical LoRA
  architecture/adapter weights when present, and other top-level .pt/.safetensors
  model files.
- Top-level runtime JSON: configuration/ledgers, evaluation evidence, subject
  registry and episodic retrieval memory. This runtime data will be included in the
  artifact you explicitly publish.
- The registered cartridge manifest and every shard it references, with existing
  bank hash/schema/finite-tensor verification. Historical unreferenced shards are
  omitted. A bank explicitly recorded by the cycle guard may be outside state and
  is normalized into the portable bundle.
- Ordered tokenizer identity/vocabulary, actual physical parameter accounting,
  architecture, loader baseline and pinned tracked source/config archive.
- Source revision, cycle count, file sizes/SHA256 hashes, launch settings and
  recorded evaluation outcomes. Source changes during a daemon lifetime require
  restart before publication.

External adapter/config files inside the state directory can be named with
repeated --extra-state relative/path flags. Unregistered external dependencies
are not guessed. Independent candidate scheduler workers must be stopped first;
their experiment registries are not the legacy daemon's inference state and are
not included. Symlinks and path escapes are refused.

The bundle is a restorable research artifact, not a native standalone executable.
Use its pinned source revision and recorded architecture with the matching
dependencies. The included v11 baseline binds reconstruction of legacy/physical
checkpoints. Restoring a different architecture needs its own explicit loader.

## Verification and latest versus best

Each unique cycle/timestamp/random tag contains bundle/, the ZIP, checksums.json
and, after publication, publication.json. Every selected file is copied under the
cycle lock and hash checked before/after copy. The model is reloaded, the tokenizer
checked, and serialized ZIP contents verified. The default total budget is 512 MiB
(--max-bytes changes it); --timeout bounds waiting for the cycle lock.

~~~sh
python -m aix_zero.aix_ctl checkpoint-verify \
  --bundle ./checkpoint-releases/TAG/bundle
~~~

Replace TAG with the tag printed by the checkpoint command. Verification checks
artifact integrity and model compatibility; it does not establish generalization
or prove the historical training data were uncontaminated.

latest.json identifies the latest local snapshot; latest_published.json is updated
only after successful publication. Existing best/champion pointers are untouched.
Releases explicitly do not take GitHub's generic Latest designation.

The release is created as a draft with the ZIP and checksums attached. Both assets
are downloaded and compared byte-for-byte by SHA256 before the draft is published.
These are separate GitHub API operations: the download check proves what was
retrieved, not platform-enforced immutability against a later authorized edit.
Use repository release immutability settings if that guarantee is required.
No existing release assets are overwritten.

## Failure and explicit recovery

Copy, compatibility, budget, permission or publication failures retain the pause.
A failed upload/download may leave a draft release for inspection. Automatic resume
never follows a failed operation. Interrupted or failed cycles are refused as
completed snapshots; investigate recovery before restarting.

After inspecting the failure, resume the previously running daemon with its
recorded remaining cycle budget and flags:

~~~sh
python -m aix_zero.aix_ctl checkpoint-resume
~~~

Or only clear the pause, leaving restart to you:

~~~sh
python -m aix_zero.aix_ctl checkpoint-resume --clear-only
~~~

--resume restarts only a daemon recorded as running before the pause. A previously
stopped model stays stopped. The old PID must exit before automatic restart.
If a restart fails the pause is restored. Publisher and cycle locks are advisory
POSIX locks, released on process exit; their files are never deleted as a shortcut.

All state writers must use the boundary protocol. The cycle and subject-discovery/
expertise mutation entry points are guarded; do not run independent legacy scripts
that bypass them concurrently with a snapshot.
