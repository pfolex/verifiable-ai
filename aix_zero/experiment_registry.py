"""Immutable artifacts and append-only per-attempt events for autonomous research.

Promotion is deliberately a separate policy: this registry never changes a champion.
"""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import uuid


class ExperimentRegistry:
    transitions = {
        'proposed': {'running', 'failed'},
        'running': {'evaluated', 'failed'},
        'evaluated': {'promoted', 'rejected', 'failed'},
        'promoted': set(), 'rejected': set(), 'failed': set(),
    }

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def digest(data):
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _exclusive(path, data):
        # Write privately, then atomically publish without replacing an existing event.
        fd, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    def snapshot(self, source):
        data = Path(source).read_bytes()
        digest = self.digest(data)
        directory = self.root / 'artifacts'
        directory.mkdir(exist_ok=True)
        destination = directory / digest
        try:
            self._exclusive(destination, data)
        except FileExistsError:
            pass
        self.verify(digest)
        return digest

    def verify(self, digest):
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid artifact digest')
        path = self.root / 'artifacts' / digest
        if self.digest(path.read_bytes()) != digest:
            raise ValueError('Corrupt artifact')
        return path

    def create(self, champion_source, metadata):
        required = {'source_revision', 'tokenizer_sha256', 'dataset_sha256', 'seed', 'hardware', 'settings'}
        if not required.issubset(metadata):
            raise ValueError('Missing reproducibility metadata')
        # Validate JSON before creating an attempt directory.
        encoded_metadata = json.loads(json.dumps(metadata, allow_nan=False))
        champion = self.snapshot(champion_source)
        attempt = uuid.uuid4().hex
        directory = self.root / 'attempts' / attempt
        directory.mkdir(parents=True)
        self._exclusive(directory / 'manifest.json', json.dumps(dict(
            attempt=attempt, champion_sha256=champion, metadata=encoded_metadata,
        ), sort_keys=True, allow_nan=False).encode())
        self._exclusive(directory / '000000.json', b'{"state":"proposed","reason":"created"}')
        return attempt

    def inspect(self, attempt):
        if len(attempt) != 32 or any(c not in '0123456789abcdef' for c in attempt):
            raise ValueError('Invalid attempt identity')
        directory = self.root / 'attempts' / attempt
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.verify(manifest['champion_sha256'])
        events = [json.loads(p.read_text()) for p in sorted(directory.glob('[0-9]' * 6 + '.json'))]
        if not events:
            raise ValueError('Incomplete attempt')
        for event in events:
            if event.get('artifact_sha256'):
                self.verify(event['artifact_sha256'])
        return dict(manifest=manifest, events=events, state=events[-1]['state'])

    def transition(self, attempt, expected_state, state, reason, artifact=None):
        current = self.inspect(attempt)
        if current['state'] != expected_state or state not in self.transitions[expected_state]:
            raise ValueError('Invalid or stale transition')
        if not reason.strip():
            raise ValueError('Decision reason is required')
        if state == 'evaluated' and artifact is None:
            raise ValueError('Evaluation evidence is required')
        event = dict(state=state, reason=reason)
        if artifact is not None:
            event['artifact_sha256'] = self.snapshot(artifact)
        path = self.root / 'attempts' / attempt / f"{len(current['events']):06d}.json"
        self._exclusive(path, json.dumps(event, sort_keys=True).encode())
        return self.inspect(attempt)
