from pathlib import Path
import tempfile
import unittest
from aix_zero.experiment_registry import ExperimentRegistry


class RegistryTests(unittest.TestCase):
    def test_snapshot_survives_source_changes_and_registry_restart(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'model'; source.write_bytes(b'best')
            registry = ExperimentRegistry(Path(d) / 'registry')
            metadata = dict(source_revision='revision', tokenizer_sha256='vocab', dataset_sha256='data', seed=42, hardware='cpu', settings={})
            attempt = registry.create(source, metadata)
            registry.transition(attempt, 'proposed', 'running', 'start')
            source.write_bytes(b'candidate')
            recovered = ExperimentRegistry(registry.root).inspect(attempt)
            self.assertEqual(recovered['state'], 'running')
            self.assertEqual(registry.verify(recovered['manifest']['champion_sha256']).read_bytes(), b'best')
            with self.assertRaises(ValueError):
                registry.transition(attempt, 'proposed', 'running', 'stale')
            with self.assertRaises(ValueError):
                registry.transition(attempt, 'running', 'evaluated', 'no evidence')
            evidence = Path(d) / 'score'; evidence.write_text('{"correct":0}')
            registry.transition(attempt, 'running', 'evaluated', 'complete', evidence)
            registry.transition(attempt, 'evaluated', 'rejected', 'no improvement')
            self.assertEqual(len(registry.inspect(attempt)['events']), 4)
            with self.assertRaises(ValueError):
                registry.transition(attempt, 'rejected', 'running', 'retry terminal')

    def test_corrupt_and_missing_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'model'; source.write_bytes(b'best')
            registry = ExperimentRegistry(Path(d) / 'registry')
            digest = registry.snapshot(source)
            artifact = registry.verify(digest)
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                registry.snapshot(source)
            artifact.unlink()
            with self.assertRaises(FileNotFoundError):
                registry.verify(digest)
            with self.assertRaises(ValueError):
                registry.verify('../escape')
