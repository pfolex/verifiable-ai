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

    def test_interrupted_initialization_and_event_gap(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'model'; source.write_bytes(b'best')
            registry = ExperimentRegistry(Path(d) / 'registry')
            metadata = dict(source_revision='rev', tokenizer_sha256='vocab', dataset_sha256='data', seed=42, hardware='cpu', settings={})
            attempt = registry.create(source, metadata)
            directory = registry.root / 'attempts' / attempt
            (directory / '000000.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                registry.inspect(attempt)
            self.assertEqual(registry.recover(attempt)['state'], 'proposed')
            registry.transition(attempt, 'proposed', 'running', 'resume')
            (directory / '000001.json').rename(directory / '000002.json')
            with self.assertRaisesRegex(ValueError, 'Non-contiguous'):
                registry.recover(attempt)

    def test_competing_publications_keep_exactly_one_event(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        with tempfile.TemporaryDirectory() as d:
            destination = Path(d) / 'event.json'
            barrier = Barrier(2)
            def publish(data):
                barrier.wait()
                try:
                    ExperimentRegistry._exclusive(destination, data)
                    return True
                except FileExistsError:
                    return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(publish, (b'first', b'second')))
            self.assertEqual(sum(results), 1)
            self.assertIn(destination.read_bytes(), (b'first', b'second'))
            self.assertEqual(list(Path(d).iterdir()), [destination])
