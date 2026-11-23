import contextlib
import errno
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aix_zero.experiment_registry import ExperimentRegistry
from aix_zero.prepare_candidate_experiment import main as prepare_main, prepare_experiment_inputs
from aix_zero.run_candidate_experiment import main as run_main


class PrepareCandidateExperimentTests(unittest.TestCase):
    def test_prepare_and_run_with_spaces_in_path_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = Path(tmp) / 'dir with spaces' / 'exp bundle'
            with contextlib.redirect_stdout(io.StringIO()):
                bundle = prepare_main([
                    '--output-dir', str(bundle_dir),
                    '--profile', 'single_skill_tiny',
                    '--steps', '1',
                    '--learning-rate', '0.001',
                    '--max-training-seconds', '60',
                ])
            self.assertFalse(bundle['independent_final_test'])
            self.assertTrue(bundle['seeded_diagnostic_templates'])
            self.assertTrue((bundle_dir / 'champion.pt').is_file())
            self.assertTrue((bundle_dir / 'split_manifest.json').is_file())
            self.assertTrue((bundle_dir / 'policy.json').is_file())
            self.assertEqual(bundle['split_counts']['train'], 4)
            self.assertEqual(bundle['split_counts']['development'], 4)
            self.assertEqual(bundle['split_counts']['regression'], 4)

            # Second invocation on the same directory refuses accidental overwrite and identifies hashes
            with self.assertRaisesRegex(FileExistsError, 'champion.pt'):
                prepare_experiment_inputs(bundle_dir)

            # Execute the prepared bundle end-to-end through run_candidate_experiment
            run_args = bundle['run_argv'][3:]
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_main(run_args)
            self.assertIn(result['state'], ('evaluated', 'rejected'))
            registry = ExperimentRegistry(bundle['registry_path'])
            self.assertEqual(registry.inspect(result['attempt'])['state'], result['state'])

    def test_missing_checkpoint_and_mocked_permission_denied_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / 'path/to/nonexistent.pt'
            with self.assertRaisesRegex(FileNotFoundError, 'Missing source checkpoint'):
                prepare_experiment_inputs(root / 'out', source_checkpoint=missing)

            def deny_mkdir(*args, **kwargs):
                raise PermissionError(errno.EPERM, 'Operation not permitted', str(root / 'protected'))

            with patch.object(Path, 'mkdir', side_effect=deny_mkdir):
                with self.assertRaisesRegex(PermissionError, 'Access denied for output directory'):
                    prepare_experiment_inputs(root / 'protected')


if __name__ == '__main__':
    unittest.main()
