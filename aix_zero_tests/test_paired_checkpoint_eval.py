from pathlib import Path
import unittest
from aix_zero.experiment_splits import build_split_manifest
from aix_zero.paired_checkpoint_eval import evaluate_checkpoints


class CheckpointBoundaryTests(unittest.TestCase):
    def test_final_split_and_invalid_budget_rejected_before_loading(self):
        splits={name:[] for name in ('train','development','regression','final')}
        manifest=build_split_manifest(splits,lambda row:row['prompt'])
        for kwargs in ({'split':'final'},{'max_new_tokens':0},{'max_new_tokens':True}):
            with self.assertRaises(ValueError):
                evaluate_checkpoints(manifest,Path('missing'),Path('missing'),**kwargs)

    def test_empty_or_tampered_manifest_rejected_before_loading(self):
        splits={name:[] for name in ('train','development','regression','final')}
        manifest=build_split_manifest(splits,lambda row:row['prompt'])
        with self.assertRaisesRegex(ValueError,'empty'):
            evaluate_checkpoints(manifest,Path('missing'),Path('missing'))
        manifest['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'digest'):
            evaluate_checkpoints(manifest,Path('missing'),Path('missing'))
