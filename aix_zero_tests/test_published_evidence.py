"""Validate committed measurements and rejected artifacts, without claiming independent reproduction."""
import hashlib
import json
from pathlib import Path
import unittest

from safetensors.torch import load_file

from aix_zero.bayesian_tasks import empirical_gate, solve_prompt
from aix_zero.train_bayesian_pilot import tensor_digest

ROOT = Path(__file__).resolve().parents[1]


class PublishedEvidenceTests(unittest.TestCase):
    def test_baseline_report_matches_checkpoint_and_recorded_answers(self):
        report = json.loads((ROOT / 'research/baseline/v11-local.json').read_text())
        checkpoint = ROOT / report['checkpoint']
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), report['checkpoint_sha256'])
        self.assertEqual(report['physical_parameters'], 817410)
        for mode in report['modes'].values():
            self.assertEqual(mode['total'], len(mode['samples']))
            self.assertEqual(mode['correct'], sum(row['prediction'] == row['gold'] for row in mode['samples']))

    def test_dispatch_report_pairing_and_cache_bounds(self):
        report = json.loads((ROOT / 'research/dispatch/local.json').read_text())
        for evaluation in report['v11_accuracy'].values():
            rows = evaluation['samples']
            self.assertEqual(evaluation['total'], len(rows))
            self.assertEqual(evaluation['same_traces'], sum(
                row['modes']['dense']['trace'] == row['modes']['sparse']['trace'] for row in rows))
            for mode in ('dense', 'sparse'):
                self.assertEqual(evaluation['modes'][mode]['correct'], sum(
                    row['modes'][mode]['prediction'] == row['gold'] for row in rows))
        for bank in report['storage']:
            for scenario in bank['scenarios'].values():
                self.assertEqual(scenario['samples'], scenario['hits'] + scenario['misses'])
                cache = scenario['final_cache']
                self.assertLessEqual(cache['entries'], cache['max_experts'])
                self.assertLessEqual(cache['bytes'], cache['max_bytes'])

    def test_pilot_capsule_hashes_oracles_decisions_and_rejected_weights(self):
        root = ROOT / 'research/bayesian-pilot'
        report = json.loads((root / 'results.json').read_text())
        data_bytes = (root / 'dataset.json').read_bytes()
        self.assertEqual(hashlib.sha256(data_bytes).hexdigest(), report['dataset_sha256'])
        dataset = json.loads(data_bytes)
        self.assertTrue(report['checkpoint_unchanged'])
        for task, result in report['tasks'].items():
            for split in ('train', 'validation', 'test'):
                for row in dataset['tasks'][task][split]:
                    self.assertEqual(solve_prompt(row['prompt']), row['gold'])
            manifest_path = root / result['candidate_manifest']
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest['dataset_sha256'], report['dataset_sha256'])
            self.assertEqual(manifest['training_base_checkpoint_sha256'], report['checkpoint_sha256'])
            entry = manifest['cartridges'][task]
            shard = manifest_path.parent / entry['cartridge_file']
            self.assertEqual(hashlib.sha256(shard.read_bytes()).hexdigest(), entry['sha256'])
            self.assertEqual(tensor_digest(load_file(str(shard))), result['training']['selected_expert_sha256'])
            self.assertTrue(result['training']['frozen_tensors_unchanged'])
            self.assertEqual(result['fallback_audit']['changed_traces'], 0)
            self.assertEqual(result['promotion_status'], 'REJECTED')
            replay = empirical_gate(result['candidate'], result['controls'],
                test_groups=len({row['group'] for row in dataset['tasks'][task]['test']}),
                fallback_changes=result['fallback_audit']['changed_traces'])
            self.assertEqual(replay, result['gate'])
            self.assertEqual(entry['empirical_gate'], replay)
            self.assertEqual(entry['sprt_gate'], 'NOT_EVALUATED')


if __name__ == '__main__':
    unittest.main()
