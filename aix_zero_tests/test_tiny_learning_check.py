import unittest

from aix_zero.benchmark_baseline import load_v11
from aix_zero.prepare_candidate_experiment import build_diagnostic_splits
from aix_zero.tiny_learning_check import (
    run_tiny_learning_check,
    verify_encoding_and_gradient_contract,
)
from pathlib import Path


class TinyLearningCheckTests(unittest.TestCase):
    def test_encoding_and_gradient_contract_and_tiny_learning_harness(self):
        base = Path(__file__).resolve().parents[1] / 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        model, tokenizer = load_v11(base)
        manifest = build_diagnostic_splits(tokenizer, model.config.max_seq_len, profile='single_skill_tiny')
        train_rows = [r['row'] for r in manifest['payload']['splits']['train']]
        contract = verify_encoding_and_gradient_contract(model, tokenizer, train_rows)
        self.assertTrue(contract['contract_verified'])
        self.assertGreater(contract['supervised_tokens_per_step'], 0)
        self.assertGreater(contract['nonzero_gradient_tensor_count'], 0)
        self.assertGreater(contract['changed_trainable_tensor_count'], 0)
        self.assertTrue(contract['frozen_parameters_unchanged'])

        # Run a fast 1-step check to exercise the full harness and verify champion preservation
        report = run_tiny_learning_check(
            manifest=manifest,
            seeds=(42,),
            steps=1,
            learning_rate=0.001,
            max_training_seconds=60.0,
            max_new_tokens=36,
        )
        self.assertTrue(report['champion_unchanged'])
        self.assertFalse(report['independent_final_test'])
        self.assertEqual(report['completed_seed_count'], 1)
        self.assertEqual(len(report['seed_runs'][0]['training']['loss_curve']), 2)
        self.assertIn(
            report['diagnostic_conclusion'],
            (
                'fitting_failed',
                'memorization_only_no_generalization',
                'target_gain_with_regression_loss',
                'fitted_and_generalized_without_regression',
            ),
        )


if __name__ == '__main__':
    unittest.main()
