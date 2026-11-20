"""Regression checks for reference leakage and solver-only capability reports."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aix_zero.benchmark_baseline import load_v11
from aix_zero.meta_ai_autopilot import (
    AutonomousSelfImprovementController,
    encode_complete_training_batch,
    build_meta_ai_self_improvement_curriculum,
    reflect_and_repair_rollout,
)


class EvaluationIntegrityTests(unittest.TestCase):
    def test_encoder_keeps_complete_targets_and_counts_overlength(self):
        class Tokenizer:
            def encode(self, text, add_bos=False, add_eos=False):
                return ([1] if add_bos else []) + [3] * len(text) + ([2] if add_eos else [])
        rows = [dict(prompt='ab', completion='c' * 80), dict(prompt='ab', completion='c' * 110)]
        ids, labels, weights, report = encode_complete_training_batch(rows, Tokenizer(), {2}, 112)
        self.assertEqual(report, dict(offered=2, encoded=1, skipped_overlength=1))
        self.assertEqual(ids.shape, (1, 112))
        self.assertEqual(labels[0, :3].tolist(), [0, 0, 0])
        self.assertEqual(labels[0, 83].item(), 2)
        self.assertEqual(weights[0, 83].item(), 3.)
        self.assertEqual(labels[0, 84:].sum().item(), 0)
        # Exactly-at-limit examples keep EOS, too.
        exact = [dict(prompt='ab', completion='c' * 108)]
        _, labels, _, _ = encode_complete_training_batch(exact, Tokenizer(), set(), 112)
        self.assertEqual(labels[0, -1].item(), 2)
        with self.assertRaisesRegex(ValueError, 'No complete examples'):
            encode_complete_training_batch(rows[1:], Tokenizer(), set(), 112)

    def test_unknown_prompt_cannot_copy_reference_into_repair(self):
        initial = '<think>unknown</think><ans>0</ans>'
        for gold in ('123456789', '987654321'):
            result = reflect_and_repair_rollout('unrecognized task', initial, gold)
            self.assertEqual(result['repaired_response'], initial)
            self.assertEqual(result['assistance_mode'], 'none')
            self.assertNotIn(gold, result['reflection_tag'])
            self.assertFalse(result['final_verified'])

    def test_known_prompt_repair_is_independent_of_reference(self):
        prompt = 'Q:MetaScale:C=144;N=4->D_opt?'
        first = reflect_and_repair_rollout(prompt, '<ans>999</ans>', '6')
        second = reflect_and_repair_rollout(prompt, '<ans>999</ans>', '123456789')
        self.assertEqual(first['repaired_response'], second['repaired_response'])
        self.assertTrue(first['final_verified'])
        self.assertFalse(second['final_verified'])
        self.assertEqual(first['assistance_mode'], 'symbolic_solver')

    def test_arena_scores_actual_generator_and_preserves_model_mode(self):
        root = Path(__file__).resolve().parents[1]
        model, tokenizer = load_v11(root / 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors')
        rows = build_meta_ai_self_improvement_curriculum()[:2]
        wrong = '<think>0=0</think><ans>99999999</ans>'
        calls = []

        def generate(prompt_ids, **kwargs):
            self.assertFalse(kwargs['inflight_alu'])
            self.assertNotIn('gold', kwargs)
            calls.append(prompt_ids)
            return prompt_ids + tokenizer.encode(wrong, add_bos=False), 0.0, 1

        with tempfile.TemporaryDirectory() as directory:
            controller = AutonomousSelfImprovementController(Path(directory))
            model.train()
            with patch('aix_zero.meta_ai_autopilot.build_meta_ai_self_improvement_curriculum', return_value=rows), patch.object(model, 'generate_verified', side_effect=generate):
                report = controller.run_frontier_arena_evaluation(model=model, tokenizer=tokenizer)
            self.assertTrue(model.training)
            self.assertEqual(len(calls), 2)
            self.assertEqual(report['overall_domain_accuracy'], 0.0)
            self.assertEqual(report['symbolic_control_accuracy'], 1.0)
            self.assertEqual(report['sprt_non_regression_status'], 'NOT_EVALUATED')
            self.assertEqual(report['evaluation_mode'], 'neural_only')
            self.assertEqual(report['swe_patch_source'], 'predefined_reference_fixtures')
            self.assertEqual(len(report['records']), 2)
            self.assertFalse(report['independent_final_test'])
            controller.request_stop()
            stopped = controller.run_frontier_arena_evaluation(model=model, tokenizer=tokenizer)
            self.assertIsNone(stopped['overall_domain_accuracy'])
            self.assertFalse(stopped['evaluation_complete'])


if __name__ == '__main__':
    unittest.main()
