"""Exact probability oracles, leakage controls, frozen training, and reversible routing."""
import copy
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.bayesian_tasks import (build_dataset, empirical_gate, parse_prompt,
    reference_joint, render, score_answer, solve_prompt)
from aix_zero.benchmark_baseline import load_v11
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
from aix_zero.train_bayesian_pilot import (explicit_route, make_batch,
    temporary_specialist, tensor_digest, train_expert, evaluate)


class ProbabilityTests(unittest.TestCase):
    def test_exact_update_complements_independent_sequence_and_boundaries(self):
        self.assertEqual(reference_joint(Fraction(1, 2), Fraction(3, 4), Fraction(1, 4), (1,)), Fraction(3, 4))
        self.assertEqual(reference_joint(Fraction(1, 2), Fraction(3, 4), Fraction(1, 4), (0,)), Fraction(1, 4))
        self.assertEqual(reference_joint(Fraction(1, 2), Fraction(3, 4), Fraction(1, 4), (1, 1)), Fraction(9, 10))
        self.assertEqual(reference_joint(Fraction(1, 3), Fraction(3, 4), Fraction(1, 4), (1, 0)), Fraction(1, 3))
        self.assertEqual(reference_joint(0, Fraction(1, 2), Fraction(1, 2), (1,)), 0)
        self.assertEqual(reference_joint(1, Fraction(1, 2), Fraction(1, 2), (1,)), 1)
        with self.assertRaises(ValueError):
            reference_joint(Fraction(1, 2), 0, 0, (1,))

    def test_parser_and_scorer_fail_closed(self):
        good = render('BP', Fraction(1, 2), Fraction(3, 4), Fraction(1, 4), (1,), 'novel')
        self.assertEqual(solve_prompt(good), '3/4')
        self.assertTrue(score_answer('6/8', '3/4'))
        self.assertFalse(score_answer('0.75', '3/4'))
        self.assertFalse(score_answer('1/0', '3/4'))
        for bad in ('Q:BP:p=1/2;a=1/2;b=1/2;e=1;p=1/2;P=',
                    'Q:BP:p=2/1;a=1/2;b=1/2;e=1;P=',
                    'Q:BP:p=1/0;a=1/2;b=1/2;e=1;P=',
                    'Q:BP:p=1/2;a=1/2;b=1/2;e=11;P=',
                    'Q:BP:p=1/2;a=1/2;b=1/2;e=2;P=',
                    'Q:Bayes:w=3,E=4,prior=5,noise=1,post='):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_prompt(bad)

    def test_dataset_groups_and_layouts_disjoint_and_oracles_agree(self):
        data = build_dataset()
        self.assertEqual(data, build_dataset())
        for task, splits in data['tasks'].items():
            groups = {split: {r['group'] for r in splits[split]} for split in ('train', 'validation', 'test')}
            self.assertEqual(len(groups['test']), 24)
            self.assertFalse(groups['train'] & groups['test'])
            self.assertFalse(groups['validation'] & groups['test'])
            self.assertFalse(groups['train'] & groups['validation'])
            self.assertNotIn('novel', {r['layout'] for r in splits['train'] + splits['validation']})
            for split in ('train', 'validation', 'test'):
                for row in splits[split]:
                    self.assertEqual(solve_prompt(row['prompt']), row['gold'])
            if task == 'BS':
                self.assertTrue(any(r['generalization'] == 'longer_sequence' for r in splits['test']))
                self.assertTrue(all(len(parse_prompt(r['prompt'])[-1]) == 2 for r in splits['train']))

    def test_prior_only_control_does_not_use_evidence(self):
        prompt = render('BP', Fraction(1, 2), Fraction(3, 4), Fraction(1, 4), (1,), 'novel')
        row = dict(prompt=prompt, gold='3/4', layout='novel', generalization='new_layout')
        result = evaluate(None, None, [row], prior_only=True)
        self.assertEqual(result['records'][0]['predicted'], '1/2')
        self.assertEqual(result['accuracy'], 0)

    def test_promotion_screen_rejects_constant_tie_and_fallback_regression(self):
        controls = {'majority': dict(accuracy=.3)}
        self.assertEqual(empirical_gate(dict(accuracy=.3, novel_layout_accuracy=.3), controls,
            test_groups=24, fallback_changes=0)['decision'], 'FAIL')
        self.assertEqual(empirical_gate(dict(accuracy=.5, novel_layout_accuracy=.5), controls,
            test_groups=24, fallback_changes=0)['decision'], 'PASS')
        self.assertEqual(empirical_gate(dict(accuracy=1, novel_layout_accuracy=1), controls,
            test_groups=24, fallback_changes=1)['decision'], 'FAIL')


class FrozenSpecialistTests(unittest.TestCase):
    def test_explicit_route_skips_router_and_restores_after_exception(self):
        model = AIXZeroModel(AIXZeroConfig(use_moe=True)).eval()
        moe = model.latent_reasoning_core.core_block.moe
        called = []
        handle = moe.router.register_forward_hook(lambda *args: called.append(True))
        try:
            with self.assertRaises(RuntimeError):
                with explicit_route(model, 0):
                    model(torch.tensor([[1, 10, 11]]))
                    self.assertFalse(called)
                    raise RuntimeError('request failed')
            self.assertIsNone(moe.forced_expert_index)
            model(torch.tensor([[1, 10, 11]]))
            self.assertTrue(called)
        finally:
            handle.remove()

    def test_training_masks_prompt_and_freezes_every_other_tensor(self):
        torch.set_num_threads(1)
        path = Path(__file__).resolve().parents[1] / 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        model, tokenizer = load_v11(path)
        task = build_dataset()['tasks']['BP']
        ids, labels = make_batch(task['train'][:2], tokenizer, model.config)
        prompt_len = len(tokenizer.encode(task['train'][0]['prompt']))
        self.assertTrue((labels[0, :prompt_len] == -100).all())
        self.assertTrue((labels[0, prompt_len:prompt_len + 2] != -100).all())
        result = train_expert(model, tokenizer, task['train'][:32], task['validation'][:8],
                              slot=0, steps=2, seed=42)
        self.assertTrue(result['frozen_tensors_unchanged'])
        self.assertEqual(result['trainable_parameters'], 49152)
        self.assertTrue(all(not p.requires_grad or name.startswith('latent_reasoning_core.core_block.moe.experts.0.')
                            for name, p in model.named_parameters()))
        self.assertIsNone(model.latent_reasoning_core.core_block.moe.forced_expert_index)

    def test_installed_candidate_session_restores_all_base_tensors(self):
        model = AIXZeroModel(AIXZeroConfig(use_moe=True)).eval()
        candidate = copy.deepcopy(model)
        with torch.no_grad():
            candidate.latent_reasoning_core.core_block.moe.experts[0].gate_proj.weight.add_(1)
        before = tensor_digest(model.state_dict())
        with tempfile.TemporaryDirectory() as folder:
            bank = SparseExpertCartridgeBank(Path(folder))
            bank.export_all_domain_cartridges(candidate, {'BP': {}}, {}, domain_expert_map={'BP': 0})
            with self.assertRaises(RuntimeError):
                with temporary_specialist(model, bank, 'BP'):
                    self.assertNotEqual(before, tensor_digest(model.state_dict()))
                    self.assertEqual(model.latent_reasoning_core.core_block.moe.forced_expert_index, 0)
                    raise RuntimeError('request failed')
        self.assertEqual(before, tensor_digest(model.state_dict()))
        self.assertIsNone(model.latent_reasoning_core.core_block.moe.forced_expert_index)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
