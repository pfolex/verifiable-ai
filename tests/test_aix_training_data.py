"""Training preparation can be checked without installing the training stack."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from aix_training_data import checkpoint_identity, prepare_examples, problem_hash, render_prompt
from train_aix import parser_for, prepare


class TrainingDataTests(unittest.TestCase):
    def test_verified_unique_training_data_is_content_disjoint_from_evaluation(self):
        examples, manifest, audit = prepare_examples(domains=['logic', 'silicon', 'systems', 'matter'],
            train_count=4, train_seed=10000, eval_count=2, eval_seed=2026, eval_difficulty=3, attempts=100)
        train_hashes = {problem_hash(example['input']) for example in examples}
        eval_hashes = {problem_hash(case['input']) for case in manifest['cases']}
        self.assertFalse(train_hashes & eval_hashes)
        self.assertEqual(len(examples), len(train_hashes))
        self.assertEqual(set(audit['training_unique_by_domain']), {'logic', 'silicon', 'systems', 'matter'})
        self.assertTrue(audit['reference_answers_verified'])
        self.assertNotIn('reference_solution', json.dumps(manifest))
        # Systems generation can repeat the same public problem across seeds.
        self.assertLess(audit['evaluation_unique_problems'], audit['evaluation_samples'])

    def test_seed_overlap_and_insufficient_distinct_data_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'seed ranges'):
            prepare_examples(domains=['logic'], train_count=1, train_seed=2026, eval_count=1, eval_seed=2026, eval_difficulty=1)
        with self.assertRaisesRegex(ValueError, 'No distinct training'):
            prepare_examples(domains=['systems'], train_count=1, train_seed=10000, eval_count=1, eval_seed=2026, eval_difficulty=1, attempts=1)

    def test_problem_identity_ignores_ids_but_not_public_specs(self):
        examples, _, _ = prepare_examples(domains=['logic'], train_count=1, train_seed=10000, eval_count=1, eval_seed=2026, eval_difficulty=1)
        task = copy.deepcopy(examples[0]['input'])
        before = problem_hash(task)
        task['task_id'] = 'another-seed'
        self.assertEqual(problem_hash(task), before)
        task['spec']['goal_clause'] = ['DIFFERENT']
        self.assertNotEqual(problem_hash(task), before)

    def test_prompt_closes_r1_thinking_identically(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return 'prompt<think>\n'
        examples, _, _ = prepare_examples(domains=['logic'], train_count=1, train_seed=10000, eval_count=1, eval_seed=2026, eval_difficulty=1)
        tokenizer = Tokenizer()
        prompt = render_prompt(tokenizer, examples[0]['input'])
        self.assertTrue(prompt.endswith('</think>\n\n'))
        self.assertNotIn('reference_solution', str(tokenizer.messages))

    def test_prepare_refuses_existing_or_base_nested_output_and_invalid_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            base, output = Path(directory) / 'base', Path(directory) / 'run'
            base.mkdir()
            (base / 'config.json').write_text(json.dumps({'num_hidden_layers': 2}))
            parser = parser_for()
            args = parser.parse_args(['prepare', '--base-model', str(base), '--run-dir', str(base / 'child')])
            with self.assertRaisesRegex(ValueError, 'separate'):
                prepare(args)
            output.mkdir()
            args.run_dir = output
            with self.assertRaisesRegex(ValueError, 'already exists'):
                prepare(args)
            args.run_dir = Path(directory) / 'new'
            args.steps = 0
            with self.assertRaisesRegex(ValueError, 'positive'):
                prepare(args)

    def test_checkpoint_digest_covers_weights_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors'):
                (root / name).write_text('test')
            before = checkpoint_identity(root)
            (root / 'model.safetensors').write_text('changed')
            self.assertNotEqual(before, checkpoint_identity(root))


if __name__ == '__main__':
    unittest.main()
