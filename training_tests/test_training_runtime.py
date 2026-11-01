"""A real, tiny Qwen2 forward/backward, adapter reload and generated-answer experiment.

No downloaded model, no projected-logit surrogate and no claim of model improvement.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aix_training_data import checkpoint_identity, load_bundle
from aix_training_runtime import encode_example, tokenizer_for, config_for


def tiny_checkpoint(path):
    torch.manual_seed(123)
    vocab = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2}
    vocab.update({f'word{i}': i + 3 for i in range(61)})
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='[EOS]', pad_token='[PAD]', unk_token='[UNK]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant\n{% endif %}"
    tokenizer.save_pretrained(path)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=2048,
        eos_token_id=1, pad_token_id=0))
    model.save_pretrained(path, safe_serialization=True)


class TrainingRuntimeTests(unittest.TestCase):
    def cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / 'train_aix.py'), *map(str, args)], cwd=ROOT, capture_output=True, text=True)

    def test_v5_export_template_and_default_rope_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'base'
            tiny_checkpoint(base)
            tokenizer_config = json.loads((base / 'tokenizer_config.json').read_text())
            template = tokenizer_config.pop('chat_template')
            (base / 'tokenizer_config.json').write_text(json.dumps(tokenizer_config))
            (base / 'chat_template.jinja').write_text(template)
            model_config = json.loads((base / 'config.json').read_text())
            model_config.pop('rope_theta', None)
            model_config['rope_parameters'] = {'rope_type': 'default', 'rope_theta': 10000}
            (base / 'config.json').write_text(json.dumps(model_config))
            config = {'base_model': str(base)}
            self.assertEqual(tokenizer_for(config).chat_template, template)
            self.assertEqual(config_for(config).rope_theta, 10000)
            model_config['rope_parameters']['rope_type'] = 'unsupported'
            (base / 'config.json').write_text(json.dumps(model_config))
            with self.assertRaisesRegex(ValueError, 'default nested RoPE'):
                config_for(config)

    def test_complete_experiment_uses_actual_gradients_and_preserves_base(self):
        with tempfile.TemporaryDirectory() as directory:
            base, run = Path(directory) / 'base', Path(directory) / 'run'
            tiny_checkpoint(base)
            before = checkpoint_identity(base)
            proc = self.cli('run', '--base-model', base, '--run-dir', run, '--domains', 'logic',
                            '--train-samples-per-domain', '2', '--eval-samples-per-domain', '1',
                            '--steps', '2', '--rank', '2', '--max-seq-length', '1024', '--max-new-tokens', '4', '--device', 'cpu', '--gradient-checkpointing')
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads((run / 'training-report.json').read_text())
            self.assertEqual(report['steps'], 2)
            self.assertTrue(all(loss > 0 for loss in report['losses']))
            self.assertTrue(report['base_unchanged'])
            self.assertEqual(checkpoint_identity(base), before)
            weights = load_file(run / 'adapter' / 'adapter_model.safetensors')
            self.assertTrue(any('lora_B' in name and bool(torch.any(tensor != 0)) for name, tensor in weights.items()))
            for which in ('baseline', 'candidate'):
                result = json.loads((run / (which + '.json')).read_text())
                self.assertEqual(result['model']['provider'], 'hf-local')
                self.assertEqual(result['summary']['sample_count'], 1)
                self.assertGreater(result['samples'][0]['output_tokens'], 0)
            self.assertTrue((run / 'comparison.md').exists())
            gate = json.loads((run / 'gate.json').read_text())
            self.assertNotEqual(gate['decision'], 'PASS')  # Random tiny LM is not evidence of improvement.
            config, examples, manifest = load_bundle(run)
            tokenizer = tokenizer_for(config)
            batch = encode_example(tokenizer, examples[0], 1024, 64)
            masked = sum(label == -100 for label in batch['labels'])
            self.assertGreater(masked, 0)
            self.assertLess(masked, len(batch['labels']))
            self.assertEqual(batch['labels'][-1], tokenizer.eos_token_id)
            self.assertIsNone(encode_example(tokenizer, examples[0], 1, 64))
            with self.assertRaisesRegex(ValueError, 'vocabulary mismatch'):
                encode_example(tokenizer, examples[0], 1024, 1)
            original = (run / 'baseline.json').read_bytes()
            proc = self.cli('evaluate', '--run-dir', run, '--which', 'baseline')
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual((run / 'baseline.json').read_bytes(), original)
            original_config = (base / 'config.json').read_bytes()
            (base / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'base checkpoint changed'):
                load_bundle(run)
            (base / 'config.json').write_bytes(original_config)
            old_comparison = (run / 'comparison.json').read_bytes()
            (run / 'adapter' / 'adapter_config.json').write_text('{}')
            proc = self.cli('report', '--run-dir', run)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual((run / 'comparison.json').read_bytes(), old_comparison)
            with (run / 'train.jsonl').open('a') as handle:
                handle.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'training data changed'):
                load_bundle(run)


if __name__ == '__main__':
    unittest.main()
