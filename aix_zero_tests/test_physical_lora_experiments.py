"""Physical-capacity integration tests: actual tensors, frozen base and paired evidence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.benchmark_baseline import load_v11
from aix_zero.candidate_checkpoint import (
    encode_candidate_checkpoint, load_candidate_checkpoint, validate_lora_settings,
)
from aix_zero.candidate_training import train_checkpoint
from aix_zero.experiment_registry import ExperimentRegistry
from aix_zero.experiment_splits import build_split_manifest
from aix_zero.meta_ai_autopilot import build_meta_ai_self_improvement_curriculum
from aix_zero.paired_checkpoint_eval import evaluate_checkpoints
from aix_zero.physical_lora import LoRALinear, attach_physical_lora_to_model
from aix_zero.run_candidate_experiment import main


class PhysicalExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[1]/'aix_zero/checkpoints/aix_zero_v11_kip.safetensors'
        model, cls.tokenizer = load_v11(base)
        cls.initial = encode_candidate_checkpoint(model, cls.tokenizer)
        cls.row = build_meta_ai_self_improvement_curriculum(cycle_index=101)[0]

    def metadata(self, rank=4, alpha=4.):
        return dict(seed=42, tokenizer_sha256=self.tokenizer.vocab_sha256,
                    settings=dict(training=dict(steps=2, learning_rate=.001, max_seconds=60,
                        physical_lora=dict(rank=rank, alpha=alpha, max_total_parameters=2_000_000))))

    def test_real_adapter_updates_frozen_base_and_expanded_continuation(self):
        original = torch.load(io.BytesIO(self.initial), weights_only=True)
        output = train_checkpoint(self.initial, [self.row], self.metadata())
        model, tokenizer, fmt = load_candidate_checkpoint(output.checkpoint)
        self.assertEqual(fmt, 'physical_lora_v2')
        adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
        expected_added = sum(4*(m.in_features+m.out_features) for m in adapters)
        self.assertGreater(expected_added, 0)
        self.assertEqual(output.metrics['added_physical_parameters'], expected_added)
        self.assertEqual(output.metrics['trainable_parameters'], expected_added)
        self.assertEqual(output.metrics['insertion_max_logit_error'], 0.)
        state = model.state_dict()
        for key, value in state.items():
            if key.endswith(('.lora_A', '.lora_B')):
                continue
            self.assertTrue(torch.equal(value, original[key.replace('.base_linear.', '.')]), key)
        self.assertTrue(any(torch.count_nonzero(m.lora_B).item() for m in adapters))
        self.assertEqual(torch.load(io.BytesIO(self.initial), weights_only=True).keys(), original.keys())
        ids = torch.tensor([tokenizer.encode(self.row['prompt'], add_bos=True, add_eos=False)])
        baseline, _, _ = load_candidate_checkpoint(self.initial)
        with torch.no_grad():
            self.assertFalse(torch.equal(model(ids)['logits'], baseline(ids)['logits']))
        restored, _, _ = load_candidate_checkpoint(encode_candidate_checkpoint(model, tokenizer))
        with torch.no_grad():
            self.assertTrue(torch.equal(model(ids)['logits'], restored(ids)['logits']))
        continued = train_checkpoint(output.checkpoint, [self.row], self.metadata(rank=8, alpha=4.))
        grown, _, _ = load_candidate_checkpoint(continued.checkpoint)
        self.assertTrue(all(m.rank == 8 for m in grown.modules() if isinstance(m, LoRALinear)))
        self.assertEqual(continued.metrics['added_physical_parameters'], expected_added)
        self.assertLess(continued.metrics['insertion_max_logit_error'], 1e-5)

    def test_caps_invalid_rank_and_shrink_fail(self):
        for change in (dict(rank=True), dict(rank=0), dict(rank=129),
                       dict(alpha=0), dict(alpha=float('nan')), dict(max_total_parameters=-1)):
            config = dict(rank=4, alpha=4., max_total_parameters=2_000_000)
            config.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_lora_settings(config)
        meta = self.metadata()
        meta['settings']['training']['physical_lora']['max_total_parameters'] = 100
        with self.assertRaisesRegex(ValueError, 'parameters'):
            train_checkpoint(self.initial, [self.row], meta)
        output = train_checkpoint(self.initial, [self.row], self.metadata())
        with self.assertRaisesRegex(ValueError, 'shrink'):
            train_checkpoint(output.checkpoint, [self.row], self.metadata(rank=2, alpha=2.))

    def test_checkpoint_provenance_shapes_and_finite_values_fail_closed(self):
        model, tokenizer, _ = load_candidate_checkpoint(self.initial)
        attach_physical_lora_to_model(model, rank=4, max_lora_rank=128, max_total_parameters=2_000_000)
        payload = torch.load(io.BytesIO(encode_candidate_checkpoint(model, tokenizer)), weights_only=True)
        mutations = [
            lambda p: p.update(tokenizer_sha256='0'*64),
            lambda p: p.update(schema_version='unknown'),
            lambda p: p['model_config'].update(d_model=10_000_000),
            lambda p: p['physical_lora_metadata'].update(target_modules=['unknown.target']),
            lambda p: p['physical_lora_metadata'].update(rank=999999),
        ]
        key = next(k for k in payload['state_dict'] if k.endswith('lora_B'))
        mutations += [
            lambda p: p['state_dict'].update({key: torch.zeros(1)}),
            lambda p: p['state_dict'][key].fill_(float('nan')),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(payload); mutate(changed)
            stream = io.BytesIO(); torch.save(changed, stream)
            with self.assertRaises((ValueError, RuntimeError)):
                load_candidate_checkpoint(stream.getvalue())

    def test_zero_adapter_paired_eval_preserves_predictions_and_counts_growth(self):
        model, tokenizer, _ = load_candidate_checkpoint(self.initial)
        attach_physical_lora_to_model(model, rank=4, max_lora_rank=128, max_total_parameters=2_000_000)
        manifest = build_split_manifest(dict(train=[], development=[self.row], regression=[], final=[]),
                                        lambda row: row['prompt'])
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a.pt'; b=Path(directory)/'b.pt'
            a.write_bytes(self.initial); b.write_bytes(encode_candidate_checkpoint(model, tokenizer))
            report=evaluate_checkpoints(manifest,a,b,max_new_tokens=8)
            self.assertTrue(report['comparison_complete'])
            self.assertEqual(report['paired']['gains'],0)
            self.assertEqual(report['paired']['losses'],0)
            self.assertEqual(report['checkpoint_formats'],dict(champion='v11_state_dict',candidate='physical_lora_v2'))
            self.assertGreater(report['physical_accounting']['candidate']['unique_physical_parameters'],
                               report['physical_accounting']['champion']['unique_physical_parameters'])
            self.assertEqual(report['arms']['champion']['records'][0]['completion'],
                             report['arms']['candidate']['records'][0]['completion'])

    def test_cli_trains_and_evaluates_an_expanded_candidate(self):
        rows={name:[] for name in ('train','development','regression','final')}; seen=set()
        for split, cycle in (('train',101),('development',2100),('regression',3100)):
            for row in build_meta_ai_self_improvement_curriculum(cycle_index=cycle):
                if row['prompt'] not in seen:
                    rows[split].append(row); seen.add(row['prompt']); break
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); champion=root/'champion.pt'; champion.write_bytes(self.initial)
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps(build_split_manifest(rows,lambda row:row['prompt'])))
            policy=root/'policy.json'
            policy.write_text(json.dumps(dict(min_cases=1,min_net_gain=1,max_regression_losses=0,max_trace_losses=0)))
            with contextlib.redirect_stdout(io.StringIO()):
                result=main(['--champion',str(champion),'--manifest',str(manifest),'--policy',str(policy),
                             '--registry',str(root/'registry'),'--steps','1','--lora-rank','4'])
            self.assertIn(result['state'], ('evaluated','rejected'))
            self.assertEqual(champion.read_bytes(),self.initial)
            registry=ExperimentRegistry(root/'registry')
            metrics=json.loads(registry.verify(result['training_metrics_sha256']).read_text())
            self.assertGreater(metrics['added_physical_parameters'],0)
            self.assertEqual(metrics['checkpoint_format'],'physical_lora_v2')
            model, _, _=load_candidate_checkpoint(registry.verify(result['candidate_sha256']).read_bytes())
            self.assertTrue(any(isinstance(m,LoRALinear) for m in model.modules()))
            evidence=json.loads((root/'registry/attempts'/result['attempt']/'evaluation.json').read_text())
            self.assertTrue(evidence['development']['comparison_complete'])
            self.assertTrue(evidence['regression']['comparison_complete'])
