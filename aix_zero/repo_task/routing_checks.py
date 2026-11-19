"""Evaluator fixtures for the disk-backed routing repository task."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from aix_zero.benchmark_baseline import load_v11
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
from aix_zero.train_bayesian_pilot import tensor_digest


class DiskRoutingChecks(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.bank = SparseExpertCartridgeBank(Path(directory), max_cached_experts=1)
        checkpoint = Path('aix_zero/checkpoints/aix_zero_v11_kip.safetensors')
        self.sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.model, tokenizer = load_v11(checkpoint)
        self.ids = torch.tensor([tokenizer.encode('Q:Word:2+3?')])
        self.moe = self.model.latent_reasoning_core.core_block.moe
        self.before = tensor_digest(self.model.state_dict())
        domains = {'ROUTE_A': 2, 'ROUTE_B': 1}
        cartridge_source, _ = load_v11(checkpoint)
        source_moe = cartridge_source.latent_reasoning_core.core_block.moe
        self.shard_digests = {}
        with torch.no_grad():
            for slot in domains.values():
                expert = source_moe.experts[slot]
                next(expert.parameters()).add_(slot / 10)
                self.shard_digests[slot] = tensor_digest(expert.state_dict())
        self.manifest = self.bank.export_all_domain_cartridges(
            cartridge_source, {key: {} for key in domains}, {}, domain_expert_map=domains,
        )
        self.manifest['base_checkpoint_sha256'] = self.sha
        for entry in self.manifest['cartridges'].values():
            entry.update(provenance='frozen_v11_single_expert_training_candidate',
                         promotion_status='APPROVED_ACTIVE',
                         training={'frozen_tensors_unchanged': True},
                         empirical_gate={'decision': 'PASS'})
        self.write_manifest()

    def write_manifest(self):
        self.bank.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def route(self, prompt, callback, **kwargs):
        kwargs.setdefault('active_base_checkpoint_sha256', self.sha)
        return self.bank.execute_routed_request(self.model, prompt, callback, **kwargs)

    def assert_restored(self):
        self.assertEqual(tensor_digest(self.model.state_dict()), self.before)
        self.assertIsNone(self.moe.forced_expert_index)
        self.assertEqual(self.moe.top_k, self.model.config.moe_top_k)
        self.assertFalse(self.model.training)

    def test_two_cartridges_route_to_distinct_slots_and_restore(self):
        for domain, slot in [('ROUTE_A', 2), ('ROUTE_B', 1), ('ROUTE_A', 2)]:
            prompt = f'Q:{domain}:task'

            def generate(model, text):
                self.assertIs(model, self.model)
                self.assertEqual(text, prompt)
                self.assertEqual(self.moe.forced_expert_index, slot)
                self.assertEqual(tensor_digest(self.moe.experts[slot].state_dict()), self.shard_digests[slot])
                return object_marker

            object_marker = object()
            self.assertIs(self.route(prompt, generate), object_marker)
            self.assert_restored()
            self.assertLessEqual(self.bank.cache_info()['entries'], 1)

    @torch.no_grad()
    def test_fallback_matches_base_output(self):
        expected = self.model(self.ids)['logits']
        self.manifest['cartridges']['ROUTE_B']['promotion_status'] = 'REJECTED'
        self.manifest['cartridges']['ROUTE_B']['empirical_gate'] = {'decision': 'FAIL'}
        self.write_manifest()
        entry = self.manifest['cartridges']['ROUTE_A']
        shard = self.bank.cartridge_dir / entry['cartridge_file']
        calls = []

        def generate(model, prompt):
            calls.append(prompt)
            self.assertIsNone(self.moe.forced_expert_index)
            return model(self.ids)['logits']

        for prompt, sha in [('unknown', self.sha), ('Q:ROUTE_B:x', self.sha),
                            ('Q:ROUTE_A:x', '0' * 64), ('Q:ROUTE_A:x', None)]:
            self.assertTrue(torch.equal(self.route(prompt, generate, active_base_checkpoint_sha256=sha), expected))
            self.assert_restored()
        shard.write_bytes(b'corrupt')
        self.assertTrue(torch.equal(self.route('Q:ROUTE_A:x', generate), expected))
        shard.unlink()
        self.assertTrue(torch.equal(self.route('Q:ROUTE_A:x', generate), expected))
        self.assertEqual(len(calls), 6)
        self.assert_restored()

    def test_callback_error_propagates_once_and_restores(self):
        calls = []

        def generate(model, prompt):
            calls.append(prompt)
            raise ValueError('callback failure')

        with self.assertRaisesRegex(ValueError, 'callback failure'):
            self.route('Q:ROUTE_A:x', generate)
        self.assertEqual(calls, ['Q:ROUTE_A:x'])
        self.assert_restored()


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
