"""Storage, installation, corruption, and compatibility contracts for expert snapshots."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetensors.torch import save
import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank


class CartridgeBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'bank'
        torch.manual_seed(42)
        self.config = AIXZeroConfig(d_model=16, n_heads=2, n_kv_heads=1, d_ff=32,
                                    use_moe=True, n_experts=3, moe_top_k=3, max_seq_len=16)
        self.model = AIXZeroModel(self.config).eval()
        self.bank = SparseExpertCartridgeBank(self.root, max_cached_experts=2)
        self.catalog = {k: {} for k in ('A', 'B', 'C')}
        self.mapping = {'A': 0, 'B': 1, 'C': 2}

    def export(self):
        return self.bank.export_all_domain_cartridges(self.model, self.catalog, {},
                                                      domain_expert_map=self.mapping)

    def rewrite_payload(self, transform):
        manifest = self.bank.get_manifest()
        meta = manifest['cartridges']['A']
        tensors = {k: v.detach().clone() for k, v in
                   self.model.latent_reasoning_core.core_block.moe.experts[0].state_dict().items()}
        transform(tensors)
        payload = save(tensors)
        (self.root / meta['cartridge_file']).write_bytes(payload)
        meta['sha256'] = hashlib.sha256(payload).hexdigest()
        meta['serialized_bytes'] = len(payload)
        meta['tensors'] = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in tensors.items()}
        meta['parameters'] = sum(v.numel() for v in tensors.values())
        self.bank.manifest_path.write_text(json.dumps(manifest))

    def test_constructor_and_missing_domain_do_not_create_files(self):
        self.assertFalse(self.root.exists())
        self.assertFalse(self.bank.page_in_cartridge('missing')['loaded'])
        self.assertFalse(self.root.exists())

    def test_exact_reload_restores_expert_and_full_model_outputs(self):
        manifest = self.export()
        self.assertEqual(manifest['cartridges']['A']['sprt_gate'], 'NOT_EVALUATED')
        self.assertIsNone(manifest['cartridges']['A']['verified_accuracy_pct'])
        self.assertIsNone(manifest['cartridges']['A']['base_model_accuracy_pct'])
        self.assertEqual(len(manifest['cartridges']['A']['sha256']), 64)
        ids = torch.tensor([[1, 10, 11]])
        before = self.model(ids)['logits'].detach().clone()
        expert = self.model.latent_reasoning_core.core_block.moe.experts[0]
        parameter_ids = [id(p) for p in expert.parameters()]
        with torch.no_grad():
            for param in expert.parameters():
                param.add_(1)
        self.assertFalse(torch.equal(before, self.model(ids)['logits']))
        result = self.bank.install_cartridge(self.model, 'A')
        self.assertEqual(result['slot'], 0)
        self.assertEqual(parameter_ids, [id(p) for p in expert.parameters()])
        self.assertTrue(torch.equal(before, self.model(ids)['logits']))

    def test_lru_count_and_byte_bounds(self):
        manifest = self.export()
        size = manifest['cartridges']['A']['parameters'] * 4
        self.bank.max_cached_bytes = size * 2
        for key in ('A', 'B', 'A', 'C'):
            self.bank.page_in_cartridge(key)
            self.assertLessEqual(self.bank.cached_bytes, size * 2)
        self.assertEqual([key[0] for key in self.bank._hot_cache], ['A', 'C'])
        self.bank.max_cached_bytes = 1
        # Fresh bank with insufficient budget serves without retaining tensors.
        bank = SparseExpertCartridgeBank(self.root, max_cached_bytes=1)
        self.assertTrue(bank.page_in_cartridge('A')['loaded'])
        self.assertEqual(bank.cache_info()['bytes'], 0)
        bank = SparseExpertCartridgeBank(self.root, max_cached_experts=0)
        bank.page_in_cartridge('A')
        self.assertEqual(bank.cache_info()['entries'], 0)

    def test_hash_checked_even_on_cache_hit(self):
        manifest = self.export()
        self.bank.page_in_cartridge('A')
        path = self.root / manifest['cartridges']['A']['cartridge_file']
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.bank.install_cartridge(self.model, 'A')

    def test_reject_paths_missing_files_and_legacy_manifest(self):
        self.export()
        manifest = self.bank.get_manifest()
        manifest['cartridges']['A']['cartridge_file'] = '../escape.safetensors'
        self.bank.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.bank.page_in_cartridge('A')
        self.export()
        manifest = self.bank.get_manifest()
        (self.root / manifest['cartridges']['A']['cartridge_file']).unlink()
        with self.assertRaises(FileNotFoundError):
            self.bank.page_in_cartridge('A')
        self.bank.manifest_path.write_text('{"cartridges": {}}')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.bank.get_manifest()

    def test_incompatible_install_does_not_change_weights(self):
        self.export()
        other = AIXZeroModel(replace(self.config, n_recurrent_steps=2)).eval()
        before = {k: v.clone() for k, v in other.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.bank.install_cartridge(other, 'A')
        self.assertTrue(all(torch.equal(v, other.state_dict()[k]) for k, v in before.items()))
        with self.assertRaisesRegex(ValueError, 'slot'):
            self.bank.install_cartridge(self.model, 'A', slot=-1)
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'eval'):
            self.bank.install_cartridge(self.model, 'A')

    def test_reject_tensor_shape_dtype_keys_and_nonfinite(self):
        transforms = [
            lambda ts: ts.__setitem__('gate_proj.weight', ts['gate_proj.weight'][:1]),
            lambda ts: ts.__setitem__('gate_proj.weight', ts['gate_proj.weight'].double()),
            lambda ts: ts.pop('gate_proj.weight'),
            lambda ts: ts['gate_proj.weight'].fill_(float('nan')),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                self.export()
                before = {k: v.clone() for k, v in self.model.state_dict().items()}
                self.rewrite_payload(transform)
                with self.assertRaises(ValueError):
                    self.bank.install_cartridge(self.model, 'A')
                self.assertTrue(all(torch.equal(v, self.model.state_dict()[k]) for k, v in before.items()))

    def test_oversized_shard_rejected_before_deserialization(self):
        self.export()
        bank = SparseExpertCartridgeBank(self.root, max_shard_bytes=1)
        with patch('aix_zero.sparse_cartridge_bank.load') as loader:
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                bank.page_in_cartridge('A')
            loader.assert_not_called()

    def test_explicit_mapping_required_and_atomic_manifest_publish(self):
        self.export()
        old = self.bank.manifest_path.read_bytes()
        with self.assertRaises(ValueError):
            self.bank.export_all_domain_cartridges(self.model, self.catalog, {}, domain_expert_map={'A': 0})
        with torch.no_grad():
            self.model.latent_reasoning_core.core_block.moe.experts[0].gate_proj.weight.add_(1)
        with patch('aix_zero.sparse_cartridge_bank.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.export()
        self.assertEqual(old, self.bank.manifest_path.read_bytes())
        # Also simulate failure after all shards have been published, at manifest replacement.
        import os
        replace_file = os.replace
        def fail_manifest(src, dst):
            if Path(dst) == self.bank.manifest_path:
                raise OSError('manifest publication failed')
            return replace_file(src, dst)
        with patch('aix_zero.sparse_cartridge_bank.os.replace', side_effect=fail_manifest):
            with self.assertRaises(OSError):
                self.export()
        self.assertEqual(old, self.bank.manifest_path.read_bytes())
        self.assertFalse(list(self.root.glob('.cartridge-*')))
        self.assertTrue(self.bank.page_in_cartridge('A')['loaded'])


class V11CheckpointTests(unittest.TestCase):
    def test_real_checkpoint_round_trip(self):
        from aix_zero.benchmark_baseline import load_v11
        root = Path(__file__).resolve().parents[1]
        model, tokenizer = load_v11(root / 'aix_zero/checkpoints/aix_zero_v11_kip.safetensors')
        self.assertEqual(model.count_parameters()['total_parameters'], 817410)
        prompt = tokenizer.encode('Q:Chain4:3*4+5-2+6=')
        ids = torch.tensor([prompt])
        with torch.no_grad():
            before = model(ids)['logits'].clone()
        with tempfile.TemporaryDirectory() as tmp:
            bank = SparseExpertCartridgeBank(Path(tmp))
            mapping = {f'E{i}': i for i in range(model.config.n_experts)}
            manifest = bank.export_all_domain_cartridges(model, {k: {} for k in mapping}, {},
                                                         domain_expert_map=mapping)
            self.assertTrue(all(m['parameters'] == 49152 for m in manifest['cartridges'].values()))
            for expert in model.latent_reasoning_core.core_block.moe.experts:
                with torch.no_grad():
                    for param in expert.parameters():
                        param.zero_()
            with torch.no_grad():
                self.assertFalse(torch.equal(before, model(ids)['logits']))
            for name in mapping:
                bank.install_cartridge(model, name)
            with torch.no_grad():
                self.assertTrue(torch.equal(before, model(ids)['logits']))
            self.assertLessEqual(bank.cache_info()['entries'], 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
