"""Run an offline two-cartridge round-trip demo using the frozen v11."""
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from aix_zero.checkpoint_registry import load_checkpoint_strict
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank


def main():
    torch.set_num_threads(1)
    checkpoint = Path(__file__).parent / 'checkpoints/aix_zero_v11_kip.safetensors'
    model, tokenizer, _ = load_checkpoint_strict(checkpoint)
    ids = torch.tensor([tokenizer.encode('Q:Word:2+3?')])
    moe = model.latent_reasoning_core.core_block.moe
    with torch.no_grad():
        expected = model(ids)['logits'].clone()
        saved = {key: value.clone() for key, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix='aix-demo-') as directory:
            bank = SparseExpertCartridgeBank(Path(directory), max_cached_experts=1)
            mapping = {'DEMO_A': 0, 'DEMO_B': 1}
            manifest = bank.export_all_domain_cartridges(
                model, {domain: {} for domain in mapping}, {}, domain_expert_map=mapping,
            )
            for domain, slot in mapping.items():
                next(moe.experts[slot].parameters()).add_(.25)
                installed = bank.install_cartridge(model, domain)
                if installed['slot'] != slot or not torch.equal(expected, model(ids)['logits']):
                    raise AssertionError('Disk round-trip did not restore the reference output')
                if bank.cache_info()['entries'] > 1:
                    raise AssertionError('Application cache exceeded the one-entry budget')
                print(f'{domain} loaded into slot {slot}; baseline output restored')
            entry = manifest['cartridges']['DEMO_A']
            shard = Path(directory) / entry['cartridge_file']
            payload = bytearray(shard.read_bytes())
            payload[-1] ^= 1
            shard.write_bytes(payload)
            try:
                bank.install_cartridge(model, 'DEMO_A')
            except ValueError as exc:
                if 'SHA-256' not in str(exc):
                    raise
                print('Corrupted cartridge rejected by SHA-256 validation')
            else:
                raise AssertionError('Corrupted cartridge was accepted')
        if any(not torch.equal(value, saved[key]) for key, value in model.state_dict().items()):
            raise AssertionError('Demo changed the serving state')
    print(json.dumps({
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'physical_parameters': model.count_parameters()['total_parameters'],
        'cartridges_checked': len(mapping), 'base_restored': True,
        'scope': 'Existing expert snapshots, explicit installation, one-entry application cache',
    }, indent=2))


if __name__ == '__main__':
    main()
