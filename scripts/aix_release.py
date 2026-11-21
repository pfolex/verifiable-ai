"""Offline entry point copied into the portable release as aix.py."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', default='Q:Word:2+3?')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 40:
        parser.error('--max-new-tokens must be between 1 and 40')
    manifest = json.loads((ROOT / 'build-manifest.json').read_text())
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported build manifest')
    for path, expected in manifest['files'].items():
        target = ROOT / path
        if target.is_symlink() or not target.resolve().is_relative_to(ROOT):
            raise ValueError('Invalid manifest path')
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Build integrity check failed for {path}')

    import torch
    from aix_zero.checkpoint_registry import load_checkpoint_strict

    torch.set_num_threads(1)
    torch.manual_seed(42)
    checkpoint = manifest['checkpoint']
    model, tokenizer, info = load_checkpoint_strict(
        ROOT / checkpoint, expected_sha256=manifest['files'][checkpoint],
    )
    prompt = tokenizer.encode(args.prompt)
    if tokenizer.decode(prompt, skip_special=True) != args.prompt:
        raise ValueError('Prompt includes characters outside the tokenizer vocabulary')
    if len(prompt) + args.max_new_tokens > model.config.max_seq_len:
        raise ValueError('Prompt and generation exceed the checkpoint context length')
    with torch.no_grad():
        logits = model(torch.tensor([prompt]))['logits']
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite inference output')
        generated, confidence, depth = model.generate_verified(
            prompt, eos_id=tokenizer.eos_id, max_new_tokens=args.max_new_tokens,
            num_candidates=1, temperature=0, tokenizer=tokenizer, inflight_alu=False,
        )
    print(json.dumps({
        'source_revision': manifest['source_revision'], 'checkpoint_sha256': info['sha256'],
        'physical_parameters': info['physical_parameters'],
        'prompt': args.prompt, 'output': tokenizer.decode(generated, skip_special=True),
        'finite_forward': True, 'generation_mode': 'neural_only',
    }, indent=2))


if __name__ == '__main__':
    main()
