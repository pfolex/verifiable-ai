"""Strict CPU checkpoint codec shared by candidate training and paired evaluation.

Legacy v11-compatible state dictionaries remain supported. Physical-LoRA payloads
carry explicit architecture, exact adapter targets and tokenizer identity.
"""
from dataclasses import asdict
import io
import math
from pathlib import Path


def validate_lora_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {'rank', 'alpha', 'max_total_parameters'}:
        raise ValueError('LoRA settings require rank, alpha and max_total_parameters')
    if type(settings['rank']) is not int or not 1 <= settings['rank'] <= 128:
        raise ValueError('Experiment LoRA rank must be between 1 and 128')
    alpha = settings['alpha']
    if type(alpha) not in (float, int) or not math.isfinite(alpha) or not 0 < alpha <= 1024:
        raise ValueError('Invalid experiment LoRA alpha')
    cap = settings['max_total_parameters']
    if type(cap) is not int or not 1 <= cap <= 10_000_000:
        raise ValueError('Invalid experiment parameter cap')


def _check_finite(state):
    import torch
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint must contain a nonempty tensor state dictionary')
    for value in state.values():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item():
            raise ValueError('Invalid or non-finite checkpoint tensor')


def load_candidate_checkpoint(data):
    import torch
    from aix_zero.benchmark_baseline import load_v11
    from aix_zero.physical_lora import load_physical_lora_payload, LoRALinear
    base = Path(__file__).resolve().parent / 'checkpoints/aix_zero_v11_kip.safetensors'
    model, tokenizer = load_v11(base)
    loaded = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if isinstance(loaded, dict) and 'schema_version' in loaded:
        if loaded['schema_version'] != '2.0.0-physical-lora':
            raise ValueError('Unsupported candidate checkpoint schema')
        if loaded.get('tokenizer_sha256') != tokenizer.vocab_sha256:
            raise ValueError('Physical checkpoint tokenizer identity differs')
        # Bound reconstruction before allocating from supplied config. Expert
        # expansion has a separate migration contract and is not enabled here.
        if loaded.get('model_config') != asdict(model.config):
            raise ValueError('Candidate architecture differs from supported v11 backbone')
        meta = loaded.get('physical_lora_metadata')
        if not isinstance(meta, dict):
            raise ValueError('Physical checkpoint adapter metadata is required')
        validate_lora_settings(dict(rank=meta['rank'], alpha=meta['alpha'],
                                    max_total_parameters=meta['max_total_parameters']))
        targets = meta.get('target_modules')
        if not isinstance(targets, list) or not targets or any(not isinstance(n, str) for n in targets):
            raise ValueError('Invalid physical checkpoint adapter targets')
        if len(set(targets)) != len(targets):
            raise ValueError('Duplicate physical checkpoint adapter targets')
        modules = dict(model.named_modules())
        if any(not isinstance(modules.get(name), torch.nn.Linear) for name in targets):
            raise ValueError('Unknown physical checkpoint adapter target')
        rank_cap = meta.get('max_lora_rank')
        if type(rank_cap) is not int or not meta['rank'] <= rank_cap <= 512:
            raise ValueError('Invalid physical checkpoint rank cap')
        state = loaded.get('state_dict')
        _check_finite(state)
        model = load_physical_lora_payload(loaded)
        actual = [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]
        if actual != targets:
            raise ValueError('Reconstructed adapter targets differ from checkpoint')
        checkpoint_format = 'physical_lora_v2'
    else:
        _check_finite(loaded)
        model.load_state_dict(loaded, strict=True)
        checkpoint_format = 'v11_state_dict'
    model.eval()
    return model, tokenizer, checkpoint_format


def encode_candidate_checkpoint(model, tokenizer):
    import torch
    from aix_zero.physical_lora import physical_lora_checkpoint_payload
    if getattr(model, '_physical_lora_metadata', None):
        payload = physical_lora_checkpoint_payload(model)
        payload['tokenizer_sha256'] = tokenizer.vocab_sha256
    else:
        payload = model.state_dict()
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()
