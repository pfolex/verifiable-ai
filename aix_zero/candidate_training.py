"""Bounded CPU candidate fine-tuning, optionally with real physical LoRA.

Time limits are cooperative checks between steps, not process-level deadlines.
Adapters are opt-in; the legacy expert-training path remains the default.
"""
from dataclasses import dataclass
import math
import time


@dataclass
class TrainingOutput:
    checkpoint: bytes
    metrics: dict


def validate_training_settings(settings):
    required = {'steps', 'learning_rate', 'max_seconds'}
    if not required <= set(settings) or set(settings) - required - {'physical_lora'}:
        raise ValueError('Training settings require steps, learning_rate, max_seconds')
    if 'physical_lora' in settings:
        from aix_zero.candidate_checkpoint import validate_lora_settings
        validate_lora_settings(settings['physical_lora'])
    if type(settings['steps']) is not int or not 1 <= settings['steps'] <= 1000:
        raise ValueError('Training steps must be between 1 and 1000')
    for key, maximum in (('learning_rate', .01), ('max_seconds', 3600)):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError('Invalid training ' + key)


def train_checkpoint(initial, rows, metadata):
    settings = metadata['settings']['training']
    validate_training_settings(settings)
    if not rows or len(rows) > 256:
        raise ValueError('Training requires 1 to 256 registered rows')
    import torch
    from aix_zero.candidate_checkpoint import load_candidate_checkpoint, encode_candidate_checkpoint
    from aix_zero.physical_lora import (LoRALinear, attach_physical_lora_to_model,
        expand_model_lora_rank, measure_physical_lora_Accounting)
    from aix_zero.meta_ai_autopilot import encode_complete_training_batch, verify_r1_reasoning_trace
    for row in rows:
        if not verify_r1_reasoning_trace(row['completion'], row['gold'])['fully_verified']:
            raise ValueError('Training completion failed trusted trace verification')
    start = time.monotonic()
    def budget():
        if time.monotonic() - start > settings['max_seconds']:
            raise TimeoutError('Cooperative training time budget exceeded')
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(metadata['seed'])
            model, tokenizer, input_format = load_candidate_checkpoint(initial)
            starting_parameters = sum(p.numel() for p in model.parameters())
            if tokenizer.vocab_sha256 != metadata['tokenizer_sha256']:
                raise ValueError('Training tokenizer identity differs')
            weights = {tokenizer.ans_id, tokenizer.end_ans_id,
                       *(tokenizer.token_to_id[ch] for ch in '0123456789*/=;-+' if ch in tokenizer.token_to_id)}
            inputs, labels, token_weights, encoding = encode_complete_training_batch(rows, tokenizer, weights, model.config.max_seq_len)
            if encoding['skipped_overlength']:
                raise ValueError('Registered training targets exceed model context')
            expansion_error = 0.
            requested = settings.get('physical_lora')
            if requested:
                budget()
                model.eval()
                with torch.no_grad():
                    initial_logits = model(inputs[:1])['logits'].clone()
                adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
                if not adapters:
                    attach_physical_lora_to_model(model, rank=requested['rank'], alpha=requested['alpha'],
                        max_lora_rank=128, max_total_parameters=requested['max_total_parameters'],
                        init_seed=metadata['seed'])
                else:
                    ranks = {m.rank for m in adapters}
                    if len(ranks) != 1 or requested['rank'] < max(ranks):
                        raise ValueError('Cannot shrink or mix physical adapter ranks')
                    if requested['rank'] == adapters[0].rank:
                        if any(m.alpha != requested['alpha'] for m in adapters):
                            raise ValueError('Same-rank alpha changes require an explicit migration')
                        if starting_parameters > requested['max_total_parameters']:
                            raise ValueError('Existing model exceeds experiment parameter cap')
                    else:
                        expand_model_lora_rank(model, requested['rank'], new_alpha=requested['alpha'],
                            max_lora_rank=128, max_total_parameters=requested['max_total_parameters'],
                            init_seed=metadata['seed'])
                with torch.no_grad():
                    expanded_logits = model(inputs[:1])['logits']
                expansion_error = float((initial_logits-expanded_logits).abs().max())
                if not torch.allclose(initial_logits, expanded_logits, atol=1e-5, rtol=1e-5):
                    raise ValueError('Physical expansion changed initial model outputs')
            has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
            for name, parameter in model.named_parameters():
                parameter.requires_grad = (
                    name.endswith(('.lora_A', '.lora_B')) if has_lora else
                    any(part in name for part in ('experts.', 'router.', 'lm_head.', 'decoder_layers.', 'memory_bank.'))
                )
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'])
            def loss_value():
                value = model(inputs, labels=labels, token_weights=token_weights)['loss']
                if not torch.isfinite(value).item():
                    raise ValueError('Non-finite training loss')
                return value
            budget()
            model.eval()
            with torch.no_grad():
                before = float(loss_value())
            model.train()
            for _ in range(settings['steps']):
                budget()
                optimizer.zero_grad(set_to_none=True)
                loss_value().backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
            budget()
            model.eval()
            with torch.no_grad():
                after = float(loss_value())
            state = model.state_dict()
            if any(not torch.isfinite(t).all().item() for t in state.values() if t.is_floating_point()):
                raise ValueError('Non-finite candidate parameters')
            checkpoint = encode_candidate_checkpoint(model, tokenizer)
            accounting = measure_physical_lora_Accounting(model, optimizer=optimizer)
            budget()
            return TrainingOutput(checkpoint, dict(
                input_checkpoint_format=input_format,
                checkpoint_format='physical_lora_v2' if has_lora else 'v11_state_dict',
                starting_physical_parameters=starting_parameters,
                added_physical_parameters=accounting['unique_physical_parameters']-starting_parameters,
                insertion_max_logit_error=expansion_error, physical_accounting=accounting,
                encoding=encoding, steps_completed=settings['steps'],
                train_loss_before=before, train_loss_after=after, elapsed_seconds=time.monotonic()-start,
                supervised_tokens_per_step=int((labels[:, 1:] != 0).sum()),
                trainable_parameters=sum(p.numel() for p in parameters),
                optimizer='AdamW', gradient_clip_norm=1., threads=1, device='cpu',
                time_limit_mode='cooperative_between_steps'))
    finally:
        torch.set_num_threads(previous_threads)
