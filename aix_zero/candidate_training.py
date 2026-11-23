"""Bounded CPU fine-tuning of a private v11-compatible checkpoint copy.

Time limits are cooperative checks between steps, not process-level deadlines.
"""
from dataclasses import dataclass
import io
import math
from pathlib import Path
import time


@dataclass
class TrainingOutput:
    checkpoint: bytes
    metrics: dict


def validate_training_settings(settings):
    if set(settings) != {'steps', 'learning_rate', 'max_seconds'}:
        raise ValueError('Training settings require steps, learning_rate, max_seconds')
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
    from aix_zero.benchmark_baseline import load_v11
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
            base = Path(__file__).resolve().parent / 'checkpoints/aix_zero_v11_kip.safetensors'
            model, tokenizer = load_v11(base)
            if tokenizer.vocab_sha256 != metadata['tokenizer_sha256']:
                raise ValueError('Training tokenizer identity differs')
            model.load_state_dict(torch.load(io.BytesIO(initial), map_location='cpu', weights_only=True), strict=True)
            weights = {tokenizer.ans_id, tokenizer.end_ans_id,
                       *(tokenizer.token_to_id[ch] for ch in '0123456789*/=;-+' if ch in tokenizer.token_to_id)}
            inputs, labels, token_weights, encoding = encode_complete_training_batch(rows, tokenizer, weights, model.config.max_seq_len)
            if encoding['skipped_overlength']:
                raise ValueError('Registered training targets exceed model context')
            for name, parameter in model.named_parameters():
                parameter.requires_grad = any(part in name for part in ('experts.', 'router.', 'lm_head.', 'decoder_layers.', 'memory_bank.'))
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
            stream = io.BytesIO()
            torch.save(state, stream)
            budget()
            return TrainingOutput(stream.getvalue(), dict(encoding=encoding, steps_completed=settings['steps'],
                train_loss_before=before, train_loss_after=after, elapsed_seconds=time.monotonic()-start,
                supervised_tokens_per_step=int((labels[:, 1:] != 0).sum()), trainable_parameters=sum(p.numel() for p in parameters),
                optimizer='AdamW', gradient_clip_norm=1., threads=1, device='cpu', time_limit_mode='cooperative_between_steps'))
    finally:
        torch.set_num_threads(previous_threads)
