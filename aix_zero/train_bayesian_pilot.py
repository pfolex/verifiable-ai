"""Frozen-v11 specialist pilot with explicit routes and held-out probability structures.

No web/council calls, vocabulary expansion, base-checkpoint writes, or automatic
promotion. Candidate exports and empirical decisions are saved in a new run folder.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F

from aix_zero.bayesian_tasks import build_dataset, empirical_gate, parse_prompt, score_answer, solve_prompt
from aix_zero.benchmark_baseline import _hardware, _report_path, load_v11
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank, _atomic_write
from aix_zero.train_from_scratch import extract_answer


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def explicit_route(model, slot):
    moe = model.latent_reasoning_core.core_block.moe
    if type(slot) is not int or not 0 <= slot < len(moe.experts):
        raise ValueError('Invalid explicit route')
    previous = moe.forced_expert_index
    moe.forced_expert_index = slot
    try:
        yield
    finally:
        moe.forced_expert_index = previous


@contextmanager
def temporary_specialist(model, bank, domain):
    """Install for one explicitly recognized task, then restore all affected weights.

    The caller owns exclusive access to model. Unrelated requests must run outside
    this context; this is not a concurrent production serving implementation.
    """
    manifest = bank.get_manifest()
    meta = manifest['cartridges'][domain]
    slot = meta['source_expert_index']
    expert = model.latent_reasoning_core.core_block.moe.experts[slot]
    before = {k: v.detach().clone() for k, v in expert.state_dict().items()}
    was_training = model.training
    try:
        model.eval()
        bank.install_cartridge(model, domain)
        with explicit_route(model, slot):
            yield
    finally:
        with torch.no_grad():
            expert.load_state_dict(before, strict=True)
        model.train(was_training)


def make_batch(rows, tokenizer, config):
    sequences, targets = [], []
    for row in rows:
        prompt = tokenizer.encode(row['prompt'])
        completion = tokenizer.encode(f"<ans>{row['gold']}</ans>", add_bos=False, add_eos=True)
        sequence = prompt + completion
        if len(sequence) > config.max_seq_len or max(sequence) >= config.vocab_size:
            raise ValueError('Task exceeds the frozen checkpoint token/length budget')
        if tokenizer.decode(prompt, skip_special=True) != row['prompt']:
            raise ValueError('Tokenizer silently dropped prompt content')
        sequences.append(sequence)
        targets.append([-100] * len(prompt) + completion)
    width = max(map(len, sequences))
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for index, (sequence, target) in enumerate(zip(sequences, targets)):
        ids[index, :len(sequence)] = torch.tensor(sequence)
        labels[index, :len(target)] = torch.tensor(target)
    return ids, labels


def completion_loss(model, ids, labels):
    logits = model(ids)['logits'][:, :-1]
    return F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def validation_loss(model, rows, tokenizer, batch_size=16):
    total, count = 0.0, 0
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            ids, labels = make_batch(rows[offset:offset + batch_size], tokenizer, model.config)
            active = int((labels[:, 1:] != -100).sum())
            total += float(completion_loss(model, ids, labels)) * active
            count += active
    return total / count


def train_expert(model, tokenizer, rows, validation, *, slot, steps, seed):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    expert = model.latent_reasoning_core.core_block.moe.experts[slot]
    for parameter in expert.parameters():
        parameter.requires_grad_(True)
    prefix = f'latent_reasoning_core.core_block.moe.experts.{slot}.'
    frozen = {k: v for k, v in model.state_dict().items() if not k.startswith(prefix)}
    frozen_before = tensor_digest(frozen)
    initial = tensor_digest(expert.state_dict())
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3, weight_decay=.01)
    generator = random.Random(seed)
    model.eval()  # No dropout; gradients are still enabled for the expert.
    logs = []
    with explicit_route(model, slot):
        best_loss = validation_loss(model, validation, tokenizer)
        initial_loss = best_loss
        best = {k: v.detach().clone() for k, v in expert.state_dict().items()}
        best_step = 0
        for step in range(1, steps + 1):
            batch = generator.sample(rows, min(16, len(rows)))
            ids, labels = make_batch(batch, tokenizer, model.config)
            optimizer.zero_grad(set_to_none=True)
            loss = completion_loss(model, ids, labels)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError('Nonfinite training state')
            optimizer.step()
            if step % 60 == 0 or step == steps:
                val = validation_loss(model, validation, tokenizer)
                logs.append(dict(step=step, train_loss=float(loss), validation_loss=val))
                print(f"  slot={slot} step={step}/{steps} validation NLL={val:.4f}", flush=True)
                if val < best_loss:
                    best_loss, best_step = val, step
                    best = {k: v.detach().clone() for k, v in expert.state_dict().items()}
        expert.load_state_dict(best)
    frozen_after = tensor_digest({k: v for k, v in model.state_dict().items() if not k.startswith(prefix)})
    if frozen_before != frozen_after:
        raise AssertionError('Frozen base tensors changed')
    return dict(steps=steps, selected_step=best_step, selection='lowest_validation_completion_NLL',
                initial_validation_loss=initial_loss, selected_validation_loss=best_loss,
                initial_expert_sha256=initial, selected_expert_sha256=tensor_digest(expert.state_dict()),
                frozen_tensor_sha256=frozen_after, frozen_tensors_unchanged=True,
                trainable_parameters=sum(p.numel() for p in expert.parameters()),
                seed=seed, optimizer=dict(name='AdamW', lr=.001, weight_decay=.01, clip_norm=1.0), logs=logs)


def evaluate(model, tokenizer, rows, *, constant=None, assisted=False, prior_only=False):
    records = []
    for row in rows:
        start = time.perf_counter()
        if prior_only:
            from aix_zero.bayesian_tasks import fraction_text
            answer, trace = fraction_text(parse_prompt(row['prompt'])[1]), ''
        elif constant is not None:
            answer, trace = constant, ''
        elif assisted:
            answer, trace = solve_prompt(row['prompt']), ''
        else:
            ids, _, _ = model.generate_verified(tokenizer.encode(row['prompt']), eos_id=tokenizer.eos_id,
                                               max_new_tokens=24, num_candidates=1,
                                               temperature=0, tokenizer=tokenizer, inflight_alu=False)
            trace = tokenizer.decode(ids, skip_special=True)
            answer = extract_answer(trace)
        records.append(dict(**row, predicted=answer, correct=score_answer(answer, row['gold']),
                            trace=trace, latency_ms=(time.perf_counter() - start) * 1000))
    novel = [r for r in records if r['layout'] == 'novel']
    categories = {key: dict(correct=sum(r['correct'] for r in records if r['generalization'] == key),
                           total=sum(r['generalization'] == key for r in records))
                  for key in sorted({r['generalization'] for r in records})}
    return dict(accuracy=sum(r['correct'] for r in records) / len(records),
                correct=sum(r['correct'] for r in records), total=len(records),
                novel_layout_accuracy=sum(r['correct'] for r in novel) / max(1, len(novel)),
                categories=categories, median_response_ms=statistics.median(r['latency_ms'] for r in records),
                records=records)


def fallback_audit(model, tokenizer, bank, domain):
    """Check existing domains before and after an on-disk specialist round trip."""
    from aix_zero.knowledge_is_power import build_21_domain_kip_v11_curriculum
    _, cases, _, _ = build_21_domain_kip_v11_curriculum(seed=42)
    before_digest = tensor_digest(model.state_dict())
    sample = bank.get_manifest()['cartridges'][domain]['evaluation_prompt']
    with temporary_specialist(model, bank, domain):
        model.generate_verified(tokenizer.encode(sample), eos_id=tokenizer.eos_id,
                                max_new_tokens=24, tokenizer=tokenizer)
    after_digest = tensor_digest(model.state_dict())
    if before_digest != after_digest:
        raise AssertionError('Specialist session failed to restore the base')
    # A second model is an untouched checkpoint reference, not a cached output oracle.
    reference_path = Path(bank.get_manifest()['training_base_checkpoint'])
    if not reference_path.is_absolute():
        reference_path = Path(__file__).resolve().parents[1] / reference_path
    reference, _ = load_v11(reference_path)
    result = {}
    for assisted in (False, True):
        mismatches, correct = 0, 0
        for row in cases:
            kwargs = dict(eos_id=tokenizer.eos_id, max_new_tokens=40, tokenizer=tokenizer,
                          inflight_alu=assisted, num_candidates=1, temperature=0)
            a = reference.generate_verified(tokenizer.encode(row['prompt']), **kwargs)[0]
            b = model.generate_verified(tokenizer.encode(row['prompt']), **kwargs)[0]
            mismatches += int(a != b)
            predicted = extract_answer(tokenizer.decode(b, skip_special=True))
            correct += int(predicted == row['gold'])
        result['symbolic_assisted' if assisted else 'neural_only'] = dict(
            total=len(cases), correct=correct, changed_traces=mismatches)
    return dict(frozen_base_restored=before_digest == after_digest, evaluations=result,
                changed_traces=sum(v['changed_traces'] for v in result.values()))


def run(output_dir, *, steps=300):
    if steps < 1:
        raise ValueError('Training steps must be positive')
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    checkpoint = Path(__file__).parent / 'checkpoints/aix_zero_v11_kip.safetensors'
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    base, tokenizer = load_v11(checkpoint)
    original_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    dataset = build_dataset()
    dataset_bytes = json.dumps(dataset, sort_keys=True, indent=2).encode()
    _atomic_write(output_dir / 'dataset.json', dataset_bytes)
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  hardware=_hardware(), torch=torch.__version__, threads=1,
                  checkpoint_sha256=checkpoint_sha, dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob('*.py'))},
                  settings=dict(steps_per_task=steps, batch_size=16, completion='exact_reduced_fraction',
                                evaluation='greedy_single_candidate_24_tokens', new_tokens=False),
                  limitations=['Narrow synthetic binary Bayesian probability models; not broad epistemology.',
                               'Assistance is an explicit full exact solver, not learned expert success.',
                               'Empirical promotion screen is not a significance test or SPRT.'], tasks={})
    for index, (task, data) in enumerate(dataset['tasks'].items()):
        print(f"TASK {task}: measuring controls", flush=True)
        slot = index
        model, _ = load_v11(checkpoint)
        controls = dict(base=evaluate(base, tokenizer, data['test']),
                        majority=evaluate(base, tokenizer, data['test'], constant=data['majority_answer']),
                        prior_only=evaluate(base, tokenizer, data['test'], prior_only=True))
        with explicit_route(model, slot):
            controls['untrained_explicit_route'] = evaluate(model, tokenizer, data['test'])
        print(f"TASK {task}: training one {sum(p.numel() for p in model.latent_reasoning_core.core_block.moe.experts[slot].parameters())}-parameter expert", flush=True)
        training = train_expert(model, tokenizer, data['train'], data['validation'],
                               slot=slot, steps=steps, seed=42 + index)
        with explicit_route(model, slot):
            candidate = evaluate(model, tokenizer, data['test'])
        assisted = evaluate(model, tokenizer, data['test'], assisted=True)
        bank = SparseExpertCartridgeBank(output_dir / 'candidates' / task)
        manifest = bank.export_all_domain_cartridges(model, {task: {'title': task + ' Bayesian candidate'}}, {},
                                                    domain_expert_map={task: slot})
        manifest['training_base_checkpoint'] = _report_path(checkpoint)
        manifest['training_base_checkpoint_sha256'] = checkpoint_sha
        manifest['dataset_sha256'] = report['dataset_sha256']
        manifest['source_sha256'] = report['source_sha256']
        manifest['routing_policy'] = 'explicit single routed expert plus the frozen shared expert'
        manifest['evaluation_protocol'] = dict(num_candidates=1, max_new_tokens=24, temperature=0, neural_only=True)
        meta = manifest['cartridges'][task]
        meta.update(provenance='frozen_v11_single_expert_training_candidate', training=training,
                    verified_accuracy_pct=100 * candidate['accuracy'],
                    evaluation_prompt=data['test'][0]['prompt'], promotion_status='PENDING_AUDIT',
                    evaluation=dict(candidate={k: v for k, v in candidate.items() if k != 'records'},
                                    controls={k: {field: v[field] for field in ('accuracy', 'novel_layout_accuracy')}
                                              for k, v in controls.items()},
                                    assistance='separate_full_exact_solver', assisted_accuracy=assisted['accuracy']))
        manifest['limitations'] = 'Candidate expert, not promoted. Base is frozen; exact task solver is reported separately.'
        _atomic_write(bank.manifest_path, json.dumps(manifest, indent=2).encode())
        print(f"TASK {task}: auditing base restoration and 21 existing domains", flush=True)
        audit = fallback_audit(base, tokenizer, bank, task)
        gate = empirical_gate(candidate, controls, test_groups=len({r['group'] for r in data['test']}),
                              fallback_changes=audit['changed_traces'])
        meta['empirical_gate'] = gate
        meta['promotion_status'] = 'ELIGIBLE_NOT_ACTIVATED' if gate['decision'] == 'PASS' else 'REJECTED'
        # Keep candidates isolated even on PASS; activation is deliberately separate.
        _atomic_write(bank.manifest_path, json.dumps(manifest, indent=2).encode())
        report['tasks'][task] = dict(training=training, controls=controls, candidate=candidate,
                                     exact_solver_assisted=assisted, fallback_audit=audit, gate=gate,
                                     candidate_manifest=str(bank.manifest_path.relative_to(output_dir)),
                                     promotion_status=meta['promotion_status'])
        print(f"TASK {task}: base={controls['base']['accuracy']:.1%}, "
              f"constant={controls['majority']['accuracy']:.1%}, candidate={candidate['accuracy']:.1%}, "
              f"gate={gate['decision']}", flush=True)
        _atomic_write(output_dir / 'results.json', json.dumps(report, indent=2, allow_nan=False).encode())
    if tensor_digest(base.state_dict()) != tensor_digest(original_state):
        raise AssertionError('Serving base changed')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_sha:
        raise AssertionError('Base checkpoint file changed')
    report['checkpoint_unchanged'] = True
    _atomic_write(output_dir / 'results.json', json.dumps(report, indent=2, allow_nan=False).encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()
    run(args.output_dir, steps=args.steps)


if __name__ == '__main__':
    main()
