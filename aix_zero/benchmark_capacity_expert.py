"""Evaluate a v11 specialist on exact active-expert parameter counts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

import torch

from aix_zero.benchmark_baseline import load_v11
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank, _atomic_write
from aix_zero.train_bayesian_pilot import evaluate, explicit_route, tensor_digest, train_expert

DOMAIN = 'ACTIVE_EXPERT_PARAMETERS'
SLOT = 0
DEFAULT_CHECKPOINT = Path(__file__).parent / 'checkpoints' / 'aix_zero_v11_kip.safetensors'


def build_cases(seed):
    # Split by the operands before rendering any layouts. Each test group gets
    # both a familiar rendering and a field order reserved for the test split.
    pairs = [(top_k, width) for top_k in (1, 2, 3)
             for width in (16, 24, 32, 48, 64, 96, 128, 192)]
    random.Random(seed).shuffle(pairs)
    splits = {}
    for split, selected in (
        ('train', pairs[:15]), ('validation', pairs[15:18]), ('test', pairs[18:])
    ):
        rows = []
        for top_k, width in selected:
            group = f'{top_k}:{width}'
            for count in (4, 8, 16, 32):
                layouts = {'standard': f'Q:Active:experts={count};top_k={top_k};params_each={width}?'}
                if split == 'test':
                    layouts['novel'] = f'Q:Active:params_each={width};experts={count};top_k={top_k}?'
                for layout, prompt in layouts.items():
                    rows.append(dict(prompt=prompt, gold=str(top_k * width), group=group,
                                     layout=layout, generalization=layout))
        splits[split] = rows
    return splits


def _arm(model, tokenizer, rows, *, bank=None, checkpoint_sha=None):
    install_ms = []
    if bank is None:
        report = evaluate(model, tokenizer, rows)
    else:
        records = []
        for row in rows:
            start = time.perf_counter()
            with bank.temporary_specialist_session(
                model, DOMAIN, active_base_checkpoint_sha256=checkpoint_sha,
                force_explicit_route=False,
            ), explicit_route(model, SLOT):
                install_ms.append((time.perf_counter() - start) * 1000)
                record = evaluate(model, tokenizer, [row])['records'][0]
            record['latency_ms'] = (time.perf_counter() - start) * 1000
            records.append(record)
        novel = [row for row in records if row['layout'] == 'novel']
        correct = sum(row['correct'] for row in records)
        report = dict(records=records, correct=correct, total=len(records),
                      accuracy=correct / len(records),
                      novel_layout_accuracy=sum(row['correct'] for row in novel) / len(novel))
    times = sorted(row['latency_ms'] for row in report['records'])
    report['p95_response_ms'] = times[math.ceil(len(times) * .95) - 1]
    if install_ms:
        report['first_install_ms'] = install_ms[0]
        report['warm_install_p95_ms'] = sorted(install_ms[1:])[math.ceil((len(install_ms) - 1) * .95) - 1]
        report['first_request_ms'] = report['records'][0]['latency_ms']
        warm_times = sorted(row['latency_ms'] for row in report['records'][1:])
        report['warm_request_p95_ms'] = warm_times[math.ceil(len(warm_times) * .95) - 1]
    return report


def run(output_dir, *, checkpoint=DEFAULT_CHECKPOINT, steps=120, seed=20261111):
    if steps < 1:
        raise ValueError('steps must be positive')
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    base, tokenizer = load_v11(checkpoint)
    candidate, _ = load_v11(checkpoint)
    frozen_digest = tensor_digest(base.state_dict())
    rows = build_cases(seed)
    banks = {}
    for name in ('untrained', 'trained'):
        if name == 'trained':
            training = train_expert(candidate, tokenizer, rows['train'], rows['validation'],
                                    slot=SLOT, steps=steps, seed=seed)
        bank = SparseExpertCartridgeBank(output_dir / name)
        manifest = bank.export_all_domain_cartridges(
            candidate, {DOMAIN: {'title': 'Active routed expert parameter count'}},
            {}, domain_expert_map={DOMAIN: SLOT},
        )
        manifest['base_checkpoint_sha256'] = checkpoint_sha
        if name == 'trained':
            manifest['cartridges'][DOMAIN].update(
                provenance='frozen_v11_single_expert_training_candidate',
                training=training, promotion_status='PENDING_AUDIT',
            )
        _atomic_write(bank.manifest_path, json.dumps(manifest, indent=2).encode())
        banks[name] = bank
    del candidate

    arms = {'v11': _arm(base, tokenizer, rows['test'])}
    for name, bank in banks.items():
        # Use a fresh application cache after export and evaluate each request
        # through the actual on-disk cartridge installation and restoration.
        arms[name] = _arm(base, tokenizer, rows['test'],
                          bank=SparseExpertCartridgeBank(bank.cartridge_dir),
                          checkpoint_sha=checkpoint_sha)
        if (tensor_digest(base.state_dict()) != frozen_digest
                or base.latent_reasoning_core.core_block.moe.forced_expert_index is not None):
            raise AssertionError('Expert session changed v11 serving tensors')

    majority = Counter(row['gold'] for row in rows['train']).most_common(1)[0][0]
    arms['majority'] = evaluate(base, tokenizer, rows['test'], constant=majority)
    moe = base.latent_reasoning_core.core_block.moe
    core = base.count_parameters()['total_parameters']
    expert = sum(p.numel() for p in moe.experts[SLOT].parameters())
    router = sum(p.numel() for p in moe.router.parameters())
    trained = arms['trained']
    improvement = trained['accuracy'] - max(arms[name]['accuracy'] for name in ('v11', 'untrained', 'majority'))
    passed = improvement >= .10 and trained['novel_layout_accuracy'] >= .25
    manifest = banks['trained'].get_manifest()
    manifest['cartridges'][DOMAIN].update(
        promotion_status='ELIGIBLE_NOT_ACTIVATED' if passed else 'REJECTED',
        verified_accuracy_pct=100 * trained['accuracy'],
        empirical_gate={'decision': 'PASS' if passed else 'FAIL', 'improvement': improvement},
    )
    _atomic_write(banks['trained'].manifest_path, json.dumps(manifest, indent=2).encode())
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = dict(checkpoint_sha256=checkpoint_sha, seed=seed, training=training, arms=arms,
                  serving_state_restored=True,
                  settings=dict(threads=1, max_new_tokens=24, num_candidates=1, temperature=0),
                  test_groups=len({row['group'] for row in rows['test']}),
                  dataset_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                  gate=dict(decision='PASS' if passed else 'FAIL', improvement=improvement,
                            min_improvement=.10, min_novel_layout_accuracy=.25),
                  parameter_counts=dict(base_resident=core, additional_trained_stored=expert,
                                        base_plus_trained_stored=core + expert,
                                        resident_with_expert_installed=core,
                                        v11_selected_per_token=core - (moe.n_experts - moe.top_k) * expert,
                                        specialist_selected_per_token=core - (moe.n_experts - 1) * expert - router),
                  process_peak_rss_bytes=peak_rss * (1 if sys.platform == 'darwin' else 1024),
                  limitations=[
                      'Six independent operand groups; examples vary a distractor and field order.',
                      'Temporary slot swap; peak RSS covers training and evaluation together.',
                      'First install starts with an empty application cache; OS disk cache is uncontrolled.',
                      'Parameter selection counts describe routing; dense v11 evaluates every expert.',
                  ])
    _atomic_write(output_dir / 'results.json', json.dumps(report, indent=2, allow_nan=False).encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()
    report = run(args.output_dir, steps=args.steps)
    for name, arm in report['arms'].items():
        print(f"{name}: {arm['correct']}/{arm['total']}")
    print(f"Promotion screen: {report['gate']['decision']}")


if __name__ == '__main__':
    main()
