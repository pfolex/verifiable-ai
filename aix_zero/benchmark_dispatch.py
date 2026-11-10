"""Paired dense/sparse CPU experiments; no training or automatic disk routing.

Synthetic expert-count scaling is a kernel experiment, not model-quality evidence.
Every disk scenario verifies hashes and uses OS-cached files; first load refers
only to an empty application cache. Report RSS in separate fresh worker processes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel, MixtureOfReasoningExperts
from aix_zero.benchmark_baseline import _hardware, load_v11
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank, _atomic_write
from aix_zero.train_from_scratch import extract_answer


def summary(values):
    ordered = sorted(values)
    return dict(samples=len(values), p50_ms=statistics.median(ordered),
                p95_ms=ordered[math.ceil(.95 * len(ordered)) - 1])


def timed(call):
    start = time.perf_counter()
    value = call()
    return (time.perf_counter() - start) * 1000, value


def memory_worker(experts, mode):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    module = MixtureOfReasoningExperts(AIXZeroConfig(n_experts=experts)).eval()
    module.dispatch_mode = mode
    x = torch.randn(4, 112, 128)
    def rss_peak():
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(raw if sys.platform == 'darwin' else raw * 1024)
    before = rss_peak()
    with torch.inference_mode():
        output = module(x)
    after = rss_peak()
    return dict(experts=experts, mode=mode, peak_rss_bytes=after,
                peak_before_forward_bytes=before, incremental_peak_bytes=max(0, after - before),
                output_finite=bool(torch.isfinite(output[0]).all()),
                resident_parameter_bytes=sum(p.numel() * p.element_size() for p in module.parameters()),
                batch=4, sequence=112)


def kernel_experiments(counts, repeats):
    results = []
    for experts in counts:
        for batch, sequence in ((1, 1), (1, 32), (4, 112)):
            torch.manual_seed(42)
            module = MixtureOfReasoningExperts(AIXZeroConfig(n_experts=experts)).eval()
            x = torch.randn(batch, sequence, 128)
            times = {'dense': [], 'sparse': []}
            outputs = {}
            with torch.inference_mode():
                for mode in times:
                    module.dispatch_mode = mode
                    for _ in range(3):
                        outputs[mode] = module(x)[0]
                torch.testing.assert_close(outputs['sparse'], outputs['dense'], rtol=1e-4, atol=2e-5)
                for repeat in range(repeats):
                    order = ('dense', 'sparse') if repeat % 2 == 0 else ('sparse', 'dense')
                    for mode in order:
                        module.dispatch_mode = mode
                        elapsed, _ = timed(lambda: module(x))
                        times[mode].append(elapsed)
            row = dict(experts=experts, top_k=module.top_k, batch=batch, sequence=sequence,
                       max_absolute_output_difference=float((outputs['dense'] - outputs['sparse']).abs().max()),
                       modes={mode: summary(values) for mode, values in times.items()},
                       routed_output_tensor_bytes=dict(dense=batch * sequence * experts * 128 * 4,
                                                       sparse=batch * sequence * module.top_k * 128 * 4))
            row['dense_over_sparse_p50'] = row['modes']['dense']['p50_ms'] / row['modes']['sparse']['p50_ms']
            results.append(row)
            print(f"kernel E={experts} B={batch} T={sequence}: "
                  f"dense/sparse speed ratio={row['dense_over_sparse_p50']:.2f}", flush=True)
    return results


def storage_experiments(counts, repeats):
    results = []
    for experts in counts:
        torch.manual_seed(42)
        model = AIXZeroModel(AIXZeroConfig(use_moe=True, n_experts=experts)).eval()
        with tempfile.TemporaryDirectory(prefix='aix-dispatch-') as folder:
            root = Path(folder)
            exporter = SparseExpertCartridgeBank(root)
            mapping = {f'E{i}': i for i in range(experts)}
            manifest = exporter.export_all_domain_cartridges(model, {k: {} for k in mapping}, {},
                                                            domain_expert_map=mapping)
            scenarios = {}
            for scenario in ('empty_app_cache', 'verified_cache_hit', 'eviction_cycle', 'install_with_eviction'):
                bank = SparseExpertCartridgeBank(root, max_cached_experts=2)
                times, hits = [], 0
                if scenario == 'verified_cache_hit':
                    bank.page_in_cartridge('E0')
                elif scenario in ('eviction_cycle', 'install_with_eviction'):
                    bank.page_in_cartridge('E1')
                    bank.page_in_cartridge('E2')
                for step in range(repeats):
                    if scenario == 'empty_app_cache':
                        bank = SparseExpertCartridgeBank(root, max_cached_experts=2)
                    domain = f'E{step % 3}' if 'eviction' in scenario else 'E0'
                    if scenario == 'install_with_eviction':
                        elapsed, result = timed(lambda: bank.install_cartridge(model, domain))
                    else:
                        elapsed, result = timed(lambda: bank.page_in_cartridge(domain))
                    times.append(elapsed)
                    hits += int(result['cache_hit'])
                    assert bank.cache_info()['entries'] <= 2
                    assert bank.cache_info()['bytes'] <= bank.max_cached_bytes
                scenarios[scenario] = dict(**summary(times), hits=hits,
                                          misses=repeats - hits, final_cache=bank.cache_info())
            results.append(dict(bank_experts=experts, scenarios=scenarios,
                                serialized_shard_bytes=manifest['cartridges']['E0']['serialized_bytes']))
            print(f"storage E={experts}: measured load, verified hit, eviction and install", flush=True)
    return results


def accuracy_experiment(checkpoint):
    from aix_zero.knowledge_is_power import build_21_domain_kip_v11_curriculum
    _, cases, _, _ = build_21_domain_kip_v11_curriculum(seed=42)
    model, tokenizer = load_v11(checkpoint)
    results = {}
    for assisted in (False, True):
        name = 'symbolic_assisted' if assisted else 'neural_only'
        records = []
        # Per-case seeds make sampling identical between modes; order alternates to
        # reduce thermal/load bias. Accuracy is not compared with earlier RNG streams.
        for index, case in enumerate(cases):
            pair = dict(domain=case['domain'], prompt=case['prompt'], gold=case['gold'], modes={})
            order = ('dense', 'sparse') if index % 2 == 0 else ('sparse', 'dense')
            for mode in order:
                model.set_expert_dispatch(mode)
                torch.manual_seed(1000 + index)
                elapsed, output = timed(lambda: model.generate_verified(
                    tokenizer.encode(case['prompt']), eos_id=tokenizer.eos_id,
                    max_new_tokens=40, num_candidates=2, temperature=.35,
                    tokenizer=tokenizer, inflight_alu=assisted))
                trace = tokenizer.decode(output[0], skip_special=True)
                prediction = extract_answer(trace)
                pair['modes'][mode] = dict(prediction=prediction, correct=prediction == case['gold'],
                                          trace=trace, latency_ms=elapsed)
            pair['same_trace'] = pair['modes']['dense']['trace'] == pair['modes']['sparse']['trace']
            records.append(pair)
        results[name] = dict(samples=records, same_traces=sum(r['same_trace'] for r in records), total=len(cases),
                             modes={mode: dict(correct=sum(r['modes'][mode]['correct'] for r in records),
                                  **summary([r['modes'][mode]['latency_ms'] for r in records]))
                                    for mode in ('dense', 'sparse')})
        print(f"v11 {name}: {results[name]['modes']}, same traces="
              f"{results[name]['same_traces']}/{len(cases)}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--skip-accuracy', action='store_true')
    parser.add_argument('--memory-worker', nargs=2, metavar=('EXPERTS', 'MODE'))
    args = parser.parse_args()
    if args.memory_worker:
        print(json.dumps(memory_worker(int(args.memory_worker[0]), args.memory_worker[1])))
        return
    if args.output is None or args.repeats < 3:
        parser.error('--output and at least 3 repeats are required')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    checkpoint = root / 'checkpoints/aix_zero_v11_kip.safetensors'
    counts = (4, 16, 64)
    report = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                  hardware=_hardware(), python=sys.version, torch=torch.__version__,
                  threads=1, repeats=args.repeats,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))},
                  limitations=['CPU F32 only; resident weights in both dispatch modes.',
                               'Synthetic 16/64 expert banks are not trained models.',
                               'Application-cache misses use OS-cached files, not cold-disk latency.',
                               'RSS includes framework/import overhead; peak increments are approximate.',
                               'No automatic disk routing or training-memory benchmark.'])
    report['kernels'] = kernel_experiments(counts, args.repeats)
    report['storage'] = storage_experiments(counts, args.repeats)
    report['memory'] = []
    for experts in counts:
        for mode in ('dense', 'sparse'):
            payload = subprocess.check_output([sys.executable, '-B', '-m', 'aix_zero.benchmark_dispatch',
                                               '--memory-worker', str(experts), mode], text=True)
            report['memory'].append(json.loads(payload))
    if not args.skip_accuracy:
        report['v11_accuracy'] = accuracy_experiment(checkpoint)
    _atomic_write(args.output, json.dumps(report, indent=2, allow_nan=False).encode())


if __name__ == '__main__':
    main()
