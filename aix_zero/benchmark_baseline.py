"""Offline v11 baseline and optional expert snapshot export (no training/network calls).

Run from the repository root with python -m aix_zero.benchmark_baseline.
Reports full response latency, including the final verifier forward pass. The first
request is reported separately; warm results follow one excluded warmup per mode.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy
import safetensors
from safetensors.torch import load_file
import torch

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank, _atomic_write
from aix_zero.tokenizer import AIXZeroTokenizer
from aix_zero.train_from_scratch import extract_answer


def load_v11(checkpoint: Path, *, expected_tokenizer_metadata=None):
    state = load_file(str(checkpoint))
    config = AIXZeroConfig(
        vocab_size=state['token_emb.weight'].shape[0], use_moe=True,
        use_shared_expert=True, use_memory_bank=True, use_prompt_pointer=True)
    model = AIXZeroModel(config)
    model.load_state_dict(state, strict=True)
    model.eval()
    tokenizer = AIXZeroTokenizer(use_word_vocab=True)
    # Existing IDs are append-only. Do not expand embeddings with random weights.
    tokenizer.verify_checkpoint_compatibility(
        config.vocab_size,
        expected_tokenizer_metadata,
        allow_append_only_superset=True,
    )
    return model, tokenizer


def _report_path(path: Path) -> str:
    """Record a portable public identity without exposing a local home directory."""
    try:
        return str(path.resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return path.name


def _hardware():
    result = dict(platform=platform.platform(), machine=platform.machine(),
                  processor=platform.processor(), cpu_count=os.cpu_count())
    if platform.system() == 'Darwin':
        for key in ('hw.model', 'hw.memsize', 'machdep.cpu.brand_string'):
            result[key] = subprocess.check_output(['sysctl', '-n', key], text=True).strip()
    return result


def run_baseline(checkpoint: Path, *, repeats: int = 1, threads: int = 1,
                 candidates: int = 2, max_new_tokens: int = 40):
    if repeats < 1 or threads < 1 or candidates < 1 or max_new_tokens < 1:
        raise ValueError('Run settings must be positive')
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    model, tokenizer = load_v11(checkpoint)
    from aix_zero.knowledge_is_power import build_21_domain_kip_v11_curriculum
    _, cases, _, _ = build_21_domain_kip_v11_curriculum(seed=42)
    if any(max(tokenizer.encode(c['prompt'])) >= model.config.vocab_size for c in cases):
        raise ValueError('Evaluation prompt uses tokens not present in checkpoint')
    source_dir = Path(__file__).resolve().parent
    source_sha256 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source_dir.glob('*.py'))}
    result = dict(
        source_sha256=source_sha256,
        schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=_report_path(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        hardware=_hardware(), versions=dict(python=platform.python_version(), torch=torch.__version__,
                                         numpy=numpy.__version__, safetensors=safetensors.__version__),
        config=asdict(model.config), physical_parameters=model.count_parameters()['total_parameters'],
        tokenizer=dict(vocab_size=tokenizer.vocab_size, vocab_sha256=tokenizer.vocab_sha256,
                       checkpoint_prefix_sha256=tokenizer.prefix_sha256(model.config.vocab_size)),
        dataset=dict(name='v11 template-family held-out cases', seed=42, cases=len(cases),
                     sha256=hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()),
        settings=dict(repeats=repeats, threads=threads, num_candidates=candidates,
                      max_new_tokens=max_new_tokens, temperature=0.35 if candidates > 1 else 0.0,
                      device='cpu', expansion=False, first_request='not OS-cache-cold'),
        limitations=['Same-template held-out arithmetic; not broad reasoning evidence.',
                     'No disk-expert routing, memory-pressure study, or OS-cache-cold guarantee.'], modes={})

    def query(case, assisted):
        start = time.perf_counter()
        ids, confidence, ponder = model.generate_verified(
            tokenizer.encode(case['prompt']), eos_id=tokenizer.eos_id,
            max_new_tokens=max_new_tokens, num_candidates=candidates,
            temperature=0.35 if candidates > 1 else 0.0, tokenizer=tokenizer,
            inflight_alu=assisted)
        elapsed = (time.perf_counter() - start) * 1000
        trace = tokenizer.decode(ids, skip_special=True)
        prediction = extract_answer(trace)
        return dict(domain=case['domain'], prompt=case['prompt'], gold=case['gold'],
                    prediction=prediction, correct=prediction == case['gold'], trace=trace,
                    verifier_confidence=confidence, ponder_depth=ponder, latency_ms=elapsed)

    for name, assisted in [('neural_only', False), ('symbolic_assisted', True)]:
        torch.manual_seed(42)
        first = query(cases[0], assisted)
        query(cases[0], assisted)  # excluded warmup
        samples = [dict(query(case, assisted), repeat=repeat)
                   for repeat in range(repeats) for case in cases]
        timings = sorted(s['latency_ms'] for s in samples)
        by_domain = {}
        for domain in sorted({s['domain'] for s in samples}):
            selected = [s for s in samples if s['domain'] == domain]
            by_domain[domain] = dict(correct=sum(s['correct'] for s in selected), total=len(selected))
        correct = sum(s['correct'] for s in samples)
        summary = dict(correct=correct, total=len(samples), accuracy_pct=100 * correct / len(samples),
                       p50_response_ms=statistics.median(timings),
                       p95_response_ms=timings[max(0, math.ceil(.95 * len(timings)) - 1)],
                       per_domain=by_domain, first_request=first, samples=samples)
        result['modes'][name] = summary
        print(f"{name}: {correct}/{len(samples)}, p50={summary['p50_response_ms']:.2f} ms, "
              f"p95={summary['p95_response_ms']:.2f} ms", flush=True)
    return result, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).parent / 'checkpoints/aix_zero_v11_kip.safetensors')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--candidates', type=int, default=2)
    parser.add_argument('--export-dir', type=Path)
    args = parser.parse_args()
    report, model = run_baseline(args.checkpoint, repeats=args.repeats, threads=args.threads,
                                 candidates=args.candidates)
    if args.export_dir:
        bank = SparseExpertCartridgeBank(args.export_dir)
        names = {f'SOURCE_EXPERT_{i}': i for i in range(model.config.n_experts)}
        manifest = bank.export_all_domain_cartridges(model, {k: {} for k in names}, {},
                                                     domain_expert_map=names)
        report['snapshot_export'] = manifest
    _atomic_write(args.output, json.dumps(report, indent=2, allow_nan=False).encode())


if __name__ == '__main__':
    main()
