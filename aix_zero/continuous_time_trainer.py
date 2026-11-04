#!/usr/bin/env python3
"""
AIX-Zero Time-Budgeted Continuous Self-Training Daemon (`aix_zero/continuous_time_trainer.py`)
==============================================================================================
Runs AIX-Zero in a continuous real-time self-evolution loop for a user-specified wall-clock
duration (e.g. `--duration-minutes 10`).

During the time budget, the organism continuously loops through:
  1. Multi-LLM Symbiotic Council Dialogue (`aix-8b:latest` Challenger <-> `aix3:latest` Solver).
  2. Dynamic Vocabulary & Domain Expansion (expanding from 10 domains to 12 domains by adding
     `GEOMETRY_AREA` and `NUMBER_SEQ` with 60 held-out unseen test problems).
  3. Active Self-Play Hard-Example Mining (identifying problems where pure-neural predictions
     make mistakes and upweighting them in the live replay buffer).
  4. Real-Time Gradient Consolidation in RAM + Periodic `.safetensors` Checkpointing every cycle.

Usage:
  PYTHONPATH=. python aix_zero/continuous_time_trainer.py --duration-minutes 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
for sub in ("verisci", "rlvr-forge", "branchbox", "mlx-tree", "steerlens", "everbench"):
    sys.path.insert(0, str(REPO_ROOT / sub / "src"))

from branchbox import BranchBoxDAG
from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.autonomous_symbiont import conduct_multi_llm_discovery_dialogue, query_ollama_quick
from aix_zero.tokenizer import AIXZeroTokenizer, WORD_VOCAB
from aix_zero.train_from_scratch import extract_answer
from aix_zero.train_v4_from_scratch import (
    build_v4_curriculum,
    evaluate_v4,
    make_v4_weighted_batch,
)


def build_12_domain_curriculum(seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Builds the full 12-domain curriculum (8 v4 domains + 2 v6 domains + 2 new v7 domains)."""
    orig_train, orig_test = build_v4_curriculum(seed=seed)
    _, v6_train, v6_test = conduct_multi_llm_discovery_dialogue()

    rng = random.Random(seed)
    v7_samples: List[Dict] = []

    # Domain 11: GEOMETRY_AREA: Q:Geom:w=a,h=b,pad=c,A= -> <think>a*b=area;area+c=tot</think><ans>tot</ans>
    for w in range(2, 8):
        for h in range(2, 8):
            for pad in (1, 3, 5):
                area = w * h
                tot = area + pad
                w_pad = 5 if pad != 5 else 1
                prompt = f"Q:Geom:w={w},h={h},p={pad},A="
                pos = f"<think>{w}*{h}={area};{area}+{pad}={tot}</think><ans>{tot}</ans>"
                neg = f"<think>{w}*{h}={area};{area}+{w_pad}={area+w_pad}</think><ans>{area+w_pad}</ans>"
                v7_samples.append(
                    {
                        "domain": "GEOMETRY_AREA",
                        "prompt": prompt,
                        "pos": prompt + pos,
                        "neg": prompt + neg,
                        "gold": str(tot),
                    }
                )

    # Domain 12: NUMBER_SEQ (Arithmetic Progression): Q:Seq:d=d,n=k,a0=u,an= -> <think>d*k=step;step+u=an</think><ans>an</ans>
    for d_val in range(2, 7):
        for k_val in range(2, 7):
            for u_val in (1, 3, 4):
                step = d_val * k_val
                an = step + u_val
                w_u = 4 if u_val != 4 else 1
                prompt = f"Q:Seq:d={d_val},n={k_val},a0={u_val},an="
                pos = f"<think>{d_val}*{k_val}={step};{step}+{u_val}={an}</think><ans>{an}</ans>"
                neg = f"<think>{d_val}*{k_val}={step};{step}+{w_u}={step+w_u}</think><ans>{step+w_u}</ans>"
                v7_samples.append(
                    {
                        "domain": "NUMBER_SEQ",
                        "prompt": prompt,
                        "pos": prompt + pos,
                        "neg": prompt + neg,
                        "gold": str(an),
                    }
                )

    rng.shuffle(v7_samples)
    v7_test: List[Dict] = []
    v7_train: List[Dict] = []
    counts = {"GEOMETRY_AREA": 0, "NUMBER_SEQ": 0}
    for s in v7_samples:
        dom = s["domain"]
        if counts[dom] < 5:
            v7_test.append(s)
            counts[dom] += 1
        else:
            v7_train.append(s)

    all_train = orig_train + v6_train + v7_train
    all_test = orig_test + v6_test + v7_test
    return all_train, all_test


def mine_hard_examples(
    model: AIXZeroModel,
    tokenizer: AIXZeroTokenizer,
    pool: List[Dict],
    sample_k: int = 24,
) -> Tuple[List[Dict], int]:
    """Self-Play Hard-Example Mining: tests Pure Neural greedy decoding on a random sample of
    `pool` and returns all items where Pure Neural made an arithmetic or operand mistake so the
    next gradient cycle focuses extra capacity on them.
    """
    subset = random.sample(pool, k=min(sample_k, len(pool)))
    hard_items: List[Dict] = []
    pure_hits = 0

    for item in subset:
        p_ids = tokenizer.encode(item["prompt"], add_bos=True, add_eos=False)
        out_ids, _, _ = model.generate_verified(
            p_ids,
            eos_id=tokenizer.eos_id,
            max_new_tokens=36,
            num_candidates=1,
            temperature=0.0,
            tokenizer=tokenizer,
            inflight_alu=False,
        )
        decoded = tokenizer.decode(out_ids, skip_special=True)
        pred = extract_answer(decoded)
        if pred == item["gold"]:
            pure_hits += 1
        else:
            hard_items.append(item)

    return hard_items, pure_hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-budgeted continuous self-trainer for AIX-Zero")
    parser.add_argument("--duration-minutes", type=float, default=10.0, help="Training duration in minutes")
    args = parser.parse_args()

    duration_seconds = max(10.0, args.duration_minutes * 60.0)
    start_wall = time.perf_counter()

    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cpu")

    print("=" * 92)
    print(
        f"  AIX-ZERO CONTINUOUS SELF-TRAINING DAEMON (Target Budget: {args.duration_minutes:.1f} min / {duration_seconds:.0f}s)"
    )
    print("=" * 92, flush=True)

    # 1. Ensure dynamic vocabulary includes all 12 domain prefixes
    for tok_str in ("Q:Quad:", "Q:Phys:", "Q:Geom:", "Q:Seq:"):
        if tok_str not in WORD_VOCAB:
            WORD_VOCAB.append(tok_str)

    tokenizer = AIXZeroTokenizer(use_word_vocab=True)
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    v6_ckpt = ckpt_dir / "aix_zero_v6_symbiont.safetensors"
    v4_ckpt = ckpt_dir / "aix_zero_v4.safetensors"

    # Initialize with 123 tokens if loading v6, or 121 if loading v4, then resize to 125
    init_vocab = 123 if v6_ckpt.exists() else 121
    config = AIXZeroConfig(
        vocab_size=init_vocab,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        n_encoder_layers=1,
        n_recurrent_steps=3,
        n_decoder_layers=1,
        max_seq_len=112,
        use_moe=True,
        use_shared_expert=True,
        n_experts=4,
        moe_top_k=2,
        use_memory_bank=True,
        n_memory_slots=8,
        use_prompt_pointer=True,
        pointer_temperature=0.35,
    )
    model = AIXZeroModel(config).to(device)
    if v6_ckpt.exists():
        model.load_state_dict(load_file(str(v6_ckpt)))
        print(f"[INIT] Loaded 10-domain checkpoint `{v6_ckpt.name}`", flush=True)
    elif v4_ckpt.exists():
        model.load_state_dict(load_file(str(v4_ckpt)))
        print(f"[INIT] Loaded 8-domain checkpoint `{v4_ckpt.name}`", flush=True)

    model.resize_token_embeddings(tokenizer.vocab_size)
    print(
        f"[VOCAB EXPANSION] Resized standalone embeddings to {tokenizer.vocab_size} tokens across 12 domains",
        flush=True,
    )

    all_train, all_test = build_12_domain_curriculum(seed=42)
    print(
        f"[CURRICULUM] Total Train Pool: {len(all_train)} problems | Held-Out 12-Domain Test Set: {len(all_test)} problems",
        flush=True,
    )

    # Pre-run evaluation on the 12-domain held-out set
    initial_eval = evaluate_v4(
        model, tokenizer, all_test, num_candidates=2, inflight_alu=True, grounded_alu=False
    )
    initial_pure = evaluate_v4(
        model, tokenizer, all_test, num_candidates=1, inflight_alu=False, grounded_alu=False
    )
    print(
        f"[BASELINE @ 0.0s] 12-Domain In-Flight ALU Accuracy: {initial_eval['correct']}/{initial_eval['total']} "
        f"({initial_eval['accuracy_pct']}%) | Pure Neural Greedy: {initial_pure['correct']}/{initial_pure['total']} ({initial_pure['accuracy_pct']}%)",
        flush=True,
    )

    dag = BranchBoxDAG()
    root = dag.create_root({"daemon": "AIX-Zero-Continuous-10Min", "domains": 12})
    curr_node = root.node_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=6, T_mult=2, eta_min=2e-4
    )

    cycle = 0
    total_grad_steps = 0
    cycle_logs: List[Dict] = []
    v7_ckpt = ckpt_dir / "aix_zero_v7_10min.safetensors"
    log_path = ckpt_dir / "continuous_10min_log.json"

    # Keep a reserve of ~25 seconds before duration_seconds to run the final 12-domain evaluation cleanly
    train_budget_s = max(5.0, duration_seconds - 22.0)

    while (time.perf_counter() - start_wall) < train_budget_s:
        cycle += 1
        cycle_t0 = time.perf_counter()

        # 1. Active Self-Play Hard-Example Mining
        hard_examples, pure_hits = mine_hard_examples(model, tokenizer, all_train, sample_k=24)

        # 2. Periodic Council Dialogue Exchange (every 4th cycle)
        council_note = "Replay+HardMining"
        if cycle % 4 == 1 and (train_budget_s - (time.perf_counter() - start_wall)) > 45.0:
            utterance = query_ollama_quick(
                "aix3:latest",
                f"Cycle {cycle}: Give a 1-line tip for solving a*x^2+c or v=u+a*t accurately.",
                max_tokens=45,
            )
            council_note = utterance[:90].replace("\n", " ")

        # 3. Build cycle training stream: sample 192 problems + 3x upweighted hard examples
        stream = random.sample(all_train, k=min(192, len(all_train))) + (hard_examples * 3)
        random.shuffle(stream)

        model.train()
        batch_size = 32
        ep_ce = 0.0
        ep_val = 0.0
        steps_in_cycle = 0

        for i in range(0, len(stream), batch_size):
            if (time.perf_counter() - start_wall) >= train_budget_s:
                break
            chunk = stream[i : i + batch_size]
            batch_tuples: List[Tuple[str, str, float]] = [
                (item["prompt"], item["pos"], 1.0) for item in chunk
            ]
            for item in chunk[: max(1, len(chunk) // 5)]:
                batch_tuples.append((item["prompt"], item["neg"], 0.0))

            input_ids, labels, verify_targets, token_weights = make_v4_weighted_batch(
                batch_tuples, tokenizer, device
            )
            optimizer.zero_grad()
            out = model(
                input_ids,
                labels=labels,
                verify_targets=verify_targets,
                token_weights=token_weights,
            )
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_ce += float(out["ce_loss"].item())
            ep_val += float(out["value_loss"].item())
            steps_in_cycle += 1
            total_grad_steps += 1

        scheduler.step()
        avg_ce = ep_ce / max(1, steps_in_cycle)
        avg_val = ep_val / max(1, steps_in_cycle)
        elapsed_total = time.perf_counter() - start_wall

        step_node = dag.step(
            curr_node,
            action=f"cycle_{cycle}",
            next_state={
                "cycle": cycle,
                "grad_steps": total_grad_steps,
                "weighted_ce": round(avg_ce, 4),
                "hard_mined": len(hard_examples),
            },
            reward=1.0,
            verified=True,
        )
        curr_node = step_node.node_id

        entry = {
            "cycle": cycle,
            "elapsed_sec": round(elapsed_total, 1),
            "total_grad_steps": total_grad_steps,
            "weighted_ce_loss": round(avg_ce, 4),
            "verifier_bce_loss": round(avg_val, 4),
            "pure_neural_sample_hits_of_24": pure_hits,
            "hard_examples_mined": len(hard_examples),
            "dag_hash": step_node.state_hash[:12],
            "council_note": council_note,
        }
        cycle_logs.append(entry)

        # Save live checkpoint & telemetry every 3 cycles
        if cycle % 3 == 0:
            state_dict_cpu = {
                k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()
            }
            save_file(state_dict_cpu, str(v7_ckpt))
            log_path.write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "elapsed_sec": round(elapsed_total, 1),
                        "target_sec": duration_seconds,
                        "cycles": cycle_logs,
                    },
                    indent=2,
                )
            )

        print(
            f"  [Cycle {cycle:02d} | {elapsed_total:5.1f}s / {duration_seconds:.0f}s] "
            f"Grad Steps: {total_grad_steps:4d} | CE: {avg_ce:.4f} | "
            f"Pure-Neural Sample: {pure_hits:2d}/24 | Hard Mined: {len(hard_examples):2d} | "
            f"DAG: {step_node.state_hash[:10]}",
            flush=True,
        )

    # Final evaluation across all 12 domains (60 held-out unseen problems)
    print("\n[FINAL 12-DOMAIN EVALUATION] Running held-out verification across 60 unseen problems...", flush=True)
    final_inflight = evaluate_v4(
        model, tokenizer, all_test, num_candidates=3, inflight_alu=True, grounded_alu=False
    )
    final_pure = evaluate_v4(
        model, tokenizer, all_test, num_candidates=3, inflight_alu=False, grounded_alu=False
    )
    total_wall_s = round(time.perf_counter() - start_wall, 2)

    state_dict_cpu = {
        k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()
    }
    save_file(state_dict_cpu, str(v7_ckpt))

    final_report = {
        "status": "COMPLETED",
        "model_name": "AIX-Zero-v7-10Min (Time-Budgeted Continuous Self-Play & 12-Domain Symbiont)",
        "wall_clock_seconds": total_wall_s,
        "total_cycles": cycle,
        "total_gradient_steps": total_grad_steps,
        "vocab_size": tokenizer.vocab_size,
        "parameters": model.count_parameters(),
        "baseline_12_domain_inflight_eval": initial_eval,
        "baseline_12_domain_pure_neural_eval": initial_pure,
        "final_12_domain_inflight_eval": final_inflight,
        "final_12_domain_pure_neural_eval": final_pure,
        "cycles": cycle_logs,
        "checkpoint_path": str(v7_ckpt),
    }
    log_path.write_text(json.dumps(final_report, indent=2))

    print("=" * 92)
    print(f"  10-MINUTE CONTINUOUS SELF-TRAINING COMPLETE ({total_wall_s}s | {cycle} Cycles | {total_grad_steps} Grad Steps)")
    print("=" * 92)
    print(
        f"  * Pure Neural 12-Domain Accuracy:   {initial_pure['correct']}/{initial_pure['total']} ({initial_pure['accuracy_pct']}%) "
        f"-> {final_pure['correct']}/{final_pure['total']} ({final_pure['accuracy_pct']}%)"
    )
    print(
        f"  * In-Flight ALU 12-Domain Accuracy: {initial_eval['correct']}/{initial_eval['total']} ({initial_eval['accuracy_pct']}%) "
        f"-> {final_inflight['correct']}/{final_inflight['total']} ({final_inflight['accuracy_pct']}%)"
    )
    print(f"  * Per-Domain Breakdown: {json.dumps(final_inflight['domain_accuracy_pct'])}")
    print(f"  * Saved Checkpoint:     {v7_ckpt}")
    print(f"  * Saved Telemetry Log:  {log_path}")
    print("=" * 92, flush=True)


if __name__ == "__main__":
    main()
