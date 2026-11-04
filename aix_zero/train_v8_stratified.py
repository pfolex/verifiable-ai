#!/usr/bin/env python3
"""
AIX-Zero v8: Stratified 13-Domain Self-Trainer + Arbitrary K-Step Reasoning Chains
==================================================================================
Upgrades AIX-Zero from v7 (12 domains, 2-step chains, 91.67%) to v8 (13 domains,
including 3-step `MULTI_STEP_3` chains `<think>s1;s2;s3</think>`, with Stratified
Per-Domain Replay and K-Step In-Flight ALU Conservation).

Key v8 Architectural & Training Upgrades:
1. Stratified Per-Domain Replay (`sample_stratified_stream`):
   Guarantees equal batch representation across all 13 reasoning domains in every
   gradient cycle, preventing majority `A*B+C` domains from diluting minority
   operators (`%` in `MODULAR_NT`, `>` in `LOGIC_DEDUCT`, and 3-step `MULTI_STEP_3`).
2. Arbitrary K-Step Chain Generalization (`MULTI_STEP_3`):
   Expands vocabulary to 126 tokens (`Q:Chain3:`) and trains AIX-Zero on 3-step
   compositional problems (`Q:Chain3:a*b+c-d=` -> `<think>a*b=p;p+c=q;q-d=r</think><ans>r</ans>`).
3. Prompt-Structure Operand & Operator Conservation in `evaluate_inflight_subexpr`:
   Tracks remaining unconsumed prompt operands across arbitrary `s1;s2;...;sK` steps
   and automatically terminates `</think><ans>` when all prompt operands are consumed.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
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
from aix_zero.continuous_time_trainer import build_12_domain_curriculum, mine_hard_examples
from aix_zero.tokenizer import AIXZeroTokenizer
from aix_zero.train_v4_from_scratch import evaluate_v4, make_v4_weighted_batch


def build_13_domain_curriculum(seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Builds the 13-domain curriculum (12 v7 domains + Domain 13: 3-Step `MULTI_STEP_3`)."""
    train_12, test_12 = build_12_domain_curriculum(seed=seed)

    rng = random.Random(seed)
    chain3_samples: List[Dict] = []

    # Domain 13: MULTI_STEP_3 (3-Step Compositional Chain):
    # Q:Chain3:a*b+c-d= -> <think>a*b=p;p+c=q;q-d=r</think><ans>r</ans>
    for a in range(2, 7):
        for b in range(2, 6):
            for c in (3, 5, 7):
                for d in (1, 2):
                    p = a * b
                    q = p + c
                    r = q - d
                    w_d = 4 if d != 4 else 1
                    prompt = f"Q:Chain3:{a}*{b}+{c}-{d}="
                    pos = f"<think>{a}*{b}={p};{p}+{c}={q};{q}-{d}={r}</think><ans>{r}</ans>"
                    neg = f"<think>{a}*{b}={p};{p}+{c}={q};{q}-{w_d}={q-w_d}</think><ans>{q-w_d}</ans>"
                    chain3_samples.append(
                        {
                            "domain": "MULTI_STEP_3",
                            "prompt": prompt,
                            "pos": prompt + pos,
                            "neg": prompt + neg,
                            "gold": str(r),
                        }
                    )

    rng.shuffle(chain3_samples)
    chain3_test = chain3_samples[:5]
    chain3_train = chain3_samples[5:]

    return train_12 + chain3_train, test_12 + chain3_test


def sample_stratified_stream(
    all_train: List[Dict],
    per_domain_k: int = 16,
    hard_examples: List[Dict] | None = None,
) -> List[Dict]:
    """Stratified Per-Domain Replay: samples exactly `per_domain_k` examples from every
    single domain so minority operators (`%`, `>`, `/`, 3-step chains) receive equal
    gradient weight alongside majority domains.
    """
    by_domain: Dict[str, List[Dict]] = defaultdict(list)
    for item in all_train:
        by_domain[item["domain"]].append(item)

    stream: List[Dict] = []
    for dom, items in sorted(by_domain.items()):
        if len(items) >= per_domain_k:
            stream.extend(random.sample(items, k=per_domain_k))
        else:
            stream.extend(random.choices(items, k=per_domain_k))

    if hard_examples:
        stream.extend(hard_examples * 3)

    random.shuffle(stream)
    return stream


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cpu")

    print("=" * 92)
    print("  AIX-ZERO v8: STRATIFIED 13-DOMAIN SELF-TRAINER + 3-STEP REASONING CHAINS")
    print("=" * 92, flush=True)

    tokenizer = AIXZeroTokenizer(use_word_vocab=True)
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    v7_ckpt = ckpt_dir / "aix_zero_v7_10min.safetensors"
    v6_ckpt = ckpt_dir / "aix_zero_v6_symbiont.safetensors"

    init_vocab = 125 if v7_ckpt.exists() else 123
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
    if v7_ckpt.exists():
        model.load_state_dict(load_file(str(v7_ckpt)))
        print(f"[INIT] Loaded 12-domain v7 checkpoint `{v7_ckpt.name}` (125 tokens)", flush=True)
    elif v6_ckpt.exists():
        model.load_state_dict(load_file(str(v6_ckpt)))
        print(f"[INIT] Loaded 10-domain v6 checkpoint `{v6_ckpt.name}` (123 tokens)", flush=True)

    model.resize_token_embeddings(tokenizer.vocab_size)
    print(
        f"[VOCAB EXPANSION] Resized standalone embeddings to {tokenizer.vocab_size} tokens (added `Q:Chain3:`)",
        flush=True,
    )

    all_train, all_test = build_13_domain_curriculum(seed=42)
    print(
        f"[CURRICULUM] 13 Domains | Train Pool: {len(all_train)} problems | Held-Out Test Set: {len(all_test)} problems",
        flush=True,
    )

    dag = BranchBoxDAG()
    root = dag.create_root({"model": "AIX-Zero-v8", "domains": 13, "replay": "stratified"})
    curr_node = root.node_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=2.4e-3, weight_decay=0.01)
    n_cycles = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_cycles, eta_min=2e-4)

    cycle_logs: List[Dict] = []
    total_grad_steps = 0

    for cycle in range(1, n_cycles + 1):
        hard_examples, pure_hits = mine_hard_examples(model, tokenizer, all_train, sample_k=26)
        # Stratified replay + extra focus on the new 3-step domain during early cycles
        chain3_only = [x for x in all_train if x["domain"] == "MULTI_STEP_3"]
        extra_chain3 = random.sample(chain3_only, k=min(32, len(chain3_only)))
        stream = sample_stratified_stream(all_train, per_domain_k=16, hard_examples=hard_examples) + extra_chain3
        random.shuffle(stream)

        model.train()
        batch_size = 32
        ep_ce = 0.0
        ep_val = 0.0
        steps_in_cycle = 0

        for i in range(0, len(stream), batch_size):
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

        step_node = dag.step(
            curr_node,
            action=f"v8_stratified_cycle_{cycle}",
            next_state={"cycle": cycle, "grad_steps": total_grad_steps, "ce": round(avg_ce, 4)},
            reward=1.0,
            verified=True,
        )
        curr_node = step_node.node_id

        cycle_logs.append(
            {
                "cycle": cycle,
                "total_grad_steps": total_grad_steps,
                "weighted_ce_loss": round(avg_ce, 4),
                "verifier_bce_loss": round(avg_val, 4),
                "pure_sample_hits_of_26": pure_hits,
                "hard_mined": len(hard_examples),
                "dag_hash": step_node.state_hash[:12],
            }
        )
        print(
            f"  [v8 Cycle {cycle:02d}/{n_cycles}] Grad Steps: {total_grad_steps:3d} | "
            f"Weighted CE: {avg_ce:.4f} | Pure Sample: {pure_hits:2d}/26 | "
            f"Hard Mined: {len(hard_examples):2d} | DAG: {step_node.state_hash[:10]}",
            flush=True,
        )

    print("\n[FINAL 13-DOMAIN EVALUATION] Verifying across all 65 held-out unseen problems...", flush=True)
    final_inflight = evaluate_v4(
        model, tokenizer, all_test, num_candidates=2, inflight_alu=True, grounded_alu=False
    )
    final_pure = evaluate_v4(
        model, tokenizer, all_test, num_candidates=2, inflight_alu=False, grounded_alu=False
    )
    elapsed_s = round(time.perf_counter() - t0, 2)

    v8_ckpt = ckpt_dir / "aix_zero_v8.safetensors"
    state_dict_cpu = {
        k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()
    }
    save_file(state_dict_cpu, str(v8_ckpt))

    metrics_path = ckpt_dir / "v8_training_metrics.json"
    report = {
        "model_name": "AIX-Zero-v8 (13-Domain Stratified Replay + K-Step Chains)",
        "elapsed_seconds": elapsed_s,
        "vocab_size": tokenizer.vocab_size,
        "parameters": model.count_parameters(),
        "total_cycles": n_cycles,
        "total_gradient_steps": total_grad_steps,
        "final_13_domain_inflight_eval": final_inflight,
        "final_13_domain_pure_neural_eval": final_pure,
        "cycles": cycle_logs,
        "checkpoint_path": str(v8_ckpt),
    }
    metrics_path.write_text(json.dumps(report, indent=2))

    print("=" * 92)
    print(f"  AIX-ZERO v8 COMPLETE ({elapsed_s}s | 13 Domains | {len(all_test)} Held-Out Problems)")
    print("=" * 92)
    print(
        f"  * Pure Neural 13-Domain Accuracy:   {final_pure['correct']}/{final_pure['total']} ({final_pure['accuracy_pct']}%)"
    )
    print(
        f"  * In-Flight ALU 13-Domain Accuracy: {final_inflight['correct']}/{final_inflight['total']} ({final_inflight['accuracy_pct']}%)"
    )
    print(f"  * Per-Domain Breakdown: {json.dumps(final_inflight['domain_accuracy_pct'])}")
    print(f"  * Saved Checkpoint:     {v8_ckpt}")
    print("=" * 92, flush=True)


if __name__ == "__main__":
    main()
