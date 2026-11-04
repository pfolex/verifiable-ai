"""Train AIX-Zero v5 100% from scratch with Causal 3-Gram Key Convolution & Domain-Agnostic In-Flight ALU.

Upgrades in AIX-Zero v5:
  1. Causal 3-Token Key Convolution (`use_key_ngram_conv=True`) inside `PromptPointerAttention`
     so multi-digit numbers in reverse-copy equations (`ax+b=c`, `x+y=S,x-y=D`) encode local
     token neighborhood context in their attention keys.
  2. Domain-Agnostic Prompt-Operand Set Conservation + Step-1 Accumulator Auto-Carry inside
     `evaluate_inflight_subexpr` (zero domain-specific regexes).
  3. Balanced 8-Domain Curriculum (`SYSTEM_2X2` expanded to 70+ problems).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.tokenizer import AIXZeroTokenizer
from aix_zero.train_v4_from_scratch import (
    build_v4_curriculum,
    evaluate_v4,
    make_v4_weighted_batch,
)


def build_v5_balanced_curriculum(seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Builds balanced 8-domain curriculum with expanded SYSTEM_2X2 coverage while keeping
    the exact same 40 held-out test problems from v4 for strict apple-to-apples comparison.
    """
    train_set, test_set = build_v4_curriculum(seed=seed)
    existing_prompts = {item["prompt"] for item in train_set} | {item["prompt"] for item in test_set}

    # Expand SYSTEM_2X2 training examples across wider integer ranges
    for x_val in range(4, 16):
        for y_val in range(1, x_val):
            s = x_val + y_val
            d = x_val - y_val
            prompt = f"Q:Sys:x+y={s},x-y={d},x="
            if prompt not in existing_prompts:
                two_x = s + d
                w_d = d + 2
                pos = f"<think>{s}+{d}={two_x};{two_x}/2={x_val}</think><ans>{x_val}</ans>"
                neg = f"<think>{s}+{w_d}={s+w_d};{s+w_d}/2={x_val+1}</think><ans>{x_val+1}</ans>"
                train_set.append(
                    {
                        "domain": "SYSTEM_2X2",
                        "prompt": prompt,
                        "pos": prompt + pos,
                        "neg": prompt + neg,
                        "gold": str(x_val),
                    }
                )
                existing_prompts.add(prompt)

    return train_set, test_set


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cpu")

    tokenizer = AIXZeroTokenizer(use_word_vocab=True)
    config = AIXZeroConfig(
        vocab_size=tokenizer.vocab_size,
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
        use_key_ngram_conv=True,
    )
    model = AIXZeroModel(config).to(device)
    param_stats = model.count_parameters()
    train_set, test_set = build_v5_balanced_curriculum(seed=42)

    print("=" * 84)
    print(f"[AIX-Zero v5] 3-Gram Key Conv + Domain-Agnostic In-Flight ALU + Balanced 8 Domains")
    print(f"[AIX-Zero v5] Vocab: {tokenizer.vocab_size} | Train: {len(train_set)} | Held-Out Test: {len(test_set)}")
    print(f"[AIX-Zero v5] Parameter Audit: {json.dumps(param_stats)}")
    print("=" * 84)

    optimizer = torch.optim.AdamW(model.parameters(), lr=4.5e-3, weight_decay=0.01)
    epochs = 24
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=3e-4)
    batch_size = 32
    t0 = time.time()
    loss_history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = list(train_set)
        random.shuffle(shuffled)
        ep_ce = 0.0
        ep_val = 0.0
        ep_ponder = 0.0
        steps = 0

        for i in range(0, len(shuffled), batch_size):
            chunk = shuffled[i : i + batch_size]
            batch_tuples: List[Tuple[str, str, float]] = [
                (item["prompt"], item["pos"], 1.0) for item in chunk
            ]
            for item in chunk[: max(1, len(chunk) // 4)]:
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
            ep_ponder += float(out["ponder_depth"].item())
            steps += 1

        scheduler.step()
        if epoch in (1, 5, 10, 16, epochs):
            avg_ce = ep_ce / max(1, steps)
            avg_val = ep_val / max(1, steps)
            avg_ponder = ep_ponder / max(1, steps)
            loss_history.append(
                {
                    "epoch": epoch,
                    "weighted_ce_loss": round(avg_ce, 4),
                    "verifier_bce_loss": round(avg_val, 4),
                    "mean_ponder_steps": round(avg_ponder, 3),
                }
            )
            print(
                f"  [Epoch {epoch:02d}/{epochs}] Operand-Weighted CE: {avg_ce:.4f} | "
                f"Verifier Loss: {avg_val:.4f} | Ponder Steps: {avg_ponder:.3f} | "
                f"Elapsed: {time.time() - t0:.1f}s"
            )

    unassisted_inflight = evaluate_v4(
        model, tokenizer, test_set, num_candidates=3, inflight_alu=True, grounded_alu=False
    )
    print(
        f"[AIX-Zero v5 - Unassisted In-Flight ALU (Zero Domain Regexes)] Held-Out 8-Domain Accuracy: "
        f"{unassisted_inflight['correct']}/{unassisted_inflight['total']} ({unassisted_inflight['accuracy_pct']}%) | "
        f"Domains: {json.dumps(unassisted_inflight['domain_accuracy_pct'])}"
    )

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    v5_ckpt = ckpt_dir / "aix_zero_v5.safetensors"
    state_dict_cpu = {
        k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()
    }
    save_file(state_dict_cpu, str(v5_ckpt))

    report = {
        "model_name": "AIX-Zero-v5 (3-Gram Key Conv + Domain-Agnostic In-Flight ALU + Auto-Carry)",
        "pretrained_weights_used": "NONE (100% From-Scratch N(0, 0.02^2) Gaussian init)",
        "vocab_size": tokenizer.vocab_size,
        "parameters": param_stats,
        "loss_history": loss_history,
        "unassisted_inflight_alu_eval": unassisted_inflight,
        "checkpoint_path": str(v5_ckpt),
    }
    report_path = ckpt_dir / "v5_training_metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[AIX-Zero v5] Saved weights to {v5_ckpt}")
    print(f"[AIX-Zero v5] Saved report to {report_path}")


if __name__ == "__main__":
    main()
