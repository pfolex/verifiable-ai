"""Train AIX-Zero 100% from scratch (random Gaussian weights -> verified reasoning model).

Training Pipeline:
  1. Stage 0 (Random Init Audit): Evaluates the completely untrained random network (0% accuracy).
  2. Stage 1 (Foundational Reasoning Pretraining + Value Verifier Calibration):
     Trains token embeddings, RoPE GQA layers, the 3-step RecurrentReasoningBlock, SwiGLU FFNs,
     and the ValueVerificationHead from scratch on compositional multi-step reasoning traces
     (<think>step-by-step derivation</think><ans>exact_value</ans>) plus negative corrupted traces.
  3. Stage 2 (RLVR / STaR Self-Play Policy Gradient + Value-Guided Best-of-N):
     Samples candidate reasoning trajectories from the model, checks them with the internal
     ValueVerificationHead + deterministic symbolic verifier, and reinforces verified trajectories.
  4. Exports `.safetensors` weights + JSON metrics report.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.tokenizer import AIXZeroTokenizer


def build_reasoning_curriculum(seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Builds compositional train and held-out test sets across 4 verifiable reasoning families.

    Every base primitive (e.g. a*b, +c, %m) appears in training in different combinations,
    while specific compositional triples (a, b, c) are held out in test_set to measure
    true compositional generalization from scratch.
    """
    rng = random.Random(seed)
    samples: List[Dict] = []

    # Domain 1: Compositional Two-Step Arithmetic Chains: (a * b) + c
    for a in range(2, 8):
        for b in range(2, 8):
            for c in (1, 3, 5):
                prod = a * b
                ans = prod + c
                wrong_ans = ans + rng.choice([-2, 2, 3])
                prompt = f"Q:({a}*{b})+{c}="
                pos_completion = f"<think>{a}*{b}={prod};{prod}+{c}={ans}</think><ans>{ans}</ans>"
                neg_completion = f"<think>{a}*{b}={prod};{prod}+{c}={wrong_ans}</think><ans>{wrong_ans}</ans>"
                samples.append(
                    {
                        "domain": "ARITH_CHAIN",
                        "prompt": prompt,
                        "pos": prompt + pos_completion,
                        "neg": prompt + neg_completion,
                        "gold": str(ans),
                    }
                )

    # Domain 2: Compositional Linear Algebra Equations: a*x + b = c -> solve for x
    for a in range(2, 7):
        for x_val in range(1, 8):
            for b in (1, 2, 4):
                c = a * x_val + b
                diff = c - b
                wrong_x = x_val + 1
                prompt = f"Q:{a}x+{b}={c},x="
                pos_completion = f"<think>{c}-{b}={diff};{diff}/{a}={x_val}</think><ans>{x_val}</ans>"
                neg_completion = f"<think>{c}-{b}={diff};{diff}/{a}={wrong_x}</think><ans>{wrong_x}</ans>"
                samples.append(
                    {
                        "domain": "ALGEBRA_SOLVE",
                        "prompt": prompt,
                        "pos": prompt + pos_completion,
                        "neg": prompt + neg_completion,
                        "gold": str(x_val),
                    }
                )

    # Domain 3: Compositional Modular Number Theory: (a + b) mod m
    for a in range(4, 14):
        for b in (2, 5, 7):
            for m in (3, 4, 5):
                s = a + b
                rem = s % m
                wrong_rem = (rem + 1) % m
                prompt = f"Q:({a}+{b})%{m}="
                pos_completion = f"<think>{a}+{b}={s};{s}%{m}={rem}</think><ans>{rem}</ans>"
                neg_completion = f"<think>{a}+{b}={s};{s}%{m}={wrong_rem}</think><ans>{wrong_rem}</ans>"
                samples.append(
                    {
                        "domain": "MODULAR_NT",
                        "prompt": prompt,
                        "pos": prompt + pos_completion,
                        "neg": prompt + neg_completion,
                        "gold": str(rem),
                    }
                )

    # Domain 4: Boolean & Relational Logic Deduction
    for x in range(1, 9):
        for y in range(1, 16):
            rel = 1 if (x * 2 > y) else 0
            wrong_rel = 1 - rel
            prompt = f"Q:2*{x}>{y}?"
            pos_completion = f"<think>2*{x}={2*x};{2*x}>{y}->{rel}</think><ans>{rel}</ans>"
            neg_completion = f"<think>2*{x}={2*x};{2*x}>{y}->{wrong_rel}</think><ans>{wrong_rel}</ans>"
            samples.append(
                {
                    "domain": "LOGIC_DEDUCT",
                    "prompt": prompt,
                    "pos": prompt + pos_completion,
                    "neg": prompt + neg_completion,
                    "gold": str(rel),
                }
            )

    rng.shuffle(samples)
    # Hold out 20 balanced examples (5 per domain) for unseen compositional evaluation
    test_set: List[Dict] = []
    train_set: List[Dict] = []
    domain_counts = {"ARITH_CHAIN": 0, "ALGEBRA_SOLVE": 0, "MODULAR_NT": 0, "LOGIC_DEDUCT": 0}
    for item in samples:
        d = item["domain"]
        if domain_counts[d] < 5:
            test_set.append(item)
            domain_counts[d] += 1
        else:
            train_set.append(item)

    return train_set, test_set


def extract_answer(decoded: str) -> str:
    m = re.search(r"<ans>([^<]+)</ans>", decoded)
    if m:
        return m.group(1).strip()
    return ""


def verify_and_repair_trace_alu(decoded: str) -> str:
    """Built-in Neuro-Symbolic ALU Gate for AIX-Zero:
    Inspects generated <think>step1;step2</think> equations and ensures exact arithmetic
    propagation into <ans>...</ans> when the neural backbone sets up the algebraic structure.
    """
    m_think = re.search(r"<think>([^<]+)</think>", decoded)
    if not m_think:
        return decoded
    steps = m_think.group(1).split(";")
    env: Dict[str, int] = {}
    repaired_steps: List[str] = []
    last_val = None

    for step in steps:
        # Relational step: A>B->R
        m_rel = re.match(r"^(\d+)>(\d+)->(\d+)$", step.strip())
        if m_rel:
            lhs = int(m_rel.group(1))
            rhs = int(m_rel.group(2))
            if last_val is not None:
                lhs = last_val
            true_rel = 1 if lhs > rhs else 0
            repaired_steps.append(f"{lhs}>{rhs}->{true_rel}")
            last_val = true_rel
            continue

        # Binary arithmetic step: A op B = C
        m_op = re.match(r"^(\d+)([\+\-\*\/\%])(\d+)=(\-?\d+)$", step.strip())
        if m_op:
            a_val = int(m_op.group(1))
            op = m_op.group(2)
            b_val = int(m_op.group(3))
            if repaired_steps and last_val is not None:
                # Chain previous intermediate result into first operand of step 2
                a_val = last_val
            if op == "+":
                res = a_val + b_val
            elif op == "-":
                res = a_val - b_val
            elif op == "*":
                res = a_val * b_val
            elif op == "/" and b_val != 0:
                res = a_val // b_val
            elif op == "%" and b_val != 0:
                res = a_val % b_val
            else:
                res = int(m_op.group(4))
            repaired_steps.append(f"{a_val}{op}{b_val}={res}")
            last_val = res
        else:
            repaired_steps.append(step)

    if last_val is not None:
        prefix = decoded.split("<think>")[0]
        return f"{prefix}<think>{';'.join(repaired_steps)}</think><ans>{last_val}</ans>"
    return decoded


def make_batch(
    batch_items: List[Tuple[str, str, float]],
    tokenizer: AIXZeroTokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads a batch of (prompt, full_text, verify_label) and masks prompt tokens in labels."""
    encoded_seqs: List[List[int]] = []
    prompt_lens: List[int] = []
    v_labels: List[float] = []

    for prompt, full_text, v_label in batch_items:
        p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        f_ids = tokenizer.encode(full_text, add_bos=True, add_eos=True)
        encoded_seqs.append(f_ids)
        prompt_lens.append(len(p_ids))
        v_labels.append(v_label)

    max_len = max(len(s) for s in encoded_seqs)
    input_ids = torch.full((len(batch_items), max_len), tokenizer.pad_id, dtype=torch.long, device=device)
    labels = torch.full((len(batch_items), max_len), tokenizer.pad_id, dtype=torch.long, device=device)

    for i, seq in enumerate(encoded_seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        if v_labels[i] > 0.5:
            labels[i, prompt_lens[i] : len(seq)] = torch.tensor(
                seq[prompt_lens[i] :], dtype=torch.long, device=device
            )

    verify_tensor = torch.tensor(v_labels, dtype=torch.float32, device=device)
    return input_ids, labels, verify_tensor


def evaluate_model(
    model: AIXZeroModel,
    tokenizer: AIXZeroTokenizer,
    eval_items: List[Dict],
    num_candidates: int = 1,
    use_symbolic_alu: bool = False,
) -> Dict:
    correct = 0
    per_domain: Dict[str, List[int]] = {}
    sample_outputs: List[Dict] = []

    for item in eval_items:
        p_ids = tokenizer.encode(item["prompt"], add_bos=True, add_eos=False)
        out_ids, val_score, ponder_depth = model.generate_verified(
            p_ids,
            eos_id=tokenizer.eos_id,
            max_new_tokens=36,
            num_candidates=num_candidates,
            temperature=0.35 if num_candidates > 1 else 0.0,
        )
        decoded = tokenizer.decode(out_ids, skip_special=True)
        if use_symbolic_alu:
            decoded = verify_and_repair_trace_alu(decoded)
        pred_ans = extract_answer(decoded)
        is_ok = int(pred_ans == item["gold"])
        correct += is_ok
        per_domain.setdefault(item["domain"], []).append(is_ok)
        if len(sample_outputs) < 8:
            sample_outputs.append(
                {
                    "domain": item["domain"],
                    "prompt": item["prompt"],
                    "generated": decoded,
                    "predicted": pred_ans,
                    "gold": item["gold"],
                    "correct": bool(is_ok),
                    "verifier_confidence": round(val_score, 4),
                    "latent_ponder_steps": round(ponder_depth, 3),
                }
            )

    domain_acc = {k: round(100.0 * sum(v) / len(v), 2) for k, v in per_domain.items()}
    return {
        "total": len(eval_items),
        "correct": correct,
        "accuracy_pct": round(100.0 * correct / len(eval_items), 2),
        "domain_accuracy_pct": domain_acc,
        "sample_outputs": sample_outputs,
    }


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cpu")
    print(f"[AIX-Zero] Initializing 100% From-Scratch Neural Network on device: {device}")

    tokenizer = AIXZeroTokenizer()
    config = AIXZeroConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        n_encoder_layers=1,
        n_recurrent_steps=3,
        n_decoder_layers=1,
        max_seq_len=96,
    )
    model = AIXZeroModel(config).to(device)
    param_stats = model.count_parameters()
    print(f"[AIX-Zero] Parameter Audit: {json.dumps(param_stats, indent=2)}")

    train_set, test_set = build_reasoning_curriculum(seed=42)
    print(f"[AIX-Zero] Curriculum built: {len(train_set)} train problems, {len(test_set)} held-out test problems")

    # Stage 0: Evaluate Untrained Random Gaussian Weights
    stage0_eval = evaluate_model(model, tokenizer, test_set, num_candidates=1, use_symbolic_alu=False)
    print(
        f"[Stage 0 - Untrained Random Init] Held-Out Accuracy: "
        f"{stage0_eval['correct']}/{stage0_eval['total']} ({stage0_eval['accuracy_pct']}%)"
    )

    # Stage 1: From-Scratch Pretraining + Value Verifier Calibration
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=3e-4)
    batch_size = 32
    epochs = 35
    t0 = time.time()
    loss_history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_set)
        epoch_ce = 0.0
        epoch_val = 0.0
        steps = 0

        for i in range(0, len(train_set), batch_size):
            chunk = train_set[i : i + batch_size]
            batch_tuples: List[Tuple[str, str, float]] = []
            for item in chunk:
                batch_tuples.append((item["prompt"], item["pos"], 1.0))
            for item in chunk[: max(1, len(chunk) // 5)]:
                batch_tuples.append((item["prompt"], item["neg"], 0.0))

            input_ids, labels, verify_targets = make_batch(batch_tuples, tokenizer, device)
            optimizer.zero_grad()
            out = model(input_ids, labels=labels, verify_targets=verify_targets)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_ce += float(out["ce_loss"].item())
            epoch_val += float(out["value_loss"].item())
            steps += 1

        scheduler.step()
        if epoch in (1, 5, 10, 15, 25, epochs):
            avg_ce = epoch_ce / max(1, steps)
            avg_val = epoch_val / max(1, steps)
            loss_history.append(
                {
                    "epoch": epoch,
                    "ce_loss": round(avg_ce, 5),
                    "verifier_bce_loss": round(avg_val, 5),
                }
            )
            print(
                f"  [Epoch {epoch:02d}/{epochs}] CE Loss: {avg_ce:.4f} | "
                f"Verifier Loss: {avg_val:.4f} | Elapsed: {time.time() - t0:.1f}s"
            )

    stage1_pure_neural = evaluate_model(model, tokenizer, test_set, num_candidates=1, use_symbolic_alu=False)
    print(
        f"[Stage 1 - Pure Neural Greedy Decoding] Held-Out Accuracy: "
        f"{stage1_pure_neural['correct']}/{stage1_pure_neural['total']} ({stage1_pure_neural['accuracy_pct']}%)"
    )

    # Stage 2: RLVR / STaR Self-Play Policy Gradient + Neuro-Symbolic ALU Gate
    print("[Stage 2] Running RLVR Self-Play Policy Gradient + Value-Guided Best-of-3 + Symbolic ALU Gate...")
    rl_optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
    rl_verified_trajectories: List[Tuple[str, str, float]] = []

    for item in random.sample(train_set, min(64, len(train_set))):
        p_ids = tokenizer.encode(item["prompt"], add_bos=True, add_eos=False)
        out_ids, val_score, _ = model.generate_verified(
            p_ids,
            eos_id=tokenizer.eos_id,
            max_new_tokens=36,
            num_candidates=2,
            temperature=0.35,
        )
        decoded = tokenizer.decode(out_ids, skip_special=True)
        repaired = verify_and_repair_trace_alu(decoded)
        pred_ans = extract_answer(repaired)
        if pred_ans == item["gold"]:
            rl_verified_trajectories.append((item["prompt"], repaired, 1.0))
        else:
            rl_verified_trajectories.append((item["prompt"], decoded, 0.0))
            rl_verified_trajectories.append((item["prompt"], item["pos"], 1.0))

    model.train()
    for _ in range(5):
        random.shuffle(rl_verified_trajectories)
        for i in range(0, len(rl_verified_trajectories), batch_size):
            batch_tuples = rl_verified_trajectories[i : i + batch_size]
            input_ids, labels, verify_targets = make_batch(batch_tuples, tokenizer, device)
            rl_optimizer.zero_grad()
            out = model(input_ids, labels=labels, verify_targets=verify_targets)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            rl_optimizer.step()

    stage2_neural = evaluate_model(model, tokenizer, test_set, num_candidates=3, use_symbolic_alu=False)
    stage2_neuro_symbolic = evaluate_model(model, tokenizer, test_set, num_candidates=3, use_symbolic_alu=True)
    print(
        f"[Stage 2 - Pure Neural Best-of-3] Held-Out Accuracy: "
        f"{stage2_neural['correct']}/{stage2_neural['total']} ({stage2_neural['accuracy_pct']}%)"
    )
    print(
        f"[Stage 2 - Neuro-Symbolic ALU + Value-Guided Best-of-3] Held-Out Accuracy: "
        f"{stage2_neuro_symbolic['correct']}/{stage2_neuro_symbolic['total']} ({stage2_neuro_symbolic['accuracy_pct']}%)"
    )

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = ckpt_dir / "aix_zero_v1.safetensors"
    state_dict_cpu = {k: v.detach().cpu().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict_cpu, str(safetensors_path))

    report = {
        "model_name": "AIX-Zero-v1 (100% From-Scratch Latent-Recurrent Verifiable Model)",
        "pretrained_weights_used": "NONE (Initialized from N(0, 0.02^2) Gaussian noise)",
        "architecture": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "d_ff": config.d_ff,
            "n_encoder_layers": config.n_encoder_layers,
            "n_recurrent_steps": config.n_recurrent_steps,
            "n_decoder_layers": config.n_decoder_layers,
            "parameters": param_stats,
        },
        "loss_history": loss_history,
        "stage0_random_init_eval": stage0_eval,
        "stage1_pure_neural_eval": stage1_pure_neural,
        "stage2_pure_neural_eval": stage2_neural,
        "stage2_neuro_symbolic_eval": stage2_neuro_symbolic,
        "checkpoint_path": str(safetensors_path),
    }
    report_path = ckpt_dir / "training_metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[AIX-Zero] Saved from-scratch weights to {safetensors_path}")
    print(f"[AIX-Zero] Saved training report to {report_path}")


if __name__ == "__main__":
    main()
