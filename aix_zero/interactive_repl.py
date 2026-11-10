#!/usr/bin/env python3
"""
AIX-Zero v8 Interactive Live REPL & Real-Time Teaching Shell (`aix_zero/interactive_repl.py`)
=============================================================================================
Interactive terminal interface for querying and teaching the standalone AIX-Zero model,
with measured inference and teaching telemetry.

Commands inside the REPL:
  * `<prompt>`                            — Run verified K-step inference on any math/logic/code/physics prompt.
  * `!teach <prompt> => <trace_and_ans>`  — Distill a new reasoning trace directly into AIX-Zero's weights in RAM.
  * `!expert <domain> [slot]`             — Install a validated expert snapshot into a routed slot.
  * `!status`                             — Show model parameter breakdown, vocabulary size, and checkpoint info.
  * `!save`                               — Save updated in-RAM weights to `aix_zero_v8.safetensors`.
  * `!quit` / `exit`                      — Exit the REPL.

Usage:
  Interactive:  PYTHONPATH=. python aix_zero/interactive_repl.py
  Demo mode:    PYTHONPATH=. python aix_zero/interactive_repl.py --demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
for sub in ("verisci", "rlvr-forge", "branchbox", "mlx-tree", "steerlens", "everbench"):
    sys.path.insert(0, str(REPO_ROOT / sub / "src"))

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel
from aix_zero.checkpoint_registry import load_checkpoint_strict, save_checkpoint_protected
from aix_zero.knowledge_is_power import CodexRetriever, WebKnowledgeFetcher
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
from aix_zero.tokenizer import AIXZeroTokenizer
from aix_zero.train_from_scratch import extract_answer
from aix_zero.train_v4_from_scratch import make_v4_weighted_batch

_CARTRIDGE_BANK = SparseExpertCartridgeBank()


def load_v8_model(
    device: torch.device,
    *,
    expand_to_tokenizer: bool = False,
) -> Tuple[AIXZeroModel, AIXZeroTokenizer, Path]:
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    v11_ckpt = ckpt_dir / "aix_zero_v11_kip.safetensors"
    v10_ckpt = ckpt_dir / "aix_zero_v10_kip.safetensors"
    v9_ckpt = ckpt_dir / "aix_zero_v9_kip.safetensors"
    v8_ckpt = ckpt_dir / "aix_zero_v8.safetensors"
    v7_ckpt = ckpt_dir / "aix_zero_v7_10min.safetensors"

    active_ckpt = v8_ckpt
    for cand in (v11_ckpt, v10_ckpt, v9_ckpt, v8_ckpt, v7_ckpt):
        if cand.exists():
            active_ckpt = cand
            break
    model, tokenizer, _ = load_checkpoint_strict(
        active_ckpt,
        device=device,
        allow_append_only_superset=True,
    )
    if expand_to_tokenizer and tokenizer.vocab_size > model.config.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
    model.eval()
    return model, tokenizer, active_ckpt


def run_query(model: AIXZeroModel, tokenizer: AIXZeroTokenizer, prompt: str) -> None:
    t0 = time.perf_counter()
    p_ids = tokenizer.encode(prompt.strip(), add_bos=True, add_eos=False)
    out_ids, val_conf, ponder_steps = model.generate_verified(
        p_ids,
        eos_id=tokenizer.eos_id,
        max_new_tokens=48,
        num_candidates=1,
        temperature=0.0,
        tokenizer=tokenizer,
        inflight_alu=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    decoded = tokenizer.decode(out_ids, skip_special=True)
    ans = extract_answer(decoded)
    think_part = ""
    if "<think>" in decoded and "</think>" in decoded:
        think_part = decoded.split("<think>", 1)[1].split("</think>", 1)[0]
    steps_count = len(think_part.split(";")) if think_part else 0
    codex_meta = CodexRetriever().lookup(prompt)

    print(f"  Prompt        : {prompt}")
    print(f"  Codex Lookup  : [{codex_meta['domain']}] {codex_meta['title']} | Law: `{codex_meta['formula']}`")
    print(f"  Full Trace    : {decoded}")
    print(f"  Answer        : {ans}  ({steps_count}-step chain)")
    print(
        f"  Telemetry     : Ponder={ponder_steps:.3f} steps | Verifier={val_conf:.4f} | Latency={elapsed_ms:.2f}ms"
    )
    print("-" * 88, flush=True)


def run_live_teach(
    model: AIXZeroModel,
    tokenizer: AIXZeroTokenizer,
    device: torch.device,
    teach_cmd: str,
) -> None:
    """Distills `prompt => <think>...</think><ans>...</ans>` into AIX-Zero's weights in RAM."""
    if "=>" not in teach_cmd:
        print("  [Usage] !teach <prompt> => <think>step1;step2</think><ans>val</ans>")
        return
    prompt_part, target_part = [x.strip() for x in teach_cmd.split("=>", 1)]
    full_pos = prompt_part + target_part

    t0 = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=0.01)
    model.train()
    batch_tuples: List[Tuple[str, str, float]] = [(prompt_part, full_pos, 1.0)] * 4
    input_ids, labels, verify_targets, token_weights = make_v4_weighted_batch(
        batch_tuples, tokenizer, device
    )

    init_loss = 0.0
    final_loss = 0.0
    for step in range(10):
        optimizer.zero_grad()
        out = model(
            input_ids,
            labels=labels,
            verify_targets=verify_targets,
            token_weights=token_weights,
        )
        loss = out["loss"]
        if step == 0:
            init_loss = float(out["ce_loss"].item())
        final_loss = float(out["ce_loss"].item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"  [LIVE PLASTICITY] Distilled `{prompt_part}` in 10 gradient steps ({elapsed_ms:.1f}ms) | "
        f"Weighted CE: {init_loss:.4f} -> {final_loss:.4f}"
    )
    # Document the newly learned lesson in the Knowledge Is Power Codex
    codex_dir = Path(__file__).resolve().parent / "knowledge_codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    live_log = codex_dir / "LIVE_TAUGHT_LESSONS.md"
    if not live_log.exists():
        live_log.write_text("# Knowledge Is Power: Live Interactive Lessons\n\n| Prompt | Verified Target Trace | Initial CE | Final CE | Latency |\n| :--- | :--- | :---: | :---: | :---: |\n")
    with live_log.open("a") as f:
        f.write(f"| `{prompt_part}` | `{target_part}` | `{init_loss:.4f}` | `{final_loss:.4f}` | `{elapsed_ms:.1f}ms` |\n")
    print(f"  [KNOWLEDGE IS POWER] Documented lesson in `{live_log.name}`")
    run_query(model, tokenizer, prompt_part)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "[EXPERIMENTAL RESEARCH INTERFACE] AIX-Zero Interactive REPL for inference, "
            "cartridge swapping, and in-RAM teaching. Source checkpoints are protected from "
            "accidental overwrite by default."
        )
    )
    parser.add_argument("--demo", action="store_true", help="Run non-interactive 13-domain + live-teaching showcase")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Disposable output checkpoint path for !save (default refuses to overwrite source checkpoint)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    model, tokenizer, ckpt_path = load_v8_model(device)
    params = model.count_parameters()

    print("=" * 88)
    print("  AIX-ZERO v8 INTERACTIVE NEURO-SYMBOLIC SHELL")
    print(
        f"  Parameters: {params['total_parameters']:,} physical ({params['effective_virtual_parameters_with_looping']:,} effective) | "
        f"Vocab: {tokenizer.vocab_size} tokens"
    )
    print("=" * 88, flush=True)

    if args.demo:
        demo_prompts = [
            "Q:(11+2)%3=",
            "Q:2*8>8?",
            "Q:Geom:w=2,h=6,p=5,A=",
            "Q:Phys:a=4,t=6,u=5,v=",
            "Q:Quad:3x^2+5@x=4=",
            "Q:Calc:d/dx(4x^2+3x)@x=5=",
            "Q:Seq:d=5,n=6,a0=3,an=",
            "Q:Chain3:4*5+7-2=",
            "Q:BridgeET:V=4,I=3,t=2,loss=5,E=",
            "Q:BridgeKW:F=5,d=4,K0=7,loss=3,E=",
            "Q:Chain4:3*4+5-2+6=",
        ]
        for p in demo_prompts:
            run_query(model, tokenizer, p)

        print("  [DEMO LIVE TEACHING] Teaching a 4-step chain live in RAM:")
        run_live_teach(
            model,
            tokenizer,
            device,
            "Q:Chain4:4*3+6-2+5= => <think>4*3=12;12+6=18;18-2=16;16+5=21</think><ans>21</ans>",
        )
        return

    while True:
        try:
            line = input("aix-zero:v11> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting AIX-Zero REPL.")
            break
        if not line:
            continue
        if line in ("!quit", "exit", "quit"):
            break
        if line == "!status":
            print(f"  Active Checkpoint : {ckpt_path}")
            print(f"  Vocab Size        : {tokenizer.vocab_size}")
            print(f"  Parameters        : {params}")
            print(f"  Expert cache      : {_CARTRIDGE_BANK.cache_info()}")
            continue
        if line.startswith("!expert "):
            fields = line.split()
            try:
                if len(fields) not in (2, 3):
                    raise ValueError("Usage: !expert <domain> [slot]")
                slot = int(fields[2]) if len(fields) == 3 else None
                print(_CARTRIDGE_BANK.install_cartridge(model, fields[1], slot=slot))
            except (ValueError, KeyError, OSError) as exc:
                print(f"  [EXPERT LOAD FAILED] {exc}")
            continue
        if line == "!save" or line.startswith("!save "):
            parts = line.split(maxsplit=1)
            target_out = Path(parts[1]) if len(parts) == 2 else (args.save_path or ckpt_path)
            try:
                saved = save_checkpoint_protected(
                    model,
                    target_out,
                    source_checkpoint_path=ckpt_path,
                    allow_overwrite_source=False,
                    allow_overwrite_indexed=False,
                )
                print(f"  [SAVED] Wrote live weights to {saved['saved_path']} (sha256={saved['sha256'][:12]})")
            except (PermissionError, ValueError, OSError) as exc:
                print(f"  [SAVE BLOCKED] {exc}")
            continue
        if line.startswith("!web "):
            topic = line[len("!web ") :].strip()
            res = WebKnowledgeFetcher.research_topic(topic)
            print(f"  [Wikipedia] {res['wikipedia']['title']} ({res['wikipedia']['url']}):\n    {res['wikipedia']['extract']}")
            print(f"  [arXiv]     {res['arxiv']['title']} ({res['arxiv']['url']}):\n    {res['arxiv']['summary']}")
            continue
        if line.startswith("!teach "):
            run_live_teach(model, tokenizer, device, line[len("!teach ") :].strip())
            continue
        run_query(model, tokenizer, line)


if __name__ == "__main__":
    main()
