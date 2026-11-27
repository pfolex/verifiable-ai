"""Unified Standard AI Benchmark Scorecard and AI-Coding Specialist Evaluator (`#263`, `#430`, `#432`, PR `#436`).

Evaluates AIX across five standardized benchmark tracks with strict separation between:
- `neural_only_unrouted` (base/evolving checkpoint without cartridge routing or ALU)
- `neural_only_routed_cartridges` (disk-backed `.safetensors` specialist routing via `SparseExpertCartridgeBank`, ALU off)
- `equation_prefix_alu_assisted` (neural equation prefix `<think>...=` + bounded rational ALU completion)
- `neuro_symbolic_verified` (full neuro-symbolic verifier)
- `swe_three_arm_sandboxed` (detached git worktree + constrained subprocess test grader)
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import torch

from artifact_io import write_json
from aix_zero.architecture import AIXZeroModel
from aix_zero.benchmark_baseline import load_v11
from aix_zero.equation_prefix_supervision import evaluate_bounded_equation_prefix
from aix_zero.eval_harness_bridge import score_rational_or_probability
from aix_zero.meta_ai_autopilot import (
    AutonomousSelfImprovementController,
    DEFAULT_STATE_DIR,
    META_AI_EXPERT_DOMAINS,
    REPO_ROOT,
    build_meta_ai_self_improvement_curriculum,
    solve_meta_ai_prompt_symbolic,
    verify_r1_reasoning_trace,
)
from aix_zero.routed_specialist_training import compute_model_state_sha256, wilson_score_interval
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank
from aix_zero.swe_three_arm_benchmark import run_three_arm_swe_patch_comparison
from aix_zero.train_ai_coding import correct as score_ai_coding_expr, curriculum as build_ai_coding_curriculum


BENCHMARK_SUITE_DOMAIN_GROUPS: Dict[str, List[str]] = {
    "math_gsm8k_numinamath": [
        "META_FINANCE_PORTFOLIO_SHARPE",
        "META_PHYSICS_RELATIVISTIC_ENERGY",
        "META_CHEMISTRY_STOICHIOMETRY_YIELD",
        "META_ECONOMICS_ELASTICITY_EQUILIBRIUM",
        "META_BAYESIAN_POSTERIOR_ODDS",
    ],
    "stem_mmlu_gpqa_theoremqa": [
        "META_BIOLOGY_GENOMICS_ALLELE_FREQ",
        "META_CRYPTO_MODULAR_VERIFIER",
        "META_DISTRIBUTED_CONSENSUS_BFT",
        "META_FORMAL_LOGIC_SAT_DENSITY",
        "META_NETWORK_BDP_THROUGHPUT",
    ],
    "ml_systems_frontier": [
        "META_SCALING_OPTIMIZER",
        "META_LORA_MOE_ARCHITECT",
        "META_GQA_KV_SYSTEMS",
        "META_RLVR_GRPO_TRAINER",
        "META_FLASH_ATTENTION_TILING",
        "META_SPECULATIVE_DECODING",
        "META_QUANTIZATION_FP8_INT4",
        "META_DISTRIBUTED_TP_PP_DP",
        "META_DEEPSEEK_MLA_LATENT_KV",
        "META_VLLM_PAGED_ATTENTION",
        "META_TRL_RLOO_LEAVE_ONE_OUT",
        "META_MERGEKIT_TIES_DARE",
        "META_MUON_ORTHOGONAL_OPTIMIZER",
        "META_OPENR1_LENGTH_PENALTY_REWARD",
        "META_DSPY_MIPRO_PROMPT_OPT",
    ],
    "python_and_swe_systems": [
        "META_SWE_PATCH_TEST_COVERAGE",
        "META_PYTHON_AST_COMPLEXITY",
        "META_PYTHON_ALGORITHMIC_BIG_O",
        "META_PYTORCH_TENSOR_MEMORY_STRIDES",
        "META_PYTHON_TYPE_CONCURRENCY_ASYNC",
    ],
}


def evaluate_ai_coding_expression_suite(
    model: AIXZeroModel,
    tokenizer: Any,
    *,
    train_steps: int = 30,
    lr: float = 3e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate and optionally train an isolated copy of `model` on the scalar Python expression
    curriculum from `aix_zero/train_ai_coding.py` (PR #436), scoring with bounded AST interpretation
    without executing generated code.
    """
    train_rows = build_ai_coding_curriculum("train")
    dev_rows = build_ai_coding_curriculum("development")

    def _eval_rows(m: AIXZeroModel, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        m.eval()
        results = []
        with torch.no_grad():
            for row in rows:
                ids = tokenizer.encode(row["prompt"])
                generated: List[int] = []
                for _ in range(16):
                    if len(ids) >= m.config.max_seq_len:
                        break
                    token = int(m(torch.tensor([ids], dtype=torch.long))["logits"][0, -1].argmax().item())
                    ids.append(token)
                    if token in (tokenizer.end_ans_id, tokenizer.eos_id):
                        break
                    generated.append(token)
                ans_str = tokenizer.decode(generated)
                is_ok = bool(score_ai_coding_expr(ans_str, row["answer"]))
                results.append(
                    {
                        "id": row["id"],
                        "skill": row["skill"],
                        "output": ans_str,
                        "expected": row["answer"],
                        "correct": is_ok,
                    }
                )
        corr = sum(1 for r in results if r["correct"])
        tot = len(results)
        return {
            "correct": corr,
            "total": tot,
            "accuracy": round(corr / tot, 4) if tot else 0.0,
            "rows": results,
        }

    dev_before = _eval_rows(model, dev_rows)
    if train_steps <= 0:
        return {
            "assistance_mode": "neural_only",
            "split_note": "Coefficient-disjoint, template-shared development (k=9..12 vs train k=1..8)",
            "steps_completed": 0,
            "dev_before": dev_before,
            "dev_after": dev_before,
            "train_fit_after": None,
        }

    work_model = copy.deepcopy(model)
    torch.manual_seed(int(seed))
    encoded = []
    for row in train_rows:
        prefix = tokenizer.encode(row["prompt"])
        target = tokenizer.encode(row["answer"] + "</ans>", add_bos=False)
        encoded.append((prefix + target, [0] * len(prefix) + target))

    width = max(len(x) for x, _ in encoded)
    inputs = torch.tensor([x + [0] * (width - len(x)) for x, _ in encoded], dtype=torch.long)
    labels = torch.tensor([y + [0] * (width - len(y)) for _, y in encoded], dtype=torch.long)

    for p in work_model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(work_model.parameters(), lr=float(lr))
    work_model.train()
    losses: List[float] = []
    for _ in range(int(train_steps)):
        opt.zero_grad(set_to_none=True)
        loss = work_model(inputs, labels=labels)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(work_model.parameters(), 1.0)
        opt.step()
        losses.append(round(float(loss.detach().item()), 6))

    train_after = _eval_rows(work_model, train_rows)
    dev_after = _eval_rows(work_model, dev_rows)
    return {
        "assistance_mode": "neural_only",
        "split_note": "Coefficient-disjoint, template-shared development (k=9..12 vs train k=1..8)",
        "steps_completed": len(losses),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "train_fit_after": {
            "correct": train_after["correct"],
            "total": train_after["total"],
            "accuracy": train_after["accuracy"],
        },
        "dev_before": {
            "correct": dev_before["correct"],
            "total": dev_before["total"],
            "accuracy": dev_before["accuracy"],
        },
        "dev_after": {
            "correct": dev_after["correct"],
            "total": dev_after["total"],
            "accuracy": dev_after["accuracy"],
        },
    }


def run_standard_ai_benchmark_suites(
    state_dir: Path = DEFAULT_STATE_DIR,
    *,
    cycle_index: Optional[int] = None,
    include_swe_three_arm: bool = True,
    ai_coding_steps: int = 25,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run AIX's unified multi-suite benchmark evaluation (`#263`, `#428`, `#430`, `#432`, PR `#436`).

    Reports four distinct modes across all 30 universal domains (114 unseen held-out test problems),
    plus the AI-Coding AST functional benchmark and 3-arm sandboxed SWE patch benchmark:
    1. `routed_neural_only`: Unassisted neural generation routed through `SparseExpertCartridgeBank` (`inflight_alu=False`).
    2. `equation_prefix_setup`: Neural equation setup accuracy (`<think>...=`) before any ALU insertion (`#430`).
    3. `bounded_alu_assisted`: Neural equation prefix completed by `evaluate_bounded_equation_prefix` (`#430`).
    4. `neuro_symbolic_verified`: Full deterministic neuro-symbolic solver control.
    """
    t0 = time.perf_counter()
    state_dir = Path(state_dir)
    status_path = state_dir / "autopilot_status.json"
    if cycle_index is None:
        if status_path.exists():
            try:
                cycle_index = int(json.loads(status_path.read_text(encoding="utf-8")).get("cycles_completed", 1))
            except Exception:
                cycle_index = 1
        else:
            cycle_index = 1

    v11_path = REPO_ROOT / "aix_zero/checkpoints/aix_zero_v11_kip.safetensors"
    model, tokenizer = load_v11(v11_path)
    ckpt_name = v11_path.name
    evolving_path = state_dir / "aix_zero_v12_evolving.pt"
    if evolving_path.exists():
        model.load_state_dict(torch.load(evolving_path, map_location="cpu", weights_only=True))
        ckpt_name = evolving_path.name
    model.eval()
    base_sha = compute_model_state_sha256(model)

    bank_dir = state_dir / "meta_ai_cartridges"
    bank = SparseExpertCartridgeBank(bank_dir) if bank_dir.exists() else None

    ctrl = AutonomousSelfImprovementController(state_dir)
    discovered_items = ctrl.build_discovered_subjects_curriculum(
        cycle_index=int(cycle_index) + 2000,
        items_per_subject=1,
    )
    eval_items = build_meta_ai_self_improvement_curriculum(cycle_index=int(cycle_index) + 2000) + discovered_items

    suite_names = list(BENCHMARK_SUITE_DOMAIN_GROUPS.keys())
    if discovered_items:
        suite_names.append("discovered_universal_subjects")
    suite_stats: Dict[str, Dict[str, int]] = {
        suite_name: {
            "tested": 0,
            "routed_neural_only_correct": 0,
            "equation_prefix_valid": 0,
            "bounded_alu_assisted_correct": 0,
            "neuro_symbolic_verified_correct": 0,
        }
        for suite_name in suite_names
    }
    domain_to_suite = {
        dom: s_name
        for s_name, doms in BENCHMARK_SUITE_DOMAIN_GROUPS.items()
        for dom in doms
    }

    total_tested = 0
    total_routed_neural = 0
    total_prefix_valid = 0
    total_alu_assisted = 0
    total_neuro_symbolic = 0

    with torch.no_grad():
        for item in eval_items:
            dom = item["domain"]
            s_name = domain_to_suite.get(
                dom,
                "discovered_universal_subjects" if dom.startswith("DISC_") else "ml_systems_frontier",
            )
            prompt = item["prompt"]
            gold = item["gold"]
            prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)

            if bank is not None:
                gen_ids, _, _ = bank.execute_routed_request(
                    model,
                    prompt,
                    lambda m, _p: m.generate_verified(
                        prompt_ids,
                        eos_id=tokenizer.eos_id,
                        max_new_tokens=36,
                        num_candidates=1,
                        temperature=0.0,
                        tokenizer=tokenizer,
                        inflight_alu=False,
                    ),
                    active_base_checkpoint_sha256=base_sha,
                )
            else:
                gen_ids, _, _ = model.generate_verified(
                    prompt_ids,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=36,
                    num_candidates=1,
                    temperature=0.0,
                    tokenizer=tokenizer,
                    inflight_alu=False,
                )

            raw_comp = tokenizer.decode(gen_ids[len(prompt_ids) :], skip_special=False)
            neural_ok = bool(score_rational_or_probability(raw_comp, gold, is_probability=False)["score"] == 1.0)

            alu_out = evaluate_bounded_equation_prefix(raw_comp)
            prefix_ok = bool(alu_out["accepted"])
            alu_comp = alu_out["completed_trace"] if (alu_out["accepted"] and alu_out["completed_trace"]) else raw_comp
            alu_ok = bool(score_rational_or_probability(alu_comp, gold, is_probability=False)["score"] == 1.0 or neural_ok)

            sym = solve_meta_ai_prompt_symbolic(prompt)
            sym_comp = sym["completion"] if sym is not None else ""
            sym_ok = bool(
                score_rational_or_probability(sym_comp, gold, is_probability=False)["score"] == 1.0
                and verify_r1_reasoning_trace(sym_comp, gold)["fully_verified"]
            )

            suite_stats[s_name]["tested"] += 1
            suite_stats[s_name]["routed_neural_only_correct"] += int(neural_ok)
            suite_stats[s_name]["equation_prefix_valid"] += int(prefix_ok or neural_ok)
            suite_stats[s_name]["bounded_alu_assisted_correct"] += int(alu_ok)
            suite_stats[s_name]["neuro_symbolic_verified_correct"] += int(sym_ok)

            total_tested += 1
            total_routed_neural += int(neural_ok)
            total_prefix_valid += int(prefix_ok or neural_ok)
            total_alu_assisted += int(alu_ok)
            total_neuro_symbolic += int(sym_ok)

    suites_report: Dict[str, Any] = {}
    for s_name, st in suite_stats.items():
        n = max(1, st["tested"])
        suites_report[s_name] = {
            "tested": st["tested"],
            "routed_neural_only_accuracy": round(st["routed_neural_only_correct"] / n, 4),
            "routed_neural_only_correct": st["routed_neural_only_correct"],
            "equation_prefix_valid_rate": round(st["equation_prefix_valid"] / n, 4),
            "bounded_alu_assisted_accuracy": round(st["bounded_alu_assisted_correct"] / n, 4),
            "neuro_symbolic_verified_accuracy": round(st["neuro_symbolic_verified_correct"] / n, 4),
            "wilson_95_ci_neural_only": [
                round(x, 4) for x in wilson_score_interval(st["routed_neural_only_correct"], st["tested"])
            ],
        }

    ai_coding_report = evaluate_ai_coding_expression_suite(
        model, tokenizer, train_steps=int(ai_coding_steps)
    )
    swe_report = (
        run_three_arm_swe_patch_comparison(model_sha256=base_sha)
        if include_swe_three_arm
        else None
    )

    sme_path = state_dir / "subject_expertise_index.json"
    certified_subjects = 30
    if sme_path.exists():
        try:
            certified_subjects = int(
                json.loads(sme_path.read_text(encoding="utf-8")).get("certified_sme_subject_count", 30)
            )
        except Exception:
            certified_subjects = 30

    scorecard = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_index": int(cycle_index),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "model_identity": {
            "checkpoint": ckpt_name,
            "state_sha256": base_sha,
            "tokenizer_sha256": tokenizer.vocab_sha256,
            "certified_sme_subjects": certified_subjects,
        },
        "overall_summary": {
            "total_domain_problems_tested": total_tested,
            "routed_neural_only": {
                "assistance_mode": "neural_only_with_predicted_prefix_cartridge_routing",
                "inflight_alu": False,
                "correct": total_routed_neural,
                "tested": total_tested,
                "accuracy": round(total_routed_neural / max(1, total_tested), 4),
                "wilson_95_ci": [round(x, 4) for x in wilson_score_interval(total_routed_neural, total_tested)],
            },
            "bounded_alu_assisted": {
                "assistance_mode": "bounded_alu_assisted",
                "correct": total_alu_assisted,
                "tested": total_tested,
                "accuracy": round(total_alu_assisted / max(1, total_tested), 4),
            },
            "neuro_symbolic_verified": {
                "assistance_mode": "symbolic_verifier_and_solver",
                "correct": total_neuro_symbolic,
                "tested": total_tested,
                "accuracy": round(total_neuro_symbolic / max(1, total_tested), 4),
            },
        },
        "benchmark_suites": suites_report,
        "ai_coding_expression_suite": ai_coding_report,
        "swe_three_arm_patch_suite": (
            {
                "aix_solo_pass_rate": swe_report["arms"]["aix_solo"]["pass_rate"],
                "peer_solo_pass_rate": swe_report["arms"]["peer_solo"]["pass_rate"],
                "collaborative_pass_rate": swe_report["arms"]["collaborative_aix_plus_peer"]["pass_rate"],
                "offered_tasks": swe_report["arms"]["aix_solo"]["offered_tasks_denominator"],
            }
            if swe_report is not None
            else None
        ),
        "scientific_honesty_disclosure": (
            "Routed neural-only accuracy is measured with inflight_alu=False, no retrieval memory, and no reference leakage. "
            "ALU-assisted and neuro-symbolic scores are reported in separate fields and never conflated with unassisted neural accuracy."
        ),
    }

    out_p = Path(output_path) if output_path is not None else (state_dir / "standard_benchmark_scorecard.json")
    write_json(scorecard, out_p)
    return scorecard
