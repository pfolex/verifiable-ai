"""Equation-Prefix Supervision and Separately Measured ALU-Trigger Evaluation (`[AIX NEURAL-03]` / Issue #430).

Teaches the model to extract operands/operators from the prompt and emit a valid checked
equation prefix (such as `<think>ratio=16/4=` or `<think>6*202=`), and evaluates three
distinct outcomes separately:
1. `neural_only_full_performance` (ALU off)
2. `neural_equation_setup_accuracy` (exact prefix match + semantic operand/operator correctness before ALU insertion)
3. `alu_assisted_final_performance` (bounded rational ALU execution; always labeled assisted)
"""
from __future__ import annotations

import copy
from fractions import Fraction
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from aix_zero.architecture import AIXZeroModel
from aix_zero.eval_harness_bridge import score_rational_or_probability
from aix_zero.meta_ai_autopilot import verify_r1_reasoning_trace
from aix_zero.routed_specialist_training import decode_generated_completion, wilson_score_interval


_ARITH_EXPR_RE = re.compile(
    r"^(?:(?P<label>[A-Za-z_][A-Za-z0-9_]*)=)?(?P<lhs>\-?\d+(?:/\d+)?)(?P<op>[\+\-\*\/\%])(?P<rhs>\-?\d+(?:/\d+)?)$"
)


def find_first_reasoning_equation_equal_char_offset(completion: str) -> int:
    """Locate the character index (inclusive) of the first arithmetic `=` inside `<think>...</think>`.

    Distinguishes optional variable-label assignment (`ratio=16/4=4` -> first `=` after `16/4`)
    from the arithmetic operator equation `=`. Never inspects the prompt.
    Raises `ValueError` if no valid `<think>...=` arithmetic equation prefix is present.
    """
    think_start = completion.find("<think>")
    think_end = completion.find("</think>")
    if think_start == -1 or think_end == -1 or think_end <= think_start:
        raise ValueError(f"Completion missing valid <think>...</think> region: {completion!r}")

    body_start = think_start + len("<think>")
    body = completion[body_start:think_end]
    first_step = body.split(";", 1)[0]

    # Find the `=` that immediately follows an arithmetic operator expression (`op1 [+|-|*|/|%] op2`)
    eq_positions = [i for i, ch in enumerate(first_step) if ch == "="]
    if not eq_positions:
        raise ValueError(f"First reasoning step contains no '=' sign: {first_step!r}")

    chosen_rel_idx: Optional[int] = None
    for pos in eq_positions:
        candidate_lhs = first_step[:pos].strip()
        if _ARITH_EXPR_RE.match(candidate_lhs):
            chosen_rel_idx = pos
            break

    if chosen_rel_idx is None:
        # Fallback if the equation has no binary operator before the first `=`, use the last `=` in first_step
        chosen_rel_idx = eq_positions[-1]

    return body_start + chosen_rel_idx


def extract_equation_prefix_and_ast(text_after_prompt: str) -> Dict[str, Any]:
    """Extract the first reasoning equation prefix (through the arithmetic `=`) and parse its operands/operator.

    Returns a structured dictionary with:
    - `has_think_open`: bool
    - `prefix_str`: str (e.g. `<think>ratio=16/4=` or `<think>6*202=`)
    - `raw_lhs_expr`: str (e.g. `ratio=16/4` or `6*202`)
    - `label`: Optional[str]
    - `operand_1`: Optional[str]
    - `operator`: Optional[str]
    - `operand_2`: Optional[str]
    - `valid_arithmetic_prefix`: bool
    - `parse_error`: Optional[str]
    """
    s = text_after_prompt.strip()
    think_idx = s.find("<think>")
    if think_idx == -1:
        return {
            "has_think_open": False,
            "prefix_str": "",
            "raw_lhs_expr": "",
            "label": None,
            "operand_1": None,
            "operator": None,
            "operand_2": None,
            "valid_arithmetic_prefix": False,
            "parse_error": "missing_think_open_tag",
        }

    after_think = s[think_idx + len("<think>") :]
    # Stop at the first `;` or `</think>` if present
    step_end = len(after_think)
    for sep in (";", "</think>", "<ans>"):
        idx = after_think.find(sep)
        if idx != -1 and idx < step_end:
            step_end = idx
    first_step = after_think[:step_end]

    eq_positions = [i for i, ch in enumerate(first_step) if ch == "="]
    if not eq_positions:
        return {
            "has_think_open": True,
            "prefix_str": "",
            "raw_lhs_expr": first_step,
            "label": None,
            "operand_1": None,
            "operator": None,
            "operand_2": None,
            "valid_arithmetic_prefix": False,
            "parse_error": "missing_equals_sign",
        }

    for pos in eq_positions:
        cand_lhs = first_step[:pos].strip()
        m = _ARITH_EXPR_RE.match(cand_lhs)
        if m:
            prefix_str = "<think>" + first_step[: pos + 1]
            return {
                "has_think_open": True,
                "prefix_str": prefix_str,
                "raw_lhs_expr": cand_lhs,
                "label": m.group("label"),
                "operand_1": m.group("lhs"),
                "operator": m.group("op"),
                "operand_2": m.group("rhs"),
                "valid_arithmetic_prefix": True,
                "parse_error": None,
            }

    first_lhs = first_step[: eq_positions[-1]].strip()
    return {
        "has_think_open": True,
        "prefix_str": "<think>" + first_step[: eq_positions[-1] + 1],
        "raw_lhs_expr": first_lhs,
        "label": None,
        "operand_1": None,
        "operator": None,
        "operand_2": None,
        "valid_arithmetic_prefix": False,
        "parse_error": "malformed_or_unsupported_operator_expression",
    }


def evaluate_bounded_equation_prefix(
    text_after_prompt: str,
    *,
    max_abs_operand: int = 10**12,
) -> Dict[str, Any]:
    """Bounded deterministic rational ALU evaluator for an equation prefix (`[AIX NEURAL-03]` / Issue #430).

    Never executes generated Python or unrestricted expressions. Supports `+`, `-`, `*`, `/`, `%`
    over bounded integers/fractions with explicit division-by-zero and magnitude checks.
    """
    t0 = time.perf_counter()
    parsed = extract_equation_prefix_and_ast(text_after_prompt)
    if not parsed["valid_arithmetic_prefix"]:
        return {
            "alu_triggered": False,
            "accepted": False,
            "rejection_reason": parsed["parse_error"] or "invalid_prefix",
            "computed_value": None,
            "completed_trace": None,
            "elapsed_us": round((time.perf_counter() - t0) * 1e6, 2),
            "parsed": parsed,
        }

    try:
        op1 = Fraction(str(parsed["operand_1"]))
        op2 = Fraction(str(parsed["operand_2"]))
    except (ValueError, ZeroDivisionError):
        return {
            "alu_triggered": True,
            "accepted": False,
            "rejection_reason": "invalid_rational_operand",
            "computed_value": None,
            "completed_trace": None,
            "elapsed_us": round((time.perf_counter() - t0) * 1e6, 2),
            "parsed": parsed,
        }

    if (
        abs(op1.numerator) > max_abs_operand
        or abs(op1.denominator) > max_abs_operand
        or abs(op2.numerator) > max_abs_operand
        or abs(op2.denominator) > max_abs_operand
    ):
        return {
            "alu_triggered": True,
            "accepted": False,
            "rejection_reason": "operand_magnitude_limit_exceeded",
            "computed_value": None,
            "completed_trace": None,
            "elapsed_us": round((time.perf_counter() - t0) * 1e6, 2),
            "parsed": parsed,
        }

    op = parsed["operator"]
    if op == "+":
        res = op1 + op2
    elif op == "-":
        res = op1 - op2
    elif op == "*":
        res = op1 * op2
    elif op == "/":
        if op2 == 0:
            return {
                "alu_triggered": True,
                "accepted": False,
                "rejection_reason": "division_by_zero",
                "computed_value": None,
                "completed_trace": None,
                "elapsed_us": round((time.perf_counter() - t0) * 1e6, 2),
                "parsed": parsed,
            }
        res = op1 / op2
    elif op == "%":
        if op2 == 0 or op1.denominator != 1 or op2.denominator != 1:
            return {
                "alu_triggered": True,
                "accepted": False,
                "rejection_reason": "invalid_modulo_operands",
                "computed_value": None,
                "completed_trace": None,
                "elapsed_us": round((time.perf_counter() - t0) * 1e6, 2),
                "parsed": parsed,
            }
        res = Fraction(op1.numerator % op2.numerator, 1)
    else:
        return {
            "alu_triggered": True,
            "accepted": False,
            "rejection_reason": f"unsupported_operator:{op}",
            "computed_value": None,
            "completed_trace": None,
            "elapsed_us": round((time.perf_counter() - t0) * 1e6, 2),
            "parsed": parsed,
        }

    val_str = str(res.numerator) if res.denominator == 1 else f"{res.numerator}/{res.denominator}"
    completed_trace = f"{parsed['prefix_str']}{val_str}</think><ans>{val_str}</ans>"
    return {
        "alu_triggered": True,
        "accepted": True,
        "rejection_reason": None,
        "computed_value": val_str,
        "completed_trace": completed_trace,
        "elapsed_us": round((time.perf_counter() - t0) * 1e6, 2),
        "parsed": parsed,
    }


def classify_equation_setup_vs_reference(
    generated_completion: str,
    reference_completion: str,
    reference_gold: str,
) -> Dict[str, Any]:
    """Compare a generated completion's equation prefix and continuation against reference (`[AIX NEURAL-03]` / #430).

    Distinguishes:
    - `exact_prefix_match`: exact string match of `<think>...=` prefix
    - `semantic_setup_correct`: matching operator and operands (allowing commutative order for `+` and `*`,
      requiring exact operand order for `-`, `/`, `%`)
    - `full_neural_answer_correct`: raw neural answer matches `reference_gold`
    - `full_neural_trace_verified`: raw neural trace passes `verify_r1_reasoning_trace`
    - `diagnostic_category`:
      - `'correct_setup_and_continuation'`
      - `'correct_setup_wrong_continuation'`
      - `'correct_arithmetic_wrong_operand_selection'`
      - `'malformed_alu_trigger'`
      - `'wrong_setup_and_wrong_answer'`
    """
    ref_parsed = extract_equation_prefix_and_ast(reference_completion)
    gen_parsed = extract_equation_prefix_and_ast(generated_completion)

    exact_prefix_match = bool(
        ref_parsed["valid_arithmetic_prefix"]
        and gen_parsed["valid_arithmetic_prefix"]
        and gen_parsed["prefix_str"] == ref_parsed["prefix_str"]
    )

    semantic_setup_correct = False
    if ref_parsed["valid_arithmetic_prefix"] and gen_parsed["valid_arithmetic_prefix"]:
        op_match = gen_parsed["operator"] == ref_parsed["operator"]
        if op_match:
            g_ops = (Fraction(str(gen_parsed["operand_1"])), Fraction(str(gen_parsed["operand_2"])))
            r_ops = (Fraction(str(ref_parsed["operand_1"])), Fraction(str(ref_parsed["operand_2"])))
            if ref_parsed["operator"] in ("+", "*"):
                semantic_setup_correct = (g_ops == r_ops) or (g_ops == (r_ops[1], r_ops[0]))
            else:
                semantic_setup_correct = (g_ops == r_ops)

    full_ans_ok = bool(
        score_rational_or_probability(generated_completion, reference_gold, is_probability=False)["score"] == 1.0
    )
    r1_check = verify_r1_reasoning_trace(generated_completion, reference_gold)
    full_trace_ok = bool(r1_check["fully_verified"])

    if not gen_parsed["valid_arithmetic_prefix"]:
        diag = "malformed_alu_trigger"
    elif semantic_setup_correct and full_ans_ok and full_trace_ok:
        diag = "correct_setup_and_continuation"
    elif semantic_setup_correct and (not full_ans_ok or not full_trace_ok):
        diag = "correct_setup_wrong_continuation"
    elif r1_check["equations_ok"] and not semantic_setup_correct:
        diag = "correct_arithmetic_wrong_operand_selection"
    else:
        diag = "wrong_setup_and_wrong_answer"

    return {
        "exact_prefix_match": exact_prefix_match,
        "semantic_setup_correct": semantic_setup_correct,
        "full_neural_answer_correct": full_ans_ok,
        "full_neural_trace_verified": full_trace_ok,
        "diagnostic_category": diag,
        "generated_parsed": gen_parsed,
        "reference_parsed": ref_parsed,
    }


def encode_equation_prefix_supervision_batch(
    items: List[Dict[str, Any]],
    tokenizer: Any,
    *,
    prefix_weight: float = 5.0,
    continuation_weight: float = 1.0,
    max_seq_len: int = 112,
) -> Dict[str, Any]:
    """Build token-aligned equation-prefix and continuation masks for training (`[AIX NEURAL-03]` / Issue #430).

    Guarantees:
    - BOS and all prompt tokens (including any `=` in the prompt!) are masked with `labels = 0` (`pad_id`),
      `prefix_mask = 0.0`, and `continuation_mask = 0.0`.
    - Completion tokens from `<think>` through the first arithmetic `=` in the reasoning region receive
      `prefix_mask = 1.0` and `token_weights = prefix_weight`.
    - Remaining completion tokens (RHS answer, `</think><ans>...</ans><eos>`) receive
      `continuation_mask = 1.0` and `token_weights = continuation_weight`.
    - Overlength items (`len(prompt_ids + completion_ids) > max_seq_len`) are skipped cleanly.
    """
    if prefix_weight <= 0.0 or continuation_weight <= 0.0:
        raise ValueError("prefix_weight and continuation_weight must be positive")
    if max_seq_len < 4:
        raise ValueError("max_seq_len must be >= 4")

    seqs: List[List[int]] = []
    lbls: List[List[int]] = []
    wgts: List[List[float]] = []
    pref_masks: List[List[float]] = []
    cont_masks: List[List[float]] = []
    skipped_overlength = 0

    for it in items:
        prompt = str(it["prompt"])
        completion = str(it["completion"])
        gold = str(it["gold"])
        # Independently verify training trace/answer labels before encoding
        r1 = verify_r1_reasoning_trace(completion, gold)
        if not r1["fully_verified"]:
            raise ValueError(f"Training item failed independent R1 equation/answer verification: {prompt!r} -> {completion!r}")

        eq_char_idx = find_first_reasoning_equation_equal_char_offset(completion)
        prefix_str = completion[: eq_char_idx + 1]
        continuation_str = completion[eq_char_idx + 1 :]

        p_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        pref_ids = tokenizer.encode(prefix_str, add_bos=False, add_eos=False)
        cont_ids = tokenizer.encode(continuation_str, add_bos=False, add_eos=True)
        c_ids = pref_ids + cont_ids
        full = p_ids + c_ids

        if len(full) > max_seq_len:
            skipped_overlength += 1
            continue

        p_len = len(p_ids)
        pref_len = len(pref_ids)
        cont_len = len(cont_ids)

        row_lbl = ([0] * p_len) + c_ids
        row_pref_mask = ([0.0] * p_len) + ([1.0] * pref_len) + ([0.0] * cont_len)
        row_cont_mask = ([0.0] * p_len) + ([0.0] * pref_len) + ([1.0] * cont_len)
        row_wgt = (
            ([0.0] * p_len)
            + ([float(prefix_weight)] * pref_len)
            + ([float(continuation_weight)] * cont_len)
        )

        if len(full) < max_seq_len:
            pad_n = max_seq_len - len(full)
            full = full + [0] * pad_n
            row_lbl = row_lbl + [0] * pad_n
            row_pref_mask = row_pref_mask + [0.0] * pad_n
            row_cont_mask = row_cont_mask + [0.0] * pad_n
            row_wgt = row_wgt + [0.0] * pad_n

        seqs.append(full)
        lbls.append(row_lbl)
        pref_masks.append(row_pref_mask)
        cont_masks.append(row_cont_mask)
        wgts.append(row_wgt)

    if not seqs:
        raise ValueError("No complete items fit within max_seq_len")

    return {
        "input_ids": torch.tensor(seqs, dtype=torch.long),
        "labels": torch.tensor(lbls, dtype=torch.long),
        "token_weights": torch.tensor(wgts, dtype=torch.float32),
        "prefix_mask": torch.tensor(pref_masks, dtype=torch.float32),
        "continuation_mask": torch.tensor(cont_masks, dtype=torch.float32),
        "prefix_weight": float(prefix_weight),
        "continuation_weight": float(continuation_weight),
        "encoding_summary": {
            "offered": len(items),
            "encoded": len(seqs),
            "skipped_overlength": skipped_overlength,
        },
    }


def compute_prefix_and_continuation_losses(
    model: AIXZeroModel,
    batch: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """Compute normalized `prefix_loss`, `continuation_loss`, and combined `total_loss`."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    token_weights = batch["token_weights"]
    prefix_mask = batch["prefix_mask"]
    continuation_mask = batch["continuation_mask"]

    out = model(input_ids, labels=labels, token_weights=token_weights)
    logits = out["logits"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_pref = prefix_mask[:, 1:].contiguous()
    shift_cont = continuation_mask[:, 1:].contiguous()

    raw_ce = F.cross_entropy(
        shift_logits.view(-1, model.config.vocab_size),
        shift_labels.view(-1),
        ignore_index=0,
        reduction="none",
    ).view(shift_labels.shape)

    active = (shift_labels != 0).float()
    pref_active = shift_pref * active
    cont_active = shift_cont * active

    prefix_loss = (raw_ce * pref_active).sum() / pref_active.sum().clamp(min=1.0)
    continuation_loss = (raw_ce * cont_active).sum() / cont_active.sum().clamp(min=1.0)

    return {
        "total_loss": out["loss"],
        "weighted_ce_loss": out["ce_loss"],
        "prefix_loss": prefix_loss,
        "continuation_loss": continuation_loss,
        "moe_aux_loss": out["moe_aux_loss"],
    }


def build_arithmetic_prefix_study_splits() -> Dict[str, List[Dict[str, Any]]]:
    """Build template/problem-family splits for equation-prefix supervision (`[AIX NEURAL-03]` / Issue #430).

    Includes:
    - `train`: canonical `META_QUANTIZATION_FP8_INT4` problems
    - `heldout_standard`: unseen operand ratios (`fp16_bits / quant_bits`)
    - `heldout_distractor_and_paraphrase`: prompts with distractor numbers (`layer_id=99`) and reversed field order
    - `adversarial_malformed_and_unsupported`: prompts/prefixes with division by zero or unsupported operators
    """
    train = [
        {
            "domain": "META_QUANTIZATION_FP8_INT4",
            "family": "canonical_quant",
            "prompt": "Q:MetaQuant:fp16_bits=16;quant_bits=4->compression?",
            "completion": "<think>ratio=16/4=4</think><ans>4</ans>",
            "gold": "4",
        },
        {
            "domain": "META_QUANTIZATION_FP8_INT4",
            "family": "canonical_quant",
            "prompt": "Q:MetaQuant:fp16_bits=16;quant_bits=8->compression?",
            "completion": "<think>ratio=16/8=2</think><ans>2</ans>",
            "gold": "2",
        },
        {
            "domain": "META_QUANTIZATION_FP8_INT4",
            "family": "canonical_quant",
            "prompt": "Q:MetaQuant:fp16_bits=32;quant_bits=4->compression?",
            "completion": "<think>ratio=32/4=8</think><ans>8</ans>",
            "gold": "8",
        },
        {
            "domain": "META_QUANTIZATION_FP8_INT4",
            "family": "canonical_quant",
            "prompt": "Q:MetaQuant:fp16_bits=32;quant_bits=8->compression?",
            "completion": "<think>ratio=32/8=4</think><ans>4</ans>",
            "gold": "4",
        },
    ]

    heldout_standard = [
        {
            "domain": "META_QUANTIZATION_FP8_INT4",
            "family": "heldout_standard_quant",
            "prompt": "Q:MetaQuant:fp16_bits=24;quant_bits=4->compression?",
            "completion": "<think>ratio=24/4=6</think><ans>6</ans>",
            "gold": "6",
        },
        {
            "domain": "META_QUANTIZATION_FP8_INT4",
            "family": "heldout_standard_quant",
            "prompt": "Q:MetaQuant:fp16_bits=24;quant_bits=8->compression?",
            "completion": "<think>ratio=24/8=3</think><ans>3</ans>",
            "gold": "3",
        },
        {
            "domain": "META_FLASH_ATTENTION_TILING",
            "family": "heldout_flash_tiling",
            "prompt": "Q:MetaFlash:seq_len=1024;tile_block=128->io_reduction?",
            "completion": "<think>tiles=1024/128=8</think><ans>8</ans>",
            "gold": "8",
        },
    ]

    heldout_distractor_and_paraphrase = [
        {
            "domain": "META_QUANTIZATION_FP8_INT4",
            "family": "distractor_number",
            "prompt": "Q:MetaQuant:layer_id=99;fp16_bits=16;quant_bits=4->compression?",
            "completion": "<think>ratio=16/4=4</think><ans>4</ans>",
            "gold": "4",
        },
        {
            "domain": "META_QUANTIZATION_FP8_INT4",
            "family": "operand_reversal_paraphrase",
            "prompt": "Q:MetaQuant:quant_bits=4;fp16_bits=32->compression?",
            "completion": "<think>ratio=32/4=8</think><ans>8</ans>",
            "gold": "8",
        },
    ]

    return {
        "train": train,
        "heldout_standard": heldout_standard,
        "heldout_distractor_and_paraphrase": heldout_distractor_and_paraphrase,
    }


def evaluate_three_way_prefix_and_alu_outcomes(
    model: AIXZeroModel,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    *,
    max_new_tokens: int = 36,
) -> Dict[str, Any]:
    """Evaluate and report the three distinct outcomes required by `[AIX NEURAL-03]` (Issue #430):

    1. `neural_only_full_performance`: Raw neural generation (`inflight_alu=False`), scoring full answer & R1 trace.
    2. `neural_equation_setup_accuracy`: Exact equation prefix match and semantic operand/operator accuracy
       before any ALU insertion.
    3. `alu_assisted_final_performance`: Bounded rational ALU completion on the model's generated prefix
       (always labeled `assistance_mode: 'bounded_alu_assisted'`, never unassisted neural reasoning).
    """
    was_training = model.training
    model.eval()

    records: List[Dict[str, Any]] = []
    by_domain: Dict[str, Dict[str, int]] = {}
    alu_calls = 0
    alu_accepted = 0
    alu_rejected = 0
    alu_rejection_reasons: Dict[str, int] = {}

    try:
        with torch.no_grad():
            for it in items:
                prompt = str(it["prompt"])
                ref_completion = str(it["completion"])
                gold = str(it["gold"])
                dom = str(it.get("domain", "UNKNOWN"))
                by_domain.setdefault(
                    dom,
                    {
                        "tested": 0,
                        "neural_only_answer_correct": 0,
                        "neural_only_trace_verified": 0,
                        "exact_prefix_match": 0,
                        "semantic_setup_correct": 0,
                        "alu_assisted_answer_correct": 0,
                        "alu_assisted_trace_verified": 0,
                    },
                )

                prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
                gen_ids, _, _ = model.generate_verified(
                    prompt_ids,
                    eos_id=tokenizer.eos_id,
                    max_new_tokens=max_new_tokens,
                    num_candidates=1,
                    temperature=0.0,
                    tokenizer=tokenizer,
                    inflight_alu=False,
                )
                dec = decode_generated_completion(
                    gen_ids,
                    prompt_ids,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    max_seq_len=model.config.max_seq_len,
                )
                raw_completion = dec["completion"]

                setup_eval = classify_equation_setup_vs_reference(raw_completion, ref_completion, gold)
                alu_res = evaluate_bounded_equation_prefix(raw_completion)
                if alu_res["alu_triggered"]:
                    alu_calls += 1
                    if alu_res["accepted"]:
                        alu_accepted += 1
                    else:
                        alu_rejected += 1
                        r_reason = str(alu_res["rejection_reason"] or "unknown")
                        alu_rejection_reasons[r_reason] = alu_rejection_reasons.get(r_reason, 0) + 1
                else:
                    alu_rejected += 1
                    r_reason = str(alu_res["rejection_reason"] or "not_triggered")
                    alu_rejection_reasons[r_reason] = alu_rejection_reasons.get(r_reason, 0) + 1

                alu_completion = alu_res["completed_trace"] if alu_res["accepted"] and alu_res["completed_trace"] else raw_completion
                alu_ans_ok = bool(
                    score_rational_or_probability(alu_completion, gold, is_probability=False)["score"] == 1.0
                )
                alu_trace_ok = bool(verify_r1_reasoning_trace(alu_completion, gold)["fully_verified"])

                by_domain[dom]["tested"] += 1
                by_domain[dom]["neural_only_answer_correct"] += int(setup_eval["full_neural_answer_correct"])
                by_domain[dom]["neural_only_trace_verified"] += int(setup_eval["full_neural_trace_verified"])
                by_domain[dom]["exact_prefix_match"] += int(setup_eval["exact_prefix_match"])
                by_domain[dom]["semantic_setup_correct"] += int(setup_eval["semantic_setup_correct"])
                by_domain[dom]["alu_assisted_answer_correct"] += int(alu_ans_ok)
                by_domain[dom]["alu_assisted_trace_verified"] += int(alu_trace_ok)

                records.append(
                    {
                        "domain": dom,
                        "family": it.get("family", "default"),
                        "prompt": prompt,
                        "gold": gold,
                        "reference_completion": ref_completion,
                        "neural_only_completion": raw_completion,
                        "generated_token_count": dec["generated_token_count"],
                        "termination_reason": dec["termination_reason"],
                        "neural_only_answer_correct": setup_eval["full_neural_answer_correct"],
                        "neural_only_trace_verified": setup_eval["full_neural_trace_verified"],
                        "exact_prefix_match": setup_eval["exact_prefix_match"],
                        "semantic_setup_correct": setup_eval["semantic_setup_correct"],
                        "diagnostic_category": setup_eval["diagnostic_category"],
                        "alu_triggered": alu_res["alu_triggered"],
                        "alu_accepted": alu_res["accepted"],
                        "alu_rejection_reason": alu_res["rejection_reason"],
                        "alu_assisted_completion": alu_completion,
                        "alu_assisted_answer_correct": alu_ans_ok,
                        "alu_assisted_trace_verified": alu_trace_ok,
                    }
                )
    finally:
        model.train(was_training)

    n = len(records)
    n_ans = sum(1 for r in records if r["neural_only_answer_correct"])
    n_trc = sum(1 for r in records if r["neural_only_trace_verified"])
    p_exact = sum(1 for r in records if r["exact_prefix_match"])
    p_sem = sum(1 for r in records if r["semantic_setup_correct"])
    a_ans = sum(1 for r in records if r["alu_assisted_answer_correct"])
    a_trc = sum(1 for r in records if r["alu_assisted_trace_verified"])

    return {
        "schema_version": "1.0.0",
        "tested_count": n,
        "outcomes": {
            "neural_only_full_performance": {
                "assistance_mode": "neural_only",
                "inflight_alu_enabled": False,
                "answer_correct_numerator": n_ans,
                "trace_verified_numerator": n_trc,
                "denominator": n,
                "answer_accuracy": round(n_ans / n, 4) if n else 0.0,
                "trace_accuracy": round(n_trc / n, 4) if n else 0.0,
                "answer_wilson_95_ci": [round(x, 4) for x in wilson_score_interval(n_ans, n)],
            },
            "neural_equation_setup_accuracy": {
                "assistance_mode": "neural_only_prefix_setup",
                "exact_prefix_numerator": p_exact,
                "semantic_setup_numerator": p_sem,
                "denominator": n,
                "exact_prefix_accuracy": round(p_exact / n, 4) if n else 0.0,
                "semantic_setup_accuracy": round(p_sem / n, 4) if n else 0.0,
                "semantic_wilson_95_ci": [round(x, 4) for x in wilson_score_interval(p_sem, n)],
                "meets_80_pct_experimental_target": (p_sem / n >= 0.80) if n else False,
            },
            "alu_assisted_final_performance": {
                "assistance_mode": "bounded_alu_assisted",
                "unassisted_neural_claim_prohibited": True,
                "answer_correct_numerator": a_ans,
                "trace_verified_numerator": a_trc,
                "denominator": n,
                "answer_accuracy": round(a_ans / n, 4) if n else 0.0,
                "trace_accuracy": round(a_trc / n, 4) if n else 0.0,
                "answer_wilson_95_ci": [round(x, 4) for x in wilson_score_interval(a_ans, n)],
            },
        },
        "alu_telemetry": {
            "alu_calls": alu_calls,
            "alu_accepted": alu_accepted,
            "alu_rejected": alu_rejected,
            "rejection_reasons": alu_rejection_reasons,
        },
        "per_domain_breakdown": by_domain,
        "records": records,
    }
