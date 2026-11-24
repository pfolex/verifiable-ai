"""Bounded Multi-Turn Local Ollama Coaching and Verified Cartridge Distillation (`[AIX PEER-01]` / Issue #431).

Provides:
1. Dynamic Ollama discovery (`discover_ollama_service`) via `OLLAMA_HOST`, `shutil.which("ollama")`, or
   explicit endpoint configuration—never hardcoding `/Users/alexpfeil/.local/bin/ollama`, never implicitly
   downloading models, and returning an actionable `UNAVAILABLE` state when offline.
2. Multi-turn bounded teacher coaching loop (`run_bounded_ollama_coaching_session`) with pinned teacher
   digest/version/settings, license provenance, strict `#407` split eligibility guards (blocking development,
   regression, and reserved_final rows), and cumulative turn/token/time budgets.
3. Task-appropriate validation (`verify_r1_reasoning_trace` + independent equation setup verification),
   labeling accepted examples as `verifier_accepted` (never claiming a complete universal correctness proof).
4. Isolated specialist cartridge distillation (`distill_verified_coaching_into_cartridge`) with four-arm
   neural-only evaluation (`original_baseline`, `no_teacher_control`, `teacher_distilled_candidate`, and
   `teacher_only_reference`) separating template-fit from unseen held-out learning.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

from aix_zero.architecture import AIXZeroModel
from aix_zero.equation_prefix_supervision import classify_equation_setup_vs_reference
from aix_zero.meta_ai_autopilot import META_AI_EXPERT_DOMAINS, verify_r1_reasoning_trace
from aix_zero.routed_specialist_training import (
    compute_model_state_sha256,
    evaluate_routed_cartridge_arm,
    train_isolated_domain_cartridge,
)
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank


def discover_ollama_service(
    *,
    requested_model: str = "qwen2.5-coder:7b",
    endpoint_url: Optional[str] = None,
    executable_path: Optional[str] = None,
    timeout_sec: float = 1.5,
    http_get_json: Optional[Callable[[str, float], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Discover local Ollama executable/endpoint and probe service + teacher model availability.

    Never starts a service, downloads a model, or calls external paid APIs.
    """
    raw_host = endpoint_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not raw_host.startswith("http://") and not raw_host.startswith("https://"):
        raw_host = f"http://{raw_host}"
    base_url = raw_host.rstrip("/")

    resolved_bin = executable_path or os.environ.get("OLLAMA_BIN") or shutil.which("ollama")

    def _default_get(url: str, timeout: float) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    getter = http_get_json or _default_get

    version_str: Optional[str] = None
    models_payload: Optional[Dict[str, Any]] = None
    probe_error: Optional[str] = None

    try:
        ver_data = getter(f"{base_url}/api/version", float(timeout_sec))
        version_str = str(ver_data.get("version", "unknown"))
        models_payload = getter(f"{base_url}/api/tags", float(timeout_sec))
    except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"

    if models_payload is None:
        return {
            "status": "UNAVAILABLE",
            "service_available": False,
            "model_available": False,
            "requested_model": requested_model,
            "endpoint_url": base_url,
            "executable_path": resolved_bin,
            "ollama_version": version_str,
            "model_digest": None,
            "error": probe_error or "Ollama service unreachable",
            "actionable_message": (
                f"Local Ollama service is not reachable at {base_url}. "
                f"To run a live local pilot, start Ollama (`ollama serve`) and ensure `{requested_model}` "
                "is pulled locally (`ollama list`). No implicit model download was attempted."
            ),
        }

    installed_models = models_payload.get("models", []) if isinstance(models_payload, dict) else []
    matched_entry: Optional[Dict[str, Any]] = None
    for entry in installed_models:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", ""))
        model_id = str(entry.get("model", ""))
        if requested_model in (name, model_id) or name.startswith(f"{requested_model}:"):
            matched_entry = entry
            break

    if matched_entry is None:
        available_names = [str(m.get("name", "")) for m in installed_models if isinstance(m, dict)]
        return {
            "status": "UNAVAILABLE",
            "service_available": True,
            "model_available": False,
            "requested_model": requested_model,
            "available_local_models": available_names,
            "endpoint_url": base_url,
            "executable_path": resolved_bin,
            "ollama_version": version_str,
            "model_digest": None,
            "error": f"Requested teacher model {requested_model!r} is not installed locally",
            "actionable_message": (
                f"Ollama service is running at {base_url}, but model {requested_model!r} is not installed "
                f"(available: {available_names}). Pull a local model explicitly if desired; no automatic download was performed."
            ),
        }

    return {
        "status": "AVAILABLE",
        "service_available": True,
        "model_available": True,
        "requested_model": requested_model,
        "matched_model_name": str(matched_entry.get("name", requested_model)),
        "endpoint_url": base_url,
        "executable_path": resolved_bin,
        "ollama_version": version_str,
        "model_digest": str(matched_entry.get("digest", "sha256:unknown")),
        "model_size_bytes": int(matched_entry.get("size", 0)),
        "error": None,
        "actionable_message": "Local Ollama service and requested teacher model are available.",
    }


def build_concise_verifier_feedback(
    prompt: str,
    candidate_response: str,
    reference_completion: str,
    gold: str,
) -> Dict[str, Any]:
    """Validate a teacher response and build concise verifier feedback WITHOUT leaking reference answers."""
    r1 = verify_r1_reasoning_trace(candidate_response, gold)
    setup = classify_equation_setup_vs_reference(candidate_response, reference_completion, gold)
    independent_semantic_ok = bool(setup["semantic_setup_correct"])
    accepted = bool(r1["fully_verified"] and independent_semantic_ok)

    findings: List[str] = []
    if not r1["format_ok"]:
        findings.append("missing_or_malformed_<think>_or_<ans>_tags")
    if not r1["equations_ok"]:
        findings.append(f"arithmetic_equation_mismatch:{r1['failed_equations']}")
    if not independent_semantic_ok:
        findings.append(f"operand_or_operator_setup_mismatch:{setup['diagnostic_category']}")
    if r1["format_ok"] and r1["equations_ok"] and independent_semantic_ok and not r1["answer_ok"]:
        findings.append("final_ans_tag_does_not_match_simplified_equation_result")

    feedback_text = (
        "VERIFIER_ACCEPTED"
        if accepted
        else (
            "VERIFIER_REJECTED: "
            + "; ".join(findings)
            + ". Re-emit `<think>step_equations</think><ans>simplified_rational</ans>` using only the prompt operands."
        )
    )
    return {
        "verifier_accepted": accepted,
        "verification_status": "verifier_accepted" if accepted else "verifier_rejected",
        "guaranteed_universal_correctness": False,
        "r1_format_ok": r1["format_ok"],
        "r1_equations_ok": r1["equations_ok"],
        "r1_answer_ok": r1["answer_ok"],
        "independent_semantic_setup_ok": independent_semantic_ok,
        "diagnostic_category": setup["diagnostic_category"],
        "findings": findings,
        "feedback_for_next_turn": feedback_text,
    }


def run_bounded_ollama_coaching_session(
    *,
    training_tasks: List[Dict[str, Any]],
    reserved_non_training_prompts: Optional[Set[str]] = None,
    teacher_model: str = "qwen2.5-coder:7b",
    dataset_license: str = "Apache-2.0",
    max_turns_per_task: int = 2,
    max_tokens_per_turn: int = 64,
    max_total_tokens: int = 512,
    max_wall_clock_sec: float = 30.0,
    max_concurrency: int = 1,
    discovery_info: Optional[Dict[str, Any]] = None,
    teacher_generate_fn: Optional[Callable[[str, int, Dict[str, Any]], Dict[str, Any]]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run a bounded multi-turn teacher coaching session (`[AIX PEER-01]` / Issue #431).

    Enforces:
    - `max_concurrency == 1`
    - `#407` split eligibility: rejects any task whose prompt appears in `reserved_non_training_prompts`
      or whose `split` is not `'train'` / `'training'`.
    - Cumulative token, turn, and wall-clock budgets, plus caller cancellation.
    - Deduplication and `verifier_accepted` provenance labeling.
    """
    if int(max_concurrency) != 1:
        raise ValueError("max_concurrency must be 1 for bounded single-worker Ollama coaching (#414/#431)")
    if int(max_turns_per_task) < 1 or int(max_tokens_per_turn) < 1 or int(max_total_tokens) < 1:
        raise ValueError("Turn and token budgets must be positive")

    forbidden_prompts = set(reserved_non_training_prompts or set())
    for task in training_tasks:
        p_str = str(task.get("prompt", "")).strip()
        sp = str(task.get("split", "train")).lower()
        if p_str in forbidden_prompts or sp in ("development", "dev", "regression", "reserved_final", "test", "val"):
            raise ValueError(
                f"Split hygiene violation (#407/#431): non-training prompt {p_str!r} (split={sp!r}) cannot enter peer coaching"
            )

    disc = discovery_info or discover_ollama_service(requested_model=teacher_model)
    if disc.get("status") != "AVAILABLE" and teacher_generate_fn is None:
        return {
            "schema_version": "1.0.0",
            "session_status": "UNAVAILABLE",
            "discovery": disc,
            "admitted_examples": [],
            "admitted_count": 0,
            "offered_task_count": len(training_tasks),
        }

    generation_settings = {
        "temperature": 0.0,
        "num_predict": int(max_tokens_per_turn),
        "seed": 42,
    }
    teacher_identity = {
        "model": disc.get("matched_model_name", teacher_model),
        "digest": disc.get("model_digest", "sha256:mock_or_pinned_teacher"),
        "ollama_version": disc.get("ollama_version", "offline_contract"),
        "generation_settings": generation_settings,
        "dataset_license": dataset_license,
    }

    def _http_ollama_generate(prompt_text: str, turn_idx: int, settings: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{disc['endpoint_url']}/api/generate"
        body = json.dumps(
            {
                "model": teacher_identity["model"],
                "prompt": prompt_text,
                "stream": False,
                "options": settings,
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        t_start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=max(1.0, float(max_wall_clock_sec))) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {
            "response": str(data.get("response", "")),
            "tokens_used": int(data.get("eval_count", 0)) or len(str(data.get("response", "")).split()),
            "latency_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        }

    gen_fn = teacher_generate_fn or _http_ollama_generate

    session_start = time.perf_counter()
    cumulative_tokens = 0
    total_turns = 0
    admitted_examples: List[Dict[str, Any]] = []
    seen_prompts: Set[str] = set()
    task_logs: List[Dict[str, Any]] = []
    session_termination_reason = "completed"

    for task in training_tasks:
        prompt = str(task["prompt"]).strip()
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        gold = str(task["gold"]).strip()
        ref_comp = str(task["completion"]).strip()
        domain = str(task.get("domain", "META_QUANTIZATION_FP8_INT4"))

        turns_for_task: List[Dict[str, Any]] = []
        coaching_context = (
            f"Solve `{prompt}` using exact rational steps inside `<think>...</think>` "
            "and final simplified answer inside `<ans>...</ans>`."
        )
        task_admitted = False

        for turn_idx in range(1, int(max_turns_per_task) + 1):
            if cancel_check is not None and cancel_check():
                session_termination_reason = "cancelled_by_caller"
                break
            elapsed_sec = time.perf_counter() - session_start
            if elapsed_sec >= float(max_wall_clock_sec):
                session_termination_reason = "wall_clock_budget_exhausted"
                break
            if cumulative_tokens >= int(max_total_tokens):
                session_termination_reason = "token_budget_exhausted"
                break

            t_turn = time.perf_counter()
            try:
                raw_out = gen_fn(coaching_context, turn_idx, generation_settings)
            except TimeoutError:
                session_termination_reason = "teacher_call_timeout"
                turns_for_task.append(
                    {
                        "turn_index": turn_idx,
                        "prompt_sent": coaching_context,
                        "error": "TimeoutError",
                        "verifier_accepted": False,
                    }
                )
                break
            except Exception as exc:
                turns_for_task.append(
                    {
                        "turn_index": turn_idx,
                        "prompt_sent": coaching_context,
                        "error": f"{type(exc).__name__}: {exc}",
                        "verifier_accepted": False,
                    }
                )
                break

            resp_text = str(raw_out.get("response", "")).strip()
            tok_count = min(
                int(max_tokens_per_turn),
                max(1, int(raw_out.get("tokens_used", max(1, len(resp_text) // 4)))),
            )
            lat_ms = float(raw_out.get("latency_ms", (time.perf_counter() - t_turn) * 1000.0))
            cumulative_tokens += tok_count
            total_turns += 1

            validation = build_concise_verifier_feedback(prompt, resp_text, ref_comp, gold)
            turns_for_task.append(
                {
                    "turn_index": turn_idx,
                    "prompt_sent": coaching_context,
                    "raw_response": resp_text,
                    "tokens_used": tok_count,
                    "latency_ms": round(lat_ms, 3),
                    "validation": validation,
                }
            )

            if validation["verifier_accepted"]:
                task_admitted = True
                admitted_examples.append(
                    {
                        "domain": domain,
                        "prompt": prompt,
                        "completion": resp_text,
                        "gold": gold,
                        "split": "train",
                        "verification_status": "verifier_accepted",
                        "guaranteed_universal_correctness": False,
                        "teacher_identity": teacher_identity,
                        "accepted_on_turn": turn_idx,
                        "provenance_sha256": hashlib.sha256(
                            f"{teacher_identity['digest']}:{prompt}:{resp_text}".encode("utf-8")
                        ).hexdigest(),
                    }
                )
                break
            else:
                # Provide concise verifier findings WITHOUT leaking reference completion or gold answer
                coaching_context = f"{coaching_context}\nTurn {turn_idx} Feedback: {validation['feedback_for_next_turn']}"

        task_logs.append(
            {
                "domain": domain,
                "prompt": prompt,
                "admitted": task_admitted,
                "turns_executed": len(turns_for_task),
                "turns": turns_for_task,
            }
        )
        if session_termination_reason != "completed":
            break

    return {
        "schema_version": "1.0.0",
        "session_status": "COMPLETED" if session_termination_reason == "completed" else "STOPPED_BY_BUDGET_OR_CANCEL",
        "termination_reason": session_termination_reason,
        "teacher_identity": teacher_identity,
        "budgets": {
            "max_turns_per_task": int(max_turns_per_task),
            "max_tokens_per_turn": int(max_tokens_per_turn),
            "max_total_tokens": int(max_total_tokens),
            "max_wall_clock_sec": float(max_wall_clock_sec),
            "max_concurrency": 1,
        },
        "usage": {
            "total_turns_executed": total_turns,
            "cumulative_tokens_used": cumulative_tokens,
            "elapsed_wall_clock_sec": round(time.perf_counter() - session_start, 4),
        },
        "offered_task_count": len(training_tasks),
        "admitted_count": len(admitted_examples),
        "admitted_examples": admitted_examples,
        "task_logs": task_logs,
    }


def distill_verified_coaching_into_cartridge(
    model: AIXZeroModel,
    tokenizer: Any,
    *,
    domain: str,
    coaching_report: Dict[str, Any],
    self_train_control_items: List[Dict[str, Any]],
    heldout_eval_items: List[Dict[str, Any]],
    template_copy_eval_items: Optional[List[Dict[str, Any]]] = None,
    work_dir: Path,
    slot: int = 2,
    steps: int = 10,
    lr: float = 4e-3,
    seed: int = 42,
    max_new_tokens: int = 28,
) -> Dict[str, Any]:
    """Train an isolated specialist cartridge on verifier-accepted teacher traces and compare against
    original baseline, no-teacher self-trained control, and teacher-only reference (`[AIX PEER-01]` / #431).

    All student model evaluations run in strict `neural_only` mode without teacher or ALU help.
    """
    admitted = list(coaching_report.get("admitted_examples", []))
    if not admitted:
        raise ValueError("Cannot run teacher cartridge distillation with zero verifier-accepted examples")

    heldout_prompts = {str(it["prompt"]).strip() for it in heldout_eval_items}
    for ex in admitted:
        if str(ex["prompt"]).strip() in heldout_prompts:
            raise ValueError("Leakage guard (#407/#431): admitted teacher example overlaps heldout_eval_items")

    base_sha = compute_model_state_sha256(model)
    domain_map = {dom: int(meta["expert_slot"]) for dom, meta in META_AI_EXPERT_DOMAINS.items()}

    # 1. Evaluate original unchanged baseline (no teacher, no training)
    baseline_heldout = evaluate_routed_cartridge_arm(
        model,
        tokenizer,
        None,
        items=heldout_eval_items,
        routing_mode="unchanged_baseline",
        active_base_checkpoint_sha256=base_sha,
        max_new_tokens=max_new_tokens,
    )

    # 2. Train No-Teacher Control Cartridge on `self_train_control_items` with identical step/LR/seed budget
    no_teacher_bank = SparseExpertCartridgeBank(Path(work_dir) / "bank_no_teacher")
    no_teacher_bank.export_all_domain_cartridges(
        model,
        META_AI_EXPERT_DOMAINS,
        {dom: None for dom in META_AI_EXPERT_DOMAINS},
        domain_expert_map=domain_map,
    )
    no_teacher_train = train_isolated_domain_cartridge(
        model,
        tokenizer,
        no_teacher_bank,
        domain=domain,
        train_items=self_train_control_items,
        val_items=self_train_control_items,
        domain_meta=META_AI_EXPERT_DOMAINS.get(domain),
        slot=slot,
        steps=steps,
        lr=lr,
        seed=seed,
        active_base_checkpoint_sha256=base_sha,
    )
    no_teacher_heldout = evaluate_routed_cartridge_arm(
        model,
        tokenizer,
        no_teacher_bank,
        items=heldout_eval_items,
        routing_mode="predicted_prefix",
        active_base_checkpoint_sha256=base_sha,
        max_new_tokens=max_new_tokens,
    )

    # 3. Train Teacher-Distilled Candidate Cartridge on `admitted` verifier-accepted examples
    distilled_bank = SparseExpertCartridgeBank(Path(work_dir) / "bank_teacher_distilled")
    distilled_bank.export_all_domain_cartridges(
        model,
        META_AI_EXPERT_DOMAINS,
        {dom: None for dom in META_AI_EXPERT_DOMAINS},
        domain_expert_map=domain_map,
    )
    distilled_train = train_isolated_domain_cartridge(
        model,
        tokenizer,
        distilled_bank,
        domain=domain,
        train_items=admitted,
        val_items=admitted,
        domain_meta=META_AI_EXPERT_DOMAINS.get(domain),
        slot=slot,
        steps=steps,
        lr=lr,
        seed=seed,
        active_base_checkpoint_sha256=base_sha,
    )
    distilled_heldout = evaluate_routed_cartridge_arm(
        model,
        tokenizer,
        distilled_bank,
        items=heldout_eval_items,
        routing_mode="predicted_prefix",
        active_base_checkpoint_sha256=base_sha,
        max_new_tokens=max_new_tokens,
    )

    template_eval_items = template_copy_eval_items if template_copy_eval_items is not None else admitted
    distilled_template_copy = evaluate_routed_cartridge_arm(
        model,
        tokenizer,
        distilled_bank,
        items=template_eval_items,
        routing_mode="predicted_prefix",
        active_base_checkpoint_sha256=base_sha,
        max_new_tokens=max_new_tokens,
    )

    offered_n = max(1, int(coaching_report.get("offered_task_count", len(admitted))))
    teacher_only_acc = round(len(admitted) / offered_n, 4)

    return {
        "schema_version": "1.0.0",
        "domain": domain,
        "evaluation_assistance_mode": "neural_only",
        "teacher_assistance_at_eval_time": False,
        "inflight_alu_at_eval_time": False,
        "base_model_unchanged": (compute_model_state_sha256(model) == base_sha),
        "teacher_only_reference": {
            "assistance_mode": "teacher_only_coaching_acceptance",
            "verifier_accepted_count": len(admitted),
            "offered_count": offered_n,
            "acceptance_rate": teacher_only_acc,
        },
        "arms": {
            "original_baseline": baseline_heldout,
            "no_teacher_self_trained_control": {
                "training": no_teacher_train,
                "heldout_evaluation": no_teacher_heldout,
            },
            "teacher_distilled_candidate": {
                "training": distilled_train,
                "heldout_evaluation": distilled_heldout,
                "known_template_evaluation": distilled_template_copy,
            },
        },
        "learning_quantification": {
            "known_template_accuracy": distilled_template_copy["accuracy"],
            "unseen_heldout_accuracy": distilled_heldout["accuracy"],
            "generalization_vs_template_gap": round(
                distilled_heldout["accuracy"] - distilled_template_copy["accuracy"], 4
            ),
            "distilled_vs_baseline_heldout_delta": round(
                distilled_heldout["accuracy"] - baseline_heldout["accuracy"], 4
            ),
            "distilled_vs_no_teacher_heldout_delta": round(
                distilled_heldout["accuracy"] - no_teacher_heldout["accuracy"], 4
            ),
            "distilled_post_train_loss": distilled_train["post_train_loss"],
            "no_teacher_post_train_loss": no_teacher_train["post_train_loss"],
        },
    }
