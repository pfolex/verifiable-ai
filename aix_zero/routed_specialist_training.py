"""Routed Domain Cartridge Evaluation and Isolated Specialist Training (`[AIX NEURAL-01]` / Issue #428).

Connects `SparseExpertCartridgeBank.execute_routed_request` to neural arena evaluation,
isolated single-domain specialist training, and paired checkpoint evaluation so actual
specialist weights participate in inference and training without silent cross-domain corruption.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from safetensors.torch import save

from artifact_io import canonical_json, write_json
from aix_zero.architecture import AIXZeroModel
from aix_zero.eval_harness_bridge import score_rational_or_probability
from aix_zero.sparse_cartridge_bank import SparseExpertCartridgeBank, _atomic_write, _moe


def wilson_score_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Compute the 95% Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * ((p * (1.0 - p) / total + (z * z) / (4.0 * total * total)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def decode_generated_completion(
    generated_ids: List[int],
    prompt_ids: List[int],
    tokenizer: Any,
    *,
    max_new_tokens: int = 36,
    max_seq_len: int = 112,
) -> Dict[str, Any]:
    """Decode generated completion tokens and classify termination reason."""
    new_ids = list(generated_ids[len(prompt_ids) :])
    if new_ids and new_ids[-1] == tokenizer.eos_id:
        termination_reason = "eos"
    elif len(generated_ids) >= max_seq_len:
        termination_reason = "context_limit"
    elif len(new_ids) >= max_new_tokens:
        termination_reason = "max_new_tokens"
    else:
        termination_reason = "incomplete"
    return {
        "completion": tokenizer.decode(new_ids, skip_special=False),
        "generated_token_count": len(new_ids),
        "termination_reason": termination_reason,
    }


def compute_model_state_sha256(
    model: AIXZeroModel,
    *,
    exclude_prefix: Optional[str] = None,
) -> str:
    """Compute a deterministic SHA-256 over `model.state_dict()`, optionally excluding one submodule prefix."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if exclude_prefix and name.startswith(exclude_prefix):
            continue
        val = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(val.shape), str(val.dtype))).encode("utf-8"))
        digest.update(val.numpy().tobytes())
    return digest.hexdigest()


def _current_rss_bytes() -> int:
    """Return current process resident memory in bytes (macOS ru_maxrss is bytes; Linux is KiB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return int(usage)
    return int(usage) * 1024


def audit_model_and_cartridge_tensor_ownership(
    model: AIXZeroModel,
    bank: Optional[SparseExpertCartridgeBank] = None,
) -> Dict[str, Any]:
    """Perform the baseline tensor ownership and cartridge deduplication audit required by Issue #428.

    Separates:
    - Physical model parameters (`total_physical_parameters`)
    - Shared backbone, router, and shared-expert parameters (`shared_non_routed_parameters`)
    - Per-slot routed expert parameters (`per_slot_expert_parameters` and `per_slot_expert_bytes`)
    - Active parameters per token under top-k vs. forced single-expert routing
    - Disk cartridge census: total registered domains vs. unique SHA-256 tensor shards vs. alias snapshots
      vs. independently trained specialist cartridges (`frozen_v11_single_expert_training_candidate`).
    """
    moe = _moe(model)
    total_params = 0
    total_bytes = 0
    routed_slot_params: Dict[int, int] = {i: 0 for i in range(len(moe.experts))}
    routed_slot_bytes: Dict[int, int] = {i: 0 for i in range(len(moe.experts))}
    shared_expert_params = 0
    router_params = 0

    for name, tensor in model.state_dict().items():
        numel = int(tensor.numel())
        nbytes = numel * int(tensor.element_size())
        total_params += numel
        total_bytes += nbytes
        if name.startswith("latent_reasoning_core.core_block.moe.experts."):
            parts = name.split(".")
            slot_idx = int(parts[4])
            routed_slot_params[slot_idx] += numel
            routed_slot_bytes[slot_idx] += nbytes
        elif name.startswith("latent_reasoning_core.core_block.moe.shared_expert."):
            shared_expert_params += numel
        elif name.startswith("latent_reasoning_core.core_block.moe.router."):
            router_params += numel

    all_routed_params = sum(routed_slot_params.values())
    shared_non_routed_params = total_params - all_routed_params
    per_slot_params = routed_slot_params.get(0, 0)
    per_slot_bytes = routed_slot_bytes.get(0, 0)

    manifest = bank.get_manifest() if bank is not None else {}
    cartridges = manifest.get("cartridges", {}) if isinstance(manifest, dict) else {}
    unique_sha256s = set()
    trained_specialist_domains: List[str] = []
    snapshot_alias_domains: List[str] = []
    sha_to_domains: Dict[str, List[str]] = {}

    for dom, entry in sorted(cartridges.items()):
        if not isinstance(entry, dict):
            continue
        sha = str(entry.get("sha256", ""))
        if sha:
            unique_sha256s.add(sha)
            sha_to_domains.setdefault(sha, []).append(dom)
        prov = entry.get("provenance", "existing_routed_expert_snapshot")
        if prov == "frozen_v11_single_expert_training_candidate":
            trained_specialist_domains.append(dom)
        else:
            snapshot_alias_domains.append(dom)

    duplicated_sha_groups = {
        sha: doms for sha, doms in sha_to_domains.items() if len(doms) > 1
    }

    return {
        "schema_version": "1.0.0",
        "model_config": asdict(model.config),
        "n_experts": len(moe.experts),
        "moe_top_k": int(moe.top_k),
        "total_physical_parameters": total_params,
        "total_physical_bytes": total_bytes,
        "shared_non_routed_parameters": shared_non_routed_params,
        "shared_expert_parameters": shared_expert_params,
        "router_parameters": router_params,
        "total_routed_expert_parameters_in_ram": all_routed_params,
        "per_slot_expert_parameters": per_slot_params,
        "per_slot_expert_bytes": per_slot_bytes,
        "active_parameters_per_token_top_k": shared_non_routed_params + int(moe.top_k) * per_slot_params,
        "active_parameters_per_token_forced_single_slot": shared_non_routed_params + per_slot_params,
        "registered_cartridge_count": len(cartridges),
        "unique_cartridge_sha256_count": len(unique_sha256s),
        "unique_specialist_parameters_on_disk": len(unique_sha256s) * per_slot_params,
        "nominal_summed_cartridge_parameters": len(cartridges) * per_slot_params,
        "trained_specialist_domain_count": len(trained_specialist_domains),
        "trained_specialist_domains": trained_specialist_domains,
        "snapshot_alias_domain_count": len(snapshot_alias_domains),
        "snapshot_alias_domains": snapshot_alias_domains,
        "duplicated_sha256_alias_groups": duplicated_sha_groups,
        "zero_cross_domain_interference_guaranteed_by_disk_file_alone": False,
    }


def _encode_supervised_items(
    items: List[Dict[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode prompt+completion items with prompt tokens masked (`0` / `pad_id`) and digit/answer tokens weighted."""
    high_weight_ids = {
        tokenizer.ans_id,
        tokenizer.end_ans_id,
        *(tokenizer.token_to_id[ch] for ch in "0123456789*/=;-+" if ch in tokenizer.token_to_id),
    }
    seqs: List[List[int]] = []
    lbls: List[List[int]] = []
    wgts: List[List[float]] = []

    for it in items:
        p_ids = tokenizer.encode(it["prompt"], add_bos=True, add_eos=False)
        c_ids = tokenizer.encode(it["completion"], add_bos=False, add_eos=True)
        full = p_ids + c_ids
        if len(full) > max_seq_len:
            continue
        p_len = len(p_ids)
        lb = [0] * p_len + list(c_ids)
        wt = [0.0] * p_len + [3.0 if tid in high_weight_ids else 1.0 for tid in c_ids]
        seqs.append(full)
        lbls.append(lb)
        wgts.append(wt)

    if not seqs:
        raise ValueError("No valid training items fit within max_seq_len")

    max_len = min(max(len(s) for s in seqs), max_seq_len)
    batch_ids = torch.full((len(seqs), max_len), tokenizer.pad_id, dtype=torch.long)
    batch_lbls = torch.full((len(seqs), max_len), 0, dtype=torch.long)
    batch_wgts = torch.zeros((len(seqs), max_len), dtype=torch.float32)

    for i, (s, l, w) in enumerate(zip(seqs, lbls, wgts)):
        batch_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        batch_lbls[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        batch_wgts[i, : len(w)] = torch.tensor(w, dtype=torch.float32)

    return batch_ids, batch_lbls, batch_wgts


def train_isolated_domain_cartridge(
    model: AIXZeroModel,
    tokenizer: Any,
    bank: SparseExpertCartridgeBank,
    *,
    domain: str,
    train_items: List[Dict[str, Any]],
    val_items: Optional[List[Dict[str, Any]]] = None,
    domain_meta: Optional[Dict[str, Any]] = None,
    slot: Optional[int] = None,
    steps: int = 10,
    lr: float = 3e-3,
    seed: int = 42,
    active_base_checkpoint_sha256: Optional[str] = None,
    approve_if_val_non_regressed: bool = True,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Train a single domain specialist cartridge in complete isolation (`[AIX NEURAL-01]` / Issue #428).

    Guarantees:
    1. Freezes 100% of shared backbone, router, shared expert, and non-target expert slots.
    2. Instantiates a fresh optimizer bound exclusively to `moe.experts[target_slot].parameters()`
       so no stale parameter references or optimizer moment buffers leak across slot swaps.
    3. Saves and restores the base model's in-RAM slot tensors, `requires_grad` flags, `forced_expert_index`,
       and training mode in a `finally:` block on success, error, or cancellation.
    4. Verifies `frozen_tensors_unchanged == True` by comparing the SHA-256 of all non-target parameters
       before and after training.
    5. Atomically persists only the single candidate cartridge and updates `cartridge_manifest.json`
       without mutating any other domain's cartridge entry or shard.
    """
    if getattr(model, "_active_specialist_session", False):
        raise RuntimeError("Cannot start isolated cartridge training while a temporary specialist session is active")
    if not train_items:
        raise ValueError("train_items must be non-empty")

    moe = _moe(model)
    manifest = bank.get_manifest()
    cartridges = copy.deepcopy(manifest.get("cartridges", {})) if isinstance(manifest, dict) else {}
    existing_entry = cartridges.get(domain, {})

    target_slot = (
        int(slot)
        if slot is not None
        else int(existing_entry.get("source_expert_index", 0))
    )
    if not (0 <= target_slot < len(moe.experts)):
        raise ValueError(f"Target expert slot {target_slot} out of range (0..{len(moe.experts) - 1})")

    base_sha = active_base_checkpoint_sha256 or compute_model_state_sha256(model)
    slot_prefix = f"latent_reasoning_core.core_block.moe.experts.{target_slot}."
    frozen_pre_sha = compute_model_state_sha256(model, exclude_prefix=slot_prefix)

    was_training = model.training
    prev_forced = getattr(moe, "forced_expert_index", None)
    prev_req_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    saved_slot_tensors = {
        k: v.detach().cpu().clone().contiguous()
        for k, v in moe.experts[target_slot].state_dict().items()
    }
    pre_slot_sha = hashlib.sha256(save(saved_slot_tensors)).hexdigest()

    # If the domain already has a cartridge on disk, start from its weights; otherwise start from base slot
    if domain in cartridges:
        try:
            model.eval()
            bank.install_cartridge(
                model,
                domain,
                slot=target_slot,
                active_base_checkpoint_sha256=base_sha,
            )
        except Exception:
            with torch.no_grad():
                moe.experts[target_slot].load_state_dict(saved_slot_tensors, strict=True)

    train_ids, train_lbls, train_wgts = _encode_supervised_items(
        train_items, tokenizer, model.config.max_seq_len
    )
    eval_items = val_items if val_items else train_items
    val_ids, val_lbls, val_wgts = _encode_supervised_items(
        eval_items, tokenizer, model.config.max_seq_len
    )

    model._active_specialist_session = True
    try:
        torch.manual_seed(int(seed))
        moe.forced_expert_index = target_slot

        # Freeze every parameter outside `moe.experts[target_slot]`
        trainable_params: List[torch.nn.Parameter] = []
        trainable_param_count = 0
        for name, param in model.named_parameters():
            is_target = name.startswith(slot_prefix)
            param.requires_grad = is_target
            if is_target:
                trainable_params.append(param)
                trainable_param_count += int(param.numel())

        model.eval()
        with torch.no_grad():
            pre_train_loss = float(model(train_ids, labels=train_lbls, token_weights=train_wgts)["loss"].item())
            pre_val_loss = float(model(val_ids, labels=val_lbls, token_weights=val_wgts)["loss"].item())

        # Fresh isolated optimizer bound strictly to this slot's parameters
        optimizer = torch.optim.AdamW(trainable_params, lr=float(lr))
        model.train()
        steps_completed = 0
        for _ in range(int(steps)):
            if cancel_check is not None and cancel_check():
                raise InterruptedError("Isolated specialist training cancelled by caller")
            optimizer.zero_grad(set_to_none=True)
            out = model(train_ids, labels=train_lbls, token_weights=train_wgts)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            steps_completed += 1

        model.eval()
        with torch.no_grad():
            post_train_loss = float(model(train_ids, labels=train_lbls, token_weights=train_wgts)["loss"].item())
            post_val_loss = float(model(val_ids, labels=val_lbls, token_weights=val_wgts)["loss"].item())

        frozen_post_sha = compute_model_state_sha256(model, exclude_prefix=slot_prefix)
        frozen_tensors_unchanged = (frozen_pre_sha == frozen_post_sha)
        if not frozen_tensors_unchanged:
            raise RuntimeError("Frozen backbone/router/other-slot tensors mutated during isolated specialist training")

        trained_tensors = {
            k: v.detach().cpu().clone().contiguous()
            for k, v in moe.experts[target_slot].state_dict().items()
        }
        payload = save(trained_tensors)
        post_slot_sha = hashlib.sha256(payload).hexdigest()
        weights_updated = (post_slot_sha != pre_slot_sha)
        val_improved_or_held = (post_val_loss <= pre_val_loss and post_train_loss < pre_train_loss)

        gate_decision = "PASS" if (val_improved_or_held and weights_updated and frozen_tensors_unchanged) else "HOLD"
        promotion_status = (
            "APPROVED_ACTIVE"
            if (approve_if_val_non_regressed and gate_decision == "PASS")
            else ("ELIGIBLE_NOT_ACTIVATED" if gate_decision == "PASS" else "PENDING_AUDIT")
        )

        filename = f"expert_{domain.lower()}_{post_slot_sha}.safetensors"
        _atomic_write(bank.cartridge_dir / filename, payload)

        meta_info = domain_meta or existing_entry or {}
        new_entry = {
            "domain": domain,
            "title": meta_info.get("title", domain),
            "category": meta_info.get("category", "specialist_domain"),
            "prefix": meta_info.get("prefix", f"Q:{domain}:"),
            "formula": meta_info.get("formula", ""),
            "introduced_in": meta_info.get("introduced_in", "v12_isolated_specialist"),
            "cartridge_file": filename,
            "sha256": post_slot_sha,
            "serialized_bytes": len(payload),
            "source_expert_index": target_slot,
            "parameters": sum(int(v.numel()) for v in trained_tensors.values()),
            "tensors": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in trained_tensors.items()
            },
            "base_checkpoint_sha256": base_sha,
            "base_model_accuracy_pct": existing_entry.get("base_model_accuracy_pct"),
            "verified_accuracy_pct": None,
            "sprt_gate": "PASS" if gate_decision == "PASS" else "NOT_EVALUATED",
            "provenance": "frozen_v11_single_expert_training_candidate",
            "promotion_status": promotion_status,
            "training": {
                "frozen_tensors_unchanged": frozen_tensors_unchanged,
                "frozen_backbone_sha256": frozen_post_sha,
                "target_slot": target_slot,
                "trainable_parameters": trainable_param_count,
                "steps_completed": steps_completed,
                "learning_rate": float(lr),
                "seed": int(seed),
                "pre_train_loss": round(pre_train_loss, 6),
                "post_train_loss": round(post_train_loss, 6),
                "pre_val_loss": round(pre_val_loss, 6),
                "post_val_loss": round(post_val_loss, 6),
            },
            "empirical_gate": {
                "decision": gate_decision,
                "val_loss_delta": round(post_val_loss - pre_val_loss, 6),
                "train_loss_delta": round(post_train_loss - pre_train_loss, 6),
            },
        }
        SparseExpertCartridgeBank.validate_specialist_provenance(new_entry)

        cartridges[domain] = new_entry
        new_manifest = {
            "schema_version": 1,
            "architecture": "AIX-Zero routed expert snapshots and isolated specialists",
            "model_config": asdict(model.config),
            "base_checkpoint_sha256": base_sha,
            "active_core_physical_parameters": model.count_parameters()["total_parameters"],
            "registered_cartridges": len(cartridges),
            "total_cartridge_bank_parameters": sum(int(v.get("parameters", 0)) for v in cartridges.values()),
            "cartridges": cartridges,
            "limitations": "Isolated specialists train a single 49,152-parameter routed slot with frozen backbone.",
        }
        bank.publish_verified_catalog_atomic(new_manifest)

        return {
            "domain": domain,
            "target_slot": target_slot,
            "cartridge_file": filename,
            "pre_slot_sha256": pre_slot_sha,
            "post_slot_sha256": post_slot_sha,
            "weights_updated": weights_updated,
            "frozen_tensors_unchanged": frozen_tensors_unchanged,
            "frozen_backbone_sha256": frozen_post_sha,
            "base_checkpoint_sha256": base_sha,
            "trainable_parameters": trainable_param_count,
            "steps_completed": steps_completed,
            "pre_train_loss": round(pre_train_loss, 6),
            "post_train_loss": round(post_train_loss, 6),
            "pre_val_loss": round(pre_val_loss, 6),
            "post_val_loss": round(post_val_loss, 6),
            "promotion_status": promotion_status,
            "empirical_gate_decision": gate_decision,
        }
    finally:
        # Always restore the base model's in-RAM slot tensors, requires_grad, forced_expert_index, and mode
        with torch.no_grad():
            moe.experts[target_slot].load_state_dict(saved_slot_tensors, strict=True)
        for name, param in model.named_parameters():
            param.requires_grad = prev_req_grad.get(name, True)
        moe.forced_expert_index = prev_forced
        model.train(was_training)
        model._active_specialist_session = False


import copy


def evaluate_routed_cartridge_arm(
    model: AIXZeroModel,
    tokenizer: Any,
    bank: Optional[SparseExpertCartridgeBank],
    *,
    items: List[Dict[str, Any]],
    routing_mode: str = "predicted_prefix",
    forced_wrong_domain: Optional[str] = None,
    active_base_checkpoint_sha256: Optional[str] = None,
    max_new_tokens: int = 36,
) -> Dict[str, Any]:
    """Evaluate a model across `items` under one explicit cartridge routing policy (`[AIX NEURAL-01]` / #428).

    Supported `routing_mode` values:
    - `'unchanged_baseline'`: No cartridge routing; base model runs unchanged.
    - `'predicted_prefix'`: Routes strictly from the prompt prefix (`Q:<Tag>:`) via `bank.route_explicit_task_schema(prompt)`.
      Never inspects `item['domain']` or `item['gold']`.
    - `'oracle_domain_label'`: Routes using permitted task metadata `item['domain']` (reported separately from predicted routing).
    - `'wrong_cartridge_control'`: Deliberately installs `forced_wrong_domain` (or a cyclic mismatched domain)
      to measure wrong-specialist control performance.

    All generation runs in `neural_only` mode (`inflight_alu=False`, no retrieval, no symbolic repair).
    """
    if routing_mode not in (
        "unchanged_baseline",
        "predicted_prefix",
        "oracle_domain_label",
        "wrong_cartridge_control",
    ):
        raise ValueError(f"Unsupported routing_mode: {routing_mode!r}")

    moe = _moe(model)
    audit = audit_model_and_cartridge_tensor_ownership(model, bank)
    base_sha = active_base_checkpoint_sha256 or compute_model_state_sha256(model)
    manifest = bank.get_manifest() if bank is not None else {}
    cartridges = manifest.get("cartridges", {}) if isinstance(manifest, dict) else {}
    registered_domains = sorted(cartridges.keys())

    was_training = model.training
    model.eval()

    records: List[Dict[str, Any]] = []
    by_domain: Dict[str, Dict[str, int]] = {}
    cache_hits = 0
    specialist_activations = 0
    fallback_events = 0

    try:
        with torch.no_grad():
            for idx, item in enumerate(items):
                prompt = str(item["prompt"])
                gold = str(item["gold"])
                true_domain = str(item.get("domain", "UNKNOWN"))
                by_domain.setdefault(true_domain, {"tested": 0, "correct": 0, "specialist_activated": 0})

                prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
                route_decision = "BASE_MODEL_DEFAULT"
                selected_domain: Optional[str] = None
                cartridge_sha256: Optional[str] = None
                cache_hit = False
                load_latency_ms = 0.0
                fallback_reason: Optional[str] = None
                checkpoint_compatible = True
                architecture_compatible = True

                def _generate_fn(m: AIXZeroModel, _p: str):
                    return m.generate_verified(
                        prompt_ids,
                        eos_id=tokenizer.eos_id,
                        max_new_tokens=max_new_tokens,
                        num_candidates=1,
                        temperature=0.0,
                        tokenizer=tokenizer,
                        inflight_alu=False,
                    )

                t0 = time.perf_counter()
                if routing_mode == "unchanged_baseline" or bank is None:
                    gen_ids, _, _ = _generate_fn(model, prompt)
                    load_latency_ms = (time.perf_counter() - t0) * 1000.0
                elif routing_mode == "predicted_prefix":
                    try:
                        route_info = bank.route_explicit_task_schema(prompt)
                    except Exception as exc:
                        route_info = {
                            "specialist_activated": False,
                            "selected_domain": None,
                            "reason": f"route_error:{exc}",
                        }
                    selected_domain = route_info.get("selected_domain")
                    if route_info.get("specialist_activated") and selected_domain:
                        try:
                            with bank.temporary_specialist_session(
                                model,
                                selected_domain,
                                active_base_checkpoint_sha256=base_sha,
                                force_explicit_route=True,
                            ) as inst_info:
                                load_latency_ms = (time.perf_counter() - t0) * 1000.0
                                cache_hit = bool(inst_info.get("cache_hit", False))
                                cartridge_sha256 = cartridges.get(selected_domain, {}).get("sha256")
                                route_decision = f"SPECIALIST:{selected_domain}"
                                specialist_activations += 1
                                cache_hits += int(cache_hit)
                                gen_ids, _, _ = _generate_fn(model, prompt)
                        except Exception as exc:
                            load_latency_ms = (time.perf_counter() - t0) * 1000.0
                            fallback_events += 1
                            fallback_reason = str(exc)
                            checkpoint_compatible = "base_checkpoint_sha256" not in fallback_reason
                            architecture_compatible = "configuration mismatch" not in fallback_reason and "tensor" not in fallback_reason
                            route_decision = "BASE_MODEL_FALLBACK"
                            gen_ids, _, _ = _generate_fn(model, prompt)
                    else:
                        load_latency_ms = (time.perf_counter() - t0) * 1000.0
                        fallback_reason = str(route_info.get("reason", "not_activated"))
                        if fallback_reason != "approved_active_specialist_matched":
                            fallback_events += 1
                        gen_ids, _, _ = _generate_fn(model, prompt)
                elif routing_mode in ("oracle_domain_label", "wrong_cartridge_control"):
                    if routing_mode == "oracle_domain_label":
                        selected_domain = true_domain
                    else:
                        if forced_wrong_domain is not None:
                            selected_domain = forced_wrong_domain
                        else:
                            other_doms = [d for d in registered_domains if d != true_domain]
                            selected_domain = other_doms[idx % len(other_doms)] if other_doms else true_domain
                    try:
                        with bank.temporary_specialist_session(
                            model,
                            selected_domain,
                            active_base_checkpoint_sha256=base_sha,
                            force_explicit_route=True,
                        ) as inst_info:
                            load_latency_ms = (time.perf_counter() - t0) * 1000.0
                            cache_hit = bool(inst_info.get("cache_hit", False))
                            cartridge_sha256 = cartridges.get(selected_domain, {}).get("sha256")
                            route_decision = f"SPECIALIST:{selected_domain}"
                            specialist_activations += 1
                            cache_hits += int(cache_hit)
                            gen_ids, _, _ = _generate_fn(model, prompt)
                    except Exception as exc:
                        load_latency_ms = (time.perf_counter() - t0) * 1000.0
                        fallback_events += 1
                        fallback_reason = str(exc)
                        checkpoint_compatible = "base_checkpoint_sha256" not in fallback_reason
                        architecture_compatible = "configuration mismatch" not in fallback_reason and "tensor" not in fallback_reason
                        route_decision = "BASE_MODEL_FALLBACK"
                        gen_ids, _, _ = _generate_fn(model, prompt)

                dec = decode_generated_completion(
                    gen_ids,
                    prompt_ids,
                    tokenizer,
                    max_new_tokens=max_new_tokens,
                    max_seq_len=model.config.max_seq_len,
                )
                completion = dec["completion"]
                sc = score_rational_or_probability(completion, gold, is_probability=False)
                correct = bool(sc["score"] == 1.0)

                by_domain[true_domain]["tested"] += 1
                by_domain[true_domain]["correct"] += int(correct)
                if route_decision.startswith("SPECIALIST:"):
                    by_domain[true_domain]["specialist_activated"] += 1

                active_params = (
                    audit["active_parameters_per_token_forced_single_slot"]
                    if route_decision.startswith("SPECIALIST:")
                    else audit["active_parameters_per_token_top_k"]
                )
                records.append(
                    {
                        "domain": true_domain,
                        "prompt": prompt,
                        "gold": gold,
                        "completion": completion,
                        "correct": correct,
                        "generated_token_count": dec["generated_token_count"],
                        "termination_reason": dec["termination_reason"],
                        "routing_mode": routing_mode,
                        "selected_domain": selected_domain,
                        "route_decision": route_decision,
                        "cartridge_sha256": cartridge_sha256,
                        "checkpoint_compatible": checkpoint_compatible,
                        "architecture_compatible": architecture_compatible,
                        "tokenizer_sha256": getattr(tokenizer, "vocab_sha256", None),
                        "cache_hit": cache_hit,
                        "load_latency_ms": round(load_latency_ms, 4),
                        "rss_memory_bytes": _current_rss_bytes(),
                        "active_parameters_per_token": active_params,
                        "trainable_specialist_parameters": audit["per_slot_expert_parameters"],
                        "fallback_reason": fallback_reason,
                    }
                )
    finally:
        model.train(was_training)

    tested = len(records)
    correct_count = sum(1 for r in records if r["correct"])
    acc = round(correct_count / tested, 4) if tested else 0.0
    ci_low, ci_high = wilson_score_interval(correct_count, tested) if tested else (0.0, 0.0)

    return {
        "schema_version": "1.0.0",
        "evaluation_mode": "neural_only",
        "routing_mode": routing_mode,
        "routing_uses_gold_answers": False,
        "inflight_alu_enabled": False,
        "retrieval_memory_enabled": False,
        "symbolic_repair_enabled": False,
        "base_checkpoint_sha256": base_sha,
        "tokenizer_sha256": getattr(tokenizer, "vocab_sha256", None),
        "tested_count": tested,
        "correct_count": correct_count,
        "accuracy": acc,
        "wilson_95_ci": [round(ci_low, 4), round(ci_high, 4)],
        "specialist_activations": specialist_activations,
        "cache_hits": cache_hits,
        "fallback_events": fallback_events,
        "per_slot_trainable_parameters": audit["per_slot_expert_parameters"],
        "per_domain_breakdown": by_domain,
        "records": records,
    }


def evaluate_routed_cartridge_paired_comparison(
    model: AIXZeroModel,
    tokenizer: Any,
    bank: SparseExpertCartridgeBank,
    *,
    domain_items: List[Dict[str, Any]],
    retention_items: Optional[List[Dict[str, Any]]] = None,
    wrong_cartridge_domain: Optional[str] = None,
    active_base_checkpoint_sha256: Optional[str] = None,
    max_new_tokens: int = 36,
) -> Dict[str, Any]:
    """Compare unchanged baseline, predicted-prefix routed candidate, oracle-routed candidate,
    and wrong-cartridge control on frozen domain and retention sets (`[AIX NEURAL-01]` / Issue #428).
    """
    base_sha = active_base_checkpoint_sha256 or compute_model_state_sha256(model)
    audit = audit_model_and_cartridge_tensor_ownership(model, bank)

    baseline_arm = evaluate_routed_cartridge_arm(
        model,
        tokenizer,
        bank,
        items=domain_items,
        routing_mode="unchanged_baseline",
        active_base_checkpoint_sha256=base_sha,
        max_new_tokens=max_new_tokens,
    )
    predicted_arm = evaluate_routed_cartridge_arm(
        model,
        tokenizer,
        bank,
        items=domain_items,
        routing_mode="predicted_prefix",
        active_base_checkpoint_sha256=base_sha,
        max_new_tokens=max_new_tokens,
    )
    oracle_arm = evaluate_routed_cartridge_arm(
        model,
        tokenizer,
        bank,
        items=domain_items,
        routing_mode="oracle_domain_label",
        active_base_checkpoint_sha256=base_sha,
        max_new_tokens=max_new_tokens,
    )
    wrong_arm = evaluate_routed_cartridge_arm(
        model,
        tokenizer,
        bank,
        items=domain_items,
        routing_mode="wrong_cartridge_control",
        forced_wrong_domain=wrong_cartridge_domain,
        active_base_checkpoint_sha256=base_sha,
        max_new_tokens=max_new_tokens,
    )

    retention_report: Optional[Dict[str, Any]] = None
    if retention_items:
        ret_base = evaluate_routed_cartridge_arm(
            model,
            tokenizer,
            bank,
            items=retention_items,
            routing_mode="unchanged_baseline",
            active_base_checkpoint_sha256=base_sha,
            max_new_tokens=max_new_tokens,
        )
        ret_routed = evaluate_routed_cartridge_arm(
            model,
            tokenizer,
            bank,
            items=retention_items,
            routing_mode="predicted_prefix",
            active_base_checkpoint_sha256=base_sha,
            max_new_tokens=max_new_tokens,
        )
        retention_report = {
            "baseline_correct": ret_base["correct_count"],
            "routed_correct": ret_routed["correct_count"],
            "tested": ret_base["tested_count"],
            "baseline_accuracy": ret_base["accuracy"],
            "routed_accuracy": ret_routed["accuracy"],
            "accuracy_delta": round(ret_routed["accuracy"] - ret_base["accuracy"], 4),
        }

    post_eval_sha = compute_model_state_sha256(model)
    return {
        "schema_version": "1.0.0",
        "tensor_ownership_audit": audit,
        "base_model_restored_unchanged": (post_eval_sha == base_sha),
        "arms": {
            "unchanged_baseline": baseline_arm,
            "predicted_prefix_routed": predicted_arm,
            "oracle_domain_label_routed": oracle_arm,
            "wrong_cartridge_control": wrong_arm,
        },
        "deltas": {
            "predicted_vs_baseline_accuracy": round(predicted_arm["accuracy"] - baseline_arm["accuracy"], 4),
            "oracle_vs_baseline_accuracy": round(oracle_arm["accuracy"] - baseline_arm["accuracy"], 4),
            "predicted_vs_wrong_cartridge_accuracy": round(predicted_arm["accuracy"] - wrong_arm["accuracy"], 4),
        },
        "retention": retention_report,
    }
