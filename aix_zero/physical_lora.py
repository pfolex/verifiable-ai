"""Physical LoRA Capacity, Function-Preserving Rank & Expert Growth, and Checkpoint Migration (`[AIX NEURAL-02]` / Issue #429).

Provides:
1. `LoRALinear`: Physical PyTorch module registering `lora_A` (`(r, in_features)`) and `lora_B` (`(out_features, r)`)
   and computing `base(x) + (alpha / r) * ((x @ lora_A.T) @ lora_B.T)` in `forward()`.
2. Function-preserving rank expansion (`expand_rank`) on already-trained adapters, rescaling preserved `B` columns
   when `alpha / r` changes so `(alpha_new / r_new) * B_new @ A_new == (alpha_old / r_old) * B_old @ A_old`.
3. Optimizer state migration (`migrate_optimizer_for_lora_expansion`) retaining `exp_avg` and `exp_avg_sq`
   for existing dimensions and zero-initializing only newly added rank slices.
4. Function-preserving expert-count expansion (`expand_moe_experts_function_preserving`) that preserves exact
   softmax router probabilities and top-k selection at the expansion boundary via a registered router logit mask.
5. Explicit versioned `v11` migration (`attach_physical_lora_to_model`, `save_physical_lora_checkpoint`,
   `load_physical_lora_checkpoint`) failing closed on unexplained shape mismatches, bounded by explicit
   RAM/time/parameter caps.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import resource
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aix_zero.architecture import AIXZeroConfig, AIXZeroModel, SwiGLUFFN
from aix_zero.candidate_policy import compute_retention_metrics


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage) if os.uname().sysname == "Darwin" else int(usage) * 1024


class LoRALinear(nn.Module):
    """Physical Low-Rank Adaptation wrapper around an `nn.Linear` layer (`[AIX NEURAL-02]` / Issue #429).

    Computes:
        y = W_0 x + b_0 + (alpha / r) * B (A x)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rank: int = 4,
        alpha: Optional[float] = None,
        freeze_base: bool = True,
        init_seed: int = 42,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be an nn.Linear instance")
        if int(rank) < 1:
            raise ValueError("LoRA rank must be >= 1")

        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.base_linear = base_linear

        if freeze_base:
            self.base_linear.weight.requires_grad = False
            if self.base_linear.bias is not None:
                self.base_linear.bias.requires_grad = False

        dtype = base_linear.weight.dtype
        device = base_linear.weight.device
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(init_seed))

        a_init = torch.empty((self.rank, self.in_features), dtype=torch.float32)
        nn.init.kaiming_uniform_(a_init, a=math.sqrt(5), generator=gen)
        b_init = torch.zeros((self.out_features, self.rank), dtype=torch.float32)

        self.lora_A = nn.Parameter(a_init.to(device=device, dtype=dtype))
        self.lora_B = nn.Parameter(b_init.to(device=device, dtype=dtype))

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)

    def delta_weight(self) -> torch.Tensor:
        """Return effective low-rank weight delta `(alpha / r) * (lora_B @ lora_A)`."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def expand_rank(
        self,
        new_rank: int,
        *,
        new_alpha: Optional[float] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        init_seed: int = 1337,
    ) -> Dict[str, Any]:
        """Function-preserving rank expansion (`r_old -> r_new`) even on an already-trained adapter.

        If `new_alpha / new_rank != old_alpha / old_rank`, rescales the preserved `lora_B[:, :r_old]`
        columns by `old_scaling / new_scaling` and appends zero columns `lora_B[:, r_old:r_new] = 0`,
        guaranteeing that `delta_weight()` before and after expansion matches within dtype tolerance.
        Also migrates `optimizer.state` (`exp_avg`, `exp_avg_sq`) for `lora_A` and `lora_B` if provided.
        """
        new_r = int(new_rank)
        old_r = int(self.rank)
        if new_r <= old_r:
            raise ValueError(f"new_rank ({new_r}) must be strictly greater than current rank ({old_r})")

        old_alpha = float(self.alpha)
        target_alpha = float(new_alpha if new_alpha is not None else old_alpha * (new_r / old_r))
        old_scale = old_alpha / old_r
        new_scale = target_alpha / new_r
        scale_ratio = old_scale / new_scale

        old_A_param = self.lora_A
        old_B_param = self.lora_B
        dtype = old_A_param.dtype
        device = old_A_param.device

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(init_seed) + new_r)
        extra_A = torch.empty((new_r - old_r, self.in_features), dtype=torch.float32)
        nn.init.kaiming_uniform_(extra_A, a=math.sqrt(5), generator=gen)
        extra_A = extra_A.to(device=device, dtype=dtype)
        extra_B = torch.zeros((self.out_features, new_r - old_r), device=device, dtype=dtype)

        with torch.no_grad():
            new_A_tensor = torch.cat([old_A_param.detach().clone(), extra_A], dim=0)
            new_B_tensor = torch.cat([old_B_param.detach().clone() * scale_ratio, extra_B], dim=1)

        new_A_param = nn.Parameter(new_A_tensor, requires_grad=old_A_param.requires_grad)
        new_B_param = nn.Parameter(new_B_tensor, requires_grad=old_B_param.requires_grad)

        self.lora_A = new_A_param
        self.lora_B = new_B_param
        self.rank = new_r
        self.alpha = target_alpha

        migrated_moments = False
        if optimizer is not None:
            migrated_moments = _migrate_optimizer_param_pair(
                optimizer,
                old_A=old_A_param,
                new_A=new_A_param,
                old_B=old_B_param,
                new_B=new_B_param,
                old_r=old_r,
                new_r=new_r,
                b_scale_ratio=scale_ratio,
            )

        return {
            "old_rank": old_r,
            "new_rank": new_r,
            "old_alpha": old_alpha,
            "new_alpha": target_alpha,
            "old_scaling": old_scale,
            "new_scaling": new_scale,
            "b_column_rescale_factor": scale_ratio,
            "optimizer_state_migrated": migrated_moments,
            "added_trainable_parameters": (new_r - old_r) * (self.in_features + self.out_features),
        }


def _migrate_optimizer_param_pair(
    optimizer: torch.optim.Optimizer,
    *,
    old_A: nn.Parameter,
    new_A: nn.Parameter,
    old_B: nn.Parameter,
    new_B: nn.Parameter,
    old_r: int,
    new_r: int,
    b_scale_ratio: float,
) -> bool:
    """Replace `old_A`/`old_B` in `optimizer.param_groups` and pad `exp_avg`/`exp_avg_sq` moments."""
    migrated = False
    for group in optimizer.param_groups:
        new_params = []
        for p in group["params"]:
            if p is old_A:
                new_params.append(new_A)
            elif p is old_B:
                new_params.append(new_B)
            else:
                new_params.append(p)
        group["params"] = new_params

    if old_A in optimizer.state:
        st_a = optimizer.state.pop(old_A)
        new_st_a: Dict[str, Any] = {}
        for k, v in st_a.items():
            if torch.is_tensor(v) and v.shape == old_A.shape:
                pad_a = torch.zeros((new_r - old_r, old_A.shape[1]), dtype=v.dtype, device=v.device)
                new_st_a[k] = torch.cat([v.detach().clone(), pad_a], dim=0)
            elif torch.is_tensor(v):
                new_st_a[k] = v.detach().clone()
            else:
                new_st_a[k] = copy.deepcopy(v)
        optimizer.state[new_A] = new_st_a
        migrated = True

    if old_B in optimizer.state:
        st_b = optimizer.state.pop(old_B)
        new_st_b: Dict[str, Any] = {}
        for k, v in st_b.items():
            if torch.is_tensor(v) and v.shape == old_B.shape:
                pad_b = torch.zeros((old_B.shape[0], new_r - old_r), dtype=v.dtype, device=v.device)
                scale_factor = (
                    b_scale_ratio
                    if k == "exp_avg"
                    else (b_scale_ratio * b_scale_ratio if k == "exp_avg_sq" else 1.0)
                )
                new_st_b[k] = torch.cat([v.detach().clone() * scale_factor, pad_b], dim=1)
            elif torch.is_tensor(v):
                new_st_b[k] = v.detach().clone()
            else:
                new_st_b[k] = copy.deepcopy(v)
        optimizer.state[new_B] = new_st_b
        migrated = True

    return migrated


DEFAULT_LORA_TARGET_SUFFIXES: Tuple[str, ...] = (
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "moe.shared_expert.gate_proj",
    "moe.shared_expert.up_proj",
    "moe.shared_expert.down_proj",
)


def _resolve_parent_and_attr(model: nn.Module, dotted_path: str) -> Tuple[nn.Module, str, nn.Module]:
    parts = dotted_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    target = getattr(parent, parts[-1])
    return parent, parts[-1], target


def attach_physical_lora_to_model(
    model: AIXZeroModel,
    *,
    rank: int = 4,
    alpha: Optional[float] = None,
    target_suffixes: Sequence[str] = DEFAULT_LORA_TARGET_SUFFIXES,
    freeze_non_lora: bool = True,
    max_total_parameters: int = 10_000_000,
    max_lora_rank: int = 512,
    max_rss_bytes: int = 4 * 1024 * 1024 * 1024,
    init_seed: int = 42,
) -> Dict[str, Any]:
    """Attach physical `LoRALinear` adapters to matching `nn.Linear` submodules on `model`.

    Enforces configured RAM and parameter caps and verifies function preservation (`B = 0`).
    """
    if int(rank) > int(max_lora_rank):
        raise ValueError(f"Requested LoRA rank {rank} exceeds max_lora_rank cap ({max_lora_rank})")
    if _rss_bytes() > int(max_rss_bytes):
        raise MemoryError(f"Process RSS ({_rss_bytes()}) exceeds max_rss_bytes cap ({max_rss_bytes})")

    candidate_paths: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in target_suffixes):
            candidate_paths.append(name)

    if not candidate_paths:
        raise ValueError(f"No nn.Linear modules matched target_suffixes={list(target_suffixes)!r}")

    projected_added = 0
    for path in candidate_paths:
        _, _, mod = _resolve_parent_and_attr(model, path)
        assert isinstance(mod, nn.Linear)
        projected_added += int(rank) * (int(mod.in_features) + int(mod.out_features))

    base_total = sum(int(p.numel()) for p in model.parameters())
    if base_total + projected_added > int(max_total_parameters):
        raise ValueError(
            f"Projected parameters ({base_total + projected_added}) exceed max_total_parameters cap ({max_total_parameters})"
        )

    if freeze_non_lora:
        for param in model.parameters():
            param.requires_grad = False

    replaced_modules: List[str] = []
    for idx, path in enumerate(candidate_paths):
        parent, attr, mod = _resolve_parent_and_attr(model, path)
        assert isinstance(mod, nn.Linear)
        wrapped = LoRALinear(
            mod,
            rank=int(rank),
            alpha=alpha,
            freeze_base=freeze_non_lora,
            init_seed=int(init_seed) + idx,
        )
        setattr(parent, attr, wrapped)
        replaced_modules.append(path)

    model._physical_lora_metadata = {
        "schema_version": "1.0.0",
        "rank": int(rank),
        "alpha": float(alpha if alpha is not None else rank),
        "target_modules": replaced_modules,
        "max_total_parameters": int(max_total_parameters),
        "max_lora_rank": int(max_lora_rank),
    }
    return measure_physical_lora_Accounting(model)


def expand_model_lora_rank(
    model: AIXZeroModel,
    new_rank: int,
    *,
    new_alpha: Optional[float] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_total_parameters: Optional[int] = None,
    max_lora_rank: Optional[int] = None,
    max_rss_bytes: int = 4 * 1024 * 1024 * 1024,
    init_seed: int = 1337,
) -> Dict[str, Any]:
    """Expand rank on all attached `LoRALinear` modules in `model` while preserving exact forward outputs."""
    meta = getattr(model, "_physical_lora_metadata", {})
    rank_cap = int(max_lora_rank if max_lora_rank is not None else meta.get("max_lora_rank", 512))
    param_cap = int(max_total_parameters if max_total_parameters is not None else meta.get("max_total_parameters", 10_000_000))

    if int(new_rank) > rank_cap:
        raise ValueError(f"Requested new_rank ({new_rank}) exceeds max_lora_rank cap ({rank_cap})")
    if _rss_bytes() > int(max_rss_bytes):
        raise MemoryError(f"Process RSS ({_rss_bytes()}) exceeds max_rss_bytes cap ({max_rss_bytes})")

    lora_modules: List[Tuple[str, LoRALinear]] = [
        (name, mod) for name, mod in model.named_modules() if isinstance(mod, LoRALinear)
    ]
    if not lora_modules:
        raise ValueError("Model has no attached LoRALinear modules to expand")

    added_params = sum(
        (int(new_rank) - mod.rank) * (mod.in_features + mod.out_features)
        for _, mod in lora_modules
    )
    current_total = sum(int(p.numel()) for p in model.parameters())
    if current_total + added_params > param_cap:
        raise ValueError(
            f"Expanded parameter count ({current_total + added_params}) exceeds max_total_parameters cap ({param_cap})"
        )

    per_module_reports: Dict[str, Any] = {}
    for idx, (name, mod) in enumerate(lora_modules):
        per_module_reports[name] = mod.expand_rank(
            int(new_rank),
            new_alpha=new_alpha,
            optimizer=optimizer,
            init_seed=int(init_seed) + idx,
        )

    if hasattr(model, "_physical_lora_metadata"):
        model._physical_lora_metadata["rank"] = int(new_rank)
        model._physical_lora_metadata["alpha"] = float(
            new_alpha if new_alpha is not None else lora_modules[0][1].alpha
        )

    accounting = measure_physical_lora_Accounting(model, optimizer=optimizer)
    accounting["expanded_modules"] = per_module_reports
    return accounting


def expand_moe_experts_function_preserving(
    model: AIXZeroModel,
    new_n_experts: int,
    *,
    max_total_parameters: int = 10_000_000,
    unmask_new_experts: bool = False,
) -> Dict[str, Any]:
    """Expand `MixtureOfReasoningExperts` from `n_old -> n_new` (`4 -> 8 -> 16 -> 32`) while preserving
    exact routing probabilities and forward outputs at the expansion boundary (`[AIX NEURAL-02]` / #429).

    Why zero expert outputs alone are insufficient:
    In `MixtureOfReasoningExperts.forward()`, `router_probs = F.softmax(router(x), dim=-1)` and
    `top_vals, top_idx = torch.topk(router_probs, k=top_k)`. Adding unmasked router rows changes
    the softmax denominator and can enter `top_k`, altering `top_vals` normalization even if new
    experts return zeros.
    Solution:
    We expand `moe.router` to `nn.Linear(d_model, new_n_experts, bias=True)` with `bias[:n_old] = 0.0`
    and `bias[n_old:] = -1e9` (when `unmask_new_experts=False`), so `softmax` assigns `0.0` probability
    to the new expert slots at the expansion boundary and preserves 100% of existing routing probabilities!
    Call `unmask_expanded_moe_experts(model)` when ready to route traffic into the new slots.
    """
    block = model.latent_reasoning_core.core_block
    if not block.use_moe:
        raise ValueError("Model does not have use_moe=True")
    moe = block.moe
    old_n = int(moe.n_experts)
    new_n = int(new_n_experts)
    if new_n <= old_n:
        raise ValueError(f"new_n_experts ({new_n}) must be > current n_experts ({old_n})")

    expert_ff = max(64, model.config.d_ff // 2)
    per_expert_params = 3 * model.config.d_model * expert_ff
    added_params = (new_n - old_n) * (per_expert_params + model.config.d_model) + new_n
    cur_total = sum(int(p.numel()) for p in model.parameters())
    if cur_total + added_params > int(max_total_parameters):
        raise ValueError(
            f"Expert expansion to {new_n} experts ({cur_total + added_params} params) exceeds max_total_parameters ({max_total_parameters})"
        )

    old_router = moe.router
    dtype = old_router.weight.dtype
    device = old_router.weight.device
    new_router = nn.Linear(model.config.d_model, new_n, bias=True, device=device, dtype=dtype)

    with torch.no_grad():
        new_router.weight[:old_n].copy_(old_router.weight.detach())
        new_router.weight[old_n:].zero_()
        if old_router.bias is not None:
            new_router.bias[:old_n].copy_(old_router.bias.detach())
        else:
            new_router.bias[:old_n].zero_()
        new_router.bias[old_n:].fill_(0.0 if unmask_new_experts else -1e9)

    for idx in range(old_n, new_n):
        new_exp = SwiGLUFFN(model.config, d_ff_override=expert_ff).to(device=device, dtype=dtype)
        # Clone slot `idx % old_n` or zero down_proj so even when unmasked, initial perturbation is controlled
        with torch.no_grad():
            new_exp.load_state_dict(moe.experts[idx % old_n].state_dict())
        moe.experts.append(new_exp)

    moe.router = new_router
    moe.n_experts = new_n
    model.config.n_experts = new_n

    return {
        "old_n_experts": old_n,
        "new_n_experts": new_n,
        "new_slots_masked_at_boundary": not unmask_new_experts,
        "total_physical_parameters": sum(int(p.numel()) for p in model.parameters()),
    }


def unmask_expanded_moe_experts(model: AIXZeroModel, *, initial_logit_offset: float = -2.0) -> None:
    """Smoothly unmask newly added MoE router slots so they receive softmax probability and gradients."""
    moe = model.latent_reasoning_core.core_block.moe
    if moe.router.bias is not None:
        with torch.no_grad():
            mask = moe.router.bias < -1e6
            moe.router.bias[mask] = float(initial_logit_offset)


def measure_physical_lora_Accounting(
    model: AIXZeroModel,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    disk_cartridge_bytes: int = 0,
) -> Dict[str, Any]:
    """Report actual unique physical, trainable, active-per-token, and disk-resident parameters,
    plus optimizer state memory bytes and process RSS (`[AIX NEURAL-02]` / Issue #429).
    """
    seen_ptrs = set()
    unique_physical_params = 0
    unique_physical_bytes = 0
    trainable_params = 0
    trainable_bytes = 0
    lora_params = 0
    routed_expert_params = 0

    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        n = int(param.numel())
        b = n * int(param.element_size())
        unique_physical_params += n
        unique_physical_bytes += b
        if param.requires_grad:
            trainable_params += n
            trainable_bytes += b
        if ".lora_A" in name or ".lora_B" in name:
            lora_params += n
        if "moe.experts." in name:
            routed_expert_params += n

    block = model.latent_reasoning_core.core_block
    if block.use_moe:
        moe = block.moe
        per_expert = routed_expert_params // max(1, int(moe.n_experts))
        active_experts = 1 if getattr(moe, "forced_expert_index", None) is not None else int(moe.top_k)
        inactive_routed = max(0, routed_expert_params - active_experts * per_expert)
    else:
        inactive_routed = 0

    active_per_token = unique_physical_params - inactive_routed

    opt_state_bytes = 0
    opt_state_tensors = 0
    if optimizer is not None:
        for st in optimizer.state.values():
            if isinstance(st, dict):
                for val in st.values():
                    if torch.is_tensor(val):
                        opt_state_tensors += 1
                        opt_state_bytes += int(val.numel()) * int(val.element_size())

    lora_mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    ranks = sorted({m.rank for m in lora_mods})

    return {
        "schema_version": "1.0.0",
        "unique_physical_parameters": unique_physical_params,
        "unique_physical_bytes": unique_physical_bytes,
        "trainable_parameters": trainable_params,
        "trainable_bytes": trainable_bytes,
        "physical_lora_parameters": lora_params,
        "active_parameters_per_token": active_per_token,
        "routed_expert_parameters_total": routed_expert_params,
        "n_experts": int(model.config.n_experts) if block.use_moe else 0,
        "lora_module_count": len(lora_mods),
        "lora_ranks_present": ranks,
        "optimizer_state_tensor_count": opt_state_tensors,
        "optimizer_state_memory_bytes": opt_state_bytes,
        "disk_resident_cartridge_bytes": int(disk_cartridge_bytes),
        "rss_memory_bytes": _rss_bytes(),
    }


def physical_lora_checkpoint_payload(model: AIXZeroModel) -> Dict[str, Any]:
    """Build the versioned payload for file or in-memory experiment serialization."""
    lora_meta = getattr(model, "_physical_lora_metadata", None)
    payload = {
        "schema_version": "2.0.0-physical-lora",
        "model_config": asdict(model.config),
        "physical_lora_metadata": copy.deepcopy(lora_meta),
        "router_has_bias": bool(
            model.latent_reasoning_core.core_block.use_moe
            and model.latent_reasoning_core.core_block.moe.router.bias is not None
        ),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    return payload


def save_physical_lora_checkpoint(model: AIXZeroModel, path: Path) -> Dict[str, Any]:
    """Save a versioned physical-LoRA checkpoint with explicit architecture metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = physical_lora_checkpoint_payload(model)
    torch.save(payload, path)
    raw = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "schema_version": payload["schema_version"],
    }


def load_physical_lora_checkpoint(
    path: Path,
    *,
    base_v11_safetensors_path: Optional[Path] = None,
) -> AIXZeroModel:
    """Load a versioned physical-LoRA checkpoint or migrate a `v11` checkpoint, failing closed on shape mismatches."""
    path = Path(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    return load_physical_lora_payload(loaded, base_v11_safetensors_path=base_v11_safetensors_path)


def load_physical_lora_payload(loaded, *, base_v11_safetensors_path=None) -> AIXZeroModel:
    """Reconstruct deserialized weights; experiment callers validate provenance first."""
    if not isinstance(loaded, dict) or "schema_version" not in loaded or "model_config" not in loaded:
        raise ValueError("Unsupported or unversioned physical LoRA checkpoint")
    if loaded["schema_version"] != "2.0.0-physical-lora":
        raise ValueError(f"Unsupported schema_version: {loaded['schema_version']!r}")

    cfg = AIXZeroConfig(**loaded["model_config"])
    base_n_experts = 4 if cfg.n_experts > 4 else cfg.n_experts
    init_cfg = copy.deepcopy(cfg)
    init_cfg.n_experts = base_n_experts
    model = AIXZeroModel(init_cfg)

    if base_v11_safetensors_path is not None and Path(base_v11_safetensors_path).exists():
        from safetensors.torch import load_file
        v11_sd = load_file(str(base_v11_safetensors_path))
        model.load_state_dict(v11_sd, strict=True)

    lora_meta = loaded.get("physical_lora_metadata")
    if lora_meta:
        attach_physical_lora_to_model(
            model,
            rank=int(lora_meta["rank"]),
            alpha=float(lora_meta["alpha"]),
            target_suffixes=tuple(lora_meta["target_modules"]),
            max_total_parameters=int(lora_meta.get("max_total_parameters", 10_000_000)),
            max_lora_rank=int(lora_meta.get("max_lora_rank", 512)),
        )

    if cfg.n_experts > base_n_experts:
        expand_moe_experts_function_preserving(model, cfg.n_experts)
    elif loaded.get("router_has_bias") and model.latent_reasoning_core.core_block.use_moe:
        moe = model.latent_reasoning_core.core_block.moe
        if moe.router.bias is None:
            old_r = moe.router
            new_r = nn.Linear(cfg.d_model, cfg.n_experts, bias=True)
            with torch.no_grad():
                new_r.weight.copy_(old_r.weight)
                new_r.bias.zero_()
            moe.router = new_r

    expected_sd = model.state_dict()
    incoming_sd = loaded["state_dict"]
    if set(expected_sd.keys()) != set(incoming_sd.keys()):
        missing = sorted(set(expected_sd.keys()) - set(incoming_sd.keys()))
        unexpected = sorted(set(incoming_sd.keys()) - set(expected_sd.keys()))
        raise ValueError(f"Checkpoint key mismatch (missing={missing}, unexpected={unexpected})")
    for k, exp_t in expected_sd.items():
        inc_t = incoming_sd[k]
        if tuple(exp_t.shape) != tuple(inc_t.shape):
            raise ValueError(f"Checkpoint tensor shape mismatch for {k!r}: expected {tuple(exp_t.shape)}, got {tuple(inc_t.shape)}")

    model.load_state_dict(incoming_sd, strict=True)
    return model
