"""AIX-Zero v4 Neural Network Architecture written 100% from scratch in PyTorch.

Core architectural modules:
1. RMSNorm + 2D Rotary Position Embeddings (RoPE) + Grouped-Query Attention (GQA) + SwiGLU FFN.
2. PromptPointerAttention (Sharp Induction/Copy Cross-Attention Head):
   Dedicated low-temperature (tau=0.25) RoPE attention mechanism inside the recurrent reasoning
   core that attends back over the sequence with a sharp induction gate so operand digits in the
   prompt (`@x=3`, `buys 2`, `%4`) are copied with high fidelity into `<think>...</think>`.
3. RecurrentReasoningBlock (Latent Ponder Core with Velocity-Aware ACT Supervision).
4. Shared-Expert + Routed-Expert MixtureOfReasoningExperts (MoRE).
5. DifferentiableMemoryBank (Working-Memory Registers).
6. Operand-Weighted Cross-Entropy Loss (`token_weights`):
   Focuses 5.0x gradient weight on the left-hand-side algebraic setup & operand-copy tokens
   while the In-Flight Symbolic ALU handles deterministic arithmetic evaluation after `=`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AIXZeroConfig:
    vocab_size: int = 121
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 256
    n_encoder_layers: int = 1
    n_recurrent_steps: int = 3
    n_decoder_layers: int = 1
    max_seq_len: int = 112
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    dropout: float = 0.0
    use_moe: bool = False
    use_shared_expert: bool = True
    n_experts: int = 4
    moe_top_k: int = 2
    use_memory_bank: bool = False
    n_memory_slots: int = 8
    use_prompt_pointer: bool = False
    pointer_temperature: float = 0.35
    use_key_ngram_conv: bool = False


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization written from scratch."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_normed * self.weight).to(orig_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) from first principles."""

    def __init__(self, head_dim: int, max_seq_len: int = 256, theta: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for 2D RoPE rotation"
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[2]
        half = x.shape[-1] // 2
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated.to(x.dtype)


class GroupedQueryAttention(nn.Module):
    """Causal Grouped-Query Attention (GQA) with RoPE."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_queries_per_kv = config.n_heads // config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope.apply_rotary(q)
        k = self.rope.apply_rotary(k)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)


class PromptPointerAttention(nn.Module):
    """Sharp Induction / Prompt-Pointer Attention Head for exact operand copying.

    Uses low-temperature (tau=0.35) causal attention over Anchor/Prompt Embeddings h_anchor
    so `<think>` reasoning steps can point directly to the exact digit tokens in the prompt
    (`@x=3`, `buys 2`, `%4`, `f(5)`) without blurring across neighboring digits.
    """

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.temperature = config.pointer_temperature
        self.use_key_ngram_conv = config.use_key_ngram_conv
        self.norm_q = RMSNorm(config.d_model, config.rms_eps)
        self.norm_kv = RMSNorm(config.d_model, config.rms_eps)
        if self.use_key_ngram_conv:
            # Depthwise causal 3-gram convolution so multi-digit tokens carry neighbor context
            self.key_conv = nn.Conv1d(
                config.d_model,
                config.d_model,
                kernel_size=3,
                padding=0,
                groups=config.d_model,
                bias=False,
            )
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.copy_gate = nn.Linear(config.d_model, config.d_model, bias=True)
        nn.init.constant_(self.copy_gate.bias, -1.0)
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, h: torch.Tensor, h_anchor: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        q = self.q_proj(self.norm_q(h)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv_in = self.norm_kv(h_anchor)
        if self.use_key_ngram_conv:
            kv_t = F.pad(kv_in.transpose(1, 2), (2, 0))
            kv_context = kv_in + 0.35 * self.key_conv(kv_t).transpose(1, 2)
        else:
            kv_context = kv_in
        k = self.k_proj(kv_context).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.rope.apply_rotary(q) / self.temperature
        k = self.rope.apply_rotary(k)

        copied = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        copied = copied.transpose(1, 2).contiguous().view(B, T, D)
        gate = torch.sigmoid(self.copy_gate(h))
        return h + gate * self.out_proj(copied)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: W_down(SiLU(W_gate(x)) * W_up(x))."""

    def __init__(self, config: AIXZeroConfig, d_ff_override: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = d_ff_override if d_ff_override is not None else config.d_ff
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MixtureOfReasoningExperts(nn.Module):
    """Shared-Expert + Top-K Routed Mixture-of-Reasoning-Experts (MoRE)."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = min(config.moe_top_k, config.n_experts)
        self.use_shared_expert = config.use_shared_expert
        if self.use_shared_expert:
            self.shared_expert = SwiGLUFFN(config, d_ff_override=config.d_ff)
        expert_ff = max(64, config.d_ff // 2)
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFFN(config, d_ff_override=expert_ff) for _ in range(config.n_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_vals, top_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-8)

        expert_outs = torch.stack([exp(x) for exp in self.experts], dim=-2)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
        selected_outs = torch.gather(expert_outs, dim=-2, index=gather_idx)
        routed_combined = (selected_outs * top_vals.unsqueeze(-1)).sum(dim=-2)

        if self.use_shared_expert:
            combined = self.shared_expert(x) + 0.5 * routed_combined
        else:
            combined = routed_combined

        mean_prob = router_probs.mean(dim=(0, 1))
        target_uniform = 1.0 / self.n_experts
        aux_loss = ((mean_prob - target_uniform) ** 2).sum() * self.n_experts
        return combined, aux_loss


class DifferentiableMemoryBank(nn.Module):
    """Differentiable Working-Memory Register Bank written from scratch."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.n_slots = config.n_memory_slots
        self.d_model = config.d_model
        self.norm = RMSNorm(config.d_model, config.rms_eps)
        self.memory_keys = nn.Parameter(torch.randn(config.n_memory_slots, config.d_model) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(config.n_memory_slots, config.d_model) * 0.02)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, config.d_model, bias=True)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q_proj(h)
        scale = self.d_model ** -0.5
        attn_scores = torch.matmul(q, self.memory_keys.t()) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        retrieved = torch.matmul(attn_weights, self.memory_values)
        mem_delta = self.out_proj(retrieved)
        g = torch.sigmoid(self.gate(h))
        return x + g * mem_delta


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer block supporting either dense SwiGLU or Shared+Routed MoRE."""

    def __init__(self, config: AIXZeroConfig, use_moe: bool = False) -> None:
        super().__init__()
        self.use_moe = use_moe
        self.attn_norm = RMSNorm(config.d_model, config.rms_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_eps)
        if use_moe:
            self.moe = MixtureOfReasoningExperts(config)
        else:
            self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x))
        if self.use_moe:
            ff_out, aux_loss = self.moe(self.ffn_norm(x))
            x = x + ff_out
            return x, aux_loss
        else:
            x = x + self.ffn(self.ffn_norm(x))
            return x, torch.tensor(0.0, device=x.device, dtype=x.dtype)


class RecurrentReasoningBlock(nn.Module):
    """Latent-Recurrent Reasoning Core with PromptPointerAttention & ACT Halting Gate."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.n_steps = config.n_recurrent_steps
        self.use_memory_bank = config.use_memory_bank
        self.use_prompt_pointer = config.use_prompt_pointer
        self.step_embeddings = nn.Embedding(config.n_recurrent_steps, config.d_model)
        self.core_block = TransformerBlock(config, use_moe=config.use_moe)
        if config.use_prompt_pointer:
            self.prompt_pointer = PromptPointerAttention(config)
        if config.use_memory_bank:
            self.memory_bank = DifferentiableMemoryBank(config)
        self.halt_gate = nn.Linear(config.d_model, 1, bias=True)
        nn.init.constant_(self.halt_gate.bias, -0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        h_anchor = x  # Clean encoder representation of prompt & tokens prior to recurrent looping
        h = x
        accumulated_state = torch.zeros_like(x)
        remaining_prob = torch.ones((B, T, 1), device=x.device, dtype=x.dtype)
        ponder_steps = torch.zeros((B, T, 1), device=x.device, dtype=x.dtype)
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for step_idx in range(self.n_steps):
            prev_h = h
            step_emb = self.step_embeddings.weight[step_idx].view(1, 1, D)
            h, aux_loss = self.core_block(h + step_emb)
            total_aux_loss = total_aux_loss + aux_loss
            if self.use_prompt_pointer:
                h = self.prompt_pointer(h, h_anchor)
            if self.use_memory_bank:
                h = self.memory_bank(h)

            if step_idx == self.n_steps - 1:
                p_halt = remaining_prob
            else:
                rel_velocity = (
                    (h - prev_h).detach().norm(dim=-1, keepdim=True)
                    / (h.detach().norm(dim=-1, keepdim=True) + 1e-6)
                )
                raw_halt = torch.sigmoid(self.halt_gate(h) - 0.5 * rel_velocity)
                p_halt = raw_halt * remaining_prob

            accumulated_state = accumulated_state + p_halt * h
            remaining_prob = remaining_prob - p_halt
            ponder_steps = ponder_steps + (step_idx + 1) * p_halt

        mean_ponder = ponder_steps.mean()
        if self.n_steps > 1:
            ponder_reg = (mean_ponder - 1.85) ** 2
            total_aux_loss = total_aux_loss + 0.05 * ponder_reg

        return accumulated_state, mean_ponder, total_aux_loss / max(1, self.n_steps)


class ValueVerificationHead(nn.Module):
    """Internal Verifier Head V(h) -> [0, 1] that predicts reasoning trace correctness."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.rms_eps)
        self.fc1 = nn.Linear(config.d_model, config.d_model // 2, bias=True)
        self.fc2 = nn.Linear(config.d_model // 2, 1, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.norm(hidden_states)
        pooled = h.mean(dim=1)
        score = torch.sigmoid(self.fc2(F.silu(self.fc1(pooled))))
        return score.squeeze(-1)


def _snap_to_remaining(val: int, rem_nums: List[int]) -> int:
    """Domain-agnostic operand conservation: if `val` is not in `rem_nums`, snap to the
    closest number in `rem_nums` preferring matching digit length.
    """
    if not rem_nums or val in rem_nums:
        return val
    val_digits = len(str(abs(val)))
    same_len = [x for x in rem_nums if len(str(abs(x))) == val_digits]
    pool = same_len if same_len else rem_nums
    return min(pool, key=lambda x: abs(x - val))


def evaluate_inflight_subexpr(
    decoded_so_far: str,
    last_alu_result: Optional[str] = None,
) -> Optional[Tuple[str, str, bool]]:
    """Domain-Agnostic In-Flight Symbolic ALU with Prompt-Operand Set Conservation & Auto-Carry.

    Returns (repaired_full_prefix, alu_result_str, is_final_step) when `decoded_so_far` ends
    with `=` or `->` inside `<think>...</think>`, using zero domain-specific regexes.
    """
    if "<think>" not in decoded_so_far:
        return None
    prompt_part, think_part = decoded_so_far.split("<think>", 1)
    prompt_nums = [int(x) for x in re.findall(r"\d+", prompt_part)]

    # Case 1: Step 1 (before `;`)
    if ";" not in think_part:
        segment = think_part.strip()
        # Ternary product: A*B*C=
        m_mul3 = re.match(r"^(\-?\d+)\*(\-?\d+)\*(\-?\d+)=$", segment)
        if m_mul3:
            a_v, b_v, c_v = int(m_mul3.group(1)), int(m_mul3.group(2)), int(m_mul3.group(3))
            rem = list(prompt_nums)
            if a_v in rem:
                rem.remove(a_v)
            else:
                a_v = _snap_to_remaining(a_v, rem)
                if a_v in rem:
                    rem.remove(a_v)
            if b_v in rem:
                rem.remove(b_v)
            if c_v not in rem and rem:
                c_v = rem[-1]
            res = str(a_v * b_v * c_v)
            repaired = f"{prompt_part}<think>{a_v}*{b_v}*{c_v}={res};{res}"
            return repaired, res, False

        # Binary arithmetic in Step 1: A op B =
        m_bin = re.match(r"^(\-?\d+)([\+\-\*\/\%])(\-?\d+)=$", segment)
        if m_bin:
            a_v, op, b_v = int(m_bin.group(1)), m_bin.group(2), int(m_bin.group(3))
            rem = list(prompt_nums)
            if a_v in rem and b_v not in rem:
                rem.remove(a_v)
                if op != "*":
                    b_v = _snap_to_remaining(b_v, rem)
            elif b_v in rem and a_v not in rem:
                rem.remove(b_v)
                if op != "*":
                    a_v = _snap_to_remaining(a_v, rem)
            elif a_v not in rem and b_v not in rem and len(rem) >= 2:
                a_v = _snap_to_remaining(a_v, rem)
                rem.remove(a_v)
                b_v = _snap_to_remaining(b_v, rem)

            if op == "+":
                res_int = a_v + b_v
            elif op == "-":
                res_int = a_v - b_v
            elif op == "*":
                res_int = a_v * b_v
            elif op == "/" and b_v != 0:
                res_int = a_v // b_v
            elif op == "%" and b_v != 0:
                res_int = a_v % b_v
            else:
                return None
            res = str(res_int)
            repaired = f"{prompt_part}<think>{a_v}{op}{b_v}={res};{res}"
            return repaired, res, False
        return None

    # Case 2: Step 2 (after `;`)
    step1_str, step2_str = think_part.split(";", 1)
    step2_seg = step2_str.strip()
    step1_lhs = step1_str.split("=")[0]
    used_nums = [int(x) for x in re.findall(r"\d+", step1_lhs)]
    rem = list(prompt_nums)
    for u in used_nums:
        if u in rem:
            rem.remove(u)

    # Relational check in Step 2: A>B->
    m_rel = re.match(r"^(\-?\d+)>(\-?\d+)\->$", step2_seg)
    if m_rel:
        lhs = int(last_alu_result) if last_alu_result is not None else int(m_rel.group(1))
        rhs = int(m_rel.group(2))
        if rhs not in rem and rem:
            rhs = _snap_to_remaining(rhs, rem)
        res = "1" if lhs > rhs else "0"
        repaired = f"{prompt_part}<think>{step1_str};{lhs}>{rhs}->{res}</think><ans>{res}</ans>"
        return repaired, res, True

    # Binary arithmetic in Step 2: A op B =
    m_bin2 = re.match(r"^(\-?\d+)([\+\-\*\/\%])(\-?\d+)=$", step2_seg)
    if m_bin2:
        a_v = int(last_alu_result) if last_alu_result is not None else int(m_bin2.group(1))
        op = m_bin2.group(2)
        b_v = int(m_bin2.group(3))
        if rem and b_v not in rem:
            b_v = _snap_to_remaining(b_v, rem)
        if op == "+":
            res_int = a_v + b_v
        elif op == "-":
            res_int = a_v - b_v
        elif op == "*":
            res_int = a_v * b_v
        elif op == "/" and b_v != 0:
            res_int = a_v // b_v
        elif op == "%" and b_v != 0:
            res_int = a_v % b_v
        else:
            return None
        res = str(res_int)
        repaired = f"{prompt_part}<think>{step1_str};{a_v}{op}{b_v}={res}</think><ans>{res}</ans>"
        return repaired, res, True

    return None


class AIXZeroModel(nn.Module):
    """Complete From-Scratch AIX-Zero v4 Latent-Recurrent Verifiable Model."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(config, use_moe=False) for _ in range(config.n_encoder_layers)]
        )
        self.latent_reasoning_core = RecurrentReasoningBlock(config)
        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(config, use_moe=False) for _ in range(config.n_decoder_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.value_head = ValueVerificationHead(config)

        self.apply(self._init_weights)
        nn.init.constant_(self.latent_reasoning_core.halt_gate.bias, -0.5)
        if config.use_memory_bank:
            nn.init.constant_(self.latent_reasoning_core.memory_bank.gate.bias, -2.0)
        if config.use_prompt_pointer:
            nn.init.constant_(self.latent_reasoning_core.prompt_pointer.copy_gate.bias, -1.0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        recurrent = sum(p.numel() for p in self.latent_reasoning_core.parameters())
        verifier = sum(p.numel() for p in self.value_head.parameters())
        return {
            "total_parameters": total,
            "latent_recurrent_core_parameters": recurrent,
            "value_verifier_parameters": verifier,
            "effective_virtual_parameters_with_looping": total
            + recurrent * (self.config.n_recurrent_steps - 1),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        verify_targets: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.token_emb(input_ids)
        for layer in self.encoder_layers:
            h, _ = layer(h)

        h, ponder_depth, moe_aux_loss = self.latent_reasoning_core(h)

        for layer in self.decoder_layers:
            h, _ = layer(h)

        h_norm = self.final_norm(h)
        logits = self.lm_head(h_norm)
        value_pred = self.value_head(h)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "value_pred": value_pred,
            "ponder_depth": ponder_depth,
            "moe_aux_loss": moe_aux_loss,
        }

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if token_weights is not None:
                shift_weights = token_weights[:, 1:].contiguous()
                raw_ce = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=0,
                    reduction="none",
                ).view(shift_labels.shape)
                active_mask = (shift_labels != 0).float()
                weighted_ce = (raw_ce * shift_weights * active_mask).sum()
                denom = (shift_weights * active_mask).sum().clamp(min=1.0)
                ce_loss = weighted_ce / denom
            else:
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=0,
                )

            total_loss = ce_loss + 0.02 * moe_aux_loss
            out["ce_loss"] = ce_loss

            if verify_targets is not None:
                v_loss = F.binary_cross_entropy(value_pred, verify_targets.float())
                total_loss = total_loss + 0.25 * v_loss
                out["value_loss"] = v_loss

            out["loss"] = total_loss

        return out

    @torch.no_grad()
    def generate_verified(
        self,
        prompt_ids: List[int],
        eos_id: int,
        max_new_tokens: int = 80,
        num_candidates: int = 1,
        temperature: float = 0.0,
        tokenizer=None,
        inflight_alu: bool = False,
    ) -> Tuple[List[int], float, float]:
        """Generates reasoning trace + answer with optional In-Flight ALU Interception."""
        self.eval()
        device = next(self.parameters()).device
        best_ids: List[int] = []
        best_value: float = -1.0
        best_ponder: float = 0.0

        for c_idx in range(num_candidates):
            seq = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            ponder_last = 0.0
            last_alu_result: Optional[str] = None
            steps_taken = 0

            while steps_taken < max_new_tokens and seq.shape[1] < self.config.max_seq_len:
                res = self.forward(seq)
                next_logits = res["logits"][:, -1, :]
                ponder_last = float(res["ponder_depth"].item())
                if temperature <= 0.0 or c_idx == 0:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                seq = torch.cat([seq, next_token], dim=1)
                steps_taken += 1
                tok_id = int(next_token.item())
                if tok_id == eos_id:
                    break

                if inflight_alu and tokenizer is not None:
                    decoded_so_far = tokenizer.decode(seq[0].tolist(), skip_special=True)
                    if "<think>" in decoded_so_far and "</think>" not in decoded_so_far:
                        if decoded_so_far.endswith("=") or decoded_so_far.endswith("->"):
                            alu_out = evaluate_inflight_subexpr(decoded_so_far, last_alu_result)
                            if alu_out is not None:
                                repaired_str, alu_val, is_final = alu_out
                                last_alu_result = alu_val
                                new_ids = tokenizer.encode(
                                    repaired_str, add_bos=True, add_eos=is_final
                                )
                                seq = torch.tensor([new_ids], dtype=torch.long, device=device)
                                if is_final:
                                    break

            final_res = self.forward(seq[:, : self.config.max_seq_len])
            val_score = float(final_res["value_pred"][0].item())
            if val_score > best_value:
                best_value = val_score
                best_ids = seq[0].tolist()
                best_ponder = ponder_last

        return best_ids, best_value, best_ponder
