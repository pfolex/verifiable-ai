"""AIX-Zero Neural Network Architecture written 100% from scratch in PyTorch.

Core architectural innovations over standard fixed-depth Transformers:
1. RMSNorm + 2D Rotary Position Embeddings (RoPE) + Grouped-Query Attention (GQA) + SwiGLU FFN.
2. RecurrentReasoningBlock (Latent Ponder Core):
   Instead of forcing every token through a static number of layers, the middle reasoning core
   iterates K steps in continuous latent space with step-indexed embeddings and a learned
   Adaptive Halting Gate p_halt = sigmoid(W_halt(h_t)), allowing deeper latent compute on
   complex mathematical states before emitting tokens.
3. ValueVerificationHead:
   A built-in verifier head V(h) -> [0, 1] trained jointly to score whether a generated
   reasoning trace is mathematically valid, enabling self-verified Best-of-N search without
   any external teacher model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AIXZeroConfig:
    vocab_size: int = 104
    d_model: int = 192
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 512
    n_encoder_layers: int = 2
    n_recurrent_steps: int = 3  # Latent recurrent thinking iterations per token
    n_decoder_layers: int = 2
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    dropout: float = 0.0


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization written from scratch:
    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
    """

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
        freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim // 2)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, T, head_dim)
        seq_len = x.shape[2]
        half = x.shape[-1] // 2
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
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


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network: W_down(SiLU(W_gate(x)) * W_up(x))."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Standard pre-RMSNorm Transformer block with GQA + SwiGLU."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RecurrentReasoningBlock(nn.Module):
    """Latent-Recurrent Reasoning Core with Adaptive Halting Gate.

    Iterates `n_recurrent_steps` times in continuous hidden space before token emission.
    At each latent step k, injects a learned step embedding e_k and computes a soft
    halting gate g_k = sigmoid(halt_proj(h_k)) that blends shallow and deep latent states.
    """

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.n_steps = config.n_recurrent_steps
        self.step_embeddings = nn.Embedding(config.n_recurrent_steps, config.d_model)
        self.core_block = TransformerBlock(config)
        self.halt_gate = nn.Linear(config.d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        h = x
        accumulated_state = torch.zeros_like(x)
        remaining_prob = torch.ones((B, T, 1), device=x.device, dtype=x.dtype)
        ponder_steps = torch.zeros((B, T, 1), device=x.device, dtype=x.dtype)

        for step_idx in range(self.n_steps):
            step_emb = self.step_embeddings.weight[step_idx].view(1, 1, D)
            h = self.core_block(h + step_emb)
            if step_idx == self.n_steps - 1:
                p_halt = remaining_prob
            else:
                p_halt = torch.sigmoid(self.halt_gate(h)) * remaining_prob
            accumulated_state = accumulated_state + p_halt * h
            remaining_prob = remaining_prob - p_halt
            ponder_steps = ponder_steps + (step_idx + 1) * p_halt

        return accumulated_state, ponder_steps.mean()


class ValueVerificationHead(nn.Module):
    """Internal Verifier Head V(h) -> [0, 1] that predicts reasoning trace correctness."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, config.rms_eps)
        self.fc1 = nn.Linear(config.d_model, config.d_model // 2, bias=True)
        self.fc2 = nn.Linear(config.d_model // 2, 1, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        h = self.norm(hidden_states)
        pooled = h.mean(dim=1)  # Sequence-pooled representation (B, D)
        score = torch.sigmoid(self.fc2(F.silu(self.fc1(pooled))))
        return score.squeeze(-1)  # (B,)


class AIXZeroModel(nn.Module):
    """Complete From-Scratch AIX-Zero Latent-Recurrent Verifiable Model."""

    def __init__(self, config: AIXZeroConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_encoder_layers)]
        )
        self.latent_reasoning_core = RecurrentReasoningBlock(config)
        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_decoder_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.rms_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.value_head = ValueVerificationHead(config)

        self.apply(self._init_weights)

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
            "effective_virtual_parameters_with_looping": total + recurrent * (self.config.n_recurrent_steps - 1),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        verify_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.token_emb(input_ids)
        for layer in self.encoder_layers:
            h = layer(h)

        h, ponder_depth = self.latent_reasoning_core(h)

        for layer in self.decoder_layers:
            h = layer(h)

        h_norm = self.final_norm(h)
        logits = self.lm_head(h_norm)
        value_pred = self.value_head(h)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "value_pred": value_pred,
            "ponder_depth": ponder_depth,
        }

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=0,  # <pad> token ID is 0
            )
            total_loss = ce_loss
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
    ) -> Tuple[List[int], float, float]:
        """Generates reasoning trace + answer and scores it with the internal ValueVerificationHead."""
        self.eval()
        device = next(self.parameters()).device
        best_ids: List[int] = []
        best_value: float = -1.0
        best_ponder: float = 0.0

        for c_idx in range(num_candidates):
            seq = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            ponder_last = 0.0
            for _ in range(max_new_tokens):
                if seq.shape[1] >= self.config.max_seq_len:
                    break
                res = self.forward(seq)
                next_logits = res["logits"][:, -1, :]
                ponder_last = float(res["ponder_depth"].item())
                if temperature <= 0.0 or c_idx == 0:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                seq = torch.cat([seq, next_token], dim=1)
                if int(next_token.item()) == eos_id:
                    break

            final_res = self.forward(seq)
            val_score = float(final_res["value_pred"][0].item())
            if val_score > best_value:
                best_value = val_score
                best_ids = seq[0].tolist()
                best_ponder = ponder_last

        return best_ids, best_value, best_ponder
