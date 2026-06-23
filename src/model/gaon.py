"""Qwen3-style decoder-only transformer, implemented from scratch in PyTorch.

Faithful to Qwen3 design choices:
  - RMSNorm (pre-norm)
  - Rotary position embeddings (RoPE)
  - Grouped-Query Attention (GQA)
  - QK-Norm: per-head RMSNorm on query and key (Qwen3-specific)
  - SwiGLU MLP
  - Tied input/output embeddings

Single-file, no external model libs, so the math is fully visible and hackable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float, device=None):
    """Return (cos, sin), each (max_seq_len, head_dim), HF/Llama rotate_half style.

    Dimension i is paired with i + head_dim/2 (NOT i with i+1). This matches
    HuggingFace Qwen3/Llama exactly, so our weights are bit-compatible with HF.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # (seq, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (seq, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, rope) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim). rope: (cos, sin) each (T_max, head_dim)."""
    cos, sin = rope
    t = x.size(2)
    cos = cos[:t].view(1, 1, t, -1)
    sin = sin[:t].view(1, 1, t, -1)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, D) -> (B, n_kv * n_rep, T, D)."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, t, d)
        .reshape(b, n_kv * n_rep, t, d)
    )


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=False)
        # Qwen3 QK-Norm: RMSNorm applied per-head over head_dim
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        cfg = self.cfg
        q = self.q_proj(x).view(b, t, cfg.num_attn_heads, cfg.head_dim)
        k = self.k_proj(x).view(b, t, cfg.num_kv_heads, cfg.head_dim)
        v = self.v_proj(x).view(b, t, cfg.num_kv_heads, cfg.head_dim)

        q = self.q_norm(q).transpose(1, 2)   # (B, n_heads, T, D)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = repeat_kv(k, cfg.n_rep)
        v = repeat_kv(v, cfg.n_rep)

        # FlashAttention via SDPA with causal masking
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, cfg.q_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Gaon(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.num_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.num_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def _chunk_loss(self, hs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hs)
        return F.cross_entropy(logits.float(), ts, ignore_index=-100, reduction="sum")

    def loss_from_hidden(self, x: torch.Tensor, targets: torch.Tensor,
                         chunk_size: int = 4096) -> torch.Tensor:
        """Memory-efficient cross-entropy.

        The LM-head logits tensor is (B*T, vocab=151936) — at batch 16 / seq 4096
        that's ~20GB in bf16 and ~40GB once cross_entropy upcasts to fp32 for the
        backward, which OOMs even a 192GB B200. We split the flattened sequence
        into chunks and gradient-checkpoint each chunk's head+CE, so peak logit
        memory is only (chunk_size, vocab) and is recomputed in backward. This lets
        the real runs use much larger batches (far higher throughput).
        """
        from torch.utils.checkpoint import checkpoint

        h = x.reshape(-1, x.size(-1))
        t = targets.reshape(-1)
        n = (t != -100).sum().clamp(min=1)
        total = x.new_zeros((), dtype=torch.float32)
        for hs, ts in zip(h.split(chunk_size), t.split(chunk_size)):
            total = total + checkpoint(self._chunk_loss, hs, ts, use_reentrant=False)
        return total / n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_logits: bool = False):
        x = self.embed(idx)
        rope = (self.rope_cos, self.rope_sin)
        for block in self.blocks:
            x = block(x, rope)
        x = self.norm(x)

        if targets is None:
            # inference: only compute last position
            return self.lm_head(x[:, -1:, :]), None

        loss = self.loss_from_hidden(x, targets)
        # logits are only materialized when explicitly requested (tests/inspection);
        # the training loop discards them, so we skip the 20GB+ allocation by default.
        logits = self.lm_head(x) if return_logits else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.lm_head.weight.numel()
        return n
