"""Model configuration for the Qwen3-style decoder-only LLM.

Defaults replicate Qwen3-0.6B exactly so we can use its tokenizer and
benchmark against the official checkpoint as a baseline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # --- core dimensions (Qwen3-0.6B) ---
    vocab_size: int = 151_936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_layers: int = 28
    num_attn_heads: int = 16
    num_kv_heads: int = 8          # GQA: 16 query heads share 8 KV heads
    head_dim: int = 128            # decoupled from hidden_size (16*128=2048 != 1024)

    # --- positional / norm ---
    max_seq_len: int = 4096        # train length; extend later via RoPE scaling
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6

    # --- regularization / init ---
    dropout: float = 0.0
    init_std: float = 0.02

    # --- weight tying ---
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.num_attn_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attn_heads ({self.num_attn_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @property
    def q_dim(self) -> int:
        return self.num_attn_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def n_rep(self) -> int:
        """How many query heads share one KV head."""
        return self.num_attn_heads // self.num_kv_heads

    def num_params(self) -> int:
        """Approximate non-embedding + embedding parameter count."""
        h, i, l, v = self.hidden_size, self.intermediate_size, self.num_layers, self.vocab_size
        per_layer = (
            self.q_dim * h + 2 * self.kv_dim * h + self.q_dim * h  # attn proj
            + 3 * i * h                                            # SwiGLU
            + 2 * h                                                # 2 RMSNorms
        )
        embed = v * h * (1 if self.tie_embeddings else 2)
        return per_layer * l + embed + h  # + final norm
