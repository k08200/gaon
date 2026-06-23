"""Fast correctness checks that run on CPU in seconds — no data/GPU needed.

  python scripts/sanity_check.py

Verifies: forward shapes, causality (future tokens can't change past logits),
parameter count is in the expected ~0.6B range, and that the model can overfit
a tiny batch (loss -> ~0), which proves the training signal flows end to end.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import ModelConfig, Gaon  # noqa: E402


def tiny_cfg() -> ModelConfig:
    # shrink everything so it runs instantly on CPU, same architecture
    return ModelConfig(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_layers=4, num_attn_heads=8, num_kv_heads=4, head_dim=32,
        max_seq_len=64,
    )


def test_shapes(cfg):
    m = Gaon(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(x)
    assert logits.shape == (2, 1, cfg.vocab_size), logits.shape
    assert loss is None
    _, loss = m(x, x)
    assert loss.ndim == 0 and loss.item() > 0
    print("  [ok] forward shapes + loss")


def test_causality(cfg):
    m = Gaon(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        full = m(x, x)[0]               # (1, 16, V)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size  # perturb last token
        pert = m(x2, x2)[0]
    diff = (full[0, :-1] - pert[0, :-1]).abs().max().item()
    assert diff < 1e-4, f"past logits changed when future token changed: {diff}"
    print("  [ok] causal masking")


def test_overfit(cfg):
    torch.manual_seed(0)
    m = Gaon(cfg).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    y = torch.randint(0, cfg.vocab_size, (4, 32))
    first = last = None
    for i in range(200):
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.2, f"failed to overfit: {first:.3f} -> {last:.3f}"
    print(f"  [ok] overfit tiny batch: {first:.3f} -> {last:.3f}")


def test_param_count():
    cfg = ModelConfig()
    m = Gaon(cfg)
    n = m.num_params()
    print(f"  [info] full Qwen3-0.6B config params: {n/1e6:.1f}M")
    assert 550e6 < n < 800e6, f"unexpected param count: {n}"
    print("  [ok] param count in ~0.6B range")


if __name__ == "__main__":
    cfg = tiny_cfg()
    print("running sanity checks (tiny model, CPU)...")
    test_shapes(cfg)
    test_causality(cfg)
    test_overfit(cfg)
    test_param_count()
    print("all checks passed ✅")
