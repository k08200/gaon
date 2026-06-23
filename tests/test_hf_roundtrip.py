"""Validate that our from-scratch weights map EXACTLY onto HuggingFace Qwen3.

If this passes, the SFT/eval path (which runs on the HF model) is operating on
the same function our pretraining produced — the weight mapping in
src.posttrain.sft.to_hf is correct, not just loadable.

    .venv/bin/python tests/test_hf_roundtrip.py
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import ModelConfig, Gaon  # noqa: E402
from src.posttrain.sft import to_hf       # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=512, hidden_size=128, intermediate_size=256,
        num_layers=4, num_attn_heads=8, num_kv_heads=4, head_dim=32,
        max_seq_len=64,
    )
    ours = Gaon(cfg).eval()

    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        logits_ours, _ = ours(x, x, return_logits=True)   # full-sequence logits

    # save a checkpoint exactly like train.save() does, then convert
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt.pt"
        torch.save({"model": ours.state_dict(), "config": cfg.__dict__, "step": 0}, ckpt)
        hf = to_hf(str(ckpt), "Qwen/Qwen3-0.6B").eval()

    with torch.no_grad():
        logits_hf = hf(x).logits

    diff = (logits_ours - logits_hf).abs().max().item()
    print(f"max logit diff (ours vs HF): {diff:.3e}")
    assert diff < 1e-3, f"weight mapping mismatch: {diff}"
    print("PASS: from-scratch weights map exactly onto HF Qwen3 ✅")


if __name__ == "__main__":
    main()
