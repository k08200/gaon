"""Load a pretrained checkpoint and generate text — quick qualitative check.

    python -m src.eval.generate --ckpt checkpoints/qwen3_0.6b/ckpt_40000.pt \
        --prompt "The history of language models" --max-new 100
"""

from __future__ import annotations

import argparse

import torch

from ..model import ModelConfig, Qwen3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig(**ck["config"])
    model = Qwen3(cfg).to(device).eval()
    # strip FSDP/DDP prefixes if present
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=False)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, args.max_new, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
