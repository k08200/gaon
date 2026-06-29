"""Interactive chat with an instruction-tuned Gaon (HF format).

    # on the box (use a free GPU so it doesn't touch training):
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m src.eval.chat --model checkpoints/sft

    # on your Mac (CPU/MPS, no GPU needed — slower):
    python -m src.eval.chat --model checkpoints/gaon-0.6b-instruct

Type a message and press enter. Type 'exit' (or Ctrl-C) to quit.
Each turn is independent (no history) — it's a small model; keep prompts simple.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir (SFT output)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[device]
    print(f"loading {args.model} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print("ready. type a message ('exit' to quit).\n", flush=True)

    while True:
        try:
            user = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in {"exit", "quit", "q"}:
            break
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=True,
                temperature=args.temperature, top_p=0.9, pad_token_id=tok.eos_token_id,
            )
        resp = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"gaon ▸ {resp}\n", flush=True)


if __name__ == "__main__":
    main()
