"""Knowledge distillation: generate instruction-response data from an open teacher.

Loads a Qwen-Instruct teacher, feeds it real instructions (Korean from kullm-v2 +
English from dolly-15k), and saves the teacher's high-quality responses as chat
'messages'. SFT-ing Gaon on this transfers the teacher's instruction-following
ability into our 0.6B base. (Open Qwen license permits distillation; Claude/GPT
outputs would not.)

    python -m src.posttrain.distill --teacher teacher_qwen25_7b \
        --out data/distill.jsonl --n-ko 6000 --n-en 3000
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_instructions(n_ko: int, n_en: int):
    from datasets import load_dataset

    insts = []
    if n_ko:
        ko = load_dataset("nlpai-lab/kullm-v2", split="train").shuffle(seed=0).select(range(n_ko))
        for ex in ko:
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            if instr:
                insts.append(instr if not inp else f"{instr}\n\n{inp}")
    if n_en:
        en = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=1).select(range(n_en))
        for ex in en:
            instr = (ex.get("instruction") or "").strip()
            ctx = (ex.get("context") or "").strip()
            if instr:
                insts.append(instr if not ctx else f"{instr}\n\n{ctx}")
    return insts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", default="data/distill.jsonl")
    ap.add_argument("--n-ko", type=int, default=6000)
    ap.add_argument("--n-en", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.teacher)
    tok.padding_side = "left"                       # required for batched generation
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    insts = seed_instructions(args.n_ko, args.n_en)
    print(f"teacher={args.teacher} | {len(insts)} instructions -> {args.out}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = 0
    with open(args.out, "w") as f:
        for i in range(0, len(insts), args.batch_size):
            batch = insts[i : i + args.batch_size]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": x}], tokenize=False, add_generation_prompt=True
                )
                for x in batch
            ]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=args.max_new, do_sample=True,
                    temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id,
                )
            gen = out_ids[:, enc.input_ids.shape[1]:]
            resps = tok.batch_decode(gen, skip_special_tokens=True)
            for x, r in zip(batch, resps):
                r = r.strip()
                if not r:
                    continue
                f.write(json.dumps(
                    {"messages": [
                        {"role": "user", "content": x},
                        {"role": "assistant", "content": r},
                    ]}, ensure_ascii=False) + "\n")
            f.flush()
            done += len(batch)
            print(f"generated {done}/{len(insts)}", flush=True)

    print(f"done. distilled dataset -> {args.out}", flush=True)
    import sys
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
