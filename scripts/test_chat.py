"""Batch chat test for an instruction-tuned Gaon (HF dir). Non-interactive.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m scripts.test_chat checkpoints/sft_1.7b
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft_1.7b"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device).eval()
print(f"loaded {model_dir} on {device}\n", flush=True)

prompts = [
    "한국의 수도는 어디인가요?",
    "인공지능이 무엇인지 초등학생도 이해할 수 있게 설명해줘.",
    "Write a haiku about the ocean.",
    "파이썬으로 1부터 10까지 더하는 코드를 작성해줘.",
    "아침에 일어나기 힘든데 어떻게 하면 좋을까?",
]
for p in prompts:
    msgs = [{"role": "user", "content": p}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    resp = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"[Q] {p}\n[A] {resp}\n{'-'*60}", flush=True)
import os
os._exit(0)
