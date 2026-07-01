# Model Card — Gaon (가온)

Bilingual (Korean + English) decoder-only language models trained from scratch.
Two sizes share one codebase and are bit-compatible with HuggingFace Qwen3.

## Models

| Model | Params | Base loss | Instruct | Format |
|---|---|---|---|---|
| Gaon-0.6B / -Instruct | 596M | 2.48 | ✅ | HF `Qwen3ForCausalLM` |
| Gaon-1.7B / -Instruct | 1.72B | 2.37 | ✅ | HF `Qwen3ForCausalLM` |

## Intended use

- Research and education: a fully-owned, reproducible small-LLM pipeline.
- Bilingual KO/EN text generation and simple instruction following.
- A base for **vertical fine-tuning** on domain-specific data.

**Not** intended as a general assistant or a drop-in replacement for frontier models.

## How to use (instruction-tuned)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

m = "checkpoints/sft_1.7b"  # or the published HF repo
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16).eval()

msgs = [{"role": "user", "content": "한국의 수도는 어디인가요?"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors="pt")
out = model.generate(**enc, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True))
```

Runs on CPU/MPS (Mac) or CUDA. On Apple Silicon use `torch_dtype=torch.float16`.

## Training data

- **Pretraining:** FineWeb-Edu (English), FineWeb-2 `kor_Hang` (Korean), and — in the
  v2 run — codeparrot-clean (Python). Tokenized with the Qwen3 tokenizer (vocab 151,936),
  packed into flat `uint32` shards. ~12B tokens (v1) / ~34B tokens (v2).
- **Instruction tuning:** ~9,000 examples produced by sequence-level knowledge
  distillation from Qwen2.5-7B-Instruct (Apache-2.0). No proprietary-API outputs were
  used for training (their ToS forbids it).

## Limitations & biases

- **Coding and strict-format tasks are weak** — inherent to the model scale and token
  budget. Do not rely on generated code without review.
- Factual recall is limited vs. models trained on 100–1000× more tokens; may hallucinate.
- Inherits biases present in web-scale pretraining data.
- Small models are more prone to repetition and off-topic drift on long generations.

## License

Code: open (see repository). Tokenizer and architecture follow Qwen3 (Apache-2.0).
Distillation teacher (Qwen2.5-7B-Instruct) is Apache-2.0, which permits using its
outputs as training data.

## Citation

```
@software{gaon2026,
  title  = {Gaon: Bilingual Language Models Trained From Scratch},
  author = {independent research},
  year   = {2026},
  url    = {https://github.com/k08200/gaon}
}
```
