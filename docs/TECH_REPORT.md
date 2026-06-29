# Gaon (가온): A Bilingual 0.6B Language Model Trained From Scratch

*A reproducible, end-to-end build of a small Korean + English LLM — pretraining
through instruction tuning — on a single GPU.*

**Repository:** https://github.com/k08200/gaon · **License:** open · **Author:** independent research

---

## Abstract

Gaon is a 596M-parameter decoder-only language model built entirely from scratch:
architecture, tokenizing/data pipeline, pretraining loop, and post-training. It is
bilingual (Korean + English) by design, since most small open models are weak in
Korean. The base model was pretrained on ~12B tokens and reached a final
cross-entropy loss of **2.48**; an instruction-tuned variant was produced by
**knowledge distillation** from an open Qwen teacher. The resulting
Gaon-0.6B-Instruct follows instructions and answers factual questions correctly in
Korean and English. The whole project runs on a single GPU and is fully
reproducible from the public repository. The goal is not to beat frontier models —
a 0.6B model cannot — but to own every stage of the pipeline and establish a base
that scales with the same code.

## 1. Architecture

Gaon is a Qwen3-compatible decoder-only transformer, implemented in a single
PyTorch file (`src/model/gaon.py`) with no external model libraries, so every
operation is visible and hackable.

| Component | Choice |
|---|---|
| Layers / hidden / heads | 28 / 1024 / 16 Q, 8 KV (GQA) |
| Head dim | 128 (decoupled from hidden) |
| Positional | RoPE (θ = 1e6), **HF rotate-half convention** |
| Attention norm | **QK-Norm** (per-head RMSNorm on q, k) — Qwen3-specific |
| MLP | SwiGLU, intermediate 3072 |
| Norm | RMSNorm (pre-norm) |
| Vocab / tokenizer | 151,936, reused from Qwen3 (multilingual) |
| Embeddings | tied | 
| **Parameters** | **596M** |

The model is **bit-compatible with HuggingFace Qwen3**: a round-trip test
(`tests/test_hf_roundtrip.py`) confirms our weights map onto `Qwen3ForCausalLM`
with a max logit difference of `0.0`. This lets us reuse the HF ecosystem for
post-training and evaluation, and required matching the exact RoPE convention
(rotate-half, not interleaved-complex).

## 2. Memory-efficient training (key technique)

The large vocabulary (151,936) makes the LM-head logits the memory bottleneck: at
batch 16 / seq 4096 the logit tensor alone is ~20 GB, and cross-entropy upcasts it
to fp32 for the backward, OOM-ing even a 192 GB GPU. Gaon bounds this with
**chunked, gradient-checkpointed cross-entropy** (`Gaon.loss_from_hidden`): the
flattened sequence is split into chunks, each chunk's head+CE is gradient-
checkpointed, so peak logit memory is `(chunk_size × vocab)` and is recomputed in
the backward. Combined with optional full activation checkpointing and 8-bit AdamW,
this lets the full 0.6B model train inside 16 GB — enabling the same code to run on
a free Colab T4 or scale up on a datacenter GPU.

## 3. Pretraining

- **Data:** FineWeb-Edu (English) + FineWeb-2 (Korean), tokenized and packed into
  flat `uint32` shards (`src/data/`). Batched tokenization runs at ~3.2M tok/s.
- **Budget:** ~12B tokens (≈ Chinchilla-optimal for 0.6B), 30,000 steps.
- **Compute:** a single NVIDIA B200, bf16, ~45k tokens/s.
- **Optimizer/schedule:** AdamW, cosine LR 3e-4 → 3e-5, 200-step warmup, grad clip 1.0.

**Loss curve (12.15 → 2.48):**

| step | 0 | 990 | 4,990 | 9,990 | 19,990 | 29,990 |
|---|---|---|---|---|---|---|
| loss | 12.15 | 3.51 | 2.89 | 2.70 | 2.69 | **2.48** |

The base model produces fluent, grammatical text in both languages (e.g. *"The
history of artificial intelligence is rich with examples of creative, insightful…"*
/ *"인공지능의 역사는 2000년대 초반…"*).

## 4. Post-training: distillation SFT

Instruction-following was added by **sequence-level knowledge distillation** from
an open teacher (Qwen2.5-7B-Instruct, Apache-2.0 — open licenses permit
distillation; proprietary-API outputs do not). The teacher generated responses to
~9,000 real instructions (Korean from kullm-v2, English from dolly-15k), saved as
chat `messages` (`src/posttrain/distill.py`). Gaon's base was then SFT'd on this
distilled set for 3 epochs (`src/posttrain/sft.py`, via TRL), train loss 2.12.

## 5. Results & honest limitations

Gaon-0.6B-Instruct, qualitatively:

| Prompt | Response | Verdict |
|---|---|---|
| 한국의 수도는 어디야? | "한국의 수도는 서울입니다." | ✅ correct |
| 인공지능을 초등학생도 이해하게 설명해줘 | structured Korean explanation w/ headings | ✅ fluent, on-task |
| 파이썬으로 피보나치 함수 짜줘 | attempts code, structure right, code broken | ⚠️ weak |

It is a real bilingual instruction-following chatbot. Factual recall and
explanation are solid; **coding and multi-step reasoning are weak — an inherent
limit of the 0.6B scale**, not a training defect (same-size open models behave
similarly). Benchmarks (MMLU/KMMLU) are not competitive with models trained on
1000× more tokens, by design.

## 6. Reproducibility

The entire pipeline is one repository and a handful of commands:

```bash
python scripts/sanity_check.py                       # architecture correctness
python -m src.data.prepare ...                       # download + pack data
python -m src.train.train --config configs/...       # pretrain
python -m src.posttrain.distill --teacher ...        # distill data
python -m src.posttrain.sft --data-jsonl ...         # instruction tune
```

## 7. Engineering lessons

- **Chunked CE** is the enabling trick for large-vocab models on small memory.
- **Match the teacher architecture's exact conventions** (RoPE rotate-half, QK-Norm)
  for HF bit-compatibility — verify with a round-trip logit test, not just "it loads."
- On a **shared GPU**, prefer single-GPU (no NCCL collective timeouts) and cap the
  process memory fraction to coexist with others' resident jobs.

## 8. Next steps

1. **Scale-up:** the same code trains 1.7B (config provided) for a clear capability jump.
2. **Vertical specialization:** the durable value of small-model skill is fine-tuning
   a strong open base on proprietary in-domain data — where a focused model can beat
   general giants on a narrow task.
