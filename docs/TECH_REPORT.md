# Gaon (가온): Bilingual Language Models Trained From Scratch (0.6B → 1.7B)

*A reproducible, end-to-end build of small Korean + English LLMs — architecture,
data, pretraining, and instruction tuning — that scales with the same code from a
single GPU to multi-GPU FSDP.*

**Repository:** https://github.com/k08200/gaon · **License:** open · **Author:** independent research

---

## Abstract

Gaon is a family of decoder-only language models built entirely from scratch:
architecture, tokenizing/data pipeline, pretraining loop, and post-training — no
external model libraries. Two base models share one codebase: **Gaon-0.6B** (596M)
and **Gaon-1.7B** (1.72B). Both are bilingual (Korean + English) by design, since
most small open models are weak in Korean. Trained on the same ~12B-token mixture,
the base models reach cross-entropy loss **2.48** (0.6B) and **2.37** (1.7B) — a
clean demonstration that the from-scratch code scales: *same data, 3× parameters,
lower loss*. Instruction-tuned variants are produced by **knowledge distillation**
from an open Qwen teacher and follow instructions in both languages. The 0.6B trains
on a single GPU; the 1.7B trains on 4× B200 with FSDP. Everything is reproducible
from the public repository. The goal is not to beat frontier models — a 1.7B model
cannot — but to own every stage of the pipeline and prove it scales.

## 1. Architecture

Gaon is a Qwen3-compatible decoder-only transformer in a single PyTorch file
(`src/model/gaon.py`), so every operation is visible and hackable. The two sizes
differ only in width/depth config — not code.

| Component | Gaon-0.6B | Gaon-1.7B |
|---|---|---|
| Layers | 28 | 28 |
| Hidden | 1024 | 2048 |
| MLP intermediate (SwiGLU) | 3072 | 6144 |
| Attn heads (GQA) | 16 Q / 8 KV | 16 Q / 8 KV |
| Head dim | 128 | 128 |
| **Parameters** | **596M** | **1.72B** |

Shared across both: RoPE (θ = 1e6, **HF rotate-half convention**), **QK-Norm**
(per-head RMSNorm on q, k — Qwen3-specific), RMSNorm pre-norm, tied embeddings,
vocab 151,936 (Qwen3 tokenizer, multilingual).

The model is **bit-compatible with HuggingFace Qwen3**: a round-trip test
(`tests/test_hf_roundtrip.py`) confirms our weights map onto `Qwen3ForCausalLM`
with a max logit difference of `0.0`. This lets us reuse the HF ecosystem for
post-training/eval, and required matching the exact RoPE convention (rotate-half,
not interleaved-complex — the first implementation was off by 0.31 logits until
corrected).

## 2. Memory-efficient training (key technique)

The large vocabulary (151,936) makes the LM-head logits the memory bottleneck: at
batch 16 / seq 4096 the logit tensor alone is ~20 GB, and cross-entropy upcasts it
to fp32 for the backward. Gaon bounds this with **chunked, gradient-checkpointed
cross-entropy** (`Gaon.loss_from_hidden`): the flattened sequence is split into
chunks, each chunk's head+CE is gradient-checkpointed, so peak logit memory is
`(chunk_size × vocab)` and is recomputed in the backward. Combined with optional
full activation checkpointing and 8-bit AdamW, this lets the 0.6B model train inside
16 GB — the same code runs on a free Colab T4 or a datacenter GPU.

## 3. Pretraining

- **Data:** FineWeb-Edu (English) + FineWeb-2 (Korean), tokenized and packed into
  flat `uint32` shards (`src/data/`). Batched tokenization runs at ~3.2M tok/s.
- **Budget:** ~12B tokens (≈ Chinchilla-optimal for 0.6B), 30,000 steps.
- **Schedule:** AdamW, cosine LR 2.5e-4 → 2.5e-5, warmup, grad clip 1.0, bf16.

| Model | Compute | Throughput | Wall-clock | Final loss |
|---|---|---|---|---|
| Gaon-0.6B | 1× B200 | ~45k tok/s | ~3 days | **2.48** |
| Gaon-1.7B | 4× B200 (FSDP) | ~128k tok/s | ~26 h | **2.37** (val 2.32) |

**Scaling result:** with identical data (12B tokens) and code, tripling parameters
(0.6B → 1.7B) lowers loss 2.48 → 2.37. The from-scratch pipeline scales as expected.

Both base models produce fluent, grammatical text in Korean and English (e.g.
*"The history of artificial intelligence is rich with…"* / *"인공지능의 역사는
2000년대 초반…"*).

## 4. Multi-GPU training (FSDP) — what it took

Gaon-1.7B trains under **FSDP (FULL_SHARD)** across 4 GPUs via `torchrun`. Running
this reliably on a **shared** box (co-located with colleagues' resident inference
servers) surfaced three failure modes worth documenting:

1. **NCCL collective timeout.** A co-located vLLM server took real inference traffic
   and monopolized one of our GPUs for over an hour; the all-reduce on that rank
   stalled and the NCCL watchdog killed the job. Fix: a bounded `init_process_group`
   timeout (20 min) so a *sustained* grab crashes-and-resumes fast instead of
   stalling while holding memory, plus frequent checkpoints + resume.
2. **Corrupt FSDP checkpoints.** Under FULL_SHARD each rank holds only a 1/N shard
   of every parameter. A plain `model.state_dict()` saved from rank 0 stores just
   that shard — resume loaded near-random weights (loss jumped to **11.9**). Fix:
   gather the full unsharded weights with
   `FSDP.state_dict_type(FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu, rank0_only))`.
   The gather is a **collective** — every rank must call `save()`, only rank 0 writes
   the file. Verified by loading the checkpoint on CPU and confirming loss matches.
3. **Orphaned workers.** Killing the `torchrun` launcher leaves the worker processes
   alive (different process name); a relaunch then overlaps two runs and doubles GPU
   memory. Fix: kill both process patterns (`torch.distributed.run` **and**
   `src.train.train`), using a bracket-glob so the kill command doesn't match its own
   shell.

**Good-neighbor policy.** To coexist with others' resident jobs, a per-process memory
cap (`torch.cuda.set_per_process_memory_fraction(0.2)`) + gradient checkpointing keeps
our footprint ~22 GB/GPU, leaving a colleague's vLLM ≥64 GB free throughout.

**Disk safety.** Unbounded checkpoints (60 × 8 GB) once filled a 1.6 TB disk to 100%,
crashing a save mid-write. Fixed with `keep_last_checkpoints` (prune all but the most
recent N after each save).

## 5. Post-training: distillation SFT

Instruction-following was added by **sequence-level knowledge distillation** from an
open teacher (Qwen2.5-7B-Instruct, Apache-2.0 — open licenses permit distillation;
proprietary-API outputs do not). The teacher generated responses to ~9,000 real
instructions (Korean + English), saved as chat `messages` (`src/posttrain/distill.py`).
Each base was then SFT'd on this distilled set for 3 epochs (`src/posttrain/sft.py`,
via TRL). The mapping from our weights to `Qwen3ForCausalLM` (`sft.py:to_hf`) reads
the checkpoint's own config, so the *same* SFT code handles both sizes.

| Model | SFT train loss |
|---|---|
| Gaon-0.6B-Instruct | 2.12 |
| Gaon-1.7B-Instruct | 1.53 |

## 6. Results & honest limitations

Gaon-1.7B-Instruct, qualitatively:

| Prompt | Response | Verdict |
|---|---|---|
| 한국의 수도는 어디인가요? | "서울입니다." | ✅ correct |
| 인공지능을 초등학생도 이해하게 설명 | structured Korean explanation | ✅ fluent, on-task |
| 아침에 일어나기 힘든데? | practical, multi-point advice | ✅ coherent |
| Write a haiku about the ocean | prose, wrong form | ⚠️ weak on strict format |
| 파이썬으로 1~10 합계 코드 | hallucinated imports, broken | ⚠️ weak coding |

The 1.7B is clearly better than the 0.6B on general Q&A — longer, more coherent,
more fluent Korean. **Coding and strict-format tasks remain weak — an inherent limit
of the scale and the modest token budget**, not a training defect. Benchmarks are not
competitive with models trained on 100–1000× more tokens, by design.

## 7. Reproducibility

The entire pipeline is one repository and a handful of commands:

```bash
python scripts/sanity_check.py                                   # architecture correctness
python -m src.data.prepare --dataset codeparrot --out data/...   # download + pack data
torchrun --standalone --nproc_per_node=4 -m src.train.train \
    --config configs/gaon_1.7b_v2.yaml                           # pretrain (FSDP)
python -m src.posttrain.distill --teacher Qwen/...               # distill data
python -m src.posttrain.sft --ckpt ... --data-jsonl ...          # instruction tune
python -m scripts.verify_ckpt checkpoints/.../latest.pt          # checkpoint sanity
```

## 8. Engineering lessons

- **Chunked CE** is the enabling trick for large-vocab models on small memory.
- **Match the teacher architecture's exact conventions** (RoPE rotate-half, QK-Norm)
  for HF bit-compatibility — verify with a round-trip logit test, not "it loads."
- **FSDP checkpoints must gather** (FULL_STATE_DICT); a plain `state_dict()` saves a
  shard and resumes to garbage. Always verify a checkpoint by *reloading and scoring*.
- On a **shared GPU**, bound the NCCL timeout, cap process memory, and prune old
  checkpoints — the failure modes are operational, not modeling.

## 9. Next steps

1. **Quality run (in progress):** Gaon-1.7B v2 — ~34B tokens (Chinchilla-optimal)
   with a Python-code mixture added, to lift coding and general quality.
2. **Vertical specialization:** the durable value of small-model skill is fine-tuning
   a strong open base on proprietary in-domain data — where a focused model beats
   general giants on a narrow task. The from-scratch build is the credential; the
   moat is the data.
