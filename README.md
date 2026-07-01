# Gaon (가온)

Real LLMs, built from scratch. *Gaon* — pure Korean for "center/core."
Rung 1 of the ladder to a frontier lab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/k08200/gaon/blob/main/notebooks/gaon_colab.ipynb)
— train on a free GPU.

**Done:** Qwen3-architecture models trained from scratch — **Gaon-0.6B** and
**Gaon-1.7B**, base + instruction-tuned, Korean + English. Same codebase, single-GPU
to 4-GPU FSDP.
**Now:** a Chinchilla-scale 1.7B quality run (+ code data).
**Endgame:** frontier. This repo is where the credibility + skill to get there is earned.

## Results

| Model | Params | Tokens | Compute | Final loss | Instruct |
|---|---|---|---|---|---|
| Gaon-0.6B | 596M | 12B | 1× B200 | 2.48 | ✅ chats KO/EN |
| Gaon-1.7B | 1.72B | 12B | 4× B200 (FSDP) | **2.37** | ✅ chats KO/EN |

**Same data + same code, 3× params → lower loss (2.48 → 2.37):** the from-scratch
pipeline scales. Full write-up: **[docs/TECH_REPORT.md](docs/TECH_REPORT.md)** —
architecture, training recipe, multi-GPU FSDP engineering, distillation, honest limits.

Why not bigger: the 235B-class "full Qwen level" needs thousands of GPUs and tens of
trillions of tokens — the capital game. The *small* models you can match on your own
GPUs **now**, while learning every stage end to end. Same architecture, same pipeline;
only scale (and money) differ.

## Architecture (faithful to Qwen3)
Decoder-only · RMSNorm (pre-norm) · RoPE · Grouped-Query Attention · **QK-Norm**
(Qwen3-specific per-head q/k RMSNorm) · SwiGLU · tied embeddings. See
[`src/model/gaon.py`](src/model/gaon.py). **Bit-compatible with HuggingFace Qwen3**
(round-trip logit diff `0.0`), so we reuse its tokenizer and the HF post-training stack.

## Layout
```
configs/            training configs (0.6b single-GPU, 1.7b 4-GPU FSDP, 1.7b v2)
src/model/          Gaon model — Qwen3-compatible architecture (config.py, gaon.py)
src/data/           prepare.py (download+tokenize+pack), loader.py (mmap batches)
src/train/          train.py (single-GPU + FSDP via torchrun, resume, disk-safe ckpts)
src/posttrain/      sft.py (TRL SFT + our->HF weight map), distill.py, dpo.py
src/eval/           generate.py (sampling), chat.py (REPL), benchmark.py
scripts/            sanity_check.py, verify_ckpt.py, test_chat.py, prepare_mixture.sh
```

## Reproduce
```bash
pip install -r requirements.txt
python scripts/sanity_check.py                                        # correctness (seconds)
python -m src.data.prepare --dataset fineweb-edu --out data/fineweb_edu
python -m src.data.prepare --dataset codeparrot  --out data/codeparrot
torchrun --standalone --nproc_per_node=4 -m src.train.train \
    --config configs/gaon_1.7b_v2.yaml                                # pretrain (FSDP)
python -m src.posttrain.sft --ckpt checkpoints/.../ckpt_30000.pt \
    --data-jsonl data/distill.jsonl --out checkpoints/sft             # instruction tune
python -m src.eval.chat --model checkpoints/sft                       # chat with it
```

## Roadmap
- [x] Architecture + HF bit-compat correctness tests
- [x] Data packing + mmap loader (EN + Korean + code)
- [x] Pretraining loop (single-GPU **and** multi-GPU FSDP, resume, disk-safe)
- [x] Post-training: distillation SFT → instruct (KO/EN chat)
- [x] **Gaon-0.6B** base + instruct (loss 2.48)
- [x] **Gaon-1.7B** base + instruct (loss 2.37) — scaling demonstrated
- [ ] **1.7B v2**: Chinchilla-scale (~34B tokens) + code mixture
- [ ] Eval harness vs Qwen3 (MMLU/KMMLU/HAERAE)
- [ ] Scale same code to 7B (when compute/capital allow)

## Compute reality (so the plan is honest)
Training FLOPs ≈ `6 × params × tokens`. Gaon-1.7B (12B tokens) ≈ 1.2e20 FLOPs, done
in ~26h on 4× B200. A Chinchilla-optimal 7B (~140B tokens) is ~5.9e21 FLOPs — roughly
6,500 GPU-hours (~$13k rented). Frontier is orders of magnitude beyond that; this repo
is the on-ramp, not the destination.
