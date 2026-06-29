# Gaon (가온)

A real LLM, built from scratch. *Gaon* — pure Korean for "center/core."
Rung 1 of the ladder to a frontier lab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/k08200/gaon/blob/main/notebooks/gaon_colab.ipynb)
— train on a free GPU.

**Now:** train a Qwen3-0.6B-architecture model from scratch (full pipeline you own).
**Next:** scale the *same code* to 1.7B → 7B → larger as compute/capital grow.
**Endgame:** frontier. This repo is where you earn the credibility + skill to get there.

Why 0.6B first: the 235B-class "Qwen full level" needs thousands of GPUs and tens
of trillions of tokens — that's the capital game. The *small* Qwen models you can
match on your own GPUs **now**, while learning every stage end to end. Same
architecture, same pipeline; only scale (and money) differ.

## Tech report
Full write-up: [docs/TECH_REPORT.md](docs/TECH_REPORT.md) — architecture, training recipe, distillation, results, and honest limitations.

## Architecture (faithful to Qwen3)
Decoder-only · RMSNorm (pre-norm) · RoPE · Grouped-Query Attention · **QK-Norm**
(Qwen3-specific per-head q/k RMSNorm) · SwiGLU · tied embeddings. See
[`src/model/gaon.py`](src/model/gaon.py). Defaults replicate Qwen3-0.6B exactly
(verified 596M params), so we reuse its tokenizer and benchmark against the
official checkpoint.

## Layout
```
configs/            training configs (start: gaon_0.6b.yaml)
src/model/          Gaon model — Qwen3-compatible architecture (config.py, gaon.py)
src/data/           prepare.py (download+tokenize+pack), loader.py (mmap batches)
src/train/          train.py (single-GPU + FSDP via torchrun)
src/posttrain/      sft.py (TRL SFT + our->HF weight map), dpo.py (TRL DPO)
src/eval/           generate.py (sampling), benchmark.py (lm-eval-harness)
scripts/            sanity_check.py, prepare_mixture.sh, run_nhn.sh (turnkey)
```

## One-command pipeline (NHN box)
```bash
pip install -r requirements.txt
bash scripts/run_nhn.sh sanity          # CPU correctness check (seconds)
bash scripts/run_nhn.sh data validate   # ~12B-token mixture (EN+KO+code)
bash scripts/run_nhn.sh pretrain        # FSDP on CUDA_VISIBLE_DEVICES
bash scripts/run_nhn.sh generate        # sample from latest checkpoint
bash scripts/run_nhn.sh sft             # instruction tuning (TRL)
bash scripts/run_nhn.sh dpo             # preference alignment (TRL)
bash scripts/run_nhn.sh eval            # benchmark vs Qwen3-0.6B
```

## Roadmap
- [x] Architecture + correctness tests
- [x] Data packing + mmap loader
- [x] Pretraining loop (FSDP)
- [x] Korean + code + multilingual data mixture (`scripts/prepare_mixture.sh`)
- [x] Post-training: SFT → DPO (`src/posttrain/`, via TRL)
- [x] Eval harness: MMLU/HellaSwag + KMMLU/HAERAE vs Qwen3-0.6B (`src/eval/benchmark.py`)
- [x] Turnkey runner (`scripts/run_nhn.sh`)
- [ ] **Run it**: validation run → quality run → tech report (credibility artifact)
- [ ] Scale same code to 1.7B / 7B

## Compute reality (so the plan is honest)
~0.6B params × 100B tokens ≈ 3.6e20 FLOPs.
A100 @ ~40% MFU ≈ 1.25e14 FLOP/s → ~33 GPU-days; **~7 days on 5 GPUs.**
The Chinchilla-optimal ~12B-token run is ~1 day on 5 GPUs — use it to validate
first, then commit to the long over-trained run for quality.
