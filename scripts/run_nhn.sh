#!/usr/bin/env bash
# Turnkey pipeline for the NHN GPU box. Run stages individually or all in order.
#   bash scripts/run_nhn.sh sanity      # CPU correctness check (seconds)
#   bash scripts/run_nhn.sh data        # build validation mixture (~12B tok)
#   bash scripts/run_nhn.sh pretrain    # FSDP pretraining on $NGPU GPUs
#   bash scripts/run_nhn.sh generate    # sample from latest checkpoint
#   bash scripts/run_nhn.sh sft         # instruction tuning
#   bash scripts/run_nhn.sh eval        # benchmark vs Qwen3-0.6B
set -euo pipefail
cd "$(dirname "$0")/.."

# --- box config (override via env) ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,6,7}"   # NHN available GPUs
NGPU="${NGPU:-5}"
CONFIG="${CONFIG:-configs/qwen3_0.6b.yaml}"
CKPT_DIR="${CKPT_DIR:-checkpoints/qwen3_0.6b}"

stage="${1:-help}"
case "$stage" in
  sanity)
    python scripts/sanity_check.py ;;
  data)
    bash scripts/prepare_mixture.sh "${2:-validate}" ;;
  pretrain)
    torchrun --standalone --nproc_per_node="$NGPU" -m src.train.train --config "$CONFIG" ;;
  generate)
    LATEST=$(ls -t "$CKPT_DIR"/ckpt_*.pt | head -1)
    python -m src.eval.generate --ckpt "$LATEST" --prompt "${2:-The history of}" --max-new 120 ;;
  sft)
    LATEST=$(ls -t "$CKPT_DIR"/ckpt_*.pt | head -1)
    python -m src.posttrain.sft --ckpt "$LATEST" --out checkpoints/sft ;;
  dpo)
    python -m src.posttrain.dpo --sft checkpoints/sft --out checkpoints/dpo ;;
  eval)
    python -m src.eval.benchmark --model "${2:-checkpoints/sft}" --tag run
    python -m src.eval.benchmark --model Qwen/Qwen3-0.6B --tag baseline ;;
  *)
    grep '^#' "$0" | sed 's/^# \{0,1\}//' ;;
esac
