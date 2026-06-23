#!/usr/bin/env bash
# Hands-off orchestrator: ensure data is prepared, then launch multi-GPU training.
# Runs prep -> train SEQUENTIALLY (no pgrep wait-loop — that deadlocked because the
# parent bash -c cmdline contains "src.data.prepare" and pgrep matched itself).
#   GPUS=3,4,5,6,7 NGPU=5 CONFIG=configs/gaon_0.6b_val.yaml \
#     nohup bash scripts/auto_run.sh > logs/auto_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-3,4,5,6,7}"
NGPU="${NGPU:-5}"
CONFIG="${CONFIG:-configs/gaon_0.6b_val.yaml}"
EN_TOK="${EN_TOK:-8000000000}"
KO_TOK="${KO_TOK:-4000000000}"
PY=".venv/bin/python"

have() { [ "$(ls "$1"/*.bin 2>/dev/null | wc -l)" -gt 0 ]; }

if ! have data/fineweb_edu; then
  echo "[auto $(date +%H:%M:%S)] preparing English data..."
  "$PY" -m src.data.prepare --dataset fineweb-edu --target-tokens "$EN_TOK" --out data/fineweb_edu
fi
if ! have data/fineweb_kor; then
  echo "[auto $(date +%H:%M:%S)] preparing Korean data..."
  "$PY" -m src.data.prepare --dataset fineweb-2-kor --target-tokens "$KO_TOK" --out data/fineweb_kor
fi

echo "[auto $(date +%H:%M:%S)] data ready (en=$(ls data/fineweb_edu/*.bin|wc -l), ko=$(ls data/fineweb_kor/*.bin|wc -l)). launching training on GPU $GPUS"
# expandable_segments avoids fragmentation OOM with the large LM-head logits.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="$GPUS" \
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" \
  -m src.train.train --config "$CONFIG"
echo "[auto $(date +%H:%M:%S)] training process exited with code $?"
