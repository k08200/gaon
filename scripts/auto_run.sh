#!/usr/bin/env bash
# Hands-off orchestrator: wait for data prep to finish, verify shards, then
# launch the 5-GPU validation training. Designed to run under nohup so the whole
# data -> train pipeline completes unattended.
#   GPUS=3,4,5,6,7 NGPU=5 CONFIG=configs/gaon_0.6b_val.yaml \
#     nohup bash scripts/auto_run.sh > logs/auto_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-3,4,5,6,7}"
NGPU="${NGPU:-5}"
CONFIG="${CONFIG:-configs/gaon_0.6b_val.yaml}"
PY=".venv/bin/python"

echo "[auto $(date +%H:%M:%S)] waiting for data prep to finish..."
while pgrep -f src.data.prepare >/dev/null; do sleep 30; done

EN=$(ls data/fineweb_edu/*.bin 2>/dev/null | wc -l)
KO=$(ls data/fineweb_kor/*.bin 2>/dev/null | wc -l)
echo "[auto $(date +%H:%M:%S)] data prep ended. fineweb_edu=$EN shards, fineweb_kor=$KO shards"
if [ "$EN" -lt 1 ] || [ "$KO" -lt 1 ]; then
  echo "[auto] ERROR: data incomplete (en=$EN ko=$KO). Aborting before training."
  exit 1
fi

echo "[auto $(date +%H:%M:%S)] launching training on GPU $GPUS with $CONFIG"
CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -m torch.distributed.run \
  --standalone --nproc_per_node="$NGPU" \
  -m src.train.train --config "$CONFIG"
echo "[auto $(date +%H:%M:%S)] training process exited with code $?"
