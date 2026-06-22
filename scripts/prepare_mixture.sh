#!/usr/bin/env bash
# Build the full pretraining data mixture: English (FineWeb-Edu) + Korean + code.
# Tune the per-source token targets to set the mixture ratio.
# Usage:  bash scripts/prepare_mixture.sh [SCALE]
#   SCALE=validate -> ~12B tokens total (1-day run, to validate the pipeline)
#   SCALE=full     -> ~100B tokens total (quality run)   [default]
set -euo pipefail
cd "$(dirname "$0")/.."

SCALE="${1:-full}"
if [ "$SCALE" = "validate" ]; then
  EN=8_000_000_000;  KO=2_500_000_000;  CODE=1_500_000_000
else
  EN=65_000_000_000; KO=20_000_000_000; CODE=15_000_000_000
fi

echo ">> mixture scale=$SCALE  EN=$EN KO=$KO CODE=$CODE"
python -m src.data.prepare --dataset fineweb-edu   --target-tokens "$EN"   --out data/fineweb_edu
python -m src.data.prepare --dataset fineweb-2-kor --target-tokens "$KO"   --out data/fineweb_kor
python -m src.data.prepare --dataset the-stack     --target-tokens "$CODE" --out data/the_stack
echo ">> done. set data_dirs in configs/qwen3_0.6b.yaml to: data/fineweb_edu, data/fineweb_kor, data/the_stack"
