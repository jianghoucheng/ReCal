#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 MODEL_PATH OUTPUT_DIR [evaluation options]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/activate_recal.sh

model="$1"
output="$2"
shift 2

python -m recal.evaluation.comprehensive_suite \
  --model "$model" \
  --output-dir "$output" \
  --profile full \
  --tasks aime gpqa_diamond ifbench livecodebench \
  --tensor-parallel-size "${RECAL_EVAL_TP:-8}" \
  --aime-n 8 \
  --aime-temperature 0.6 \
  --aime-top-p 0.95 \
  --max-tokens 20480 \
  --max-model-len 21504 \
  --gpu-memory-utilization 0.84 \
  "$@"
