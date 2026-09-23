#!/usr/bin/env bash
# Reproduce the 15/25/35% SFT-stage sweep for Minitron, Wanda, and FLAP.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/activate_recal.sh

if [[ $# -gt 0 ]]; then
  models=("$@")
else
  models=(qwen3_4b qwen3_8b)
fi
criteria=(minitron wanda flap)
ratios=(0.15 0.25 0.35)

for model in "${models[@]}"; do
  for ratio in "${ratios[@]}"; do
    tag="$(python -c "print(f'r{round(float(\"$ratio\") * 100):02d}')")"
    for criterion in "${criteria[@]}"; do
      for method in "$criterion" "${criterion}_recal"; do
        generated="configs/generated/${model}/${method}_${tag}.yaml"
        mkdir -p "$(dirname "$generated")"
        python scripts/materialize_ratio_config.py \
          --config "configs/${model}/${method}.yaml" \
          --ratio "$ratio" \
          --output "$generated"
        bash scripts/run.sh "$generated"
      done
    done
  done
done
