#!/usr/bin/env bash
# Reproduce every 25% Base/ReCal pair. Base must precede ReCal because its
# pruned checkpoint is the criterion-specific ReCal probe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -gt 0 ]]; then
  models=("$@")
else
  models=(qwen3_4b qwen3_8b)
fi
criteria=(minitron wanda flap llm_pruner)

for model in "${models[@]}"; do
  for criterion in "${criteria[@]}"; do
    bash scripts/run.sh "configs/${model}/${criterion}.yaml"
    bash scripts/run.sh "configs/${model}/${criterion}_recal.yaml"
  done
done
