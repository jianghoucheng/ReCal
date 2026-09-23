#!/usr/bin/env bash
# Run one Base or Base+ReCal configuration through:
# data preparation -> pruning statistics -> masks -> prune -> SFT -> OPD -> evaluation.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG.yaml [recal-pipeline options]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/activate_recal.sh

config="$1"
shift

readarray -t values < <(python - "$config" <<'PY'
import sys
from recal.common import load_yaml

c = load_yaml(sys.argv[1])
p = c["pruning"]
r = p.get("recal", {})
print(p["method"])
print(p["importance_dir"])
print(p["mask_dir"])
print(" ".join(map(str, p["ratios"])))
print(p["hardware_multiple"])
print(c["model"].get("student_path", c["model"]["path"]))
print(r.get("statistics_dir", p["importance_dir"]))
print(c["data"]["sft_teacher_path"])
print(p.get("calibration_path", ""))
print(r.get("probe_path", ""))
print(max(int(c["sft"].get("n_gpus_per_node", 8)), int(c["opd"].get("n_gpus_per_node", 8))))
PY
)

method="${values[0]}"
importance_dir="${values[1]}"
mask_dir="${values[2]}"
ratios="${values[3]}"
hardware_multiple="${values[4]}"
model="${values[5]}"
statistics_dir="${values[6]}"
teacher_targets="${values[7]}"
calibration_path="${values[8]}"
probe_path="${values[9]}"
nproc="${RECAL_NPROC:-${values[10]}}"

case "$method" in
  minitron|wanda|flap|llm_pruner)
    ;;
  minitron_recal|wanda_recal|flap_recal|llm_pruner_recal)
    if [[ ! -s "$probe_path/config.json" ]]; then
      echo "Missing ReCal probe: $probe_path" >&2
      echo "Run the matching Base configuration first." >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown pruning.method: $method" >&2
    exit 2
    ;;
esac

if [[ "${RECAL_SKIP_DATA:-0}" != "1" ]]; then
  echo "==> Preparing disjoint SFT and OPD pools"
  python scripts/prepare_recovery_data.py --config "$config" --output-dir data/recovery_pools

  if [[ ! -s "$teacher_targets" ]]; then
    echo "==> Generating and filtering dense-teacher SFT trajectories"
    python -m recal.data.teacher_targets --config "$config"
  fi
fi

case "$method" in
  minitron|wanda|flap|llm_pruner)
    if [[ ! -s "$calibration_path" ]]; then
      python scripts/prepare_calibration_data.py --config "$config"
    fi
    ;;
esac

echo "==> Collecting pruning statistics: $method"
if [[ ! -s "$statistics_dir/channel_order.json" ]]; then
  case "$method" in
    minitron)
      torchrun --standalone --nnodes=1 --nproc-per-node="$nproc" \
        -m recal.pruning.minitron_collector --config "$config"
      ;;
    wanda|flap)
      torchrun --standalone --nnodes=1 --nproc-per-node="$nproc" \
        -m recal.pruning.baseline_collector --config "$config"
      ;;
    llm_pruner)
      torchrun --standalone --nnodes=1 --nproc-per-node="$nproc" \
        -m recal.pruning.llm_pruner_collector --config "$config"
      ;;
    minitron_recal|wanda_recal|flap_recal)
      torchrun --standalone --nnodes=1 --nproc-per-node="$nproc" \
        -m recal.pruning.recal_collector --config "$config"
      ;;
    llm_pruner_recal)
      torchrun --standalone --nnodes=1 --nproc-per-node="$nproc" \
        -m recal.pruning.llm_pruner_recal_collector --config "$config"
      ;;
  esac
fi

echo "==> Building structured FFN masks"
python -m recal.pruning.nested_masks \
  --importance-dir "$statistics_dir" \
  --ratios $ratios \
  --output-dir "$mask_dir" \
  --hardware-multiple "$hardware_multiple" \
  --ratio-basis total_parameters \
  --model "$model"

echo "==> Running preflight"
output_dir="$(python - "$config" <<'PY'
import sys
from recal.common import load_yaml
print(load_yaml(sys.argv[1])["experiment"]["output_dir"])
PY
)"
mkdir -p "$output_dir"
python -m recal.orchestration.preflight \
  --config "$config" --output "$output_dir/preflight.json"

echo "==> Running pruning -> SFT -> OPD -> evaluation"
python -m recal.orchestration.pipeline --config "$config" "$@"
