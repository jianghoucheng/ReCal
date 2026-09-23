#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install -e .
export NLTK_DATA="${NLTK_DATA:-${ROOT}/.cache/nltk_data}"
NLTK_ALLOW_PROXIED_URLOPEN="${NLTK_ALLOW_PROXIED_URLOPEN:-1}" \
python -m nltk.downloader -d "$NLTK_DATA" \
  punkt punkt_tab stopwords averaged_perceptron_tagger_eng

python - <<'PY'
import torch
import transformers
import vllm
import verl
from recal.evaluation.ifbench import run_eval
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from importlib.util import find_spec
from pathlib import Path

expected = (Path("src") / "verl").resolve()
loaded = Path(verl.__file__).resolve()
assert loaded.parent == expected, f"Expected integrated verl at {expected}, got {loaded}"
assert find_spec("recal.evaluation.ifbench.run_eval") is not None
assert find_spec("lcb_runner.evaluation.compute_code_generation_metrics") is not None
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("integrated verl", loaded)
print("integrated IFBench", run_eval.__file__)
print("integrated LiveCodeBench", codegen_metrics.__module__)
print("cuda available", torch.cuda.is_available())
PY

echo
echo "Installation complete."
echo "Run 'source scripts/activate_recal.sh' before launching experiments."
