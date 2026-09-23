# ReCal

ReCal is a recoverability-aware calibration method for structured FFN pruning
of language models. It can be applied to Minitron, Wanda, FLAP, and
LLM-Pruner without changing the target architecture or the downstream recovery
recipe.

This repository contains the complete runnable implementation:

- ReCal and all four pruning baselines;
- SFT and OPD training;
- IFBench and LiveCodeBench evaluation;
- data preparation and contamination filtering;
- pruning, SFT, OPD, checkpoint export, and evaluation;
- Qwen3-4B-Instruct-2507 and Qwen3-8B experiment configurations;
- 15%, 25%, and 35% pruning-ratio reproduction scripts.

## Method

Given a dense teacher \(p_T\) and a provisional model \(p_P\) pruned by the
selected baseline criterion, ReCal measures pruning damage along dense-teacher
trajectories:

\[
d_t =
D_{\mathrm{KL}}\left(
p_T(\cdot\mid x_{\leq t})
\parallel
p_P(\cdot\mid x_{\leq t})
\right).
\]

The implementation computes the divergence on the teacher's top-\(k\)
probabilities plus an `OTHER` bucket. The token signal is shifted to the
activation that produced the affected next-token prediction and normalized
inside each trajectory:

\[
w_t = \frac{d_t}{\sum_{j\in\mathcal R}d_j},
\]

where \(\mathcal R\) denotes assistant-response positions. ReCal then replaces
uniform token aggregation in the original pruning criterion:

\[
I_j^{\mathrm{ReCal}}
=
\operatorname{Aggregate}_t
\left(w_t\,s_j(x_{\leq t})\right).
\]

The underlying statistic \(s_j\) remains Minitron activation importance,
Wanda-SP, FLAP-WIFN, or LLM-Pruner Taylor importance.

## Source layout

```text
src/
├── recal/                  # ReCal, pruning, data, pipeline, and evaluation
├── verl/                   # SFT and OPD training
└── lcb_runner/             # code execution evaluation

src/recal/evaluation/ifbench/
                            # instruction-following scorer

configs/
├── qwen3_4b/
└── qwen3_8b/

scripts/
├── install.sh
├── download_models.py
├── run.sh
├── evaluate.sh
├── reproduce_main.sh
├── reproduce_ratio_sweep.sh
├── prepare_recovery_data.py
├── prepare_calibration_data.py
├── materialize_ratio_config.py
├── compare_masks.py
└── summarize_results.py

results/
├── main_results.json
├── ratio_sweep.json
├── mask_summary.csv
└── mask_layers.csv
```

## Installation

The reference environment uses Python 3.12, PyTorch 2.7.1,
Transformers 4.54.1, vLLM 0.10.0, TensorDict 0.10.0, and Ray 2.57.0.

Create an isolated environment with a CUDA-compatible PyTorch installation,
then run:

```bash
bash scripts/install.sh
source scripts/activate_recal.sh
```

The install command installs the complete repository in editable mode.

The full 8B configuration was run on eight high-memory GPUs. To use a smaller
setup, adjust GPU count, tensor parallelism, sequence lengths, and batch sizes in
the YAML configuration.

## Models

Download the reference model revisions:

```bash
python scripts/download_models.py
```

This creates:

```text
models/Qwen3-8B
models/Qwen3-4B-Instruct-2507
```

Custom local model paths can be set through:

```yaml
model:
  path: /path/to/model
  teacher_path: /path/to/dense/teacher
  student_path: /path/to/model
```

## Supported configurations

Each model directory contains:

```text
minitron.yaml
minitron_recal.yaml
wanda.yaml
wanda_recal.yaml
flap.yaml
flap_recal.yaml
llm_pruner.yaml
llm_pruner_recal.yaml
```

All methods use homogeneous per-layer FFN width reduction so that model shapes
and parameter counts are directly comparable.

## Data preparation

The default setup builds:

- 12,000 SFT prompts;
- 24,000 disjoint OPD prompts;
- balanced mathematics, code, science, and instruction-following domains;
- dense-teacher SFT trajectories;
- contamination checks against all evaluation benchmarks.

Data preparation is automatically invoked by `scripts/run.sh`. It can also be
run explicitly:

```bash
python scripts/prepare_recovery_data.py \
  --config configs/qwen3_8b/wanda.yaml \
  --output-dir data/recovery_pools

python -m recal.data.teacher_targets \
  --config configs/qwen3_8b/wanda.yaml

python scripts/prepare_calibration_data.py \
  --config configs/qwen3_8b/wanda.yaml
```

## Run one experiment

Run the baseline first:

```bash
bash scripts/run.sh configs/qwen3_8b/wanda.yaml
```

Then run ReCal:

```bash
bash scripts/run.sh configs/qwen3_8b/wanda_recal.yaml
```

The baseline must run first because its `pruned_initial` checkpoint is used as
the criterion-specific probe:

```yaml
pruning:
  method: wanda_recal
  recal:
    probe_path: outputs/qwen3_8b/wanda_r25/stages/stage_01_r25/pruned_initial
    weighting: forward_kl
```

The full workflow is:

```text
data preparation
→ dense-teacher trajectories
→ pruning statistics
→ structured mask
→ pruning
→ evaluation
→ SFT
→ evaluation
→ OPD
→ evaluation
```

Completed artifacts are reused when the same command is restarted.

## Reproduce the main grid

Run all 25% configurations for one model:

```bash
bash scripts/reproduce_main.sh qwen3_8b
```

or:

```bash
bash scripts/reproduce_main.sh qwen3_4b
```

Without arguments, the script runs both models.

## SFT and OPD

SFT uses filtered dense-teacher trajectories for two epochs. OPD uses:

```yaml
opd:
  topk: 16
  top_k_strategy: only_stu
  reward_weight_mode: student_p
  rollout_n: 4
  learning_rate: 1.0e-6
  max_training_steps: 200
```

The relevant entry points are:

```text
src/verl/trainer/sft_trainer.py
src/verl/trainer/main_ppo.py
src/verl/model_merger/
```

`scripts/activate_recal.sh` places `src/` on `PYTHONPATH`.

## Evaluation

Evaluate any Hugging Face checkpoint with the reference 20,480-token setup:

```bash
bash scripts/evaluate.sh \
  /path/to/checkpoint \
  outputs/my_evaluation \
  --thinking-enabled
```

For Qwen3-4B-Instruct-2507:

```bash
bash scripts/evaluate.sh \
  /path/to/checkpoint \
  outputs/my_evaluation \
  --no-thinking-enabled
```

The evaluator includes:

- AIME 2024, 2025, and 2026, with eight samples per problem;
- GPQA-Diamond;
- IFBench strict and loose metrics;
- LiveCodeBench-v5 pass@1;
- truncation, repeated-ngram, and unclosed-reasoning diagnostics.

Generation shards and task results are cached for resumption.

## Compression-ratio sweep

Run the 15%, 25%, and 35% SFT-stage sweep:

```bash
bash scripts/reproduce_ratio_sweep.sh qwen3_8b
```

Create a single ratio-specific configuration:

```bash
python scripts/materialize_ratio_config.py \
  --config configs/qwen3_8b/wanda_recal.yaml \
  --ratio 0.35 \
  --output configs/generated/qwen3_8b/wanda_recal_r35.yaml

bash scripts/run.sh configs/generated/qwen3_8b/wanda_recal_r35.yaml
```

Use `--full-recovery` when materializing the configuration to include OPD.

## Mask comparison

```bash
python scripts/compare_masks.py \
  --base-mask artifacts/qwen3_8b/wanda_masks/width_prune_25.json \
  --recal-mask artifacts/qwen3_8b/wanda_recal_masks/width_prune_25.json \
  --output results/wanda_mask_layers.csv
```

## Reference results

The complete machine-readable results are in:

```text
results/main_results.json
results/ratio_sweep.json
results/mask_summary.csv
results/mask_layers.csv
```

The main result JSON includes AIME 2024, AIME 2025, AIME 2026, AIME average,
GPQA-Diamond, IFBench, LiveCodeBench, truncation, and repetition metrics.

### Qwen3-8B, 25% pruning, final recovery

| Method | AIME24 | AIME25 | AIME26 | AIME avg | GPQA-D | IFBench-S | LCB-v5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Minitron | 0.167 | 0.100 | 0.154 | 0.140 | 0.359 | 0.212 | 0.199 |
| Minitron + ReCal | 0.354 | 0.283 | 0.229 | 0.289 | 0.364 | 0.195 | 0.229 |
| Wanda | 0.467 | 0.367 | 0.417 | 0.417 | 0.424 | 0.195 | 0.259 |
| Wanda + ReCal | 0.529 | 0.429 | 0.496 | 0.485 | 0.404 | 0.183 | 0.283 |
| FLAP | 0.317 | 0.254 | 0.300 | 0.290 | 0.434 | 0.201 | 0.247 |
| FLAP + ReCal | 0.450 | 0.300 | 0.371 | 0.374 | 0.374 | 0.198 | 0.241 |
| LLM-Pruner | 0.188 | 0.079 | 0.104 | 0.124 | 0.348 | 0.224 | 0.175 |
| LLM-Pruner + ReCal | 0.138 | 0.050 | 0.087 | 0.092 | 0.449 | 0.195 | 0.283 |

### Qwen3-4B-Instruct-2507, 25% pruning, final recovery

| Method | AIME24 | AIME25 | AIME26 | AIME avg | GPQA-D | IFBench-S | LCB-v5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.633 | 0.458 | 0.542 | 0.544 | 0.606 | 0.343 | 0.319 |
| Minitron | 0.062 | 0.021 | 0.062 | 0.049 | 0.354 | 0.302 | 0.217 |
| Minitron + ReCal | 0.275 | 0.175 | 0.196 | 0.215 | 0.364 | 0.270 | 0.259 |
| Wanda | 0.358 | 0.325 | 0.287 | 0.324 | 0.444 | 0.297 | 0.247 |
| Wanda + ReCal | 0.425 | 0.342 | 0.350 | 0.372 | 0.470 | 0.288 | 0.253 |
| FLAP | 0.279 | 0.175 | 0.192 | 0.215 | 0.389 | 0.308 | 0.223 |
| FLAP + ReCal | 0.292 | 0.304 | 0.242 | 0.279 | 0.354 | 0.270 | 0.241 |
| LLM-Pruner | 0.087 | 0.008 | 0.087 | 0.061 | 0.359 | 0.279 | 0.151 |
| LLM-Pruner + ReCal | 0.171 | 0.050 | 0.163 | 0.128 | 0.434 | 0.276 | 0.217 |

## Acknowledgements

This implementation incorporates and adapts components from OPD and verl for
SFT and on-policy distillation, IFBench for instruction-following evaluation,
LiveCodeBench for code evaluation, and NVIDIA Model Optimizer for the Minitron
reference formulation.
