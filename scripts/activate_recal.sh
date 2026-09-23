#!/usr/bin/env bash

RECAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RECAL_ROOT

if [[ -n "${RECAL_ENV:-}" && -x "${RECAL_ENV}/bin/python" ]]; then
  export PATH="${RECAL_ENV}/bin:${PATH}"
fi

export PYTHONPATH="${RECAL_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${RECAL_ROOT}/.cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-${RECAL_ROOT}/.cache/nltk_data}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
