#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark residual add + RMSNorm + FP8 quant fusion (PR reproduction).
# Usage:
#   export MODEL_PATH=/path/to/Qwen3-4B-FP8
#   export LLM_MODELS_ROOT=/path/to/models
#   ./scripts/benchmark_add_norm_quant_fusion.sh before|after
set -euo pipefail

PHASE="${1:-after}"
RESULT_ROOT="${RESULT_ROOT:-/tmp/add_norm_quant_fusion_bench}"
# Default HF id (download/cache handled by serve/bench); override with a local dir if needed.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-FP8}"
CONFIG="${CONFIG:-$(dirname "$0")/bench_config_add_norm_quant_fusion.yaml}"
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"

mkdir -p "${RESULT_ROOT}/${PHASE}"

if [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* ]]; then
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: Local MODEL_PATH is not a directory: ${MODEL_PATH}" >&2
    exit 1
  fi
fi

bench_serving() {
  local tag="$1"
  local concurrency="$2"
  local num_prompts="$3"
  local out="${RESULT_ROOT}/${PHASE}/${tag}.json"

  python3 -m tensorrt_llm.serve.scripts.benchmark_serving \
    --backend openai \
    --model "${MODEL_PATH}" \
    --dataset-name random \
    --random-input-len 1000 \
    --random-output-len 1000 \
    --random-prefix-len 0 \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${concurrency}" \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --save-result \
    --result-dir "${RESULT_ROOT}/${PHASE}" \
    --result-filename "${tag}.json"

  echo "Wrote ${out}"
}

bench_trtllm_bench() {
  local tag="$1"
  local concurrency="$2"
  local num_requests="$3"
  local dataset="${RESULT_ROOT}/ds_1000_${num_requests}.jsonl"

  if [[ ! -f "${dataset}" ]]; then
    trtllm-bench --model Qwen3/Qwen3-4B prepare-dataset \
      --output "${dataset}" token-norm-dist \
      --input-mean 1000 --output-mean 1000 \
      --input-stdev 0 --output-stdev 0 \
      --num-requests "${num_requests}"
  fi

  trtllm-bench --model Qwen3/Qwen3-4B \
    --model_path "${MODEL_PATH}" \
    throughput \
    --dataset "${dataset}" \
    --backend pytorch \
    --concurrency "${concurrency}" \
    --streaming \
    --config "${CONFIG}" \
    --report_json "${RESULT_ROOT}/${PHASE}/${tag}_trtllm_bench.json"
}

echo "=== Phase: ${PHASE} | branch: ${GIT_BRANCH} | model: ${MODEL_PATH} ==="

if [[ "${BENCH_MODE:-serve}" == "trtllm-bench" ]]; then
  bench_trtllm_bench con1 1 10
  bench_trtllm_bench con8 8 80
else
  bench_serving con1 1 10
  bench_serving con8 8 80
fi

python3 "$(dirname "$0")/parse_fusion_bench_results.py" \
  --result-dir "${RESULT_ROOT}" \
  --phase "${PHASE}" \
  --branch "${GIT_BRANCH}"
