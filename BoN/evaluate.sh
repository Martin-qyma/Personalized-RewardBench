#!/bin/bash
# Evaluate BoN best-of-k outputs using an LLM judge.
#
# Usage:
#   bash BoN/evaluate.sh [GPU_IDS] [TENSOR_PARALLEL_SIZE]
#
# Examples:
#   bash BoN/evaluate.sh 0,1,2,3 4
#   bash BoN/evaluate.sh 0,1 2

set -e

GPU="${1:-0,1,2,3}"
TP="${2:-4}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

CUDA_VISIBLE_DEVICES=${GPU} python BoN/evaluate.py \
    --model "${JUDGE_MODEL}" \
    --tensor-parallel-size "${TP}" \
    --batch-size 8

echo "Done. Results saved to BoN/data/"
