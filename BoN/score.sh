#!/bin/bash
# Score BoN samples and select best-of-k outputs.
#
# Usage:
#   bash BoN/score.sh <MODEL> <MODEL_TYPE> <GPUS> [TENSOR_PARALLEL_SIZE]
#
# Args:
#   MODEL                HuggingFace name/path of the reward model
#   MODEL_TYPE           discriminative | generative
#   GPUS                 Comma-separated GPU IDs (e.g., "0,1,2,3")
#   TENSOR_PARALLEL_SIZE vLLM TP size, generative only (default: 1)
#
# Examples:
#   bash BoN/score.sh Skywork/Skywork-Reward-V2-Llama-3.2-3B discriminative 0
#   bash BoN/score.sh Qwen/Qwen2.5-7B-Instruct generative 0,1 2

set -e

MODEL="${1:?MODEL required}"
MODEL_TYPE="${2:?MODEL_TYPE required (discriminative | generative)}"
GPUS="${3:?GPUS required}"
TP_SIZE="${4:-1}"

python BoN/score.py \
    --model "${MODEL}" \
    --model-type "${MODEL_TYPE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpus "${GPUS}"

echo "Done. Outputs saved to BoN/data/"
