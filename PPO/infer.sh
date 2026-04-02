#!/bin/bash
# Run inference for one or more (subset, model) combinations.
#
# Usage:
#   bash PPO/infer.sh [GPU_IDS]
#
# Edit MODEL_PATHS and SUBSETS below before running.

GPU="${1:-0}"

# Paths to PPO-trained model checkpoints under PPO/output/
MODEL_PATHS=(
    "PPO/output/Qwen2.5-0.5B-Instruct_Art_and_Entertainment/hf"
    "PPO/output/Qwen2.5-0.5B-Instruct_Society_and_Culture/hf"
    "PPO/output/Qwen2.5-0.5B-Instruct_Lifestyle_and_Personal_Development/hf"
)

SUBSETS=(
    "Art_and_Entertainment"
    "Lifestyle_and_Personal_Development"
    "Society_and_Culture"
)

for model_path in "${MODEL_PATHS[@]}"; do
    for subset in "${SUBSETS[@]}"; do
        tag=$(basename "$(dirname "${model_path}")")
        echo "Inference: ${model_path} | ${subset}"
        CUDA_VISIBLE_DEVICES=${GPU} python PPO/infer.py \
            --model-path "${model_path}" \
            --subset "${subset}" \
            --output-tag "${tag}"
        echo ""
    done
done
