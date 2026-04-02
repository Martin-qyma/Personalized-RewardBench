#!/bin/bash
# Evaluate generated responses using an LLM judge.
#
# Usage:
#   bash PPO/evaluate.sh [GPU_IDS]
#
# Edit SUBSETS and OUTPUT_TAGS below to match your infer.sh outputs.

GPU="${1:-0,1,2,3}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

SUBSETS=(
    "Art_and_Entertainment"
    # "Lifestyle_and_Personal_Development"
    # "Society_and_Culture"
)

# Tags must match what was used in infer.sh
OUTPUT_TAGS=(
    "Qwen2.5-0.5B-Instruct_Art_and_Entertainment"
    # ""
)

for subset in "${SUBSETS[@]}"; do
    for tag in "${OUTPUT_TAGS[@]}"; do
        echo "========================================="
        echo "Evaluating: ${subset} | tag: ${tag}"
        echo "  Judge: ${JUDGE_MODEL}"
        echo "========================================="

        CUDA_VISIBLE_DEVICES=${GPU} python PPO/evaluate.py \
            --model "${JUDGE_MODEL}" \
            --subset "${subset}" \
            --output-tag "${tag}" \
            --tensor-parallel-size 4 \
            --batch-size 8

        [ $? -eq 0 ] && echo "Done." || echo "FAILED."
        echo ""
    done
done
