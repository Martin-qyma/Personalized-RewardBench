#!/bin/bash
# Convert a verl FSDP checkpoint to HuggingFace format.
#
# Usage:
#   bash convert.sh <CHECKPOINT_DIR> <OUTPUT_DIR>
#
# Example:
#   bash PPO/convert.sh PPO/output/Qwen2.5-0.5B-Instruct_Art_and_Entertainment/global_step_59/actor \
#                       PPO/output/Qwen2.5-0.5B-Instruct_Art_and_Entertainment/hf

set -e

CHECKPOINT_DIR="${1:?Usage: bash convert.sh <CHECKPOINT_DIR> <OUTPUT_DIR>}"
OUTPUT_DIR="${2:?Usage: bash convert.sh <CHECKPOINT_DIR> <OUTPUT_DIR>}"

echo "Converting checkpoint:"
echo "  Source: ${CHECKPOINT_DIR}"
echo "  Target: ${OUTPUT_DIR}"

python -m verl.model_merger merge \
    --local_dir "${CHECKPOINT_DIR}" \
    --target_dir "${OUTPUT_DIR}" \
    --backend fsdp

echo "Conversion complete — model saved to ${OUTPUT_DIR}"
