#!/bin/bash
# Generate N samples for Best-of-N selection.
#
# Usage:
#   bash BoN/sample.sh <MODEL> <GPUS> [NUM_SAMPLES] [K]
#
# Args:
#   MODEL        HuggingFace name/path of the policy model
#   GPUS         Comma-separated GPU IDs (e.g., "0,1,2,3")
#   NUM_SAMPLES  Number of samples per query (default: 16)
#   K            Number of profile posts to include (default: 10)
#
# Example:
#   bash BoN/sample.sh Qwen/Qwen2.5-0.5B-Instruct 0,1,2,3 16 10

set -e

MODEL="${1:?MODEL required}"
GPUS="${2:?GPUS required}"
NUM_SAMPLES="${3:-16}"
K="${4:-10}"

SUBSETS=(
    "Art_and_Entertainment"
    "Lifestyle_and_Personal_Development"
    "Society_and_Culture"
)

mkdir -p BoN/data

for SUBSET in "${SUBSETS[@]}"; do
    echo "Sampling: ${SUBSET}"
    python BoN/sample.py \
        --model "${MODEL}" \
        --subset "${SUBSET}" \
        --num-samples "${NUM_SAMPLES}" \
        --k "${K}" \
        --gpus "${GPUS}"
done

echo "Done. Samples saved to BoN/data/"
