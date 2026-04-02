#!/bin/bash
# Rank and retrieve profiles for all dataset subsets and splits.
#
# Usage:
#   bash run_rank_dataset.sh [GPU_ID]

GPU="${1:-0}"

SUBSETS=(
    "Lifestyle_and_Personal_Development"
    "Art_and_Entertainment"
    "Society_and_Culture"
)

SPLITS=(
    "train"
    "test"
)

for subset in "${SUBSETS[@]}"; do
    for split in "${SPLITS[@]}"; do
        echo "Processing: ${subset} | ${split}"
        python ./retrieve/rank_dataset.py \
            --dataset_config "${subset}" \
            --split "${split}" \
            --gpu "${GPU}"
        echo "Done: ./data/${subset}_${split}.jsonl"
    done
done
