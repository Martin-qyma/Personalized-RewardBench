#!/bin/bash
# PPO Training with verl Framework
#
# Usage:
#   bash verl.sh <DATASET> <POLICY_MODEL> [GPU_IDS]
#
# Args:
#   DATASET       Full subset name (e.g. Art_and_Entertainment)
#   POLICY_MODEL  HuggingFace model path for the policy (e.g. Qwen/Qwen2.5-0.5B-Instruct)
#   GPU_IDS       Comma-separated GPU IDs (default: 0,1,2,3)
#
# Examples:
#   bash verl.sh Art_and_Entertainment Qwen/Qwen2.5-0.5B-Instruct 0,1,2,3
#   bash verl.sh Society_and_Culture meta-llama/Llama-3.2-3B-Instruct 4,5,6,7

set -e

DATASET="${1:?Usage: bash verl.sh <DATASET> <POLICY_MODEL> [GPU_IDS]}"
POLICY_MODEL="${2:?Usage: bash verl.sh <DATASET> <POLICY_MODEL> [GPU_IDS]}"
GPUS="${3:-0,1,2,3}"

PPO_DIR="PPO"
DATA_DIR="${PPO_DIR}/data"
OUTPUT_BASE="${PPO_DIR}/output"
WANDB_PROJECT="RewardBench"

# Derive a filesystem-safe short name from the model path (e.g. "Qwen2.5-0.5B-Instruct")
MODEL_SHORT=$(basename "${POLICY_MODEL}")
OUTPUT_DIR="${OUTPUT_BASE}/${MODEL_SHORT}_${DATASET}"
WANDB_RUN="${MODEL_SHORT}_${DATASET}"

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Prepare data if not already done
mkdir -p "${DATA_DIR}"
PARQUET="${DATA_DIR}/${DATASET}.parquet"
if [ ! -f "${PARQUET}" ]; then
    echo "Preparing dataset: ${DATASET}"
    python "${PPO_DIR}/prepare_data.py" --subset "${DATASET}" --output-dir "${DATA_DIR}"
fi

echo "=========================================="
echo "PPO Training"
echo "  Policy model: ${POLICY_MODEL}"
echo "  Dataset:      ${DATASET}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  GPUs:         ${GPUS} (${NUM_GPUS} total)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${GPUS} python -m verl.trainer.main_ppo \
    data.train_files="${PARQUET}" \
    data.val_files="${PARQUET}" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=8192 \
    data.filter_overlong_prompts=True \
    data.max_response_length=512 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    actor_rollout_ref.model.path="${POLICY_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    custom_reward_function.path="${PPO_DIR}/reward_function.py" \
    custom_reward_function.name=compute_score \
    critic.model.path="${POLICY_MODEL}" \
    critic.optim.lr=5e-6 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${WANDB_RUN}" \
    trainer.total_epochs=1 \
    trainer.logger=['wandb','console'] \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.save_freq=10000

echo "=========================================="
echo "Training complete — model saved to ${OUTPUT_DIR}"
echo "=========================================="
