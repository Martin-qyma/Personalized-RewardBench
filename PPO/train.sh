#!/bin/bash
# Master Training Script
# Starts the reward model server, runs PPO training, then shuts the server down.
#
# Usage:
#   bash run_training.sh <REWARD_MODEL> <MODEL_TYPE> <DATASET> <POLICY_MODEL> \
#                        [SERVER_GPUS] [TRAINING_GPUS]
#
# Args:
#   REWARD_MODEL    HuggingFace name/path of the reward model
#   MODEL_TYPE      discriminative | generative
#   DATASET         Full subset name (e.g. Art_and_Entertainment)
#   POLICY_MODEL    HuggingFace name/path of the policy model
#   SERVER_GPUS     GPU IDs for the reward server (default: 4)
#   TRAINING_GPUS   GPU IDs for PPO training     (default: 0,1,2,3)
#
# Examples:
#   bash run_training.sh Skywork/Skywork-Reward-V2-Llama-3.2-3B discriminative \
#        Art_and_Entertainment Qwen/Qwen2.5-0.5B-Instruct 4 0,1,2,3
#
#   bash run_training.sh Qwen/Qwen2.5-7B-Instruct generative \
#        Lifestyle_and_Personal_Development Qwen/Qwen2.5-0.5B-Instruct 6,7 0,1,2,3

set -e

REWARD_MODEL="${1:?REWARD_MODEL required}"
MODEL_TYPE="${2:?MODEL_TYPE required (discriminative | generative)}"
DATASET="${3:?DATASET required}"
POLICY_MODEL="${4:?POLICY_MODEL required}"
SERVER_GPUS="${5:-4}"
TRAINING_GPUS="${6:-0,1,2,3}"

SERVER_HOST="127.0.0.1"
SERVER_PORT=8000
SCORE_TEMPLATE="prompts/sample.txt"

# Validate model type
if [[ ! "${MODEL_TYPE}" =~ ^(discriminative|generative)$ ]]; then
    echo "Error: MODEL_TYPE must be one of: discriminative, generative"
    exit 1
fi

# Validate dataset
VALID_DATASETS=("Lifestyle_and_Personal_Development" "Art_and_Entertainment" "Society_and_Culture")
if [[ ! " ${VALID_DATASETS[*]} " =~ " ${DATASET} " ]]; then
    echo "Error: DATASET must be one of: ${VALID_DATASETS[*]}"
    exit 1
fi

# Tensor parallel size = number of server GPUs (for generative models)
IFS=',' read -ra SERVER_GPU_ARRAY <<< "${SERVER_GPUS}"
TP_SIZE=${#SERVER_GPU_ARRAY[@]}

echo "=========================================="
echo "Training Configuration"
echo "  Reward model:  ${REWARD_MODEL}"
echo "  Model type:    ${MODEL_TYPE}"
echo "  Dataset:       ${DATASET}"
echo "  Policy model:  ${POLICY_MODEL}"
echo "  Server GPUs:   ${SERVER_GPUS} (TP=${TP_SIZE})"
echo "  Training GPUs: ${TRAINING_GPUS}"
echo "=========================================="

# ------------------------------------------------------------------
# Step 1: Start the reward model server
# ------------------------------------------------------------------
echo ""
echo "Step 1: Starting reward model server..."

CUDA_VISIBLE_DEVICES=${SERVER_GPUS} python PPO/reward_server.py \
    --model-name "${REWARD_MODEL}" \
    --model-type "${MODEL_TYPE}" \
    --gpu-ids "${SERVER_GPUS}" \
    --tensor-parallel-size ${TP_SIZE} \
    --score-template "${SCORE_TEMPLATE}" \
    --host "${SERVER_HOST}" \
    --port ${SERVER_PORT} \
    > PPO/reward_server.log 2>&1 &
SERVER_PID=$!
echo "Server started (PID: ${SERVER_PID}). Waiting for readiness..."

# Wait up to 10 minutes for the server to be ready
MAX_WAIT=60  # iterations × 10s = 600s
WAIT_COUNT=0
until curl -sf "http://${SERVER_HOST}:${SERVER_PORT}/health" > /dev/null; do
    sleep 10
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ ${WAIT_COUNT} -ge ${MAX_WAIT} ]; then
        echo "ERROR: Server did not start within $((MAX_WAIT * 10)) seconds"
        kill ${SERVER_PID} 2>/dev/null || true
        exit 1
    fi
done
echo "Server is ready!"

# ------------------------------------------------------------------
# Step 2: PPO training
# ------------------------------------------------------------------
echo ""
echo "Step 2: Starting PPO training..."
bash PPO/verl.sh "${DATASET}" "${POLICY_MODEL}" "${TRAINING_GPUS}"

# ------------------------------------------------------------------
# Step 3: Shut down the server
# ------------------------------------------------------------------
echo ""
echo "Step 3: Stopping reward model server..."
if kill -0 ${SERVER_PID} 2>/dev/null; then
    kill ${SERVER_PID}
    wait ${SERVER_PID} 2>/dev/null || true
    echo "Server stopped."
else
    echo "Server already exited."
fi

echo ""
echo "Pipeline complete!"
echo "  Reward model: ${REWARD_MODEL}"
echo "  Policy model: ${POLICY_MODEL}"
echo "  Dataset:      ${DATASET}"
