# PPO Training for Personalized Response Generation

This directory contains the full pipeline for training a policy model with PPO using a reward model, then generating the output using trained checkpoints and evaluating the results. The scripts are designed to be run sequentially, but you can also run individual steps as needed.

All commands assume **`personalized-rewardbench/`** as the working directory.

---

## Pipeline Overview

```
retrieve/rank_dataset.sh       # 1. Rank user profiles by relevance
      ↓
PPO/prepare_data.py            # 2. Convert to parquet for verl
      ↓
PPO/train.sh                   # 3. PPO training
  ├── reward_server.py         #    3a. Reward model server (runs in background)
  └── verl.sh                  #    3b. verl PPO trainer
      ↓
PPO/convert.sh                 # 4. Convert FSDP checkpoint → HuggingFace format
      ↓
PPO/infer.sh                   # 5. Generate responses with trained model
      ↓
PPO/evaluate.sh                # 6. Score responses with LLM judge
```

---

## Step-by-Step

### Step 1 — Rank profiles

Use Contriever to rank each user's profile posts by relevance to their question. Outputs JSONL files to `PPO/data/`.

```bash
bash retrieve/rank_dataset.sh <GPU_ID>
```

Edit `retrieve/rank_dataset.sh` to select subsets. GPU defaults to `0`.

---

### Step 2 — Prepare training data

Convert the ranked JSONL into parquet format expected by verl. Fills in the generation prompt template and packages `query`/`profile` into `extra_info` for the reward server.

```bash
python PPO/prepare_data.py --subset Art_and_Entertainment
```

Key options:

| Arg | Default | Description |
|-----|---------|-------------|
| `--subset` | *(required)* | Dataset subset name |
| `--data-dir` | `PPO/data` | Directory with ranked JSONL files |
| `--output-dir` | `PPO/data` | Directory to save parquet |
| `--k` | `10` | Number of profile posts to include per example |
| `--template` | `PPO/prompts/sample.txt` | Generation prompt template |

This step is also run automatically by `verl.sh` if the parquet does not exist yet.

---

### Step 3 — PPO training

`train.sh` orchestrates the full training run:
1. Starts the reward model server (`reward_server.py`) in the background
2. Runs PPO training via verl (`verl.sh`)
3. Shuts the server down on completion

```bash
bash PPO/train.sh <REWARD_MODEL> <MODEL_TYPE> <DATASET> <POLICY_MODEL> [SERVER_GPUS] [TRAINING_GPUS]
```

**`MODEL_TYPE`** choices:
- `discriminative` — classification-head reward model (e.g. Skywork)
- `generative` — LLM that outputs a numeric score (e.g. Qwen, Llama)

**Examples:**
```bash
# Discriminative reward model
bash PPO/train.sh \
    Skywork/Skywork-Reward-V2-Llama-3.2-3B discriminative \
    Art_and_Entertainment Qwen/Qwen2.5-0.5B-Instruct \
    4 0,1,2,3

# Generative reward model
bash PPO/train.sh \
    Qwen/Qwen2.5-7B-Instruct generative \
    Society_and_Culture Qwen/Qwen2.5-0.5B-Instruct \
    6,7 0,1,2,3
```

Checkpoints are saved to `PPO/output/<policy_model>_<dataset>/`.
Training logs are written to `PPO/reward_server.log` and WandB.

**How the reward works:**
During each PPO rollout, verl calls `reward_function.py:compute_score()` for every generated response. This function sends the response, query, and profile to the reward server over HTTP and returns a normalized score. The reward server must be running before training starts — `train.sh` handles this automatically.

---

### Step 4 — Convert checkpoint

verl saves checkpoints in FSDP format. Convert to HuggingFace format before inference.

```bash
bash PPO/convert.sh \
    PPO/output/Qwen2.5-0.5B-Instruct_Art_and_Entertainment/global_step_59/actor \
    PPO/output/Qwen2.5-0.5B-Instruct_Art_and_Entertainment/hf
```

---

### Step 5 — Generate responses

Run the trained model on the test set.

```bash
bash PPO/infer.sh <GPU_ID>
```

Edit `MODEL_PATHS` and `SUBSETS` in `infer.sh` before running. Results are saved to `PPO/result/` as `{subset_short}_{tag}_inference.jsonl`.

---

### Step 6 — Evaluate

Score the generated responses against rubric aspects using an LLM judge.

```bash
bash PPO/evaluate.sh <GPU_IDS>
```

Edit `SUBSETS` and `OUTPUT_TAGS` in `evaluate.sh` to match the tags used in Step 5. Results and summary statistics are saved to `PPO/result/` as `{subset_short}_{tag}_evaluation.jsonl` and `.txt`.

---

## Directory Structure

```
personalized-rewardbench/
├── retrieve/
│   ├── rank_dataset.py        # Profile ranking with Contriever
│   └── rank_dataset.sh        # Runner for all subsets
└── PPO/
    ├── prompts/
    │   ├── sample.txt          # Generation prompt template
    │   └── evaluate.txt        # Evaluation prompt template
    ├── data/                   # Ranked JSONL + parquet files (generated)
    ├── output/                 # PPO checkpoints (generated)
    ├── result/                 # Inference + evaluation outputs (generated)
    ├── prepare_data.py         # Step 2: JSONL → parquet
    ├── reward_server.py        # Step 3a: reward model HTTP server
    ├── reward_function.py      # Step 3a: verl reward callback
    ├── verl.sh                 # Step 3b: verl PPO training
    ├── train.sh                # Step 3: master training script
    ├── convert.sh              # Step 4: FSDP → HF conversion
    ├── infer.py                # Step 5: response generation
    ├── infer.sh                # Step 5: runner
    ├── evaluate.py             # Step 6: LLM judge scoring
    └── evaluate.sh             # Step 6: runner
```
