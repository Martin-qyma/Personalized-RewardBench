# Best-of-N (BoN) Sampling

Best-of-N is a inference-time scaling method: generate **N** candidate responses per query, score all of them with a reward model, and select the best one. By varying N (equivalently k), you can trace a scaling curve showing how response quality improves with more samples.

## Pipeline Overview

```
data/{subset}_test.jsonl
        │
        ▼
  [1] sample.py       — generate N responses per query
        │  BoN/data/{subset}_samples.jsonl
        ▼
  [2] score.py        — score every sample with a reward model;
        │               select best-of-k for each k value
        │  BoN/data/{subset}_scored.jsonl
        │  BoN/data/{subset}_k{k}.jsonl  (one per k)
        ▼
  [3] evaluate.py     — LLM judge scores best-of-k outputs
        │  BoN/data/{subset}_eval_k{k}.jsonl
        │  BoN/data/evaluation_summary.json
```

All scripts assume **`personalized-rewardbench/`** as the working directory.

---

## Step 1 — Generate N Samples

```bash
bash BoN/sample.sh <MODEL> <GPUS> [NUM_SAMPLES] [K]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `MODEL` | HuggingFace policy model | — |
| `GPUS` | Comma-separated GPU IDs | — |
| `NUM_SAMPLES` | Samples per query (N) | `16` |
| `K` | Profile posts to include | `10` |

**Example:**
```bash
bash BoN/sample.sh Qwen/Qwen2.5-0.5B-Instruct 0,1,2,3 16 10
```

Loops all 3 subsets. Output: `BoN/data/{Lifestyle|Art|Society}_samples.jsonl`

**Direct usage:**
```bash
python BoN/sample.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --subset Art_and_Entertainment \
    --num-samples 16 \
    --k 10 \
    --gpus 0,1,2,3
```

---

## Step 2 — Score and Select Best-of-k

```bash
bash BoN/score.sh <MODEL> <MODEL_TYPE> <GPUS> [TENSOR_PARALLEL_SIZE]
```

| Argument | Description | Default |
|----------|-------------|---------|
| `MODEL` | HuggingFace reward model | — |
| `MODEL_TYPE` | `discriminative` or `generative` | — |
| `GPUS` | Comma-separated GPU IDs | — |
| `TENSOR_PARALLEL_SIZE` | vLLM TP size (generative only) | `1` |

**Reward model types:**

- **`discriminative`** — `AutoModelForSequenceClassification` models (e.g. Skywork, InternLM). Score is extracted from logits.
- **`generative`** — Instruction-tuned LLMs that output a numeric score (e.g. Qwen, Llama). Requires a scoring template with `{question}`, `{profile}`, `{answer}` placeholders (pass via `--score-template`).

**Examples:**
```bash
# Discriminative reward model
bash BoN/score.sh Skywork/Skywork-Reward-V2-Llama-3.2-3B discriminative 0

# Generative reward model (2 GPUs)
bash BoN/score.sh Qwen/Qwen2.5-7B-Instruct generative 0,1 2
```

Outputs for each k in `[1, 2, 4, 8, 16, 32, 64, 128]`:
- `BoN/data/{subset}_scored.jsonl` — all samples with scores
- `BoN/data/{subset}_k{k}.jsonl` — best-of-k selected response per query

**Direct usage:**
```bash
python BoN/score.py \
    --model Skywork/Skywork-Reward-V2-Llama-3.2-3B \
    --model-type discriminative \
    --gpus 0 \
    --k-values 1 2 4 8 16
```

---

## Step 3 — Evaluate with LLM Judge

```bash
bash BoN/evaluate.sh [GPU_IDS] [TENSOR_PARALLEL_SIZE]
```

Uses an LLM judge (`Qwen/Qwen2.5-32B-Instruct` by default) to score each best-of-k output against the rubric aspects from `prompts/evaluation.txt`.

**Examples:**
```bash
bash BoN/evaluate.sh 0,1,2,3 4

# Override judge model
JUDGE_MODEL=Qwen/Qwen2.5-72B-Instruct bash BoN/evaluate.sh 0,1,2,3 4
```

Outputs:
- `BoN/data/{subset}_eval_k{k}.jsonl` — per-example scores for each k
- `BoN/data/evaluation_summary.json` — mean/std across all subsets and k values

**Direct usage:**
```bash
python BoN/evaluate.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --subset Art_and_Entertainment \
    --k-values 1 2 4 8 16 \
    --tensor-parallel-size 4
```

---

## Directory Structure

```
BoN/
├── sample.py          # Step 1: generate N candidates
├── sample.sh
├── score.py           # Step 2: score + best-of-k selection
├── score.sh
├── evaluate.py        # Step 3: LLM judge evaluation
├── evaluate.sh
└── data/
    ├── {subset}_samples.jsonl      # N samples per query
    ├── {subset}_scored.jsonl       # all samples with RM scores
    ├── {subset}_k{k}.jsonl         # best-of-k output
    └── {subset}_eval_k{k}.jsonl    # judge scores per k
```

## Prompts

| File | Used by | Placeholders |
|------|---------|--------------|
| `prompts/sample.txt` | `sample.py` | `{question}`, `{profile}` |
| `prompts/evaluation.txt` | `evaluate.py` | `{question}`, `{answer}`, `{aspects}` |
| custom score template | `score.py` (generative only) | `{question}`, `{profile}`, `{answer}` |
