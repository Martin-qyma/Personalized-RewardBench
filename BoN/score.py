"""
Score BoN samples with a reward model and select the best-of-k output.

Supports two reward model types:
  discriminative  Classification-head models (e.g. Skywork, InternLM)
  generative      Instruction-tuned LLMs that output a numeric score

Input:  BoN/data/{subset_short}_samples.jsonl  (from sample.py)
Output: BoN/data/{subset_short}_scored.jsonl
        BoN/data/{subset_short}_k{k}.jsonl  (one per k value)

Usage:
  python BoN/score.py --model <HF_MODEL> --model-type <TYPE> [options]
"""

import os
import re
import json
import argparse
import torch
from collections import defaultdict
from typing import List, Dict, Tuple
from tqdm import tqdm


SUBSET_SHORT = {
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Art_and_Entertainment": "Art",
    "Society_and_Culture": "Society",
}

K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]


def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_samples(path: str) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# Discriminative scoring
# ---------------------------------------------------------------------------

def score_discriminative(
    samples: List[Dict], model_name: str, gpu_ids: str, batch_size: int
) -> List[float]:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    model.eval()
    print(f"Discriminative model loaded on {device}")

    scores = []
    for i in tqdm(range(0, len(samples), batch_size), desc="Scoring"):
        batch = samples[i : i + batch_size]
        convs_formatted = []
        for item in batch:
            conv = [
                {"role": "user", "content": item["query"] + item["profile"]},
                {"role": "assistant", "content": item["generated_response"]},
            ]
            convs_formatted.append(tokenizer.apply_chat_template(conv, tokenize=False))

        inputs = tokenizer(
            convs_formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            if logits.shape[-1] == 1:
                batch_scores = logits[:, 0].tolist()
            else:
                batch_scores = torch.softmax(logits, dim=-1)[:, -1].tolist()

        scores.extend(batch_scores)

    return scores


# ---------------------------------------------------------------------------
# Generative scoring
# ---------------------------------------------------------------------------

def score_generative(
    samples: List[Dict],
    model_name: str,
    gpu_ids: str,
    tensor_parallel_size: int,
    score_template: str,
    batch_size: int,
) -> List[float]:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.65,
    )
    print(f"Generative model loaded (TP={tensor_parallel_size})")

    params = SamplingParams(max_tokens=10, temperature=0.0, top_p=1.0)
    scores = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Scoring"):
        batch = samples[i : i + batch_size]
        prompts = []
        for item in batch:
            prompt_text = score_template.replace("{question}", item["query"])
            prompt_text = prompt_text.replace("{profile}", item["profile"])
            prompt_text = prompt_text.replace("{answer}", item["generated_response"])
            messages = [
                {"role": "system", "content": "You are a fair and insightful judge."},
                {"role": "user", "content": prompt_text},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        outputs = llm.generate(prompts, params)
        for output in outputs:
            text = output.outputs[0].text.strip()
            try:
                scores.append(max(0.0, min(1.0, float(text))))
            except ValueError:
                nums = re.findall(r"0?\.\d+|[01]\.?\d*", text)
                scores.append(max(0.0, min(1.0, float(nums[0]))) if nums else 0.5)

    return scores


# ---------------------------------------------------------------------------
# Best-of-k selection
# ---------------------------------------------------------------------------

def select_best_of_k(samples: List[Dict], scores: List[float], k: int) -> List[Dict]:
    """For each unique ID, take the first k samples and pick the highest-scored one."""
    groups: Dict[str, List[Tuple[float, Dict]]] = defaultdict(list)
    for sample, score in zip(samples, scores):
        groups[sample["id"]].append((score, sample))

    best = []
    for item_id, scored in groups.items():
        top_k = scored[:k]
        best_score, best_sample = max(top_k, key=lambda x: x[0])
        best.append({
            "id": item_id,
            "query": best_sample["query"],
            "output": best_sample["generated_response"],
            "aspects": best_sample.get("aspects", ""),
            "score": best_score,
        })
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score BoN samples with a reward model")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["discriminative", "generative"],
                        help=(
                            "discriminative: classification-head RM (e.g. Skywork, InternLM); "
                            "generative: LLM that outputs a numeric score"
                        ))
    parser.add_argument("--subset", type=str, default=None,
                        choices=list(SUBSET_SHORT.keys()),
                        help="Dataset subset (default: all subsets)")
    parser.add_argument("--score-template", type=str, default="prompts/evaluation.txt",
                        help="Scoring prompt template with {question}/{profile}/{answer} placeholders "
                             "(generative only, default: prompts/evaluation.txt)")
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES,
                        help="K values for best-of-k selection (default: 1 2 4 8 16 32 64 128)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Scoring batch size (default: 16)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel size (generative only, default: 1)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g. '0,1,2,3')")
    args = parser.parse_args()

    subsets = [args.subset] if args.subset else list(SUBSET_SHORT.keys())
    score_template = (
        load_template(args.score_template) if args.model_type == "generative" else None
    )

    for subset in subsets:
        subset_short = SUBSET_SHORT[subset]
        samples_path = f"BoN/data/{subset_short}_samples.jsonl"

        if not os.path.exists(samples_path):
            print(f"Warning: {samples_path} not found, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Scoring subset: {subset}")
        print(f"{'='*60}")

        samples = load_samples(samples_path)
        print(f"Loaded {len(samples)} samples")

        if args.model_type == "discriminative":
            scores = score_discriminative(samples, args.model, args.gpus, args.batch_size)
        else:
            scores = score_generative(
                samples, args.model, args.gpus,
                args.tensor_parallel_size, score_template, args.batch_size,
            )

        # Save scored samples
        os.makedirs("BoN/data", exist_ok=True)
        scored_path = f"BoN/data/{subset_short}_scored.jsonl"
        with open(scored_path, "w", encoding="utf-8") as f:
            for sample, score in zip(samples, scores):
                record = dict(sample)
                record["score"] = score
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Scored samples saved to {scored_path}")

        # Select best-of-k for each k value
        for k in sorted(args.k_values):
            best = select_best_of_k(samples, scores, k)
            out_path = f"BoN/data/{subset_short}_k{k}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for item in best:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"  k={k:3d}: {len(best)} examples → {out_path}")


if __name__ == "__main__":
    main()
