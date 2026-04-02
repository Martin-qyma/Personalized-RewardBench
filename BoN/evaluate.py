"""
Evaluate best-of-k outputs using an LLM judge.

Reads BoN/data/{subset_short}_k{k}.jsonl (from score.py) and scores each
response against rubric aspects. Loops over all specified k values.

Usage:
  python BoN/evaluate.py --subset <SUBSET> [options]
"""

import os
import re
import json
import argparse
import numpy as np
from typing import List, Dict
from vllm import LLM, SamplingParams
from tqdm import tqdm


SUBSET_SHORT = {
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Art_and_Entertainment": "Art",
    "Society_and_Culture": "Society",
}

K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]


def load_template(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_model(model_name: str, tensor_parallel_size: int, pipeline_parallel_size: int) -> LLM:
    print(f"Loading judge model: {model_name}")
    return LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )


def prepare_prompts(dataset: List[Dict], template: str) -> List[Dict]:
    prompts = []
    for example in dataset:
        prompt = template.replace("{question}", example["query"])
        prompt = prompt.replace("{answer}", example["output"])
        prompt = prompt.replace("{aspects}", example.get("aspects", ""))

        prompts.append({
            "id": example["id"],
            "query": example["query"],
            "answer": example["output"],
            "aspects": example.get("aspects", ""),
            "prompt": prompt,
        })
    return prompts


def evaluate_answers(llm: LLM, prompts: List[Dict], batch_size: int) -> List[Dict]:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Scoring"):
        batch = prompts[i : i + batch_size]
        messages_batch = [
            [
                {"role": "system", "content": "You are a fair and insightful judge."},
                {"role": "user", "content": item["prompt"]},
            ]
            for item in batch
        ]
        outputs = llm.chat(messages=messages_batch, sampling_params=sampling_params, use_tqdm=False)

        for j, output in enumerate(outputs):
            score_text = output.outputs[0].text.strip()
            try:
                list_match = re.search(r"\[([^\]]+)\]", score_text)
                numbers = re.findall(r"-?\d+\.?\d*", list_match.group(1) if list_match else score_text)
                scores = [float(n) for n in numbers]
                average_score = float(np.mean(scores)) if scores else 0.0
            except Exception:
                scores = [0.0]
                average_score = 0.0

            results.append({
                "id": batch[j]["id"],
                "query": batch[j]["query"],
                "answer": batch[j]["answer"],
                "aspects": batch[j]["aspects"],
                "scores": scores,
                "average_score": average_score,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BoN outputs with an LLM judge")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                        help="Judge model (default: Qwen/Qwen2.5-32B-Instruct)")
    parser.add_argument("--subset", type=str, default=None,
                        choices=list(SUBSET_SHORT.keys()),
                        help="Dataset subset (default: all subsets)")
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES,
                        help="K values to evaluate (default: 1 2 4 8 16 32 64 128)")
    parser.add_argument("--data-dir", type=str, default="BoN/data",
                        help="Directory containing scored k-output files (default: BoN/data)")
    parser.add_argument("--template", type=str, default="prompts/evaluation.txt",
                        help="Path to evaluation prompt template (default: prompts/evaluation.txt)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    args = parser.parse_args()

    subsets = [args.subset] if args.subset else list(SUBSET_SHORT.keys())
    template = load_template(args.template)
    llm = load_model(args.model, args.tensor_parallel_size, args.pipeline_parallel_size)

    all_summary = {}

    for subset in subsets:
        subset_short = SUBSET_SHORT[subset]
        print(f"\n{'='*60}")
        print(f"Subset: {subset}")
        print(f"{'='*60}")

        k_summary = {}
        for k in sorted(args.k_values):
            input_path = os.path.join(args.data_dir, f"{subset_short}_k{k}.jsonl")
            if not os.path.exists(input_path):
                print(f"  k={k}: {input_path} not found, skipping.")
                continue

            print(f"\n--- k={k} ({input_path}) ---")
            dataset = load_jsonl(input_path)
            prompts = prepare_prompts(dataset, template)
            results = evaluate_answers(llm, prompts, batch_size=args.batch_size)

            averages = [r["average_score"] for r in results]
            summary = {
                "k": k,
                "num_examples": len(results),
                "mean_score": float(np.mean(averages)),
                "std_score": float(np.std(averages)),
                "min_score": float(np.min(averages)),
                "max_score": float(np.max(averages)),
            }
            k_summary[k] = summary
            print(f"  Mean: {summary['mean_score']:.4f} ± {summary['std_score']:.4f}")

            # Save per-k results
            out_path = os.path.join(args.data_dir, f"{subset_short}_eval_k{k}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Saved to {out_path}")

        all_summary[subset] = k_summary

    # Print and save overall summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for subset, k_results in all_summary.items():
        print(f"\n{subset}:")
        print(f"  {'k':>4}  {'Mean':>8}  {'Std':>8}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*8}")
        for k in sorted(k_results):
            r = k_results[k]
            print(f"  {k:>4}  {r['mean_score']:>8.4f}  {r['std_score']:>8.4f}")

    summary_path = os.path.join(args.data_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
