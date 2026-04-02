"""
Evaluate generated responses using an LLM judge.

Reads inference output from infer.py and scores each response
against the rubric aspects using a large judge model.

Usage:
  python PPO/evaluate.py --subset <SUBSET> --output-tag <TAG> [options]
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


def load_template(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    print(f"Loading data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} entries")
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


def format_aspects(aspects) -> str:
    if isinstance(aspects, list):
        return ", ".join(a["aspect"] for a in aspects)
    return str(aspects)


def prepare_prompts(dataset: List[Dict], template: str) -> List[Dict]:
    prompts = []
    for example in dataset:
        question_text = example["question"]
        answer_text = example["evaluation"]
        aspects = format_aspects(example["aspects"])

        prompt = template.replace("{question}", question_text)
        prompt = prompt.replace("{answer}", answer_text)
        prompt = prompt.replace("{aspects}", aspects)

        prompts.append({
            "id": example["id"],
            "query": question_text,
            "aspects": aspects,
            "answer": answer_text,
            "prompt": prompt,
        })
    return prompts


def evaluate_answers(llm: LLM, prompts: List[Dict], batch_size: int) -> List[Dict]:
    print(f"Scoring {len(prompts)} answers...")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Scoring"):
        batch = prompts[i:i + batch_size]
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


def save_results(results: List[Dict], output_file: str):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_file}")


def print_and_save_summary(results: List[Dict], summary_file: str, meta: Dict):
    averages = [r["average_score"] for r in results]
    lines = [
        "=== Evaluation Summary ===",
        f"Total entries:  {len(results)}",
        f"Average score:  {np.mean(averages):.4f}",
        f"Std dev:        {np.std(averages):.4f}",
        f"Min:            {np.min(averages):.4f}",
        f"Max:            {np.max(averages):.4f}",
    ]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")

    print("\n" + "\n".join(lines))

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated responses with an LLM judge")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct",
                        help="Judge model for evaluation")
    parser.add_argument("--subset", type=str, required=True,
                        help="Dataset subset (e.g. Art_and_Entertainment)")
    parser.add_argument("--output-tag", type=str, default="",
                        help="Tag matching the one used in infer.sh")
    parser.add_argument("--input-file", type=str, default="",
                        help="Path to inference JSONL (default: PPO/result/<subset_short>_<tag>_inference.jsonl)")
    parser.add_argument("--result-dir", type=str, default="PPO/result",
                        help="Directory for input and output files")
    parser.add_argument("--template", type=str, default="PPO/prompts/evaluate.txt",
                        help="Path to the evaluation prompt template")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    args = parser.parse_args()

    subset_short = SUBSET_SHORT.get(args.subset, args.subset)
    tag = f"_{args.output_tag}" if args.output_tag else ""

    input_file = args.input_file or os.path.join(args.result_dir, f"{subset_short}{tag}_inference.jsonl")
    output_file = os.path.join(args.result_dir, f"{subset_short}{tag}_evaluation.jsonl")
    summary_file = output_file.replace(".jsonl", ".txt")

    template = load_template(args.template)
    dataset = load_jsonl(input_file)
    prompts = prepare_prompts(dataset, template)

    llm = load_model(args.model, args.tensor_parallel_size, args.pipeline_parallel_size)
    results = evaluate_answers(llm, prompts, batch_size=args.batch_size)

    save_results(results, output_file)
    print_and_save_summary(results, summary_file, meta={"judge": args.model, "subset": args.subset})


if __name__ == "__main__":
    main()
