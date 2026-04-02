"""
Generate personalized responses using a (PPO-trained) model.

Usage:
  python PPO/infer.py --model-path <PATH> --subset <SUBSET> [options]
"""

import os
import json
import argparse
from vllm import LLM, SamplingParams
from typing import List, Dict
from tqdm import tqdm


SUBSET_SHORT = {
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Art_and_Entertainment": "Art",
    "Society_and_Culture": "Society",
}


def load_template(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def format_profile(profile_list: List[Dict]) -> str:
    formatted = []
    for i, item in enumerate(profile_list[:5], 1):
        formatted.append(f"Post {i} (Category: {item['category']}):\n{item['text']}")
    return "\n".join(formatted)


def prepare_prompts(dataset: List[Dict], template: str) -> List[Dict]:
    prompts = []
    for example in dataset:
        profile_text = format_profile(example["profile"])
        question_text = example["question"]

        prompt = template.replace("{profile}", profile_text)
        prompt = prompt.replace("{question}", question_text)

        prompts.append({
            "id": example["id"],
            "question": question_text,
            "profile": profile_text,
            "aspects": example["rubric_aspects"],
            "prompt": prompt,
        })
    return prompts


def load_model(model_path: str, tensor_parallel_size: int, pipeline_parallel_size: int) -> LLM:
    print(f"Loading model: {model_path}")
    return LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )


def batch_generate(llm: LLM, prompts: List[Dict], batch_size: int) -> List[Dict]:
    print(f"Generating responses for {len(prompts)} examples...")

    sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=512)
    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i:i + batch_size]
        messages_batch = [
            [
                {"role": "system", "content": "You are a helpful assistant designed to generate personalized responses to user questions."},
                {"role": "user", "content": item["prompt"]},
            ]
            for item in batch
        ]
        outputs = llm.chat(messages=messages_batch, sampling_params=sampling_params, use_tqdm=False)

        for j, output in enumerate(outputs):
            results.append({
                "id": batch[j]["id"],
                "question": batch[j]["question"],
                "profile": batch[j]["profile"],
                "evaluation": output.outputs[0].text,
                "aspects": batch[j]["aspects"],
            })

    return results


def save_results(results: List[Dict], output_file: str):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate personalized responses")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the (PPO-trained) model checkpoint")
    parser.add_argument("--subset", type=str, required=True,
                        help="Dataset subset (e.g. Art_and_Entertainment)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use (default: test)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing the JSONL dataset files")
    parser.add_argument("--output-dir", type=str, default="PPO/result",
                        help="Directory to save generated responses")
    parser.add_argument("--output-tag", type=str, default="",
                        help="Optional tag appended to the output filename")
    parser.add_argument("--template", type=str, default="PPO/prompts/sample.txt",
                        help="Path to the generation prompt template")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    args = parser.parse_args()

    subset_short = SUBSET_SHORT.get(args.subset, args.subset)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    output_file = os.path.join(args.output_dir, f"{subset_short}{tag}_inference.jsonl")

    template = load_template(args.template)

    data_path = os.path.join(args.data_dir, f"{args.subset}_test.jsonl")
    print(f"Loading dataset from {data_path}...")
    dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    print(f"Loaded {len(dataset)} examples")

    prompts = prepare_prompts(dataset, template)
    llm = load_model(args.model_path, args.tensor_parallel_size, args.pipeline_parallel_size)
    results = batch_generate(llm, prompts, batch_size=args.batch_size)
    save_results(results, output_file)

    print(f"\nDone — {len(results)} examples written to {output_file}")


if __name__ == "__main__":
    main()
