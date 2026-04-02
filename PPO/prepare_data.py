"""
Prepare training data for verl PPO.

Converts a JSONL dataset into parquet format expected by verl,
filling in the generation prompt template for each example.

Usage:
  python prepare_data.py --subset <SUBSET> [--data-dir <DIR>] [--output-dir <DIR>]
"""

import os
import json
import argparse
import datasets
from typing import List, Dict


def load_template(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_profile(profile_list: List[Dict], k: int = 10) -> str:
    """Format the first k profile posts into a readable string."""
    formatted = []
    for i, item in enumerate(profile_list[:k], 1):
        formatted.append(f"Post {i} (Category: {item['category']}):\n{item['text']}")
    return "\n".join(formatted)


def process_example(example: Dict, idx: int, gen_template: str, data_source: str, k: int = 10) -> Dict:
    query = example["question"]
    profile_text = format_profile(example["profile"], k=k)

    prompt_text = gen_template.replace("{question}", query)
    prompt_text = prompt_text.replace("{profile}", profile_text)

    return {
        "data_source": data_source,
        "prompt": [
            {
                "role": "system",
                "content": "You are a helpful assistant designed to generate personalized responses to user questions.",
            },
            {"role": "user", "content": prompt_text},
        ],
        "reward_model": {"ground_truth": ""},
        "extra_info": {
            "split": "train",
            "index": idx,
            "query": query,
            "profile": profile_text,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True,
                        help="Dataset subset name (e.g. Art_and_Entertainment)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing the JSONL files")
    parser.add_argument("--output-dir", type=str, default="./PPO/data",
                        help="Directory to save the parquet output")
    parser.add_argument("--template", type=str, default="./prompts/sample.txt",
                        help="Path to the generation prompt template")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of profile posts to include per example (default: 10)")
    args = parser.parse_args()

    data_file = os.path.join(args.data_dir, f"{args.subset}_train.jsonl")
    print(f"Loading data from {data_file}")
    raw_data = load_jsonl(data_file)

    gen_template = load_template(args.template)
    data_source = f"local:{data_file}"

    processed = [process_example(ex, idx, gen_template, data_source, k=args.k)
                 for idx, ex in enumerate(raw_data)]

    dataset = datasets.Dataset.from_list(processed)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.subset}.parquet")
    dataset.to_parquet(output_file)
    print(f"Saved {len(processed)} examples to {output_file}")


if __name__ == "__main__":
    main()
