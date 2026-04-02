import os
import json
import random
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from typing import List, Dict
from tqdm import tqdm


SUBSET_SHORT = {
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Art_and_Entertainment": "Art",
    "Society_and_Culture": "Society",
}

HF_DATASET = "QiyaoMa/Personalized-RewardBench"


def load_template(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_model(model_name: str, tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1) -> LLM:
    print(f"Loading model {model_name} with vLLM (TP={tensor_parallel_size})")
    return LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )


def prepare_pairwise_prompts(dataset, user_template: str, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    prompts = []

    for item in dataset:
        pos_in_a = random.random() < 0.5
        if pos_in_a:
            response_a, response_b = item['chosen'], item['rejected']
            correct_answer = "A"
        else:
            response_a, response_b = item['rejected'], item['chosen']
            correct_answer = "B"

        prompt = user_template.replace("{question}", item['question'])
        prompt = prompt.replace("{Response_A}", response_a)
        prompt = prompt.replace("{Response_B}", response_b)

        prompts.append({
            'id': item['id'],
            'query': item['question'],
            'response_a': response_a,
            'response_b': response_b,
            'correct_answer': correct_answer,
            'pos_in_a': pos_in_a,
            'prompt': prompt,
        })

    return prompts


def evaluate_pairwise(llm: LLM, prompts: List[Dict], system_prompt: str,
                      batch_size: int = 8) -> tuple[List[Dict], int]:
    print(f"Evaluating {len(prompts)} pairwise comparisons...")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    results = []
    failed_parse_count = 0

    for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating pairs"):
        batch = prompts[i:i + batch_size]
        messages_batch = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item['prompt']},
            ]
            for item in batch
        ]
        outputs = llm.chat(messages=messages_batch, sampling_params=sampling_params, use_tqdm=False)

        for j, output in enumerate(outputs):
            prediction_text = output.outputs[0].text.strip()
            prediction_text = prediction_text.split("</think>")[-1].strip()
            prediction_text = prediction_text.replace("\"", "").replace("*", "").strip()

            prediction = None
            if "result:" in prediction_text.lower():
                result_section = prediction_text.lower().split("result:")[1].strip()
                if result_section.startswith("a"):
                    prediction = "A"
                elif result_section.startswith("b"):
                    prediction = "B"

            parse_failed = prediction is None
            if parse_failed:
                prediction = "A"
                failed_parse_count += 1
                print(f"Warning: Failed to parse prediction for id {batch[j]['id']}: '{prediction_text[:100]}'...")

            results.append({
                'id': batch[j]['id'],
                'query': batch[j]['query'],
                'pos_in_a': batch[j]['pos_in_a'],
                'prediction': prediction,
                'correct_answer': batch[j]['correct_answer'],
                'correct': prediction == batch[j]['correct_answer'],
                'parse_failed': parse_failed,
                'raw_output': prediction_text,
            })

    return results, failed_parse_count


def calculate_accuracy(results: List[Dict]) -> Dict:
    total = len(results)
    correct = sum(1 for r in results if r['correct'])

    pos_in_a = [r for r in results if r['pos_in_a']]
    pos_in_b = [r for r in results if not r['pos_in_a']]

    def acc(subset): return sum(1 for r in subset if r['correct']) / len(subset) if subset else 0.0

    pred_a = sum(1 for r in results if r['prediction'] == 'A')
    pred_b = sum(1 for r in results if r['prediction'] == 'B')

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0.0,
        'pos_in_a_count': len(pos_in_a),
        'pos_in_a_accuracy': acc(pos_in_a),
        'pos_in_b_count': len(pos_in_b),
        'pos_in_b_accuracy': acc(pos_in_b),
        'pred_a_count': pred_a,
        'pred_b_count': pred_b,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Judge model (HuggingFace name or local path)")
    parser.add_argument("--subset", type=str, required=True,
                        choices=list(SUBSET_SHORT.keys()),
                        help="Dataset subset name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--system-template", type=str, default="prompts/pairwise_system.txt",
                        help="Path to pairwise system prompt (default: prompts/pairwise_system.txt)")
    parser.add_argument("--user-template", type=str, default="prompts/pairwise_user.txt",
                        help="Path to pairwise user template (default: prompts/pairwise_user.txt)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for position shuffling (default: 42)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g. '0,1')")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    subset_short = SUBSET_SHORT[args.subset]
    model_short = os.path.basename(args.model.rstrip("/"))

    print("=" * 70)
    print("PAIRWISE EVALUATION")
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {HF_DATASET} / {args.subset} / {args.split}")
    print("=" * 70)

    dataset = load_dataset(HF_DATASET, args.subset, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    system_prompt = load_template(args.system_template)
    user_template = load_template(args.user_template)
    llm = load_model(args.model, args.tensor_parallel_size, args.pipeline_parallel_size)

    prompts = prepare_pairwise_prompts(dataset, user_template, seed=args.seed)
    results, failed_parses = evaluate_pairwise(llm, prompts, system_prompt, batch_size=args.batch_size)
    metrics = calculate_accuracy(results)

    position_bias = abs(metrics['pos_in_a_accuracy'] - metrics['pos_in_b_accuracy'])

    summary = [
        "=" * 70,
        "PAIRWISE EVALUATION COMPLETE",
        "=" * 70,
        f"Subset:   {args.subset}",
        f"Split:    {args.split}",
        f"Model:    {args.model}",
        f"Seed:     {args.seed}",
        "",
        f"Overall Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})",
        "",
        "Position-specific Accuracy:",
        f"  Pos in A: {metrics['pos_in_a_accuracy']:.2%} ({metrics['pos_in_a_count']} pairs)",
        f"  Pos in B: {metrics['pos_in_b_accuracy']:.2%} ({metrics['pos_in_b_count']} pairs)",
        f"  Position bias: {position_bias:.2%}",
        "",
        "Prediction Distribution:",
        f"  Predicted A: {metrics['pred_a_count']} ({metrics['pred_a_count']/metrics['total']:.2%})",
        f"  Predicted B: {metrics['pred_b_count']} ({metrics['pred_b_count']/metrics['total']:.2%})",
        "",
        f"Failed parses: {failed_parses}",
        "=" * 70,
    ]

    print("\n" + "\n".join(summary))

    output_file = f"evaluation/results/{subset_short}_{model_short}_pairwise.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary))
    print(f"\nSummary saved to: {output_file}")


if __name__ == "__main__":
    main()
