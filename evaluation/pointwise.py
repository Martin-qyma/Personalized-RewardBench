import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from tqdm import tqdm


SUBSET_SHORT = {
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Art_and_Entertainment": "Art",
    "Society_and_Culture": "Society",
}

HF_DATASET = "QiyaoMa/Personalized-RewardBench"


def load_model(model_name: str):
    print(f"Loading reward model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def prepare_scoring_inputs(dataset) -> List[Dict]:
    """Expand each example into two entries: one for chosen, one for rejected."""
    inputs = []
    for item in dataset:
        inputs.append({
            'id': item['id'],
            'query': item['question'],
            'answer': item['chosen'],
            'label': 'chosen',
        })
        inputs.append({
            'id': item['id'],
            'query': item['question'],
            'answer': item['rejected'],
            'label': 'rejected',
        })
    return inputs


def score_answers(model, tokenizer, prompts: List[Dict], batch_size: int = 8) -> List[Dict]:
    print(f"Scoring {len(prompts)} answers...")
    results = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Scoring answers"):
        batch = prompts[i:i + batch_size]

        conversations = [
            [
                {"role": "user", "content": item['query']},
                {"role": "assistant", "content": item['answer']},
            ]
            for item in batch
        ]
        conv_formatted = [
            tokenizer.apply_chat_template(conv, tokenize=False)
            for conv in conversations
        ]

        inputs = tokenizer(
            conv_formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        )
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            for j in range(len(batch)):
                if logits.shape[-1] == 1:
                    score = logits[j, 0].item()
                else:
                    score = torch.softmax(logits[j], dim=-1)[-1].item()

                results.append({
                    'id': batch[j]['id'],
                    'query': batch[j]['query'],
                    'answer': batch[j]['answer'],
                    'label': batch[j]['label'],
                    'score': score,
                })

    return results


def compare_scores(scored_answers: List[Dict], subset_short: str,
                   model_short: str, output_file: str):
    scores_by_id = {}
    for r in scored_answers:
        scores_by_id.setdefault(r['id'], {})[r['label']] = r['score']

    chosen_wins = rejected_wins = ties = total = 0
    diffs = []

    for scores in scores_by_id.values():
        if 'chosen' in scores and 'rejected' in scores:
            total += 1
            d = scores['chosen'] - scores['rejected']
            diffs.append(d)
            if scores['chosen'] > scores['rejected']:
                chosen_wins += 1
            elif scores['rejected'] > scores['chosen']:
                rejected_wins += 1
            else:
                ties += 1

    chosen_scores = [r['score'] for r in scored_answers if r['label'] == 'chosen']
    rejected_scores = [r['score'] for r in scored_answers if r['label'] == 'rejected']

    avg_chosen = sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0.0
    avg_rejected = sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0.0
    avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
    chosen_win_rate = chosen_wins / total * 100 if total else 0.0

    lines = [
        "=" * 70,
        "POINTWISE SCORE COMPARISON",
        "=" * 70,
        f"Model:       {model_short}",
        f"Subset:      {subset_short}",
        f"Comparisons: {total}",
        "",
        f"Chosen win rate:   {chosen_win_rate:.2f}% ({chosen_wins}/{total})",
        f"Rejected wins:     {rejected_wins}  Ties: {ties}",
        "",
        f"Avg chosen score:   {avg_chosen:.4f}",
        f"Avg rejected score: {avg_rejected:.4f}",
        f"Avg diff (chosen - rejected): {avg_diff:.4f}",
        "=" * 70,
    ]

    print("\n" + "\n".join(lines))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\nSummary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Discriminative reward model (HuggingFace name or local path)")
    parser.add_argument("--subset", type=str, required=True,
                        choices=list(SUBSET_SHORT.keys()),
                        help="Dataset subset name")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for scoring (default: 8)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g. '0,1')")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    subset_short = SUBSET_SHORT[args.subset]
    model_short = os.path.basename(args.model.rstrip("/"))

    print("=" * 70)
    print("POINTWISE EVALUATION")
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {HF_DATASET} / {args.subset} / {args.split}")
    print("=" * 70)

    dataset = load_dataset(HF_DATASET, args.subset, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    model, tokenizer = load_model(args.model)

    prompts = prepare_scoring_inputs(dataset)
    scored = score_answers(model, tokenizer, prompts, batch_size=args.batch_size)

    output_file = f"evaluation/results/{subset_short}_{model_short}_pointwise.txt"
    compare_scores(scored, subset_short, model_short, output_file)


if __name__ == "__main__":
    main()
