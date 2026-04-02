import os
import json
import argparse
from vllm import LLM, SamplingParams
from typing import List, Dict
from tqdm import tqdm


subset_map = {
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Art_and_Entertainment": "Art",
    "Society_and_Culture": "Society",
}


def load_template(file_path: str) -> str:
    """Load template from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        template = f.read().strip()
    return template


def format_profile(profile_list: List[Dict], k) -> str:
    """Format the profile list into a readable string."""
    formatted = []
    for i, item in enumerate(profile_list[:k], 1):
        formatted.append(f"Post {i} (Category: {item['category']}):\n{item['text']}")
    return "\n".join(formatted)


def load_model(model_name: str, tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1) -> LLM:
    """Load model with vLLM using model parallelism.

    Args:
        model_name: Name or path of the model
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of GPUs for pipeline parallelism
    """
    total_gpus = tensor_parallel_size * pipeline_parallel_size
    print(f"Loading model {model_name} with vLLM:")
    print(f"  - Tensor Parallel Size: {tensor_parallel_size}")
    print(f"  - Pipeline Parallel Size: {pipeline_parallel_size}")
    print(f"  - Total GPUs: {total_gpus}")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )

    return llm



def format_plan(rubric_aspects: List[Dict]) -> str:
    """Format the rubric aspects into a readable string."""
    formatted = []
    for aspect in rubric_aspects:
        formatted.append(f"{aspect['aspect']}")
    return ", ".join(formatted)


def prepare_generation_prompts(dataset, gen_template: str, num_samples: int = 1, k: int = 10) -> List[Dict]:
    """Prepare prompts for answer generation.

    Args:
        dataset: Dataset with query and profile fields
        gen_template: Generation template with {question} and {profile} placeholders
        num_samples: Number of answers to generate per query

    Returns:
        List of dicts with id, query, profile, prompt, and sample_idx
    """
    prompts = []

    for example in dataset:
        query = example['question']
        aspects = format_plan(example['rubric_aspects'])
        # Format profile from list of dicts
        profile_text = format_profile(example['profile'], k)

        # Fill in the template
        prompt = gen_template.replace("{question}", query)
        prompt = prompt.replace("{profile}", profile_text)

        # Generate N samples for each query
        for sample_idx in range(num_samples):
            prompts.append({
                'id': example['id'],
                'query': query,
                'profile': profile_text,
                'prompt': prompt,
                'sample_idx': sample_idx,
                'aspects': aspects,
            })
    return prompts


def generate_n_answers(llm: LLM, prompts: List[Dict], batch_size: int = 8, temperature: float = 0.8) -> List[Dict]:
    """Generate N answers for each (query, profile) pair.

    Args:
        llm: vLLM model instance
        prompts: List of prompt dicts with pre-filled templates
        batch_size: Batch size for generation
        temperature: Sampling temperature

    Returns:
        List of dicts with id, query, profile, sample_idx, and generated_response
    """
    print(f"Generating {len(prompts)} answers...")

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=512
    )

    results = []

    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating answers"):
        batch = prompts[i:i + batch_size]

        # Prepare messages for batch
        messages_batch = []
        for item in batch:
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to generate personalized responses to user questions."},
                {"role": "user", "content": item['prompt']}
            ]
            messages_batch.append(messages)

        # Generate responses for the batch
        outputs = llm.chat(
            messages=messages_batch,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        # Extract and store results
        for j, output in enumerate(outputs):
            result = {
                'id': batch[j]['id'],
                'query': batch[j]['query'],
                'profile': batch[j]['profile'],
                'sample_idx': batch[j]['sample_idx'],
                'aspects': batch[j]['aspects'],
                'generated_response': output.outputs[0].text
            }
            results.append(result)
        
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        type=str,
        default="Lifestyle_and_Personal_Development",
        help="choose from ['Art_and_Entertainment', 'Lifestyle_and_Personal_Development', 'Society_and_Culture']"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of answers to generate per query (N in Best-of-N, default: 16)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for answer generation (default: 1.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation and evaluation (default: 8)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for pipeline parallelism (default: 1)"
    )
    parser.add_argument(
        "--gen-template",
        type=str,
        default="prompts/sample.txt",
        help="Path to generation template (default: prompts/sample.txt)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model to use for generation (default: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of profile posts to include per example (default: 10)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3'). If not specified, uses all available GPUs."
    )
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES if GPU selection is specified
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")

    # Load templates
    print("Loading templates...")
    gen_template = load_template(args.gen_template)

    # Load dataset from JSONL file
    data_path = f"data/{args.subset}_{args.split}.jsonl"
    print(f"Loading dataset from {data_path}...")
    dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    print(f"Dataset loaded: {len(dataset)} examples")

    # Generate N answers for each (query, profile)
    print("=" * 70)
    print(f"STEP 1: GENERATING N={args.num_samples} ANSWERS")
    print("=" * 70)
    generation_prompts = prepare_generation_prompts(dataset, gen_template, num_samples=args.num_samples, k=args.k)

    llm = load_model(args.model, args.tensor_parallel_size, args.pipeline_parallel_size)
    answers = generate_n_answers(llm, generation_prompts, batch_size=args.batch_size, temperature=args.temperature)

    subset_short_name = subset_map[args.subset]
    output_file = f"BoN/data/{subset_short_name}_samples.jsonl"
    os.makedirs("BoN/data", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for answer in answers:
            f.write(json.dumps(answer) + '\n')
    print(f"Generated answers saved to {output_file}")

if __name__ == "__main__":
    main()