"""
Unified reward model server for PPO training.

Supports two model types via --model-type:
  discriminative  Classification-head models (e.g. Skywork, InternLM)
  generative      Instruction-tuned LLMs that output a numeric score

Usage:
  python reward_server.py --model-name <HF_MODEL> --model-type <TYPE> --gpu-ids <IDS>
"""

import os
import re
import argparse
import uvicorn
import torch
from fastapi import FastAPI

app = FastAPI()

# Populated at startup via command-line args
_model_name: str = None
_model_type: str = None
_gpu_ids: str = None
_tensor_parallel_size: int = 1
_score_template_path: str = None

# Runtime state
_model = None
_tokenizer = None
_score_template: str = None
_device = None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    print("[Server] Initializing reward model...")
    _initialize_model()
    print("[Server] Ready to accept requests!")


def _initialize_model():
    global _model, _tokenizer, _score_template, _device

    if _gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_ids
        _device = torch.device("cuda:0")
        print(f"[Server] Using GPU(s): {_gpu_ids}")
    else:
        _device = torch.device("cpu")

    if _model_type == "generative":
        _load_generative()
    else:  # discriminative
        _load_discriminative()

    _score_template = _load_template(_score_template_path)
    print(f"[Server] Score template loaded from {_score_template_path}")


def _load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_generative():
    """Load a generative reward model via vLLM."""
    global _model, _tokenizer
    from vllm import LLM
    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(_model_name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # vLLM may modify CUDA_VISIBLE_DEVICES internally — save and restore
    saved_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_ids

    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    gpu_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.65"))

    _model = LLM(
        model=_model_name,
        tensor_parallel_size=_tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
    )
    print(f"[Server] Generative model loaded (TP={_tensor_parallel_size})")

    if saved_cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]


def _load_discriminative():
    """Load a discriminative reward model via Transformers."""
    global _model, _tokenizer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _tokenizer = AutoTokenizer.from_pretrained(_model_name, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = _device if torch.cuda.is_available() else None

    _model = AutoModelForSequenceClassification.from_pretrained(
        _model_name, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    _model.eval()
    print(f"[Server] Discriminative model loaded on {_device}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/score")
async def score_response(input_dict: dict):
    response_text = input_dict["response_text"]
    query = input_dict["query"]
    profile = input_dict["profile"]

    if _model_type == "generative":
        score = _score_generative(query, profile, response_text)
    else:
        score = _score_discriminative(query, profile, response_text)

    return {"score": score}


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _score_generative(query: str, profile: str, response_text: str) -> float:
    from vllm import SamplingParams

    prompt_text = _score_template.replace("{question}", query)
    prompt_text = prompt_text.replace("{profile}", profile)
    prompt_text = prompt_text.replace("{answer}", response_text)

    messages = [
        {"role": "system", "content": "You are a fair and insightful judge."},
        {"role": "user", "content": prompt_text},
    ]
    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    params = SamplingParams(max_tokens=10, temperature=0.0, top_p=1.0)
    output_text = _model.generate([prompt], params)[0].outputs[0].text.strip()

    try:
        return max(0.0, min(1.0, float(output_text)))
    except ValueError:
        nums = re.findall(r"0?\.\d+|[01]\.?\d*", output_text)
        return max(0.0, min(1.0, float(nums[0]))) if nums else 0.5


def _score_discriminative(query: str, profile: str, response_text: str) -> float:
    conv = [
        {"role": "user", "content": query + profile},
        {"role": "assistant", "content": response_text},
    ]
    conv_formatted = _tokenizer.apply_chat_template(conv, tokenize=False)
    inputs = _tokenizer(
        conv_formatted, return_tensors="pt", padding=True, truncation=True, max_length=8192
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits
        if logits.shape[-1] == 1:
            score = logits[0, 0].item()
        else:
            score = torch.softmax(logits, dim=-1)[0, -1].item()

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Reward Model Server")
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=["discriminative", "generative"],
        help=(
            "discriminative: classification-head reward model (e.g. Skywork, InternLM); "
            "generative: LLM that outputs a numeric score (e.g. Qwen, Llama)"
        ),
    )
    parser.add_argument("--gpu-ids", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g. '0' or '4,5')")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor parallel size (generative models only)")
    parser.add_argument("--score-template", type=str, default="score.txt",
                        help="Path to the scoring prompt template")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _model_name = args.model_name
    _model_type = args.model_type
    _gpu_ids = args.gpu_ids
    _tensor_parallel_size = args.tensor_parallel_size
    _score_template_path = args.score_template

    print(f"[Config] Model:   {_model_name}")
    print(f"[Config] Type:    {_model_type}")
    print(f"[Config] GPU IDs: {_gpu_ids}")

    uvicorn.run(app, host=args.host, port=args.port)
