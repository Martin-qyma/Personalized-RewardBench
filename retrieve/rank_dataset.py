import os
import torch
from transformers import AutoModel, AutoTokenizer
import json
import tqdm
import argparse
import datasets

def batchify(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, k, batch_size, device="cuda:0"):
    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to(device)
    output_query = contriver(**query_tokens)
    output_query = mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
    scores = []
    batched_corpus = batchify(corpus, batch_size)
    for batch in batched_corpus:
        tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs_batch = contriver(**tokens_batch)
        outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
        temp_scores = output_query.squeeze() @ outputs_batch.T
        scores.extend(temp_scores.tolist())
    topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
    return [profile[m] for m in topk_indices.tolist()]

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='facebook/contriever-msmarco')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--split", type=str, default="test")
parser.add_argument(
        "--dataset_config",
        type=str,
        default="Lifestyle_and_Personal_Development",
        help="choose from 'Lifestyle_and_Personal_Development', 'Society_and_Culture', 'Art_and_Entertainment'"
)
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (default: 0)")


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = datasets.load_dataset("alireza7/LaMP-QA", args.dataset_config)
    dataset = dataset[args.split]

    device = f"cuda:{args.gpu}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    contriver = AutoModel.from_pretrained(args.model_name).to(device)
    contriver.eval()

    dataset_new = []
    for data in tqdm.tqdm(dataset):
        profile = data['profile']
        corpus = [x['text'] for x in profile]
        query = data['question']
        ids = [x['id'] for x in profile]
       
        ranked_profile = retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, len(profile), args.batch_size, device)
        ranked_profile = [{'id': ids[i], 'category': ranked_profile[i]['category'], 'text': ranked_profile[i]['text']} for i in range(len(ranked_profile))]
        data['profile'] = ranked_profile
        dataset_new.append(data)

    os.makedirs("./data", exist_ok=True)
    with open(f"./data/{args.dataset_config}_{args.split}.jsonl", "w") as file:
        for item in dataset_new:
            file.write(json.dumps(item) + "\n")