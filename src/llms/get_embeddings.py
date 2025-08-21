"""
Generate embeddings for Java code snippets using an LLM.
Saves embeddings and labels to a .npz file for training/eval.

Example Usage:
    python src/llms/get_embeddings.py \
        --dataset data/raw/java_snippets.csv \
        --out data/embeddings/qwen2.5_train_embeddings.npz \
        --model Qwen/Qwen2.5-Coder-3B-Instruct \
        --batch_size 8 \
        --max_length 1024

Test Usage (sanity check with dummy.java):
    python src/llms/get_embeddings.py --test
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def embed_batch(tokenizer, model, texts: List[str], device: torch.device, max_length: int):
    """
    Tokenize and run model to extract embeddings.
    Uses mean pooling over hidden states (excluding padding).
    """
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # request hidden states
            return_dict=True
        )

        # Use the last hidden layer
        last_hidden = outputs.hidden_states[-1]   # (batch, seq_len, hidden_dim)

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1)       # (batch, seq_len, 1)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / lengths
        return pooled.cpu().numpy()



def generate_embeddings(csv_path: str, out_path: str, model_name: str, batch_size: int, max_length: int):
    device = get_device()
    print(f"[INFO] Using device: {device}")

    tokenizer, model = load_model_and_tokenizer(model_name, device)
    print(f"[INFO] Loaded model & tokenizer: {model_name}")

    df = pd.read_csv(csv_path)
    assert "code" in df.columns and "label" in df.columns, "CSV must contain 'code' and 'label' columns."

    all_embs, all_labels, all_ids = [], [], []
    texts = df["code"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    ids = df.get("id", pd.Series(range(len(df)))).tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        emb = embed_batch(tokenizer, model, batch_texts, device, max_length=max_length)
        all_embs.append(emb)
        all_labels.extend(batch_labels)
        all_ids.extend(batch_ids)

    all_embs = np.vstack(all_embs)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_ids = np.array(all_ids)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, embeddings=all_embs, labels=all_labels, ids=all_ids)
    print(f"[INFO] Saved embeddings to {out_path} (shape={all_embs.shape})")


def test_dummy_code():
    """
    Sanity check: load dummy.java and generate embeddings from all models.
    """
    dummy_file = "data/raw/HelloWorld.java"
    assert os.path.exists(dummy_file), f"Missing test file: {dummy_file}"

    dummy_code = open(dummy_file).read()
    models = [
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "stabilityai/stable-code-instruct-3b",
        "microsoft/Phi-3.5-mini-instruct"
    ]

    device = get_device()
    print(f"[TEST] Running dummy test on {device}")

    for model_name in models:
        print(f"\n--- Testing {model_name} ---")
        tokenizer, model = load_model_and_tokenizer(model_name, device)
        emb = embed_batch(tokenizer, model, [dummy_code], device, max_length=512)
        print(f"Embedding shape: {emb.shape}, dtype: {emb.dtype}")
        print(f"First 5 values: {emb[0][:5].tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="CSV file with columns: id, code, label")
    parser.add_argument("--out", type=str, help="Output .npz path")
    parser.add_argument("--model", type=str, help="Hugging Face model name or local path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--test", action="store_true", help="Run dummy test mode")
    args = parser.parse_args()

    if args.test:
        test_dummy_code()
    else:
        assert args.dataset and args.out and args.model, "Must provide --dataset, --out, and --model"
        generate_embeddings(args.dataset, args.out, args.model, args.batch_size, args.max_length)
