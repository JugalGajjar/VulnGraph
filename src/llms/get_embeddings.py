"""
Generate embeddings for Java code snippets using multiple LLMs.
Saves embeddings and labels to a .npz file for each model.

Data format: id, code, cfg, label

Example Usage:
    python src/llms/get_embeddings.py --dataset data/parquet/cleaned_data_with_cfg.parquet \
        --out_dir data/embeddings \
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
    print(f"[INFO] Loading model & tokenizer: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    print(f"[INFO] Model loaded on device: {device}")
    return tokenizer, model


def embed_batch(tokenizer, model, texts: List[str], device: torch.device, max_length: int):
    """
    Tokenize and run model to extract embeddings.
    Uses mean pooling over hidden states (excluding padding).
    """
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / lengths
        return pooled.cpu().numpy()


def generate_embeddings_for_models(data_path: str, out_dir: str, models: List[str], batch_size: int, max_length: int):
    device = get_device()
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading dataset: {data_path} ...")
    df = pd.read_parquet(data_path)
    assert "code" in df.columns and "label" in df.columns, "Dataset must contain 'code' and 'label' columns."
    print(f"[INFO] Total samples in dataset: {len(df)}")

    texts = df["code"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    ids = df["id"].tolist()

    os.makedirs(out_dir, exist_ok=True)

    for model_name in models:
        print(f"\n[INFO] Generating embeddings for model: {model_name} ...")
        tokenizer, model = load_model_and_tokenizer(model_name, device)
        all_embs = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i:i + batch_size]
            emb = embed_batch(tokenizer, model, batch_texts, device, max_length=max_length)
            all_embs.append(emb)

        all_embs = np.vstack(all_embs)
        all_labels = np.array(labels, dtype=np.int64)
        all_ids = np.array(ids)

        print(f"[INFO] Embedding dimensions for {model_name}: {all_embs.shape} (num_samples, feature_size)")

        model_safe_name = model_name.split("/")[-1].replace("-", "_")
        out_path = os.path.join(out_dir, f"{model_safe_name}_embeddings.npz")
        np.savez_compressed(out_path, embeddings=all_embs, labels=all_labels, ids=all_ids)
        print(f"[INFO] Saved embeddings to {out_path}")


def test_dummy_code():
    dummy_file = input("Enter path to dummy.java file for testing: ").strip()
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
    parser.add_argument("--dataset", type=str, required=True, help="Data file with columns: id, code, cfg, label")
    parser.add_argument("--out_dir", type=str, default="data/embeddings/", help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--test", action="store_true", help="Run dummy test mode")
    args = parser.parse_args()

    if args.test:
        test_dummy_code()
    else:
        selected_models = [
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "stabilityai/stable-code-instruct-3b",
            "microsoft/Phi-3.5-mini-instruct"
        ]

        generate_embeddings_for_models(args.dataset, args.out_dir, selected_models, args.batch_size, args.max_length)
