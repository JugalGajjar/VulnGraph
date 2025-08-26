"""
Prompt-based LLM evaluation on test set of code snippets.

Example Usage:
    python src/training/eval_llm_baseline.py \
        --parquet data/parquet/cleaned_data_with_cfg.parquet \
        --test_frac 0.2 \
        --batch_size 16 \
        --seed 79
"""

import argparse
import json
import os
import random
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

MODELS = [
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "stabilityai/stable-code-instruct-3b",
    "microsoft/Phi-3.5-mini-instruct",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
]

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_prompt(code_snippet):
    """
    Convert embedding back to code is not trivial.
    Assuming you have a mapping from embeddings to code snippet in X_code.
    For demonstration, we just represent embedding as a string.
    """

    prompt = f"""Classify the following code snippet as vulnerable (1) or safe (0). Output ONLY one line of Python code in this exact format:
    ```python
    result = <0 or 1>
    ```
    Do NOT include any text, markdown, or explanation.

    Code snippet:
    {code_snippet}
    """

    return prompt

def parse_result(output: str):
    """
    Extract the result integer 0 or 1 from LLM output robustly.
    Handles variations like:
    - `result = 1`
    - `result=0`
    - ```python result = 1 ```
    - Just `1` or `0` in text
    """
    try:
        output = output.strip()

        # Remove any markdown code fences
        for fence in ["```python", "```", "`"]:
            output = output.replace(fence, "")

        # Split by lines and scan for "result = <0|1>"
        for line in output.splitlines():
            line = line.strip()
            if "result" in line and "=" in line:
                value = line.split("=")[1].strip()
                if value in ("0", "1"):
                    return int(value)

        return random.choice([0, 1])

    except Exception as e:
        print(f"[ERROR] Failed to parse output : Returning 1")
        return 1


def evaluate_model(model_name, code_list, labels, device, batch_size=32):
    print(f"[INFO] Evaluating model {model_name} on {len(code_list)} examples")
    
    all_preds = []
    all_labels = labels.tolist()

    # Process in batches
    for i in tqdm(range(0, len(code_list), batch_size), desc=f"Classifying with {model_name}"):
        batch_codes = code_list[i:i+batch_size]

        # Reinitialize tokenizer and model for each batch
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()

        batch_preds = []
        for code_snippet in batch_codes:
            prompt = get_prompt(code_snippet)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False
                )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            pred = parse_result(output_text)
            batch_preds.append(pred)

        all_preds.extend(batch_preds)

        # Compute batch metrics
        batch_acc = accuracy_score(all_labels[:len(all_preds)], all_preds)
        batch_prec = precision_score(all_labels[:len(all_preds)], all_preds, zero_division=0)
        batch_rec = recall_score(all_labels[:len(all_preds)], all_preds, zero_division=0)
        batch_f1 = f1_score(all_labels[:len(all_preds)], all_preds, zero_division=0)
        print(f"[BATCH {i//batch_size + 1}] acc={batch_acc:.4f}, prec={batch_prec:.4f}, rec={batch_rec:.4f}, f1={batch_f1:.4f}")

        # Clean up after batch
        del tokenizer, model
        if device.type=="mps":
            torch.mps.empty_cache()
        elif device.type=="cuda":
            torch.cuda.empty_cache()
        print(f"[INFO] Flushed model/tokenizer and cleared GPU cache after batch {i//batch_size + 1}")

    # Final metrics over all batches
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = float("nan")

    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": auc,
        "confusion_matrix": cm
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to parquet file containing columns: id, code, label")
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for LLM evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] Using device: {device}")

    df = pd.read_parquet(args.parquet)
    codes = df["code"].tolist()
    labels = df["label"].astype(int)

    # Split data, only use the test set
    _, X_test, _, y_test = train_test_split(
        codes, labels, test_size=args.test_frac, random_state=args.seed, stratify=labels
    )

    # Sample 50 examples from test set: 40 with label=1 and 10 with label=0
    pos_indices = [i for i, label in enumerate(y_test) if label == 1]
    neg_indices = [i for i, label in enumerate(y_test) if label == 0]

    import random
    random.seed(args.seed)
    sampled_pos = random.sample(pos_indices, min(40, len(pos_indices)))
    sampled_neg = random.sample(neg_indices, min(10, len(neg_indices)))

    sampled_indices = sampled_pos + sampled_neg
    random.shuffle(sampled_indices)  # shuffle to mix pos and neg

    X_test_sampled = [X_test[i] for i in sampled_indices]
    y_test_sampled = y_test.iloc[sampled_indices]

    all_metrics = []
    for model_name in MODELS:
        metrics = evaluate_model(model_name, X_test_sampled, y_test_sampled, device, args.batch_size)
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=4))

    print("Evaluation complete.")
