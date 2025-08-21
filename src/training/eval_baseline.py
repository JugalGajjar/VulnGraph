"""
Evaluate the trained MLP on test embeddings.

Example Usage:
    python src/training/eval_baseline.py \
        --embeddings data/embeddings/qwen2.5_test_embeddings.npz \
        --model models/mlp_qwen2_5.pth \
        --out results/qwen2_5_eval.json
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def load_embeddings(npz_path: str):
    data = np.load(npz_path)
    X = data["embeddings"]
    y = data["labels"]
    ids = data.get("ids")
    return X, y, ids


def evaluate(model_path: str, embeddings_path: str, batch_size: int = 256):
    device = get_device()

    # Load test embeddings
    X, y, ids = load_embeddings(embeddings_path)
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()

    # Load model + metadata
    checkpoint = torch.load(model_path, map_location=device)
    in_dim = checkpoint["in_dim"]
    hidden_dim = checkpoint.get("hidden_dim", 512)
    out_dim = checkpoint["out_dim"]
    dropout = checkpoint.get("dropout", 0.2)

    # Rebuild model exactly as trained
    model = MLPClassifier(in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # Sanity check: ensure test embeddings match expected dim
    assert X_t.shape[1] == in_dim, (
        f"Embedding dimension mismatch: test has {X_t.shape[1]}, model expects {in_dim}. "
        f"Did you mix embeddings from different LLMs?"
    )

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)

    preds, probs, gts = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            preds.extend(pred.tolist())
            if out_dim == 2:
                probs.extend(prob[:, 1].tolist())  # positive class prob
            else:
                probs.extend(prob.tolist())
            gts.extend(yb.numpy().tolist())

    # Metrics
    acc = accuracy_score(gts, preds)
    avg = "binary" if out_dim == 2 else "macro"
    prec, rec, f1, _ = precision_recall_fscore_support(gts, preds, average=avg, zero_division=0)
    cm = confusion_matrix(gts, preds).tolist()

    auc = None
    try:
        if out_dim == 2:
            auc = roc_auc_score(gts, probs)
        else:
            auc = "N/A for multi-class (use macro-avg or one-vs-rest AUC)."
    except Exception:
        auc = "Error computing AUC"

    metrics = {
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
    parser.add_argument("--embeddings", required=True, help="Path to test .npz embeddings")
    parser.add_argument("--model", required=True, help="Trained MLP model path (pth)")
    parser.add_argument("--out", required=False, help="Output JSON path for metrics", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    metrics = evaluate(args.model, args.embeddings, args.batch_size)
    print(json.dumps(metrics, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved metrics to {args.out}")
