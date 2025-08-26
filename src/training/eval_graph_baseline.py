"""
Evaluate a saved classifier on graph embeddings.

Usage:
    python src/training/eval_graph_baseline.py \
        --embeddings data/embeddings/graph/graph_embeddings.npz \
        --model models/graph_classifier.pt \
        --scaler models/graph_classifier.scaler.pkl
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import joblib

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scaler", required=True)
    args = ap.parse_args()

    # Load
    pack = np.load(args.embeddings)
    X = pack["embeddings"].astype(np.float32)
    y = pack["labels"].astype(np.int64)

    # Scaler & model
    scaler: StandardScaler = joblib.load(args.scaler)
    X = scaler.transform(X)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    checkpoint = torch.load(args.model, map_location=device)
    in_dim = checkpoint["in_dim"]

    model = MLPClassifier(in_dim=in_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y, preds)

    print("=== Evaluation ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

if __name__ == "__main__":
    main()
