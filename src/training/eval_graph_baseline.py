"""
Evaluate a saved classifier on the held-out test set of graph embeddings.

Example Usage:
    python src/training/eval_graph_baseline.py \
        --embeddings data/embeddings/graph_embeddings_node2vec.npz \
        --model models/graph_classifier_node2vec.pt \
        --scaler models/graph_classifier_node2vec.scaler.pkl \
        --val_frac 0.2 \
        --test_frac 0.1
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


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
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load dataset
    pack = np.load(args.embeddings)
    X = pack["embeddings"].astype(np.float32)
    y = pack["labels"].astype(np.int64)

    # Test split (same as train script)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_frac, random_state=args.seed, stratify=y
    )

    print(f"[INFO] Using test set of size {len(X_test)}")

    unique, counts = np.unique(y_test, return_counts=True)
    print("[INFO] Label distribution in test set:")
    for u, c in zip(unique, counts):
        print(f"Label {u}: {c}")

    # Load scaler and transform test set
    scaler: StandardScaler = joblib.load(args.scaler)
    X_test = scaler.transform(X_test)

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    in_dim = checkpoint["in_dim"]

    model = MLPClassifier(in_dim=in_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run inference
    X_t = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_test, preds)

    # Print report
    print("=== Test Set Evaluation ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"AUC      : {auc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Bar chart for metrics
    metrics_dict = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }
    plt.figure(figsize=(6, 5))
    sns.barplot(x=list(metrics_dict.keys()), y=list(metrics_dict.values()), palette="viridis")
    plt.ylim(0, 1)
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.show()


if __name__ == "__main__":
    main()
