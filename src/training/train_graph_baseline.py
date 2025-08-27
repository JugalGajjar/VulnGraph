"""
Train a classifier on precomputed graph embeddings with early stopping.
Keeps a hold-out test set unused for final evaluation.

Example Usage:
    python src/training/train_graph_baseline.py \
        --embeddings data/embeddings/graph_embeddings_node2vec.npz \
        --out_model models/graph_classifier_node2vec.pt \
        --scaler_out models/graph_classifier_node2vec.scaler.pkl \
        --val_frac 0.2 \
        --test_frac 0.1
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import joblib


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)  # binary classification
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help=".npz with arrays: embeddings, labels, ids")
    ap.add_argument("--out_model", default="models/graph_classifier_gcb+sage.pt")
    ap.add_argument("--scaler_out", default="models/graph_classifier_gcb+sage.scaler.pkl")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    # Load embeddings
    pack = np.load(args.embeddings)
    X = pack["embeddings"].astype(np.float32)  # [N, D]
    y = pack["labels"].astype(np.int64)        # [N]

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_frac, random_state=args.seed, stratify=y
    )

    # Second split: train vs val (from train+val set)
    val_size = args.val_frac / (1 - args.test_frac)  # adjust fraction relative to remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=args.seed, stratify=y_temp
    )

    print(f"Dataset split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} (held out)")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Torch tensors
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    # Model
    model = MLPClassifier(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()

    best_val_f1 = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_train_t)
        loss = crit(logits, y_train_t)
        loss.backward()
        opt.step()

        # Eval
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            probs_val = torch.sigmoid(logits_val).cpu().numpy()
            preds_val = (probs_val >= 0.5).astype(int).ravel()
            yv = y_val_t.cpu().numpy().astype(int)

            acc = accuracy_score(yv, preds_val)
            f1 = f1_score(yv, preds_val)
            prec = precision_score(yv, preds_val, zero_division=0)
            rec = recall_score(yv, preds_val, zero_division=0)
            try:
                auc = roc_auc_score(yv, probs_val)
            except ValueError:
                auc = float("nan")

        print(
            f"[Epoch {epoch:03d}] loss={loss.item():.4f} | "
            f"val: acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} auc={auc:.4f}"
        )

        # Early stopping check
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_state = {
                "model_state_dict": model.state_dict(),
                "in_dim": X.shape[1]
            }
            patience_counter = 0  # reset
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                break

    # Save best
    torch.save(best_state, args.out_model)
    joblib.dump(scaler, args.scaler_out)
    print(f"Saved classifier to {args.out_model}")
    print(f"Saved scaler to {args.scaler_out}")
    print(f"[INFO] Test set ({len(X_test)} samples) held out for evaluation in another script.")


if __name__ == "__main__":
    main()
