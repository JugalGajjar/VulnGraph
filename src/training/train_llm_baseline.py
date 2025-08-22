"""
Train a simple MLP classifier on precomputed LLM embeddings.

Example Usage:
    python src/training/train_baseline.py \
        --embeddings data/embeddings/qwen2.5_train_embeddings.npz \
        --out_model models/mlp_qwen2_5.pth \
        --epochs 20 \
        --batch_size 64
"""

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_embeddings(npz_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(npz_path)
    embeddings = data["embeddings"]  # (N, D)
    labels = data["labels"]          # (N,)
    return torch.from_numpy(embeddings).float(), torch.from_numpy(labels).long()


def train(model, device, train_loader, val_loader, epochs, lr, out_model_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[EPOCH {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # ---- Save best w/ metadata ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "in_dim": model.net[0].in_features,
                "hidden_dim": model.net[0].out_features,
                "out_dim": model.net[-1].out_features,
                "dropout": getattr(model.net[2], "p", 0.0),
                "lr": lr,
                "epochs_trained": epoch,
            }
            os.makedirs(os.path.dirname(out_model_path) or ".", exist_ok=True)
            torch.save(ckpt, out_model_path)
            print(f"[INFO] Saved best model with metadata to {out_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to .npz embeddings (embeddings, labels, ids)")
    parser.add_argument("--out_model", required=True, help="Path to save trained MLP (pth)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    X, y = load_embeddings(args.embeddings)
    dataset = TensorDataset(X, y)

    # Split train/val
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = X.shape[1]
    out_dim = int(y.max().item() + 1)

    model = MLPClassifier(in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim).to(device)

    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    train(model, device, train_loader, val_loader, args.epochs, args.lr, args.out_model)

    print("[INFO] Training complete.")
