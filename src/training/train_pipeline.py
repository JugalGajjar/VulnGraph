"""
Train a fusion model using precomputed graph and LLM embeddings (.npz)
and optional CFG info in a parquet for Laplacian regularization.

Loss = BCE + lambda_nce * InfoNCE + lambda_lap * Laplacian

Saves best checkpoint on validation F1.
"""

import argparse
import os
import json
import ast
import numpy as np
from collections import OrderedDict
from typing import Dict, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel

from src.graph_llm_integration.norm_and_project import l2_normalize_np, Projector
from src.graph_llm_integration.fusion_module import ConcatFusion, TwoWayGatingFusion, QKVCrossAttentionFusion
from src.graph_llm_integration.mlp import BinaryMLP


# Device utils
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def drop_nan_embeddings(embeddings, labels, node_ids):
    nan_mask = ~np.isnan(embeddings).any(axis=1)
    clean_embeddings = embeddings[nan_mask]
    clean_labels = [label for i, label in enumerate(labels) if nan_mask[i]]
    clean_node_ids = [node_id for i, node_id in enumerate(node_ids) if nan_mask[i]]
    return clean_embeddings, np.array(clean_labels), np.array(clean_node_ids)


# Load + align npz embeddings
def load_npz_align(graph_npz: str, llm_npz: str, debug=False, debug_size=50, seed=42):
    g = np.load(graph_npz)
    l = np.load(llm_npz)

    g_ids, g_labels, g_embs = g["ids"], g["labels"], g["embeddings"]
    l_ids, l_labels, l_embs = l["ids"], l.get("labels", None), l["embeddings"]

    # --- Drop NaN LLM embeddings before alignment ---
    l_embs, l_labels, l_ids = drop_nan_embeddings(l_embs, l_labels if l_labels is not None else [0]*len(l_ids), l_ids)

    idx_map = {int(id_): i for i, id_ in enumerate(l_ids)}

    # If debug, preselect subset of graph IDs stratified on labels
    if debug:
        from sklearn.model_selection import train_test_split
        selected_ids, _, _, _ = train_test_split(
            g_ids, g_labels,
            train_size=min(debug_size, len(g_labels)),
            stratify=g_labels,
            random_state=seed
        )
        selected_set = set(int(x) for x in selected_ids)
    else:
        selected_set = None

    aligned_graph_embs, aligned_llm_embs, aligned_labels, aligned_ids = [], [], [], []
    for i, gid in enumerate(g_ids):
        gid_i = int(gid)
        if gid_i not in idx_map:
            continue
        if selected_set is not None and gid_i not in selected_set:
            continue

        aligned_ids.append(gid_i)
        aligned_graph_embs.append(g_embs[i])
        aligned_llm_embs.append(l_embs[idx_map[gid_i]])
        aligned_labels.append(int(g_labels[i]))

    if len(aligned_ids) == 0:
        raise RuntimeError("No aligned IDs between graph and LLM npz files!")

    return (np.stack(aligned_graph_embs),
            np.stack(aligned_llm_embs),
            np.array(aligned_labels, dtype=np.int64),
            np.array(aligned_ids, dtype=np.int64))


# Losses
def info_nce_loss(h_g: torch.Tensor, h_l: torch.Tensor, tau: float = 0.07):
    h_g = F.normalize(h_g, p=2, dim=1)
    h_l = F.normalize(h_l, p=2, dim=1)
    logits = torch.matmul(h_g, h_l.T) / tau
    targets = torch.arange(h_g.size(0), device=h_g.device)
    loss_g2l = F.cross_entropy(logits, targets)
    loss_l2g = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_g2l + loss_l2g)


# CFG + Laplacian
def parse_cfg(cfg_str_or_obj):
    if isinstance(cfg_str_or_obj, dict):
        return cfg_str_or_obj
    try:
        return json.loads(cfg_str_or_obj)
    except Exception:
        try:
            return ast.literal_eval(cfg_str_or_obj)
        except Exception:
            return None


class NodeEmbedCache:
    """
    Memory-friendly cache of node embeddings with LRU eviction.
    """
    def __init__(self, parquet_df, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 device=None, batch_size=64, max_cache=100):
        self.df_map = {int(r["id"]): r["cfg"] for _, r in parquet_df.iterrows()}
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.batch_size = batch_size
        self.max_cache = max_cache

    @torch.no_grad()
    def embed_node_texts(self, texts: List[str]) -> torch.Tensor:
        out_chunks = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            res = self.model(**enc)
            last = res.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out_chunks.append(pooled.cpu())
        return torch.cat(out_chunks, dim=0)

    def get_node_embeddings(self, gid: int) -> torch.Tensor:
        if gid in self.cache:
            self.cache.move_to_end(gid)
            return self.cache[gid]
        cfg = self.df_map.get(int(gid))
        parsed = parse_cfg(cfg) if cfg is not None else None
        if parsed is None or "nodes" not in parsed or len(parsed["nodes"]) == 0:
            emb = torch.zeros((1, self.model.config.hidden_size), dtype=torch.float32)
        else:
            texts = [str(n.get("label", "")) for n in parsed["nodes"]]
            emb = self.embed_node_texts(texts)
        # LRU eviction
        if len(self.cache) >= self.max_cache:
            self.cache.popitem(last=False)
        self.cache[gid] = emb
        return emb


def compute_laplacian_term(node_cache: NodeEmbedCache, ids_batch: List[int], device: torch.device):
    total = 0.0
    count = 0
    for gid in ids_batch:
        parsed = parse_cfg(node_cache.df_map.get(int(gid)))
        if parsed is None or "edges" not in parsed:
            continue
        edges = parsed["edges"]
        if len(edges) == 0:
            continue
        node_emb = node_cache.get_node_embeddings(gid).to(device)
        u = torch.tensor([e["source"] for e in edges if "source" in e and "target" in e], device=device)
        v = torch.tensor([e["target"] for e in edges if "source" in e and "target" in e], device=device)
        if u.numel() == 0 or v.numel() == 0:
            continue
        diffs = node_emb[u] - node_emb[v]
        s = (diffs ** 2).sum(dim=1).mean()
        total += s
        count += 1
    return (total / max(count, 1)).to(device)


# Training
def train(args):
    device = get_device()
    print("[INFO] Using device:", device)

    Xg, Xl, y, ids = load_npz_align(
        args.graph_npz, args.llm_npz,
        debug=args.debug, debug_size=args.debug_size, seed=args.seed
    )
    print(f"[INFO] Loaded aligned {len(ids)} samples (debug={args.debug})")

    # normalize
    Xg = l2_normalize_np(Xg)
    Xl = l2_normalize_np(Xl)
    print("[INFO] Normalized embeddings")

    # split
    X_temp_g, X_test_g, X_temp_l, X_test_l, y_temp, y_test, ids_temp, ids_test = train_test_split(
        Xg, Xl, y, ids, test_size=args.test_frac, random_state=args.seed, stratify=y
    )
    X_train_g, X_val_g, X_train_l, X_val_l, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp_g, X_temp_l, y_temp, ids_temp, test_size=args.val_frac, random_state=args.seed, stratify=y_temp
    )

    # datasets
    train_ds = TensorDataset(torch.from_numpy(X_train_g), torch.from_numpy(X_train_l),
                             torch.from_numpy(y_train), torch.from_numpy(ids_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_g), torch.from_numpy(X_val_l),
                           torch.from_numpy(y_val), torch.from_numpy(ids_val))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # model
    proj_g = Projector(Xg.shape[1], args.dproj, use_ln=True).to(device)
    proj_l = Projector(Xl.shape[1], args.dproj, use_ln=True).to(device)
    if args.fusion == "concat":
        fusion = ConcatFusion(2 * args.dproj, hidden=args.hidden, out_dim=args.fusion_out, dropout=args.dropout)
    elif args.fusion == "two_way":
        fusion = TwoWayGatingFusion(args.dproj, hidden=args.hidden, out_dim=args.fusion_out)
    else:
        fusion = QKVCrossAttentionFusion(args.dproj, out_dim=args.fusion_out)
    fusion = fusion.to(device)
    classifier = BinaryMLP(args.fusion_out, hidden=args.class_hidden, dropout=args.dropout).to(device)

    # Laplacian
    node_cache = None
    if args.lambda_lap > 0 and args.parquet:
        import pandas as pd
        df = pd.read_parquet(args.parquet)
        node_cache = NodeEmbedCache(df, model_name=args.node_embed_model,
                                    device=device, batch_size=args.node_batch)

    opt = torch.optim.AdamW(list(proj_g.parameters()) + list(proj_l.parameters())
                            + list(fusion.parameters()) + list(classifier.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1, patience = -1, 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        classifier.train(); fusion.train(); proj_g.train(); proj_l.train()
        running_loss = 0.0
        for bg, bl, by, bid in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            bg, bl, by = bg.to(device).float(), bl.to(device).float(), by.to(device).float()
            bid_list = [int(x.item()) for x in bid]

            h_g, h_l = proj_g(bg), proj_l(bl)
            fused = fusion(h_g, h_l) if args.fusion != "concat" else fusion(torch.cat([h_g, h_l], dim=1))
            logits = classifier(fused)

            loss_cls = F.binary_cross_entropy_with_logits(logits, by)
            loss_nce = info_nce_loss(h_g, h_l, tau=args.tau) if args.lambda_nce > 0 else torch.tensor(0., device=device)
            loss_lap = compute_laplacian_term(node_cache, bid_list, device) if (args.lambda_lap > 0 and node_cache) else torch.tensor(0., device=device)
            loss = loss_cls + args.lambda_nce * loss_nce + args.lambda_lap * loss_lap

            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item() * bg.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # --- Validate ---
        classifier.eval(); fusion.eval(); proj_g.eval(); proj_l.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for bg, bl, by, bid in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                bg, bl = bg.to(device).float(), bl.to(device).float()
                h_g, h_l = proj_g(bg), proj_l(bl)
                fused = fusion(h_g, h_l) if args.fusion != "concat" else fusion(torch.cat([h_g, h_l], dim=1))
                probs = torch.sigmoid(classifier(fused)).cpu().numpy()
                preds = (probs >= 0.5).astype(int).tolist()
                val_preds.extend(preds); val_gts.extend(by.numpy().tolist())

        val_f1 = f1_score(val_gts, val_preds)
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.6f}, val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, patience = val_f1, 0
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "proj_g": proj_g.state_dict(),
                "proj_l": proj_l.state_dict(),
                "fusion": fusion.state_dict(),
                "classifier": classifier.state_dict(),
                "args": vars(args)
            }, args.out)
            print(f"[INFO] Saved best checkpoint to {args.out}")
        else:
            patience += 1
            if patience >= args.patience:
                print("[EARLY STOPPING]")
                break

    print("[TRAIN] Best val F1:", best_val_f1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_npz", required=True)
    parser.add_argument("--llm_npz", required=True)
    parser.add_argument("--parquet", required=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fusion", choices=["concat", "two_way", "qkv"], default="concat")
    parser.add_argument("--dproj", type=int, default=128)
    parser.add_argument("--fusion_out", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--class_hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_nce", type=float, default=0.1)
    parser.add_argument("--lambda_lap", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--node_embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--node_batch", type=int, default=64)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (small stratified subset)")
    parser.add_argument("--debug_size", type=int, default=50, help="Number of samples for debug mode")
    args = parser.parse_args()
    
    train(args)
