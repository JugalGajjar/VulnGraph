"""
Build graph embeddings for Java CFGs with:
- Node features: GraphCodeBERT embeddings of node 'label'
- Graph encoder: GraphSAGE (PyTorch Geometric), GCN (PyTorch Geometric)
- Training objective: node-feature denoising (unsupervised)
Saves graph-level embeddings to data/embeddings/*.npz

Example Usage:
    python src/graphs/get_embeddings.py \
        --dataset data/parquet/cleaned_data_with_cfg.parquet \
        --out_dir data/embeddings \
        --epochs 10 \
        --batch_size 4
"""

import os
import ast
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import SAGEConv, global_mean_pool
# from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.nn import GATConv, global_mean_pool

# Embeddings: GraphCodeBERT
class GraphCodeBERTEmbedder:
    def __init__(self, model_name="microsoft/graphcodebert-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts, batch_size=32):
        """Embed a list of strings -> tensor [N, 768] (CLS)"""
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(self.device)
            outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]  # CLS token
            embs.append(cls.detach().cpu())
        return torch.cat(embs, dim=0)

# CFG -> PyG Data
def cfg_to_data(cfg_json: str, embedder: GraphCodeBERTEmbedder):
    """Convert a single CFG (JSON string) into a PyG Data object with x, edge_index, y(optional)"""
    if isinstance(cfg_json, dict):
        cfg = cfg_json
    else:
        try:
            # Try strict JSON first
            cfg = json.loads(cfg_json)
        except json.JSONDecodeError:
            # Fallback: interpret as Python dict string
            cfg = ast.literal_eval(cfg_json)

    # Node texts
    node_texts = [n.get("label", "") for n in cfg.get("nodes", [])]
    if len(node_texts) == 0:
        # Edge case: empty graph -> 1 dummy node
        node_texts = [""]

    # Node features via GraphCodeBERT
    x = embedder.embed_texts(node_texts)  # [num_nodes, 768]

    # Edges
    edges = cfg.get("edges", [])
    if len(edges) == 0:
        # no edges -> single-node self-loop (PyG can handle empty but SAGE benefits from some connectivity)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        src = [e["source"] for e in edges]
        dst = [e["target"] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data

# Graph Encoder (GraphSAGE)
# class GraphSAGEEncoder(nn.Module):
#     def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.1):
#         super().__init__()
#         self.conv1 = SAGEConv(in_dim, hidden_dim)
#         self.conv2 = SAGEConv(hidden_dim, hidden_dim)
#         self.conv3 = SAGEConv(hidden_dim, out_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.norm3 = nn.LayerNorm(out_dim)

#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = self.norm1(F.relu(x))
#         x = self.dropout(x)

#         x = self.conv2(x, edge_index)
#         x = self.norm2(F.relu(x))
#         x = self.dropout(x)

#         x = self.conv3(x, edge_index)
#         x = self.norm3(x)

#         # node_emb: [num_nodes_total_in_batch, out_dim]
#         node_emb = x
#         # graph_emb: [num_graphs_in_batch, out_dim]
#         graph_emb = global_mean_pool(node_emb, batch)
#         return node_emb, graph_emb

# Graph Encoder (GCN)
# class GCNEncoder(nn.Module):
#     def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, dropout=0.1):
#         super().__init__()
#         self.conv1 = GraphConv(in_dim, hidden_dim)
#         self.conv2 = GraphConv(hidden_dim, hidden_dim)
#         self.conv3 = GraphConv(hidden_dim, out_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.norm3 = nn.LayerNorm(out_dim)

#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = self.norm1(F.relu(x))
#         x = self.dropout(x)

#         x = self.conv2(x, edge_index)
#         x = self.norm2(F.relu(x))
#         x = self.dropout(x)

#         x = self.conv3(x, edge_index)
#         x = self.norm3(x)

#         node_emb = x
#         graph_emb = global_mean_pool(node_emb, batch)
#         return node_emb, graph_emb

# # Denoising Head (unsupervised training)
class DenoiseHead(nn.Module):
    """Decode node embeddings back to original feature space (768) and match x_orig."""
    def __init__(self, emb_dim=128, out_dim=768):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, out_dim),
        )

    def forward(self, node_emb):
        return self.decoder(node_emb)

# Graph Encoder (GAT)
class GATEncoder(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, out_dim=128, heads=4, dropout=0.1):
        super().__init__()
        # multi-head attention
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, out_dim // heads, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.norm1(F.elu(x))
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(F.elu(x))
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)

        node_emb = x
        graph_emb = global_mean_pool(node_emb, batch)
        return node_emb, graph_emb


# Training loop (unsupervised denoising)
def train_encoder(encoder, head, loader, epochs=10, lr=1e-3, device="cpu"):
    encoder.train()
    encoder.to(device)
    head.train()
    head.to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_data in loader:
            batch_data = batch_data.to(device)
            # Corrupt node features (dropout noise) for denoising
            x_orig = batch_data.x
            x_noisy = F.dropout(x_orig, p=0.2, training=True)

            node_emb, _ = encoder(x_noisy, batch_data.edge_index, batch_data.batch)
            x_rec = head(node_emb)
            loss = F.mse_loss(x_rec, x_orig)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_data.num_graphs

        avg = total_loss / len(loader.dataset)
        print(f"[Epoch {epoch:02d}] Denoise MSE: {avg:.6f}")
    print("Training complete.")

# Extract & Save Embeddings
@torch.no_grad()
def extract_graph_embeddings(encoder, loader, device="cpu"):
    encoder.eval()
    all_graph_emb = []
    all_ids = []
    all_labels = []

    for batch_data in loader:
        batch_data = batch_data.to(device)
        node_emb, graph_emb = encoder(batch_data.x, batch_data.edge_index, batch_data.batch)
        all_graph_emb.append(graph_emb.cpu())

        # stash ids & labels stored on Data
        all_ids.extend(batch_data.gid.cpu().tolist())
        all_labels.extend(batch_data.y.cpu().tolist())

    embs = torch.cat(all_graph_emb, dim=0).numpy()  # [N_graphs, emb_dim]
    ids = np.array(all_ids)
    labels = np.array(all_labels).astype(int)
    return embs, ids, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset with columns id,code,cfg,label")
    ap.add_argument("--out_dir", default="data/embeddings", help="Output directory for .npz")
    ap.add_argument("--model_out", default="models/graph_encoder_gcb+gat.pt", help="(Optional) save encoder weights")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load Dataset
    df = pd.read_parquet(args.dataset)
    required = {"id", "code", "cfg", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data missing columns: {missing}")

    embedder = GraphCodeBERTEmbedder(device=device)

    # Build PyG dataset
    graphs = []
    print("Building graphs & embedding nodes with GraphCodeBERT...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        data = cfg_to_data(row["cfg"], embedder)
        # attach label and id for later
        y = int(row["label"])
        gid = int(row["id"])
        data.y = torch.tensor([y], dtype=torch.long)
        data.gid = torch.tensor([gid], dtype=torch.long)
        graphs.append(data)

    print("Loading data into DataLoader...")
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)

    # Create encoder & denoising head
    # print("Initializing GraphSAGE encoder and denoising head...")
    # encoder = GraphSAGEEncoder(in_dim=768, hidden_dim=args.hidden_dim, out_dim=args.emb_dim).to(device)
    # print("Initializing GCN encoder and denoising head...")
    # encoder = GCNEncoder(in_dim=768, hidden_dim=args.hidden_dim, out_dim=args.emb_dim).to(device)
    print("Initializing GAT encoder and denoising head...")
    encoder = GATEncoder(in_dim=768, hidden_dim=args.hidden_dim, out_dim=args.emb_dim, heads=4).to(device)
    head = DenoiseHead(emb_dim=args.emb_dim, out_dim=768).to(device)

    # Train encoder (unsupervised)
    print("Training encoder with node-feature denoising...")
    train_encoder(encoder, head, loader, epochs=args.epochs, lr=args.lr, device=device)

    # Extract graph embeddings (with *clean* inputs)
    print("Extracting graph embeddings...")
    clean_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    graph_embs, ids, labels = extract_graph_embeddings(encoder, clean_loader, device=device)

    # Save embeddings
    out_path = os.path.join(args.out_dir, "graph_embeddings_gcb+gat.npz")
    np.savez_compressed(out_path, ids=ids, labels=labels, embeddings=graph_embs)
    print(f"Saved graph embeddings to: {out_path}")

    # Save encoder weights
    torch.save({"encoder_state_dict": encoder.state_dict(),
                "in_dim": 768, "hidden_dim": args.hidden_dim, "out_dim": args.emb_dim},
               args.model_out)
    print(f"Saved encoder to: {args.model_out}")

if __name__ == "__main__":
    main()
