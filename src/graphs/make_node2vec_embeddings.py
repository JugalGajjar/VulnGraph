import argparse
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from tqdm import tqdm
import os, json, ast

def parse_cfg(cfg_entry):
    """Parse cfg entry (dict or JSON string) and return edge list."""
    if isinstance(cfg_entry, dict):
        cfg = cfg_entry
    else:
        try:
            cfg = json.loads(cfg_entry)
        except json.JSONDecodeError:
            cfg = ast.literal_eval(cfg_entry)

    nodes = cfg.get("nodes", [])
    edges = cfg.get("edges", [])
    edge_list = [(e["source"], e["target"]) for e in edges]
    return edge_list

def main(args):
    # Load dataset
    df = pd.read_parquet(args.input_file)
    print(f"Loaded {len(df)} rows from {args.input_file}")

    embeddings_list = []
    labels = []
    ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CFGs"):
        code_id = row["id"]
        label = row["label"]
        cfg = row["cfg"]

        edge_list = parse_cfg(cfg)

        # --- Build graph ---
        G = nx.DiGraph()
        G.add_edges_from(edge_list)

        if len(G.nodes) == 0:
            emb = np.zeros(args.dim)
        else:
            # Node2Vec embedding
            node2vec = Node2Vec(
                G,
                dimensions=args.dim,
                walk_length=args.walk_length,
                num_walks=args.num_walks,
                workers=1
            )
            model = node2vec.fit(window=args.window_size, min_count=1, batch_words=4)

            # Aggregate node embeddings -> graph embedding
            node_embs = [model.wv[str(n)] for n in G.nodes()]
            emb = np.mean(node_embs, axis=0)

        embeddings_list.append(emb)
        labels.append(label)
        ids.append(code_id)

    embeddings = np.array(embeddings_list)
    labels = np.array(labels)
    ids = np.array(ids)

    # Save in npz format
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.savez(args.output_file, ids=ids, embeddings=embeddings, labels=labels)

    print(f"Saved embeddings to {args.output_file}")
    print(f"Shape: {embeddings.shape}, Labels: {labels.shape}, IDs: {ids.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Parquet file with columns [id, code, cfg, label]")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output .npz file")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--walk_length", type=int, default=20, help="Length of random walk")
    parser.add_argument("--num_walks", type=int, default=200, help="Number of walks per node")
    parser.add_argument("--window_size", type=int, default=10, help="Skip-gram window size")
    args = parser.parse_args()

    main(args)
