"""
Evaluate saved fusion checkpoint on the test set (same split recipe as training).
Produces metrics, ROC curve, confusion matrix, attention/gating visualizations, and top-k saliency.

Usage:
python -m src.training.eval_pipeline \
    --graph_npz data/embeddings/graph_embeddings_gcb+sage.npz \
    --llm_npz data/embeddings/deepseek_coder_1.3b_instruct_embeddings.npz \
    --checkpoint models/fusion_best_2way_deepseekcoder.pt \
    --out_dir results/2way_deepseekcoder \
    --generate_llm_explanations deepseek-ai/deepseek-coder-1.3b-instruct \
    --debug --debug_size 50
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.graph_llm_integration.norm_and_project import Projector, l2_normalize_np
from src.graph_llm_integration.fusion_module import ConcatFusion, TwoWayGatingFusion, QKVCrossAttentionFusion
from src.graph_llm_integration.mlp import BinaryMLP
from src.graph_llm_integration.explanation import input_gradient_saliency, generate_llm_justification

# -------------------
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


def build_model_from_ckpt(ckpt_path, fusion_name, dproj, fusion_out, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # re-create modules
    # We don't know original input dims here; projectors should be provided externally by user
    if fusion_name == "concat":
        fusion = ConcatFusion(2 * dproj, hidden=256, out_dim=fusion_out)
    elif fusion_name == "two_way":
        fusion = TwoWayGatingFusion(dproj, hidden=256, out_dim=fusion_out)
    else:
        fusion = QKVCrossAttentionFusion(dproj, out_dim=fusion_out)
    classifier = BinaryMLP(in_dim=fusion_out, hidden=128)
    fusion.load_state_dict(ckpt["fusion"])
    classifier.load_state_dict(ckpt["classifier"])
    fusion = fusion.to(device).eval()
    classifier = classifier.to(device).eval()
    return fusion, classifier


def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0,1], [0,1], color="gray", linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.title("ROC")
    plt.savefig(out_path)
    plt.close()


def plot_confusion(cm, out_path):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()


def visualize_two_way_weights(fusion_module, g_batch, l_batch, out_path):
    """
    For TwoWayGatingFusion, compute soft weights a_g and a_l per sample and plot histogram
    """
    pair_gl = torch.cat([g_batch, l_batch], dim=1)
    pair_lg = torch.cat([l_batch, g_batch], dim=1)
    e_g = fusion_module.score_g(pair_gl).squeeze(-1)
    e_l = fusion_module.score_l(pair_lg).squeeze(-1)
    a = torch.softmax(torch.stack([e_g, e_l], dim=1), dim=1).cpu().detach().numpy()
    plt.figure(figsize=(6,4))
    plt.hist(a[:,0], bins=20, alpha=0.6, label="a_g (graph weight)")
    plt.hist(a[:,1], bins=20, alpha=0.6, label="a_l (llm weight)")
    plt.legend()
    plt.title("Two-way gating weight distribution")
    plt.savefig(out_path)
    plt.close()


def visualize_qkv_attention(fusion_module, g_batch, l_batch, out_path_prefix):
    """
    For QKV fusion: re-compute attention matrices for the batch and save heatmaps.
    NOTE: attention matrices are NxN (batch cross-attention) — careful with large batches.
    We'll save one heatmap for g->l and one for l->g.
    """
    with torch.no_grad():
        qg = fusion_module.qg(g_batch)
        kl = fusion_module.kl(l_batch)
        vl = fusion_module.vl(l_batch)
        scale = fusion_module.scale
        attn_gl = torch.softmax((qg @ kl.T) / scale, dim=1).cpu().numpy()  # (B,B)
        plt.figure(figsize=(6,6))
        sns.heatmap(attn_gl, annot=True, cmap="viridis")
        plt.title("Attention: graph->llm (rows: graph samples, cols: llm samples)")
        plt.savefig(out_path_prefix + "_g2l.png")
        plt.close()

        ql = fusion_module.ql(l_batch)
        kg = fusion_module.kg(g_batch)
        vg = fusion_module.vg(g_batch)
        attn_lg = torch.softmax((ql @ kg.T) / scale, dim=1).cpu().numpy()
        plt.figure(figsize=(6,6))
        sns.heatmap(attn_lg, annot=True, cmap="viridis")
        plt.title("Attention: llm->graph (rows: llm samples, cols: graph samples)")
        plt.savefig(out_path_prefix + "_l2g.png")
        plt.close()


def main(args):
    device = get_device()
    print("[INFO] Device:", device)

    Xg, Xl, y, ids = load_npz_align(
        args.graph_npz, args.llm_npz,
        debug=args.debug, debug_size=args.debug_size, seed=args.seed
    )
    print(f"[INFO] Loaded aligned {len(ids)} samples (debug={args.debug})")

    # normalize
    Xg = l2_normalize_np(Xg)
    Xl = l2_normalize_np(Xl)

    # split
    X_temp_g, X_test_g, X_temp_l, X_test_l, y_temp, y_test, ids_temp, ids_test = train_test_split(
        Xg, Xl, y, ids, test_size=args.test_frac, random_state=args.seed, stratify=y
    )

    # split train/val on temp to mimic training but we only need test set
    # projectors should match training dproj; user must provide dproj and projector weights are in ckpt if required
    dproj = args.dproj

    # Recreate projectors with right input dims (from npz)
    proj_g = Projector(Xg.shape[1], dproj, use_ln=True).to(device)
    proj_l = Projector(Xl.shape[1], dproj, use_ln=True).to(device)

    # load ckpt to get fusion/classifier weights and optionally proj weights if saved there.
    ckpt = torch.load(args.checkpoint, map_location=device)
    # load projector states if present
    if "proj_g" in ckpt:
        proj_g.load_state_dict(ckpt["proj_g"])
    if "proj_l" in ckpt:
        proj_l.load_state_dict(ckpt["proj_l"])

    # build fusion & classifier
    loaded_args = ckpt.get("args", {})
    fusion_name = loaded_args["fusion"] if loaded_args else args.fusion
    fusion_out = args.fusion_out
    if fusion_name == "concat":
        fusion = ConcatFusion(2 * dproj, hidden=256, out_dim=fusion_out).to(device)
    elif fusion_name == "two_way":
        fusion = TwoWayGatingFusion(dproj, hidden=256, out_dim=fusion_out).to(device)
    else:
        fusion = QKVCrossAttentionFusion(dproj, out_dim=fusion_out).to(device)
    classifier = BinaryMLP(in_dim=fusion_out, hidden=128).to(device)
    fusion.load_state_dict(ckpt["fusion"])
    classifier.load_state_dict(ckpt["classifier"])
    fusion.eval(); classifier.eval(); proj_g.eval(); proj_l.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # Create test loader
    test_ds = TensorDataset(torch.from_numpy(X_test_g).float(), torch.from_numpy(X_test_l).float(),
                            torch.from_numpy(y_test).long(), torch.from_numpy(ids_test).long())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_probs, all_preds, all_labels, all_ids = [], [], [], []
    # For visualization collect first batch for attention viz
    first_batch_for_viz = None

    with torch.no_grad():
        for bg, bl, by, bid in tqdm(test_loader, desc="Testing"):
            bg = bg.to(device); bl = bl.to(device)
            h_g = proj_g(bg); h_l = proj_l(bl)
            # fusion
            if fusion_name == "concat":
                inp = torch.cat([h_g, h_l], dim=1)
                fused = fusion(inp)
            elif fusion_name == "two_way":
                fused = fusion(h_g, h_l)
            else:
                fused = fusion(h_g, h_l)
            logits = classifier(fused)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int).tolist()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds)
            all_labels.extend(by.numpy().tolist())
            all_ids.extend([int(x.item()) for x in bid])

            if first_batch_for_viz is None:
                first_batch_for_viz = (h_g.cpu(), h_l.cpu(), fused.cpu())

    # compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec),
               "f1": float(f1), "auc": float(auc), "confusion_matrix": cm.tolist()}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # ROC and confusion matrix
    plot_roc(all_labels, all_probs, os.path.join(args.out_dir, "roc.png"))
    plot_confusion(cm, os.path.join(args.out_dir, "confusion.png"))

    # Visualize attention / gating on first batch
    if first_batch_for_viz is not None:
        hg_batch, hl_batch, fused_batch = first_batch_for_viz
        hg_batch = hg_batch.to(device); hl_batch = hl_batch.to(device)
        if fusion_name == "two_way":
            visualize_two_way_weights(fusion, hg_batch, hl_batch, os.path.join(args.out_dir, "two_way_weights.png"))
        elif fusion_name == "qkv":
            visualize_qkv_attention(fusion, hg_batch, hl_batch, os.path.join(args.out_dir, "qkv_attn"))
        else:
            # For concat, we can visualize correlation between modalities
            sim = torch.matmul(F.normalize(hg_batch, dim=1), F.normalize(hl_batch, dim=1).T).cpu().numpy()
            plt.figure(figsize=(6,6)); sns.heatmap(sim, annot=True, cmap="coolwarm"); plt.title("Graph-LLM similarity (batch)"); plt.savefig(os.path.join(args.out_dir, "concat_sim.png")); plt.close()

    # Saliency: compute gradient saliency for a small set (first B samples)
    # We'll use the projection + fusion+classifier as a wrapper to compute gradient wrt g_proj
    # Build a simple wrapper model that takes (g_proj,l_proj) and returns logits
    class Wrapper(torch.nn.Module):
        def __init__(self, fusion, classifier):
            super().__init__()
            self.fusion = fusion
            self.classifier = classifier
        def forward(self, g_proj, l_proj):
            if args.fusion == "concat":
                inp = torch.cat([g_proj, l_proj], dim=1)
                return self.classifier(self.fusion(inp))
            else:
                return self.classifier(self.fusion(g_proj, l_proj))

    wrapper = Wrapper(fusion, classifier).to(device)
    # pick first min(32, len(test)) samples
    sample_n = min(args.saliency_n, len(all_ids))
    sample_idx = list(range(sample_n))
    sample_g = torch.from_numpy(X_test_g[sample_idx]).float()
    sample_l = torch.from_numpy(X_test_l[sample_idx]).float()
    g_proj = proj_g(sample_g.to(device))
    l_proj = proj_l(sample_l.to(device))
    sal = input_gradient_saliency(wrapper, proj_g, g_proj, l_proj, device)  # (B, dproj)
    # aggregate per-node (dimension) saliency to top dims
    avg_sal = sal.mean(dim=0).numpy()
    topk = np.argsort(-avg_sal)[:args.topk_saliency]
    print(f"[SAL] top-{args.topk_saliency} projection dims:", topk.tolist())
    # Save saliency plot
    plt.figure(figsize=(8,3))
    plt.plot(avg_sal); plt.scatter(topk, avg_sal[topk], color="red"); plt.title("Avg abs gradient saliency across projection dims")
    plt.savefig(os.path.join(args.out_dir, "saliency_proj_dims.png")); plt.close()

    # LLM one-sentence justifications for a handful of examples
    if args.generate_llm_explanations:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # use the same model name the user prefers
        model_name = args.generate_llm_explanations
        tok = AutoTokenizer.from_pretrained(model_name)
        m = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        m.eval()
        justifications = {}
        # read parquet to get code text
        import pandas as pd
        df = pd.read_parquet(args.parquet)
        id2code = {int(r["id"]): r["code"] for _, r in df.iterrows()}
        for i, gid in enumerate(all_ids[: args.explanations_n]):
            code = id2code.get(int(gid), "")
            pred = int(all_preds[i])
            # generate short prompt
            prompt = f"One sentence: Explain why the following Java code is {'vulnerable' if pred==1 else 'safe'}. Do not add extra text.\n\nCode:\n{code}\n\nOne-sentence explanation:"
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=256, do_sample=False)
            text = tok.decode(out[0], skip_special_tokens=True)
            # extract tail
            just = text.split("One-sentence explanation:")[-1].strip()
            justifications[int(gid)] = {"pred": pred, "explanation": just}
        with open(os.path.join(args.out_dir, "llm_justifications.json"), "w") as f:
            json.dump(justifications, f, indent=2)
        print(f"[INFO] Saved LLM justifications for {len(justifications)} examples")

    print("[INFO] Evaluation artifacts saved in", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_npz", required=True)
    parser.add_argument("--llm_npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", required=False, default="data/parquet/cleaned_data_with_cfg.parquet")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fusion", choices=["concat", "two_way", "qkv"], default="concat")
    parser.add_argument("--dproj", type=int, default=128)
    parser.add_argument("--fusion_out", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--saliency_n", type=int, default=32)
    parser.add_argument("--topk_saliency", type=int, default=10)
    parser.add_argument("--explanations_n", type=int, default=10)
    parser.add_argument("--generate_llm_explanations", type=str, help="Generate LLM justifications using model name")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (subset eval data)")
    parser.add_argument("--debug_size", type=int, default=50, help="Number of samples for debug mode")
    args = parser.parse_args()

    main(args)
