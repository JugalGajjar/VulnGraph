"""
Explanation helpers:

 - input_gradient_saliency: returns gradient-based saliency scores over projected graph embedding dims.
 - get_top_graph_nodes_by_saliency: given per-node node_features and an importance vector per node,
   returns top-k node indices (this helper expects node-level features or labels available).
 - generate_llm_justification: uses an HF causal LM to generate a one-sentence justification for the prediction.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def input_gradient_saliency(model: torch.nn.Module, projector_g: torch.nn.Module,
                            g_proj: torch.Tensor, l_proj: torch.Tensor, device: torch.device):
    """
    Compute gradient of model output wrt g_proj (projected graph features).
    Returns importance scores per feature (L2 across dims) for each sample in batch.

    Args:
        model: fusion + classifier wrapper that accepts (g_proj, l_proj) or single fused vector.
        projector_g: projection module (used if needed).
        g_proj, l_proj: (B, d')
    Returns:
        saliency: (B, d') absolute gradient magnitudes for g_proj
    """
    # Ensure requires_grad
    g = g_proj.detach().to(device).requires_grad_(True)
    l = l_proj.detach().to(device)
    model = model.to(device).eval()

    logits = model(g, l) if callable(model) else model(torch.cat([g, l], dim=1))
    # logits shape: (B,)
    probs = torch.sigmoid(logits).sum()  # scalar
    probs.backward()
    grads = g.grad  # (B, d')
    saliency = grads.abs().cpu()
    return saliency


def generate_llm_justification(model_name: str, code: str, pred: int, device: torch.device,
                               max_new_tokens: int = 64, temperature: float = 0.0) -> str:
    """
    Generates a single-sentence justification using a causal HF model.

    Note: This loads a model — heavy. Use for a small number of examples only.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    prompt = f"One sentence: explain why the following Java code is {'vulnerable' if pred==1 else 'safe'}. Do not add extra text.\n\nCode:\n{code}\n\nOne-sentence explanation:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # extract after prompt
    if "One-sentence explanation:" in text:
        return text.split("One-sentence explanation:")[-1].strip()
    # fallback: return whole generation after code
    return text
