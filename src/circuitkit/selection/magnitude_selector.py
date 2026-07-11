"""Magnitude selector — weight-norm baseline.

Per-component score is the RMS weight magnitude (Frobenius norm / sqrt(numel)),
NOT the raw Frobenius norm: ‖W‖_F grows with sqrt(parameter count), so a raw
norm makes MLP components (≈8-10x more weights than a head) systematically
outrank heads regardless of importance. RMS is intensive (per-weight), so heads
and MLPs are comparable in the pooled cross-component pruning ranking — matching
how the attribution selectors' per-node scores behave.
"""
import torch
from circuitkit.selection import register

@register("magnitude")
def magnitude_selector(model, task_name: str, config: dict) -> dict:
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    scores = {}
    for L in range(nL):
        WO = model.W_O[L]
        for h in range(nH):
            scores[f"A{L}.{h}"] = float((WO[h].norm() / WO[h].numel() ** 0.5).item())
        mlp = model.blocks[L].mlp
        w = mlp.W_out if hasattr(mlp, "W_out") else mlp.c_proj.weight
        scores[f"MLP {L}"] = float((w.norm() / w.numel() ** 0.5).item())
    vals = list(scores.values())
    mx, mn = max(vals), min(vals)
    if mx > mn:
        for k in scores: scores[k] = (scores[k] - mn) / (mx - mn)
    return scores
