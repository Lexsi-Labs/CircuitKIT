"""GPTQ-derived diagonal-Hessian saliency proxy (NOT the GPTQ algorithm).

HONEST FRAMING
--------------
GPTQ (Frantar et al. 2022, "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers"; reference repo ``/tmp/gptq``) is a
*quantizer*: it quantizes weights column-by-column using the inverse Hessian
``H^-1 = (2 X X^T + lambda I)^-1`` of the layer's calibration activations to
compensate for quantization error. GPTQ does **not** produce a per-component
importance ranking — it has no notion of "which head/MLP matters most".

This selector is therefore **not GPTQ**. It is a *GPTQ-derived diagonal-Hessian
saliency proxy*: it borrows only the diagonal of GPTQ's calibration Hessian
(``diag(H)_j = E[X_j^2]``, the mean squared activation of input channel ``j``)
and combines it with the squared output-projection weights to form a saliency
score::

    saliency(component) = sum_j ||W_:,j||_2^2 * E[X_j^2]

This is the leading (diagonal) term of the OBS/GPTQ second-order error
expansion, used here purely as an importance heuristic. It is a defensible
*proxy inspired by* GPTQ's Hessian, but it must not be cited or described as
"the GPTQ algorithm".

Calibration data is **general text** (WikiText-2), matching GPTQ's paper
(C4 / WikiText2), NOT the downstream task's EAP dataloader.
"""

from __future__ import annotations

import logging

import torch

from circuitkit.selection import register

logger = logging.getLogger(__name__)

PAPER_LABEL = "GPTQ-Hessian-diag"


def _get_calibration_batches(model, config):
    """General-text (WikiText-2) calibration windows — see calibration.py."""
    from circuitkit.data.wikitext_calibration import wikitext_calibration_batches

    n_samples = config.get("num_examples", 128)
    seqlen = config.get("calib_seqlen", None)
    return wikitext_calibration_batches(model, n_samples=n_samples, seqlen=seqlen)


@register("gptq")
def gptq_selector(model, task_name: str, config: dict) -> dict:
    """GPTQ-derived diagonal-Hessian saliency proxy (see module docstring).

    Computes the per-channel saliency
        score(component) = sum_j ||W_:,j||_2^2 * E[X_j^2]
    where j indexes the *input channels* of the component's output projection
    (d_head for an attention head, d_mlp for an MLP block). Each channel's
    column norm is paired with that channel's Hessian-diagonal entry
    E[X_j^2] — the calibration mean-squared activation of input channel j.
    This is the diagonal term of the OBS/GPTQ second-order expansion.

    Implementation note (correctness): the calibration accumulator is kept at
    full channel dimensionality (a [d_head] / [d_mlp] tensor per component)
    so the per-channel pairing with the column norms is preserved. An earlier
    version collapsed both terms to scalars and multiplied them — that gives
    the product of marginals, not the per-channel inner product, and is wrong.

    ``task_name`` is accepted for API compatibility but unused: like GPTQ,
    calibration uses general text, not the task.
    """
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    device = next(model.parameters()).device

    # Per-channel running sums of squared input activations. Lazy alloc on
    # first batch so we don't have to know d_head / d_mlp up front.
    attn_scaler = [[None] * nH for _ in range(nL)]   # each becomes [d_head]
    mlp_scaler = [None] * nL                          # each becomes [d_mlp]
    n_tokens = 0

    batches = _get_calibration_batches(model, config)
    max_batches = config.get("max_batches", len(batches))

    nB = 0
    for tokens in batches:
        if nB >= max_batches:
            break
        tokens = tokens.to(device)
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: n.endswith("attn.hook_z")
                or n.endswith("mlp.hook_post"),
            )
        for L in range(nL):
            # Attention head: linear layer = W_O[L][h], input channels = hook_z.
            hook_z = f"blocks.{L}.attn.hook_z"
            if hook_z in cache:
                z = cache[hook_z]
                if z.dim() == 4:
                    # [batch, pos, n_heads, d_head]
                    for h in range(nH):
                        ssq = (z[..., h, :].to(torch.float32) ** 2).sum(dim=(0, 1))
                        if attn_scaler[L][h] is None:
                            attn_scaler[L][h] = ssq
                        else:
                            attn_scaler[L][h] += ssq
                else:
                    d_head = z.size(-1) // nH
                    for h in range(nH):
                        slc = z[..., h * d_head:(h + 1) * d_head]
                        ssq = (slc.to(torch.float32) ** 2).sum(dim=(0, 1))
                        if attn_scaler[L][h] is None:
                            attn_scaler[L][h] = ssq
                        else:
                            attn_scaler[L][h] += ssq
            # MLP block: linear layer = W_out, input channels = hook_post (the
            # post-activation hidden units W_out actually consumes).
            hook_post = f"blocks.{L}.mlp.hook_post"
            if hook_post in cache:
                post = cache[hook_post]
                ssq = (post.to(torch.float32) ** 2).sum(dim=(0, 1))  # [d_mlp]
                if mlp_scaler[L] is None:
                    mlp_scaler[L] = ssq
                else:
                    mlp_scaler[L] += ssq
        # token count = batch * pos, same per layer
        n_tokens += tokens.shape[0] * tokens.shape[1]
        nB += 1

    if n_tokens == 0:
        raise RuntimeError("GPTQ-proxy selector: no calibration tokens processed.")

    scores = {}
    for L in range(nL):
        wo = model.W_O[L]                                       # [n_heads, d_head, d_model]
        for h in range(nH):
            if attn_scaler[L][h] is None:
                scores[f"A{L}.{h}"] = 0.0
                continue
            hessian_diag = attn_scaler[L][h] / n_tokens          # [d_head]
            col_norms_sq = (wo[h].to(torch.float32) ** 2).sum(dim=-1)  # [d_head]
            scores[f"A{L}.{h}"] = float((col_norms_sq * hessian_diag).sum().item())
        # MLP — align orientation: rows index d_mlp (input channels), so the
        # nn.Linear fallback (weight is [d_model, d_mlp]) needs transpose, same
        # convention as the wanda_selector fix.
        mlp = model.blocks[L].mlp
        if hasattr(mlp, "W_out"):
            W_out = mlp.W_out.to(torch.float32)                  # [d_mlp, d_model]
        else:
            W_out = mlp.c_proj.weight.to(torch.float32).T        # -> [d_mlp, d_model]
        if mlp_scaler[L] is None:
            scores[f"MLP {L}"] = 0.0
            continue
        hessian_diag = mlp_scaler[L] / n_tokens                  # [d_mlp]
        col_norms_sq = (W_out ** 2).sum(dim=-1)                  # [d_mlp]
        scores[f"MLP {L}"] = float((col_norms_sq * hessian_diag).sum().item())

    vals = list(scores.values())
    mx, mn = max(vals), min(vals)
    if mx > mn:
        for k in scores:
            scores[k] = (scores[k] - mn) / (mx - mn)
    return scores
