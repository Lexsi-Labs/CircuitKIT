"""Wanda selector — the real Wanda metric, aggregated to component granularity.

Canonical reference: Sun et al. 2023, "A Simple and Effective Pruning Approach
for Large Language Models" (Wanda). Reference implementation:
``/tmp/wanda/lib/prune.py:166`` and ``/tmp/wanda/lib/layerwrapper.py:33``.

Wanda's pruning metric is **per weight**::

    W_metric_ij = |W_ij| * sqrt( E[ ||X_j||_2^2 ] )

where ``W_ij`` is a weight of a linear layer, ``X_j`` is the j-th
input-channel activation feeding that layer, and the expectation
``E[||X_j||_2^2]`` is Wanda's ``scaler_row`` — the mean over calibration
tokens of the squared L2 norm of input channel ``j`` (see ``WrappedGPT.add_batch``).
In Wanda the comparison is done *within output rows* to decide which weights to
prune.

ADAPTATION TO COMPONENT GRANULARITY
-----------------------------------
The CircuitKit audit operates on *components* ("A{l}.{h}" attention heads,
"MLP {l}" blocks), not individual weights. We therefore compute the genuine
Wanda per-weight metric on the output-projection layer that *consumes* each
component's activations:

  * Attention head ``A{l}.{h}``: the linear layer is the head's slice of the
    output projection ``W_O[l][h]`` (shape ``[d_head, d_model]``). Its input
    channels are the head's ``hook_z`` activations.
  * MLP block ``MLP {l}``: the linear layer is ``W_out`` (shape
    ``[d_mlp, d_model]``). Its input channels are the post-activation hidden
    units (``mlp.hook_post``).

For each component we compute the full Wanda per-weight metric and then
**sum** it over every weight belonging to that component to obtain a single
per-component score. This is "Wanda aggregated to component granularity" —
faithful to Wanda's per-weight statistic, adapted to the component space the
audit's interventions act on.

Calibration data is **general text** (WikiText-2), matching Wanda's paper,
NOT the downstream task's EAP dataloader.
"""

from __future__ import annotations

import logging

import torch

from circuitkit.selection import register

logger = logging.getLogger(__name__)

PAPER_LABEL = "Wanda-component"

# Number of MLP hidden units processed per chunk when forming the per-weight
# metric, to bound peak memory on large d_mlp.
_MLP_CHUNK = 512


def _get_calibration_batches(model, config):
    """General-text (WikiText-2) calibration windows — see calibration.py."""
    from circuitkit.data.wikitext_calibration import wikitext_calibration_batches

    n_samples = config.get("num_examples", 128)
    seqlen = config.get("calib_seqlen", None)
    return wikitext_calibration_batches(model, n_samples=n_samples, seqlen=seqlen)


@register("wanda")
def wanda_selector(model, task_name: str, config: dict) -> dict:
    """Real Wanda per-weight metric, summed over each head / MLP block.

    ``task_name`` is accepted for API compatibility but is intentionally
    unused: Wanda calibrates on general text, not the task.
    """
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    d_head = model.cfg.d_head
    device = next(model.parameters()).device

    # hook_z is per-head and always available; hook_post gives the MLP's
    # post-activation hidden units (the inputs to W_out).
    def _names_filter(n: str) -> bool:
        return n.endswith("attn.hook_z") or n.endswith("mlp.hook_post")

    # Wanda's scaler_row: per-input-channel running mean of ||X_j||_2^2,
    # accumulated over calibration tokens (matching WrappedGPT.add_batch).
    attn_scaler = [
        torch.zeros(nH, d_head, dtype=torch.float32, device=device)
        for _ in range(nL)
    ]
    mlp_scaler = [None] * nL  # lazily sized to d_mlp on first batch
    n_tokens = 0

    batches = _get_calibration_batches(model, config)
    max_batches = config.get("max_batches", len(batches))

    for bi, tokens in enumerate(batches):
        if bi >= max_batches:
            break
        tokens = tokens.to(device)
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=_names_filter)

        for L in range(nL):
            z_key = f"blocks.{L}.attn.hook_z"
            if z_key in cache:
                z = cache[z_key]
                # z: [batch, pos, n_heads, d_head] -> sum of squares over tokens
                zf = z.to(torch.float32)
                # sum over batch+pos for each (head, channel)
                attn_scaler[L] += (zf ** 2).sum(dim=(0, 1))

            post_key = f"blocks.{L}.mlp.hook_post"
            if post_key in cache:
                post = cache[post_key]
                pf = post.to(torch.float32)
                ssq = (pf ** 2).sum(dim=(0, 1))  # [d_mlp]
                if mlp_scaler[L] is None:
                    mlp_scaler[L] = torch.zeros_like(ssq)
                mlp_scaler[L] += ssq

        # token count: batch * pos (same per layer)
        n_tokens += tokens.shape[0] * tokens.shape[1]

    if n_tokens == 0:
        raise RuntimeError("Wanda selector: no calibration tokens processed.")

    scores: dict[str, float] = {}
    for L in range(nL):
        # --- attention heads: linear layer = W_O[L][h], inputs = hook_z ---
        # scaler_row_j = E[||X_j||^2] = sum(X_j^2) / n_tokens
        attn_scaler_mean = attn_scaler[L] / n_tokens  # [n_heads, d_head]
        sqrt_scaler = torch.sqrt(attn_scaler_mean)     # [n_heads, d_head]
        WO = model.W_O[L].to(torch.float32)            # [n_heads, d_head, d_model]
        for h in range(nH):
            # Wanda per-weight metric for this head's output projection:
            #   |W_ij| * sqrt(scaler_row_j),  input channel j indexes d_head.
            metric = WO[h].abs() * sqrt_scaler[h].unsqueeze(-1)  # [d_head, d_model]
            # mean (not sum): per-weight saliency density. A raw sum scales with
            # the component's weight count, so MLPs (~8-10x more weights than a
            # head) would dominate the pooled cross-component ranking purely by
            # size. mean makes heads and MLPs comparable.
            scores[f"A{L}.{h}"] = float(metric.mean().item())

        # --- MLP block: linear layer = W_out, inputs = hook_post ---
        mlp = model.blocks[L].mlp
        if hasattr(mlp, "W_out"):
            W_out = mlp.W_out.to(torch.float32)             # [d_mlp, d_model]
        else:
            # nn.Linear fallback: weight is [out, in] = [d_model, d_mlp];
            # transpose so rows index d_mlp, aligning with sqrt_mlp (and the
            # `d_mlp = W_out.shape[0]` below) regardless of which branch ran.
            W_out = mlp.c_proj.weight.to(torch.float32).T   # -> [d_mlp, d_model]
        if mlp_scaler[L] is None:
            scores[f"MLP {L}"] = 0.0
            continue
        mlp_scaler_mean = mlp_scaler[L] / n_tokens          # [d_mlp]
        sqrt_mlp = torch.sqrt(mlp_scaler_mean)              # [d_mlp]
        d_mlp = W_out.shape[0]
        total = 0.0
        for start in range(0, d_mlp, _MLP_CHUNK):
            end = min(start + _MLP_CHUNK, d_mlp)
            chunk = W_out[start:end].abs() * sqrt_mlp[start:end].unsqueeze(-1)
            total += float(chunk.sum().item())
        # mean per weight (total / element count) — intensive, comparable with
        # the per-head scores above and with the other selectors' node scores.
        scores[f"MLP {L}"] = total / (W_out.shape[0] * W_out.shape[1])

    # Min-max normalize to [0, 1] for comparability with other selectors.
    vals = list(scores.values())
    mx, mn = max(vals), min(vals)
    if mx > mn:
        for k in scores:
            scores[k] = (scores[k] - mn) / (mx - mn)
    return scores
