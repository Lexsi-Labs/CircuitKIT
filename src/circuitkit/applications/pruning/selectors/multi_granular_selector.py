"""Multi-granular node pruning — scores at block, head, and neuron levels.

Single-input importance (same family as ``taylor``): one forward pass on
the clean batch, importance = |grad(W) * W| accumulated at each weight
matrix. No clean-vs-patched contrast, so the KL metric is semantically
inapplicable — see the explicit guard below.

Padding hygiene: tokens are produced by ``tokenize_plus`` with explicit
``padding_side='left'`` rather than ``model.to_tokens``. ``input_lengths`` is
threaded through to ``metric_fn`` as a per-example tensor so the metric
indexes the last *real* token for every sample under either padding side
(legacy ``model.to_tokens`` + ``tokens.size(1)`` silently depended on the
tokenizer's global ``padding_side`` and produced wrong scores under right
padding).
"""

import torch

from circuitkit.backends.eap.eap_utils import tokenize_plus
from circuitkit.selection import register
from circuitkit.tasks.registry import get_task


@register("multi_granular")
def multi_granular_selector(model, task_name: str, config: dict) -> dict:
    # Guard: the KL metric requires a clean-vs-patched contrast (KL(patched ||
    # clean)). multi_granular computes single-input |grad(W) * W| importance —
    # there is no second forward pass to contrast against. Asking for the KL
    # metric here would silently compute KL(logits || logits) = 0 → all-zero
    # scores. Fail loudly instead.
    if config.get("metric_type") == "kl":
        raise ValueError(
            "multi_granular selector computes single-input |grad(W) * W| "
            "importance; the 'kl' metric requires a clean-vs-patched logit "
            "contrast and is not applicable here. Use the default metric for "
            "this task or pick a contrastive selector (eap / eap-ig / eap-gp)."
        )

    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    device = next(model.parameters()).device
    task_spec = get_task(task_name)
    bs = config.get("batch_size", 4)
    discovery_cfg = {
        "algorithm": "eap",
        "task": task_name,
        "level": "node",
        "batch_size": bs,
        "model_name": config.get("model_name", "unknown"),
        "data_params": {
            "num_examples": config.get("num_examples", 128),
            "seed": config.get("seed", 42),
            # Held-out calibration half (the selector-audit grid sets this to
            # "discovery"); "all" keeps the default library behaviour.
            "data_partition": config.get("discovery_data_partition", "all"),
        },
    }
    dataloader = task_spec.build_dataloader(model, discovery_cfg, device)
    metric_fn = task_spec.metric_fn()
    templated = getattr(dataloader, "templated", False)

    head_scores = [[0.0] * nH for _ in range(nL)]
    mlp_scores = [0.0] * nL
    nB = 0
    for batch in dataloader:
        clean, corrupted, labels = batch
        if isinstance(clean[0], str):
            # Padding-safe: tokenize_plus returns per-example input_lengths
            # and pins padding_side='left' explicitly.
            tokens, _attn_mask, input_lengths, _n_pos = tokenize_plus(
                model, list(clean), padding_side="left", templated=templated
            )
            input_lengths = input_lengths.to(device)
        else:
            tokens = clean.to(device)
            input_lengths = torch.full(
                (tokens.size(0),), tokens.size(-1),
                dtype=torch.long, device=device,
            )
        logits = model(tokens)
        # Non-contrastive metrics ignore the second positional arg, but pass
        # logits for it to match the standard metric signature; the KL guard
        # above prevents the silently-zero failure mode.
        metric = metric_fn(logits, logits, input_lengths, labels)
        if isinstance(metric, torch.Tensor) and metric.requires_grad:
            metric.sum().backward()
            for L in range(nL):
                block = model.blocks[L]
                if hasattr(block.attn, "W_O") and block.attn.W_O.grad is not None:
                    wo_grad = block.attn.W_O.grad
                    wo_val = block.attn.W_O.detach()
                    for h in range(nH):
                        head_scores[L][h] += float((wo_grad[h] * wo_val[h]).abs().sum().item())
                # Score the MLP layer with the same Taylor importance metric
                # used for heads: |grad(W) * W| summed over the layer's
                # projection weights (W_in + W_out). Previously the MLP key was
                # read from `scores` before it existed, so every MLP scored 0.0
                # and all MLPs tied at the global minimum after normalization.
                mlp = model.blocks[L].mlp
                for wname in ("W_in", "W_out"):
                    w = getattr(mlp, wname, None)
                    if w is not None and w.grad is not None:
                        mlp_scores[L] += float((w.grad * w.detach()).abs().sum().item())
            model.zero_grad()
        nB += 1
        if nB >= config.get("max_batches", 5):
            break

    scores = {}
    for L in range(nL):
        for h in range(nH):
            scores[f"A{L}.{h}"] = head_scores[L][h] / max(nB, 1)
        scores[f"MLP {L}"] = mlp_scores[L] / max(nB, 1)

    vals = list(scores.values())
    mx, mn = max(vals), min(vals)
    if mx > mn:
        for k in scores:
            scores[k] = (scores[k] - mn) / (mx - mn)
    return scores
