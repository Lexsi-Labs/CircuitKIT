"""Taylor — gradient*activation saliency.

Single-input importance: the selector runs *one* forward pass on the clean
batch and uses |grad(W) * W| at each weight matrix as the per-component
importance. There is no clean-vs-patched contrast, so the KL metric is
semantically inapplicable — see the explicit guard below.

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


@register("taylor")
def taylor_selector(model, task_name: str, config: dict) -> dict:
    # Guard: the KL metric requires a clean-vs-patched contrast (KL(patched ||
    # clean)). Taylor importance is single-input by construction — there is
    # no second forward pass to contrast against. Asking for the KL metric
    # here would silently compute KL(logits || logits) = 0 → all-zero scores
    # and a degenerate ranking. Fail loudly instead.
    if config.get("metric_type") == "kl":
        raise ValueError(
            "Taylor selector computes single-input |grad(W) * W| importance; "
            "the 'kl' metric requires a clean-vs-patched logit contrast and is "
            "not applicable here. Use the default metric for this task "
            "(typically logit_diff / prob / suffix_loglik) or pick a "
            "contrastive selector (eap / eap-ig / eap-gp)."
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

    accum_h = [[0.0] * nH for _ in range(nL)]
    accum_m = [0.0] * nL
    nB = 0
    for batch in dataloader:
        clean, corrupted, labels = batch
        if isinstance(clean[0], str):
            # Padding-safe: tokenize_plus returns per-example input_lengths
            # and lets us pin padding_side='left' explicitly so the metric
            # indexes the last real token for every sample.
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
                        accum_h[L][h] += float((wo_grad[h] * wo_val[h]).abs().sum().item())
                mlp = block.mlp
                if hasattr(mlp, "W_out") and mlp.W_out.grad is not None:
                    accum_m[L] += float((mlp.W_out.grad * mlp.W_out).abs().sum().item())
            model.zero_grad()
        nB += 1
        if nB >= config.get("max_batches", 10):
            break
    scores = {}
    for L in range(nL):
        for h in range(nH):
            scores[f"A{L}.{h}"] = accum_h[L][h] / max(nB, 1)
        scores[f"MLP {L}"] = accum_m[L] / max(nB, 1)
    vals = list(scores.values())
    mx, mn = max(vals), min(vals)
    if mx > mn:
        for k in scores:
            scores[k] = (scores[k] - mn) / (mx - mn)
    return scores
