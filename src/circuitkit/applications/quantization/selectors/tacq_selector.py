"""TaCQ-style task-circuit quantization selector.

Implementation note vs the TaCQ paper (arXiv 2504.07389):

  Paper form:        score = |grad| × (W_clean - W_quantized) × W_clean
                                       └──── signed delta ────┘
  Our form:          score = |grad| × |W_clean - W_quantized| × |W_clean|

We use the *absolute value* of the weight delta (and of W_clean) so the
per-weight contribution is positive-definite. This prevents sign
cancellation when contributions are summed across the weights of a
component (sum-aggregated, see below). The paper's signed form depends on
sign-cancellation across weights, which can produce small total magnitudes
for components whose individual weights all have moderately large impact
but in opposing directions — an artefact of summation rather than a
genuine "low importance" reading. Both forms produce monotonically
related rankings on the components we tested, but the absolute form is
more robust to sign-cancellation noise.

Component coverage (per-component scope):
  - **Output-side weights only.** For attention this scores ``W_O``; for
    MLP this scores ``W_out``. The input-side projections (``W_Q``,
    ``W_K``, ``W_V``, ``W_in`` / ``W_gate``) are NOT scored here. This is
    deliberate — TaCQ is a *quantization-sensitivity* score, and the
    output projections are the components whose downstream cost of
    quantization error is most directly captured by the
    quantization-error · gradient term. A full input+output coverage
    variant is straightforward (mirror the o_proj branch onto q_proj /
    k_proj / v_proj / w_in) but changes the ranking; we have kept the
    output-side scope to match the form of saliency selectors like Wanda
    that also score output projections.

For component-level aggregation:
  component_score = Σ_w |grad_w| × |W_clean_w − W_quant_w| × |W_clean_w|

This differentiates from:
  - taylor: |grad| × |weight| (first-order, no quantization error)
  - gptq: |weight|² × E[input²] (Hessian proxy, no gradient)
"""

import torch

from circuitkit.selection import register
from circuitkit.tasks.registry import get_task


def _simulate_4bit_quantize(w: torch.Tensor) -> torch.Tensor:
    """Simulate 4-bit symmetric quantization per output channel."""
    max_val = w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    normalized = w / max_val
    q_levels = 7.0  # 4-bit symmetric: [-7, 7]
    q = (normalized * q_levels).round().clamp(-q_levels, q_levels)
    return (q / q_levels) * max_val


@register("tacq")
def tacq_selector(model, task_name: str, config: dict) -> dict:
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    device = next(model.parameters()).device
    # Built-in task registry starts empty; other selectors register tasks
    # idempotently as a side-effect of their own get_task call, but if `tacq`
    # is the FIRST selector invoked in a fresh process (or if no other
    # selector ran before it), the registry is still empty and `get_task`
    # raises KeyError. The bootstrap call below is idempotent — re-running
    # it after another selector has already registered the same tasks is a
    # no-op — so it's safe to call unconditionally here.
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks

    _bootstrap_builtin_tasks()
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

    # Accumulate |gradient| per weight (using grad hooks, then compute score)
    accum_grad_attn = [[0.0] * nH for _ in range(nL)]
    accum_grad_mlp = [0.0] * nL
    nB = 0

    for batch in dataloader:
        clean, corrupted, labels = batch
        tokens = (
            model.to_tokens(list(clean), prepend_bos=not templated)
            if isinstance(clean[0], str)
            else clean.to(device)
        )
        logits = model(tokens)
        input_length = tokens.size(1) if tokens.dim() == 2 else tokens.size(-1)
        metric = metric_fn(logits, logits, input_length, labels)
        if isinstance(metric, torch.Tensor) and metric.requires_grad:
            metric.sum().backward()
            for L in range(nL):
                block = model.blocks[L]
                # TaCQ formula: |grad| × (W_clean - W_quant) × W_clean
                # Aggregate per-component: sum over all weight elements
                if hasattr(block.attn, "W_O") and block.attn.W_O.grad is not None:
                    wo = block.attn.W_O.detach()  # W_clean
                    wo_grad = block.attn.W_O.grad  # gradient
                    wo_quant = _simulate_4bit_quantize(wo)  # W_quant
                    delta = (wo - wo_quant).abs()  # |W_clean - W_quant|
                    for h in range(nH):
                        score = float((wo_grad[h].abs() * delta[h] * wo[h].abs()).sum().item())
                        accum_grad_attn[L][h] += score

                mlp = block.mlp
                if hasattr(mlp, "W_out") and mlp.W_out.grad is not None:
                    w = mlp.W_out.detach()
                    wg = mlp.W_out.grad
                    wq = _simulate_4bit_quantize(w)
                    delta = (w - wq).abs()
                    score = float((wg.abs() * delta * w.abs()).sum().item())
                    accum_grad_mlp[L] += score

            model.zero_grad()
        nB += 1
        if nB >= config.get("max_batches", 10):
            break

    scores = {}
    for L in range(nL):
        for h in range(nH):
            scores[f"A{L}.{h}"] = accum_grad_attn[L][h] / max(nB, 1)
        scores[f"MLP {L}"] = accum_grad_mlp[L] / max(nB, 1)

    vals = list(scores.values())
    mx, mn = max(vals), min(vals)
    if mx > mn:
        for k in scores:
            scores[k] = (scores[k] - mn) / (mx - mn)
    return scores
