"""AWQ-derived activation-salience selector.

HONEST FRAMING
--------------
AWQ (Lin et al. 2023, "AWQ: Activation-aware Weight Quantization for LLM
Compression and Acceleration"; reference repo ``/tmp/llm-awq``) is a
*quantizer*: it identifies a small set of *salient* weight channels and scales
them up before quantization to protect them from quantization error. The
signal AWQ uses to decide which channels are salient is the **per-channel
activation magnitude**::

    x_max = mean(|X|)        # /tmp/llm-awq/awq/quantize/auto_scale.py:29
                             # get_act_scale(x): x.abs().view(-1, x.shape[-1]).mean(0)

i.e. the mean absolute activation of each input channel. AWQ's protect-set is
exactly the channels with the largest ``x_max``.

This selector keeps that genuine AWQ signal: it scores each component by the
mean absolute magnitude of its output activations. Because AWQ's salience
criterion *is* per-channel activation magnitude, ranking components by their
activation magnitude is a defensible **AWQ-derived activation-salience**
selector — it uses AWQ's actual importance signal, lifted to component
granularity. It is not the full AWQ quantization procedure (no scaling search,
no quantization), so it is framed as "AWQ-derived" rather than "AWQ".

Calibration data is **general text** (WikiText-2), matching AWQ's paper
(general-text / Pile calibration), NOT the downstream task's EAP dataloader.
"""
from __future__ import annotations

import torch

from circuitkit.selection import register


def _get_calibration_batches(model, config):
    """General-text (WikiText-2) calibration windows — see calibration.py."""
    from circuitkit.data.wikitext_calibration import wikitext_calibration_batches

    n_samples = config.get("num_examples", 128)
    seqlen = config.get("calib_seqlen", None)
    return wikitext_calibration_batches(model, n_samples=n_samples, seqlen=seqlen)


@register("awq")
def awq_selector(model, task_name: str, config: dict) -> dict:
    """AWQ-derived activation-salience score per head / MLP block.

    ``task_name`` is accepted for API compatibility but unused: like AWQ,
    calibration uses general text, not the task.
    """
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    device = next(model.parameters()).device

    # Per-head activations are read from blocks.{L}.attn.hook_result, a hook
    # that only fires when use_attn_result is enabled. Without this flag the
    # cache lookup silently misses and all attention-head scores come out 0.0.
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)
    else:
        model.cfg.use_attn_result = True

    act_mag_attn = [[0.0] * nH for _ in range(nL)]
    act_mag_mlp = [0.0] * nL
    nB = 0

    batches = _get_calibration_batches(model, config)
    max_batches = config.get("max_batches", len(batches))

    for tokens in batches:
        if nB >= max_batches:
            break
        tokens = tokens.to(device)
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: n.endswith("hook_result") or n.endswith("hook_mlp_out"),
            )
        for L in range(nL):
            attn_name = f"blocks.{L}.attn.hook_result"
            if attn_name in cache:
                # hook_result has shape [batch, pos, n_heads, d_model]; the head
                # axis is dim 2. Index it directly rather than slicing d_model.
                act = cache[attn_name]
                for h in range(nH):
                    head_act = act[:, :, h, :]
                    # AWQ's x_max = mean(|X|) over tokens, per channel; we then
                    # aggregate channels for a per-head salience scalar.
                    act_mag_attn[L][h] += float(head_act.abs().mean().item())
            mlp_name = f"blocks.{L}.hook_mlp_out"
            if mlp_name in cache:
                act_mag_mlp[L] += float(cache[mlp_name].abs().mean().item())
        nB += 1

    if nB == 0:
        raise RuntimeError("AWQ-salience selector: no calibration batches processed.")

    scores = {}
    for L in range(nL):
        for h in range(nH):
            scores[f"A{L}.{h}"] = act_mag_attn[L][h] / max(nB, 1)
        scores[f"MLP {L}"] = act_mag_mlp[L] / max(nB, 1)

    vals = list(scores.values())
    mx, mn = max(vals), min(vals)
    if mx > mn:
        for k in scores:
            scores[k] = (scores[k] - mn) / (mx - mn)
    return scores
