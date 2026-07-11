"""CD-T adapter: runs the full contextual-decomposition forward-pass
importance metric on the TransformerLens checkpoint already loaded by
CircuitKit's discover_circuit pipeline.

CD-T (contextual decomposition for transformers, Singh et al. 2018;
transformer extension Sun et al. 2024) is a gradient-free forward-pass
method that propagates a (rel, irrel) split through every linear,
activation, and layer-norm.  The implementation in
``backends.cdt.propagation`` is TL-native and works for every
architecture TransformerLens supports (GPT-2, GPT-J, Llama-1/2/3,
Mistral, Gemma, Gemma-2, Qwen2, Falcon, Pythia, InternLM2, ...).

With ``use_full_propagation=True`` (default), this adapter calls
``cd_propagate_tl`` directly on the TL model -- no separate HF
checkpoint is loaded.  The simplified TL-cache metric is kept as a
fallback.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def run_cdt_discovery(
    tl_model,
    dataloader: DataLoader,
    *,
    device: str = "cuda",
    max_seq_len: int = 128,
    n_examples: int = 16,
    use_full_propagation: bool = True,
) -> Dict[str, float]:
    """Run CD-T contextual importance against the model.

    With ``use_full_propagation=True`` (default), this runs the full
    rel/irrel propagation via ``cd_propagate_tl`` directly on the TL
    model -- no additional HF checkpoint is loaded.  The attention
    pattern is frozen from a single clean forward pass; bilinear
    cross-terms in gated MLPs are split 50/50 (Singh et al. 2018).
    Falls back to the simplified TL-cache metric only on unexpected
    exceptions.
    """
    templated = getattr(dataloader, "templated", False)
    if use_full_propagation:
        try:
            return _run_full_cdt(
                tl_model,
                dataloader,
                device=device,
                max_seq_len=max_seq_len,
                n_examples=n_examples,
                templated=templated,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Full CD-T propagation failed ({type(exc).__name__}: "
                f"{str(exc)[:80]}); falling back to simplified TL forward "
                f"importance metric."
            )
    return _run_simple_cdt(
        tl_model,
        dataloader,
        device=device,
        max_seq_len=max_seq_len,
        n_examples=n_examples,
        templated=templated,
    )


def _run_full_cdt(
    tl_model,
    dataloader: DataLoader,
    *,
    device: str,
    max_seq_len: int,
    n_examples: int,
    templated: bool = False,
):
    """Full CD-T propagation via cd_propagate_tl on the TL model."""
    from .propagation import cd_propagate_tl

    n_layers = tl_model.cfg.n_layers
    n_heads = tl_model.cfg.n_heads
    tok = tl_model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    head_scores = torch.zeros((n_layers, n_heads), dtype=torch.float32)
    mlp_scores = torch.zeros((n_layers,), dtype=torch.float32)

    seen = 0
    succeeded = 0
    last_exc: Exception | None = None
    for clean, corrupted, _ in dataloader:
        for clean_str in clean:
            if seen >= n_examples:
                break
            try:
                in_ids = tok(
                    clean_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                    add_special_tokens=not templated,
                )["input_ids"].to(device)
                if in_ids.numel() < 2:
                    continue
                rel_indices = [in_ids.shape[1] - 1]
                contribs = cd_propagate_tl(tl_model, in_ids, rel_indices)
                for k, v in contribs.items():
                    if k.startswith("MLP "):
                        lyr = int(k.split()[1])
                        mlp_scores[lyr] += float(v)
                    elif k.startswith("A"):
                        lyr, h = k[1:].split(".")
                        head_scores[int(lyr), int(h)] += float(v)
                seen += 1
                succeeded += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"cd_propagate_tl failed on example {seen}: {exc}")
                last_exc = exc
                seen += 1
        if seen >= n_examples:
            break

    # If cd_propagate_tl failed for *every* example, CD-T has produced nothing
    # meaningful — surface a clear error instead of silently returning an
    # all-zeros (empty) circuit.
    if succeeded == 0:
        detail = (
            f" Last failure: {type(last_exc).__name__}: {last_exc}" if last_exc is not None else ""
        )
        raise RuntimeError(
            f"CD-T full propagation failed on all {seen} example(s); "
            f"cd_propagate_tl never succeeded.{detail}"
        )

    head_scores /= succeeded
    mlp_scores /= succeeded

    result: Dict[str, float] = {}
    for lyr in range(n_layers):
        for h in range(n_heads):
            result[f"A{lyr}.{h}"] = float(head_scores[lyr, h].item())
        result[f"MLP {lyr}"] = float(mlp_scores[lyr].item())
    return result


def _run_simple_cdt(
    tl_model,
    dataloader: DataLoader,
    *,
    device: str,
    max_seq_len: int,
    n_examples: int,
    templated: bool = False,
):
    """Simplified CD-T-inspired importance via the TL cache.

    Per attention head (l, h):
        score(l, h) = E_x [ ||attn_pattern[l, h, target_pos, :] · V[l, h]||_2 ]
    Per MLP layer l:
        score(l) = E_x [ ||mlp_out[l, target_pos, :]||_2 ]
    """
    n_layers = tl_model.cfg.n_layers
    n_heads = tl_model.cfg.n_heads

    head_scores = torch.zeros((n_layers, n_heads), dtype=torch.float32)
    mlp_scores = torch.zeros((n_layers,), dtype=torch.float32)

    seen = 0
    succeeded = 0
    tok = tl_model.tokenizer
    tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id

    for clean, corrupted, _ in dataloader:
        for clean_str in clean:
            if seen >= n_examples:
                break
            try:
                in_ids = tok(
                    clean_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                    add_special_tokens=not templated,
                )["input_ids"].to(device)
                if in_ids.numel() < 2:
                    continue
                target_pos = in_ids.shape[1] - 1

                # Run the model with cache so we can read attention patterns
                # and value vectors at the target position. TransformerLens'
                # run_with_cache is the cheapest way to get all components.
                with torch.inference_mode():
                    _, cache = tl_model.run_with_cache(in_ids)

                for layer in range(n_layers):
                    # Attention pattern: [batch, head, query_pos, key_pos]
                    pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
                    # Value matrix: [batch, key_pos, head, d_head]
                    value = cache[f"blocks.{layer}.attn.hook_v"]
                    # Per-head contribution at the target position:
                    # weighted V where weights are pattern[target_pos, :].
                    # Shape: [batch, head, key_pos] @ [batch, key_pos, head, d_head]
                    # -> per head: pattern[h, t, :] @ value[:, h, :]
                    for h in range(n_heads):
                        pat_th = pattern[0, h, target_pos, :]  # [key_pos]
                        v_h = value[0, :, h, :]  # [key_pos, d_head]
                        contribution = (pat_th.unsqueeze(-1) * v_h).sum(dim=0)  # [d_head]
                        head_scores[layer, h] += float(contribution.norm().item())
                    # MLP output at target_pos: [batch, pos, d_model]
                    mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]
                    mlp_scores[layer] += float(mlp_out[0, target_pos, :].norm().item())

                seen += 1
                succeeded += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"CD-T failed on example {seen}: {exc}")
                seen += 1
        if seen >= n_examples:
            break

    # Normalize by the number of examples that actually contributed (mirrors
    # _run_full_cdt above). Earlier code divided by `seen`, which includes
    # examples whose exceptions zeroed the score accumulation — dormant in
    # our grid (0 per-example failures in the logs) but real under any model
    # where exceptions occur.
    if succeeded > 0:
        head_scores /= succeeded
        mlp_scores /= succeeded

    result: Dict[str, float] = {}
    for lyr in range(n_layers):
        for h in range(n_heads):
            result[f"A{lyr}.{h}"] = float(head_scores[lyr, h].item())
        result[f"MLP {lyr}"] = float(mlp_scores[lyr].item())

    return result
