"""Full CD-T propagation using TransformerLens's unified weight API.

Works for any model TransformerLens supports — GPT-2, GPT-J, Llama-1/2/3,
Mistral, Gemma, Gemma-2, Qwen2, Falcon, Pythia, InternLM2, … — because TL
exposes the same parameter names regardless of the underlying HF architecture:

  blocks[l].attn.W_Q/K/V [n_heads, d_model, d_head]
  blocks[l].attn.b_Q/K/V [n_heads, d_head]
  blocks[l].attn.W_O      [n_heads, d_head, d_model]
  blocks[l].attn.b_O      [d_model]
  blocks[l].ln1/ln2.w/b   [d_model]   (b absent for RMSNorm)
  blocks[l].mlp.W_in/b_in, W_out/b_out
  blocks[l].mlp.W_gate/b_gate          (gated MLPs only)
  embed.W_E               [vocab, d_model]
  pos_embed.W_pos         [n_ctx, d_model]  (absent / unused for RoPE models)

Design follows adelaidehsu/CD_Circuit (reference repo) with three additions:
  1. normalize_rel_irrel after every residual add AND after the MLP pre-act
     linear — prevents sign divergence (reference §normalize_rel_irrel).
  2. Bias allocated by |rel_t| / (|rel_t| + |irrel_t| + ε) at every linear.
  3. Gated-MLP cross-terms split 50/50 (Singh et al. 2018 Half-rule for the
     bilinear gated product; matches RelP's Half-rule on gated paths).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

# ── Numerical stability ───────────────────────────────────────────────────────


def _normalize_rel_irrel(rel: torch.Tensor, irrel: torch.Tensor) -> tuple:
    """In-place stability rule from adelaidehsu/CD_Circuit.

    Where rel and irrel have opposite signs, zero out the smaller-magnitude
    component and assign the combined total to the larger one.
    Preserves rel + irrel = tot exactly.
    """
    tot = rel + irrel
    conflict = (rel * irrel) < 0
    rel_larger = conflict & (rel.abs() >= irrel.abs())
    irrel_larger = conflict & ~rel_larger
    rel, irrel = rel.clone(), irrel.clone()
    rel[rel_larger] = tot[rel_larger]
    rel[irrel_larger] = 0.0
    irrel[irrel_larger] = tot[irrel_larger]
    irrel[rel_larger] = 0.0
    return rel, irrel


# ── Primitive decompositions ──────────────────────────────────────────────────


def _prop_linear(
    rel: torch.Tensor,
    irrel: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    tol: float = 1e-8,
) -> tuple:
    """Linear layer: weight is [d_in, d_out] for left-multiply (x @ W)."""
    r = rel @ weight
    i = irrel @ weight
    if bias is None:
        return r, i
    rw, iw = r.abs(), i.abs()
    tw = rw + iw + tol
    return r + bias * (rw / tw), i + bias * (iw / tw)


def _prop_act(rel: torch.Tensor, irrel: torch.Tensor, act_fn) -> tuple:
    """ACD activation rule (reference prop_act): irrel gets act(irrel),
    rel gets act(total) - act(irrel).  Works for GELU, SiLU, ReLU, etc."""
    ir_act = act_fn(irrel)
    return act_fn(rel + irrel) - ir_act, ir_act


def _prop_tl_norm(rel: torch.Tensor, irrel: torch.Tensor, ln, tol: float = 1e-8) -> tuple:
    """CD decomposition for TL's LayerNorm / RMSNorm, including the *folded*
    ``LayerNormPre`` / ``RMSNormPre`` variants.

    When CircuitKit enables ``use_attn_result`` / ``use_split_qkv_input`` /
    ``use_hook_mlp_in`` (which discover_circuit always does for the EAP/CD-T
    family), TransformerLens folds the LayerNorm scale/bias into the adjacent
    weight matrices and swaps in a parameter-free ``LayerNormPre``/``RMSNormPre``
    module. Those modules have no ``.w`` / ``.b`` attributes — accessing
    ``ln.w`` raised ``AttributeError`` and forced the adapter into an
    all-zeros fallback. Treat a missing scale as identity (1) and a missing
    bias as zero, and decide LayerNorm-vs-RMSNorm from the class name (the
    "Pre" variants still subtract the mean for LayerNorm).

    Variance / RMS is computed from tot = rel + irrel so that both components
    see the same scale (matching the reference prop_layer_norm).
    """
    tot = rel + irrel
    w = getattr(ln, "w", None)  # None for the folded *Pre variants
    b = getattr(ln, "b", None)
    eps = getattr(ln, "eps", 1e-5)

    # RMSNorm omits mean-centering; LayerNorm (incl. LayerNormPre) keeps it.
    is_rmsnorm = "RMS" in type(ln).__name__

    if not is_rmsnorm:
        # LayerNorm / LayerNormPre: mean-center each component; shared std.
        tot_mn = tot.mean(dim=-1, keepdim=True)
        rel_mn = rel.mean(dim=-1, keepdim=True)
        irr_mn = irrel.mean(dim=-1, keepdim=True)
        var = (tot - tot_mn).pow(2).mean(dim=-1, keepdim=True)
        inv_std = (var + eps).rsqrt()
        r = (rel - rel_mn) * inv_std
        i = (irrel - irr_mn) * inv_std
        if w is not None:
            r, i = r * w, i * w
        if b is not None:
            rw, iw = rel.abs(), irrel.abs()
            tw = rw + iw + tol
            r = r + b * (rw / tw)
            i = i + b * (iw / tw)
        return r, i
    else:
        # RMSNorm / RMSNormPre: divide by shared RMS; no mean subtraction.
        rms = (tot.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
        r, i = rel / rms, irrel / rms
        if w is not None:
            r, i = r * w, i * w
        return r, i


# ── Attention ─────────────────────────────────────────────────────────────────


def _prop_tl_attn(
    rel: torch.Tensor, irrel: torch.Tensor, attn, pattern: torch.Tensor, tol: float = 1e-8
):
    """CD through one TL attention block.

    TL standard shapes (identical for all architectures):
      W_V [n_heads, d_model, d_head]  b_V [n_heads, d_head]
      W_O [n_heads, d_head, d_model]  b_O [d_model]

    The attention pattern is frozen (computed from the full forward pass
    and treated as a constant matrix).  Only the V branch is decomposed;
    Q and K don't contribute to the value-weighted output once pattern
    is fixed — consistent with the reference prop_attention_no_output_hh.

    Returns: (rel_out, irrel_out, rel_z) where rel_z [b, q, n_heads, d_head]
    is used for per-head attribution scoring.
    """
    # V projection with abs-magnitude bias split.
    r_vt = torch.einsum("bpd,hde->bphe", rel, attn.W_V)
    i_vt = torch.einsum("bpd,hde->bphe", irrel, attn.W_V)
    rw, iw = r_vt.abs(), i_vt.abs()
    tw = rw + iw + tol
    r_v = r_vt + attn.b_V * (rw / tw)
    i_v = i_vt + attn.b_V * (iw / tw)

    # Apply frozen pattern [b, n_heads, q, k] over V [b, k, n_heads, d_head].
    r_z = torch.einsum("bnqk,bkne->bqne", pattern, r_v)
    i_z = torch.einsum("bnqk,bkne->bqne", pattern, i_v)

    # O projection with abs-magnitude bias split.
    r_at = torch.einsum("bqne,ned->bqd", r_z, attn.W_O)
    i_at = torch.einsum("bqne,ned->bqd", i_z, attn.W_O)
    rw, iw = r_at.abs(), i_at.abs()
    tw = rw + iw + tol
    r_a = r_at + attn.b_O * (rw / tw)
    i_a = i_at + attn.b_O * (iw / tw)

    return r_a, i_a, r_z


# ── MLP ───────────────────────────────────────────────────────────────────────


def _prop_tl_mlp(rel: torch.Tensor, irrel: torch.Tensor, mlp, tol: float = 1e-8) -> tuple:
    """CD through TL's non-gated MLP (GPT-2, Pythia, Falcon, …).

    Applies normalize_rel_irrel after W_in, before the activation,
    as in the reference prop_GPT_layer.
    """
    r, i = _prop_linear(rel, irrel, mlp.W_in, mlp.b_in, tol)
    r, i = _normalize_rel_irrel(r, i)
    r, i = _prop_act(r, i, mlp.act_fn)
    return _prop_linear(r, i, mlp.W_out, mlp.b_out, tol)


def _prop_tl_gated_mlp(rel: torch.Tensor, irrel: torch.Tensor, mlp, tol: float = 1e-8) -> tuple:
    """CD through TL's gated MLP (Llama, Mistral, Gemma, Qwen2, …).

    Gate path: W_gate → normalize → act_fn (SiLU / GELU)
    Up path:   W_in   (linear, no act)
    Product:   gate * up  — bilinear interaction split 50/50 (Half-rule,
               matching RelP Mohebbi et al. 2025 for gated architectures).
    Down path: W_out  (linear)
    """
    # Gate branch.  TL's GatedMLP exposes no ``b_gate`` parameter at all for
    # bias-free gated architectures (Llama, Mistral, Gemma, Qwen2, …) — only
    # ``W_gate``. Accessing ``mlp.b_gate`` raises AttributeError; use getattr
    # with a None default so the linear simply skips the bias split.
    b_gate = getattr(mlp, "b_gate", None)
    b_in = getattr(mlp, "b_in", None)
    b_out = getattr(mlp, "b_out", None)
    r_g, i_g = _prop_linear(rel, irrel, mlp.W_gate, b_gate, tol)
    r_g, i_g = _normalize_rel_irrel(r_g, i_g)
    r_gate, i_gate = _prop_act(r_g, i_g, mlp.act_fn)

    # Up branch.
    r_up, i_up = _prop_linear(rel, irrel, mlp.W_in, b_in, tol)

    # Bilinear product with 50/50 cross-term split.
    cross = r_gate * i_up + i_gate * r_up
    r_mid = r_gate * r_up + 0.5 * cross
    i_mid = i_gate * i_up + 0.5 * cross

    return _prop_linear(r_mid, i_mid, mlp.W_out, b_out, tol)


# ── Main entry point ──────────────────────────────────────────────────────────


def cd_propagate_tl(tl_model, input_ids: torch.Tensor, rel_indices: List[int]) -> Dict[str, float]:
    """Full CD-T forward-pass decomposition on a TransformerLens model.

    Supports every architecture TL loads (GPT-2, GPT-J, Llama-1/2/3,
    Mistral, Gemma, Gemma-2, Qwen2, Falcon, Pythia, InternLM2, …).
    No HF model loading required — uses TL's normalized weight interface.

    Args:
        tl_model:    HookedTransformer, eval mode, on device.
        input_ids:   LongTensor [1, seq_len].
        rel_indices: Token positions to treat as the relevant source.

    Returns:
        {"A{l}.{h}": float, "MLP {l}": float} — L2 norm of the rel
        component's contribution to the target position's representation.
    """
    device = input_ids.device

    # CD-T is a gradient-free forward decomposition: run the whole pass under
    # no_grad so model weights (leaf tensors with requires_grad=True) do not
    # pull the einsums into the autograd graph.
    grad_guard = torch.no_grad()
    grad_guard.__enter__()
    try:
        return _cd_propagate_tl_impl(tl_model, input_ids, rel_indices, device)
    finally:
        grad_guard.__exit__(None, None, None)


def _cd_propagate_tl_impl(tl_model, input_ids, rel_indices, device):
    # Freeze the attention patterns from a single clean forward pass.
    # Use no_grad (not inference_mode): inference-mode tensors cannot later be
    # consumed by ops that allocate autograd-visible buffers, and the cached
    # pattern is reused inside einsum below.
    with torch.no_grad():
        _, cache = tl_model.run_with_cache(
            input_ids,
            names_filter=lambda n: n.endswith("hook_pattern"),
        )

    n_layers = tl_model.cfg.n_layers
    n_heads = tl_model.cfg.n_heads
    tgt = input_ids.shape[1] - 1  # target position (last token)
    tol = 1e-8

    # All rel/irrel tensors and frozen-pattern constants must match the model
    # weight dtype.  On a 16-bit model (bf16/fp16) the cached ``hook_pattern``
    # is float32 — TransformerLens captures it via the HookPoint *before* the
    # final ``pattern.to(cfg.dtype)`` cast (abstract_attention.py: hook fires
    # at line 280, cast at line 281).  An einsum of that float32 pattern
    # against the bf16 weights raises "expected scalar type BFloat16 but found
    # Float".  ``model_dtype`` is used to coerce every introduced tensor.
    model_dtype = tl_model.cfg.dtype

    # Initial hidden state from TL's embedding modules.
    ids = input_ids[0]  # [seq]
    hidden = tl_model.embed.W_E[ids].unsqueeze(0)  # [1, seq, d]
    # Add positional embeddings only for models that use them (not RoPE).
    if hasattr(tl_model, "pos_embed") and tl_model.cfg.positional_embedding_type != "rotary":
        pos = torch.arange(ids.shape[0], device=device)
        hidden = hidden + tl_model.pos_embed.W_pos[pos].unsqueeze(0)

    # Defensively coerce the residual stream to the model dtype so every
    # downstream rel/irrel tensor inherits it (W_E is already cfg.dtype, but
    # this also covers any folded-embedding edge cases).
    hidden = hidden.to(dtype=model_dtype)

    # Split into relevant and irrelevant.
    rel = torch.zeros_like(hidden)
    irrel = hidden.clone()
    for p in rel_indices:
        if 0 <= p < hidden.shape[1]:
            rel[:, p, :] = hidden[:, p, :]
            irrel[:, p, :] = 0.0

    contributions: Dict[str, float] = {}

    for lyr in range(n_layers):
        block = tl_model.blocks[lyr]

        # Pre-attention norm (LayerNorm or RMSNorm).
        r_n, i_n = _prop_tl_norm(rel, irrel, block.ln1, tol)

        # Attention with frozen pattern (cloned to a plain tensor so it is
        # never an inference tensor when reused across the einsum below).
        # Cast to the model dtype: on 16-bit models the cached pattern is
        # float32 (captured before TL's final cast), which would otherwise
        # make the V-branch einsum fail against the bf16/fp16 weights.
        pattern = (
            cache[f"blocks.{lyr}.attn.hook_pattern"].clone().to(dtype=model_dtype)
        )  # [1, H, q, k]
        r_a, i_a, r_z = _prop_tl_attn(r_n, i_n, block.attn, pattern, tol)

        # Per-head attribution at the target position.
        for h in range(n_heads):
            contributions[f"A{lyr}.{h}"] = float(r_z[0, tgt, h, :].norm().item())

        rel = rel + r_a
        irrel = irrel + i_a
        rel, irrel = _normalize_rel_irrel(rel, irrel)

        # Pre-MLP norm.
        r_n, i_n = _prop_tl_norm(rel, irrel, block.ln2, tol)

        # MLP — gated or non-gated detected via TL attribute.
        if hasattr(block.mlp, "W_gate"):
            r_m, i_m = _prop_tl_gated_mlp(r_n, i_n, block.mlp, tol)
        else:
            r_m, i_m = _prop_tl_mlp(r_n, i_n, block.mlp, tol)

        contributions[f"MLP {lyr}"] = float(r_m[0, tgt, :].norm().item())

        rel = rel + r_m
        irrel = irrel + i_m
        rel, irrel = _normalize_rel_irrel(rel, irrel)

    return contributions


# ── Backward-compat shim (HF-based, kept for any external callers) ────────────


def cd_propagate(model, encoding: dict, rel_indices: List[int]) -> Dict[str, float]:
    """Legacy entry point for HF causal-LM models.

    Prefer ``cd_propagate_tl`` for all new code; that function works for every
    TL-supported architecture without requiring a separate HF model load.
    This shim raises NotImplementedError for unsupported architectures so the
    adapter can fall back to the simplified TL metric.
    """
    raise NotImplementedError(
        "cd_propagate (HF-based) is deprecated. Use cd_propagate_tl with a "
        "TransformerLens HookedTransformer instead — it supports all TL "
        "architectures without loading a second copy of the model."
    )
