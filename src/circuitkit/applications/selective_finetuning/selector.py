"""
selector.py — Component selection for selective finetuning.

Given the normalised score dicts produced by score_loader.load_scores(), this
module selects the top-X% most important components (attention heads and/or MLP
layers/neurons), resolves them to concrete weight-matrix index ranges, and
provides random and baseline selection conditions that mirror the same structure.

Output interface
----------------
SelectionResult.attn : Dict[str, Dict[str, List[int] | None]]
    Outer key  : "attn_{layer}"
    Inner keys : "q", "k", "v", "o"  (node-level)
                 "o"                  (neuron-level — only output projection)
    Values     : Sorted list of row indices (q/k/v) or column indices (o)
                 into the corresponding weight matrix.
                 None → train the entire projection without masking (baseline).

SelectionResult.mlp : Dict[str, List[int] | None]
    Key    : "mlp_{layer}"
    Value  : Sorted list of down_proj column indices (neuron-level)
             None → train all columns without masking (node-level or baseline).

Projection index arithmetic
---------------------------
  q_proj shape : [n_q_heads  * head_dim, d_model]  — rows selected per Q-head
  k_proj shape : [n_kv_heads * head_dim, d_model]  — rows selected per KV-head
  v_proj shape : [n_kv_heads * head_dim, d_model]  — rows selected per KV-head
  o_proj shape : [d_model,  n_q_heads  * head_dim] — cols selected per Q-head
  down_proj    : [d_model,  d_mlp]                 — cols selected per neuron

GQA: group_size = n_q_heads // n_kv_heads
     kv_head    = q_head // group_size
     KV indices are deduplicated when multiple Q-heads share one KV group.

Usage
-----
    head_scores, mlp_scores, metadata = load_scores(scores_path)

    circuit  = select_components(head_scores, mlp_scores, metadata,
                                 top_frac=0.1, scope="both", ...)
    rand     = random_selection(head_scores, mlp_scores, metadata,
                                circuit_result=circuit, ...)
    baseline = build_baseline_selection(head_scores, mlp_scores, metadata,
                                        scope="both", ...)

    print_selection_summary(circuit, rand, baseline, head_dim=head_dim)
"""

from __future__ import annotations

import random as _random
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

HeadScores = Dict[Tuple[int, int], float]
MLPScores = Dict[int, Union[float, torch.Tensor]]

# "attn_{layer}" -> {"q"|"k"|"v"|"o" -> sorted index list or None}
AttnResult = Dict[str, Dict[str, Optional[List[int]]]]
# "mlp_{layer}" -> sorted column index list or None
MLPResult = Dict[str, Optional[List[int]]]


# ---------------------------------------------------------------------------
# SelectionResult
# ---------------------------------------------------------------------------


class SelectionResult(NamedTuple):
    """
    Resolved component selection for one condition (circuit / random / baseline).

    attn : Dict["attn_{layer}", Dict["q"|"k"|"v"|"o", List[int] | None]]
    mlp  : Dict["mlp_{layer}",  List[int] | None]

    A None index list means: enable the full weight matrix with no gradient
    mask (used for node-level MLP and the baseline condition).
    """

    attn: AttnResult
    mlp: MLPResult


# ---------------------------------------------------------------------------
# Layer exclusion filter
# ---------------------------------------------------------------------------


def _filter_by_layer_exclusion(
    head_scores: HeadScores,
    mlp_scores: MLPScores,
    exclude_first_n: int,
    exclude_last_n: int,
    n_layers: int,
) -> Tuple[HeadScores, MLPScores]:
    """
    Return filtered copies of both dicts containing only layers in
    [exclude_first_n, n_layers - 1 - exclude_last_n].

    Called before any ranking so excluded layers never influence selection.
    """
    if exclude_first_n < 0 or exclude_last_n < 0:
        raise ValueError(
            f"exclude_first_n and exclude_last_n must be non-negative, "
            f"got {exclude_first_n} and {exclude_last_n}."
        )
    first_allowed = exclude_first_n
    last_allowed = n_layers - 1 - exclude_last_n
    if first_allowed > last_allowed:
        raise ValueError(
            f"Layer exclusion leaves no layers: "
            f"first_allowed={first_allowed} > last_allowed={last_allowed}. "
            f"Reduce exclude_first_n or exclude_last_n."
        )

    filtered_head: HeadScores = {
        (lyr, h): s for (lyr, h), s in head_scores.items() if first_allowed <= lyr <= last_allowed
    }
    filtered_mlp: MLPScores = {
        lyr: s for lyr, s in mlp_scores.items() if first_allowed <= lyr <= last_allowed
    }
    return filtered_head, filtered_mlp


# ---------------------------------------------------------------------------
# GQA index resolution
# ---------------------------------------------------------------------------


def _resolve_gqa_indices(
    layer_head_pairs: List[Tuple[int, int]],
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    neuron_level: bool,
) -> AttnResult:
    """
    Convert a list of (layer, head) pairs into per-layer, per-projection index
    lists ready for gradient masking.

    Parameters
    ----------
    layer_head_pairs : List of (layer, q_head_index) tuples to resolve.
    n_q_heads        : Total Q heads per layer.
    n_kv_heads       : Total KV heads per layer (≤ n_q_heads).
    head_dim         : Dimension per attention head (d_model // n_q_heads).
    neuron_level     : If True, only populate "o"; q/k/v are omitted because
                       attention scores were aggregated from neuron-level data
                       and only the output projection is masked.

    Returns
    -------
    Dict["attn_{layer}", Dict["q"|"k"|"v"|"o", List[int]]]
    """
    assert n_q_heads % n_kv_heads == 0, (
        f"n_q_heads ({n_q_heads}) must be divisible by "
        f"n_kv_heads ({n_kv_heads}) for GQA grouping."
    )
    group_size = n_q_heads // n_kv_heads

    # Group selected heads by layer.
    by_layer: Dict[int, List[int]] = {}
    for layer, head in layer_head_pairs:
        by_layer.setdefault(layer, []).append(head)

    result: AttnResult = {}

    for layer, heads in by_layer.items():
        heads = sorted(set(heads))

        # o_proj column indices — same arithmetic as q_proj rows.
        # o_proj shape: [d_model, n_q_heads * head_dim]  → columns selected.
        o_cols: List[int] = sorted(
            {col for h in heads for col in range(h * head_dim, (h + 1) * head_dim)}
        )
        assert o_cols[-1] < n_q_heads * head_dim, (
            f"o_proj column index {o_cols[-1]} out of bounds "
            f"(n_q_heads={n_q_heads}, head_dim={head_dim}, "
            f"max_valid={n_q_heads * head_dim - 1})"
        )

        if neuron_level:
            # Only output projection is masked at neuron level.
            result[f"attn_{layer}"] = {"o": o_cols}
            continue

        # q_proj row indices — identical arithmetic to o_proj columns.
        # q_proj shape: [n_q_heads * head_dim, d_model]  → rows selected.
        q_rows: List[int] = o_cols  # same index range

        # k_proj / v_proj row indices — GQA grouped.
        # Multiple Q-heads share one KV-head; deduplicate via set.
        # k/v_proj shape: [n_kv_heads * head_dim, d_model]  → rows selected.
        kv_rows: List[int] = sorted(
            {
                row
                for h in heads
                for kv_h in [h // group_size]
                for row in range(kv_h * head_dim, (kv_h + 1) * head_dim)
            }
        )
        assert kv_rows[-1] < n_kv_heads * head_dim, (
            f"k/v_proj row index {kv_rows[-1]} out of bounds "
            f"(n_kv_heads={n_kv_heads}, head_dim={head_dim}, "
            f"max_valid={n_kv_heads * head_dim - 1})"
        )

        result[f"attn_{layer}"] = {
            "q": q_rows,
            "k": kv_rows,
            "v": kv_rows,  # k and v share the same index range
            "o": o_cols,
        }

    return result


# ---------------------------------------------------------------------------
# MLP resolution helpers
# ---------------------------------------------------------------------------


def _resolve_mlp_node(selected_layers: List[int]) -> MLPResult:
    """
    Node-level MLP: selected layers → None (train full down_proj, no masking).
    """
    return {f"mlp_{layer}": None for layer in selected_layers}


def _resolve_mlp_neuron(
    mlp_scores: MLPScores,
    candidate_layers: List[int],
    top_frac: float,
) -> MLPResult:
    """
    Neuron-level MLP: rank all neurons across all candidate layers globally by
    absolute score, take top top_frac, re-group into per-layer column index lists.

    Parameters
    ----------
    mlp_scores       : Full filtered MLP scores dict (Tensor values expected).
    candidate_layers : Layers to consider (already layer-exclusion filtered).
    top_frac         : Fraction of total neurons to select (e.g. 0.1 = top 10%).

    Returns
    -------
    Dict["mlp_{layer}", List[int]]  — sorted down_proj column indices per layer.
                                      Only layers that received ≥1 neuron are
                                      included (sparse layers may be absent).
    """
    # Collect (abs_score, layer, neuron_col) triples from all candidate layers.
    triples: List[Tuple[float, int, int]] = []
    for layer in candidate_layers:
        tensor = mlp_scores.get(layer)
        if tensor is None or not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"mlp_scores[{layer}] must be a Tensor for neuron-level MLP, "
                f"got {type(tensor)}. Check metadata['mlp_neuron_level']."
            )
        for col, score in enumerate(tensor.tolist()):
            triples.append((abs(score), layer, col))

    if not triples:
        return {}

    triples.sort(key=lambda x: x[0], reverse=True)
    k = max(1, round(len(triples) * top_frac))
    selected = triples[:k]

    # Re-group by layer into sorted column lists.
    by_layer: Dict[int, List[int]] = {}
    for _, layer, col in selected:
        by_layer.setdefault(layer, []).append(col)

    return {f"mlp_{layer}": sorted(cols) for layer, cols in by_layer.items()}


# ---------------------------------------------------------------------------
# Head count helper  (used by random_selection to mirror circuit counts)
# ---------------------------------------------------------------------------


def _count_selected_heads(attn_result: AttnResult, head_dim: int) -> int:
    """
    Count total unique (layer, head) pairs encoded in an AttnResult.

    Works for both node-level (q/k/v/o keys) and neuron-level (o key only).
    Uses the "o" entry which always holds one contiguous block per Q-head.
    """
    total = 0
    for proj_dict in attn_result.values():
        o_cols = proj_dict.get("o")
        if o_cols is not None:
            # Each head contributes exactly head_dim consecutive column indices.
            total += len(o_cols) // head_dim
    return total


# ---------------------------------------------------------------------------
# Main selection
# ---------------------------------------------------------------------------


def select_components(
    head_scores: HeadScores,
    mlp_scores: MLPScores,
    metadata: Dict[str, Any],
    top_frac: float,
    scope: str,
    n_layers: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    exclude_first_n: int = 0,
    exclude_last_n: int = 0,
) -> SelectionResult:
    """
    Select the top-X% circuit components ranked by absolute attribution score.

    Parameters
    ----------
    head_scores     : Dict[(layer, head), float] from load_scores().
    mlp_scores      : Dict[layer, float | Tensor] from load_scores().
    metadata        : Dict from load_scores() — must carry 'level' and
                      'mlp_neuron_level'.
    top_frac        : Fraction in (0, 1] of components to select.
                      Applied globally (not per-layer) within each component type.
    scope           : Which component types to process:
                        "attn" — attention heads only
                        "mlp"  — MLP layers/neurons only
                        "both" — both attention and MLP
    n_layers        : Total transformer layer count.
    n_q_heads       : Q heads per layer.
    n_kv_heads      : KV heads per layer (≤ n_q_heads; equal for MHA).
    head_dim        : Dimension per attention head.
    exclude_first_n : Layers [0, exclude_first_n) are skipped.
    exclude_last_n  : Layers (n_layers - 1 - exclude_last_n, n_layers) are skipped.

    Returns
    -------
    SelectionResult(attn, mlp)
    """
    if not (0.0 < top_frac <= 1.0):
        raise ValueError(f"top_frac must be in (0, 1], got {top_frac}.")
    if scope not in ("attn", "mlp", "both"):
        raise ValueError(f"scope must be 'attn', 'mlp', or 'both', got {scope!r}.")

    neuron_level = metadata.get("level") == "neuron"
    mlp_neuron_level = metadata.get("mlp_neuron_level", False)

    # ── Step 1: apply layer exclusion ─────────────────────────────────────
    filt_head, filt_mlp = _filter_by_layer_exclusion(
        head_scores, mlp_scores, exclude_first_n, exclude_last_n, n_layers
    )

    attn_result: AttnResult = {}
    mlp_result: MLPResult = {}

    # ── Step 2a: attention — global ranking of (layer, head) pairs ────────
    if scope in ("attn", "both") and filt_head:
        ranked = sorted(filt_head.items(), key=lambda x: abs(x[1]), reverse=True)
        k = max(1, round(len(ranked) * top_frac))
        selected_pairs: List[Tuple[int, int]] = [pair for pair, _ in ranked[:k]]
        attn_result = _resolve_gqa_indices(
            selected_pairs,
            n_q_heads,
            n_kv_heads,
            head_dim,
            neuron_level=neuron_level,
        )

    # ── Step 2b: MLP — node or neuron path ────────────────────────────────
    if scope in ("mlp", "both") and filt_mlp:
        if mlp_neuron_level:
            # Global neuron ranking across ALL allowed MLP layers.
            attn_result_neuron = _resolve_mlp_neuron(
                filt_mlp,
                candidate_layers=list(filt_mlp.keys()),
                top_frac=top_frac,
            )
            mlp_result = attn_result_neuron
        else:
            # Rank layers by their scalar score, take top top_frac layers.
            ranked_mlp = sorted(filt_mlp.items(), key=lambda x: abs(x[1]), reverse=True)
            k_mlp = max(1, round(len(ranked_mlp) * top_frac))
            selected_layers = [layer for layer, _ in ranked_mlp[:k_mlp]]
            mlp_result = _resolve_mlp_node(selected_layers)

    return SelectionResult(attn=attn_result, mlp=mlp_result)


# ---------------------------------------------------------------------------
# Random selection
# ---------------------------------------------------------------------------


def random_selection(
    head_scores: HeadScores,
    mlp_scores: MLPScores,
    metadata: Dict[str, Any],
    circuit_result: SelectionResult,
    n_layers: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    exclude_first_n: int = 0,
    exclude_last_n: int = 0,
    seed: int = 42,
) -> SelectionResult:
    """
    Mirror the circuit selection with equal component counts drawn uniformly.

    Uses two independent RNG instances to ensure attention and MLP draws do
    not interfere:
      rng_attn — seeded with seed     (attention head sampling)
      rng_mlp  — seeded with seed + 1 (MLP layer / neuron sampling)

    The resolved structure (GQA index arithmetic, neuron-level flags) is
    identical to select_components so finetune_utils.py sees the same API.
    """
    neuron_level = metadata.get("level") == "neuron"
    mlp_neuron_level = metadata.get("mlp_neuron_level", False)

    rng_attn = _random.Random(seed)
    rng_mlp = _random.Random(seed + 1)

    filt_head, filt_mlp = _filter_by_layer_exclusion(
        head_scores, mlp_scores, exclude_first_n, exclude_last_n, n_layers
    )

    attn_result: AttnResult = {}
    mlp_result: MLPResult = {}

    # ── Attention: match head count from circuit ───────────────────────────
    if circuit_result.attn and filt_head:
        n_circuit_heads = _count_selected_heads(circuit_result.attn, head_dim)
        all_pairs = list(filt_head.keys())
        n_select = min(n_circuit_heads, len(all_pairs))
        selected_pairs = rng_attn.sample(all_pairs, n_select)
        attn_result = _resolve_gqa_indices(
            selected_pairs,
            n_q_heads,
            n_kv_heads,
            head_dim,
            neuron_level=neuron_level,
        )

    # ── MLP: match layer or neuron count from circuit ─────────────────────
    if circuit_result.mlp and filt_mlp:
        if mlp_neuron_level:
            # Count total neurons in circuit selection.
            n_circuit_neurons = sum(
                len(idxs) for idxs in circuit_result.mlp.values() if idxs is not None
            )
            # Pool all available (layer, col) pairs from filtered MLP layers.
            all_neurons: List[Tuple[int, int]] = []
            for layer, tensor in filt_mlp.items():
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"mlp_scores[{layer}] must be a Tensor for neuron-level random selection, "
                        f"got {type(tensor)}. Check metadata['mlp_neuron_level']."
                    )
                all_neurons.extend((layer, col) for col in range(tensor.numel()))
            n_select = min(n_circuit_neurons, len(all_neurons))
            selected_neurons = rng_mlp.sample(all_neurons, n_select)

            by_layer: Dict[int, List[int]] = {}
            for layer, col in selected_neurons:
                by_layer.setdefault(layer, []).append(col)
            mlp_result = {f"mlp_{layer}": sorted(cols) for layer, cols in by_layer.items()}
        else:
            # Match number of layers selected in circuit.
            n_circuit_layers = len(circuit_result.mlp)
            all_layers = list(filt_mlp.keys())
            n_select = min(n_circuit_layers, len(all_layers))
            selected_layers = rng_mlp.sample(all_layers, n_select)
            mlp_result = _resolve_mlp_node(selected_layers)

    return SelectionResult(attn=attn_result, mlp=mlp_result)


# ---------------------------------------------------------------------------
# Baseline selection
# ---------------------------------------------------------------------------


def build_baseline_selection(
    head_scores: HeadScores,
    mlp_scores: MLPScores,
    metadata: Dict[str, Any],
    scope: str,
    n_layers: int,
    exclude_first_n: int = 0,
    exclude_last_n: int = 0,
) -> SelectionResult:
    """
    Include all non-excluded in-scope layers with None index lists.

    None means: enable the weight matrix with requires_grad=True but register
    no gradient hook — all rows/columns update freely.  This is the upper-bound
    condition against which circuit and random finetuning are compared.

    The same projection key structure is used as circuit/random so
    finetune_utils.py can iterate all three conditions with a uniform API.
    """
    if scope not in ("attn", "mlp", "both"):
        raise ValueError(f"scope must be 'attn', 'mlp', or 'both', got {scope!r}.")

    neuron_level = metadata.get("level") == "neuron"

    filt_head, filt_mlp = _filter_by_layer_exclusion(
        head_scores, mlp_scores, exclude_first_n, exclude_last_n, n_layers
    )

    attn_result: AttnResult = {}
    mlp_result: MLPResult = {}

    if scope in ("attn", "both") and filt_head:
        # Include every non-excluded layer that has attention scores.
        # Projection values are None → no masking.
        layers = sorted({lyr for lyr, _ in filt_head})
        for layer in layers:
            if neuron_level:
                attn_result[f"attn_{layer}"] = {"o": None}
            else:
                attn_result[f"attn_{layer}"] = {"q": None, "k": None, "v": None, "o": None}

    if scope in ("mlp", "both") and filt_mlp:
        for layer in sorted(filt_mlp.keys()):
            mlp_result[f"mlp_{layer}"] = None

    return SelectionResult(attn=attn_result, mlp=mlp_result)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_selection_summary(
    circuit: SelectionResult,
    random: SelectionResult,
    baseline: SelectionResult,
    head_dim: int,
) -> None:
    """
    Print a human-readable breakdown of all three selection conditions.

    Shows per-condition head/neuron/layer counts and which layers are touched,
    so the user can quickly verify the selection makes sense before training.
    """
    width = 64
    logger.info(f"\n{'=' * width}")
    logger.info("  SELECTION SUMMARY")
    logger.info(f"{'=' * width}")

    conditions = [("Circuit", circuit), ("Random", random), ("Baseline", baseline)]

    for label, result in conditions:
        logger.info(f"\n  [{label}]")

        # ── Attention summary ──────────────────────────────────────────────
        if result.attn:
            layers_touched = sorted(int(key.split("_")[1]) for key in result.attn)
            # Count heads — None entries (baseline) are reported separately.
            total_heads = 0
            has_none = False
            for proj_dict in result.attn.values():
                o_val = proj_dict.get("o")
                if o_val is None:
                    has_none = True
                else:
                    total_heads += len(o_val) // head_dim

            head_str = "ALL (no mask)" if has_none else str(total_heads)
            proj_keys = sorted({k for pd in result.attn.values() for k in pd})
            logger.info(
                f"    Attention : {len(result.attn)} layers | "
                f"{head_str} heads | "
                f"projections: {proj_keys}\n"
                f"               layers touched: {layers_touched}"
            )
        else:
            logger.info("    Attention : (not in scope)")

        # ── MLP summary ───────────────────────────────────────────────────
        if result.mlp:
            layers_touched = sorted(int(key.split("_")[1]) for key in result.mlp)
            any_none = any(v is None for v in result.mlp.values())
            if any_none:
                logger.info(
                    f"    MLP       : {len(result.mlp)} layers | "
                    f"ALL neurons (no mask)\n"
                    f"               layers touched: {layers_touched}"
                )
            else:
                total_neurons = sum(len(v) for v in result.mlp.values() if v is not None)
                logger.info(
                    f"    MLP       : {len(result.mlp)} layers | "
                    f"{total_neurons} neurons total\n"
                    f"               layers touched: {layers_touched}"
                )
        else:
            logger.info("    MLP       : (not in scope)")

    logger.info(f"\n{'=' * width}\n")
