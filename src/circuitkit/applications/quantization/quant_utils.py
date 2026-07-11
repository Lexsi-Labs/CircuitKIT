"""
quant_utils.py — Circuit-guided mixed-precision quantization via optimum-quanto.

Uses circuit discovery scores (produced by score_extractor.py in the pruning
application) to assign each transformer layer to a quantization tier.
High-importance layers get higher precision; low-importance layers are
quantized more aggressively.

Design
------
* Layer-level scores are aggregated from per-head attention scores and
  per-neuron MLP scores (same format produced by score_extractor).
* Layers are ranked by importance and split into tiers (high / mid / low).
* Each tier is quantized separately using optimum-quanto's ``quantize()``
  with non-overlapping ``include=`` pattern lists so no layer is quantized
  twice.
* An optional calibration step uses the circuit discovery task data (already
  collected during discovery) to optimise activation scales.
* ``compute_ppl`` provides a standard perplexity metric for comparing
  quantization quality against a wikitext-2 held-out set.

Optimum-quanto path note
------------------------
The optimum-quanto root must be on sys.path before importing from this
module. Example scripts in this package add it automatically.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------


def compute_layer_scores(
    q_head_scores: Dict[Tuple[int, int], float],
    mlp_scores: Dict[int, torch.Tensor],
    n_layers: int,
    aggregation: str = "mean",
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Aggregate per-head and per-neuron circuit scores to per-layer scalars.

    Parameters
    ----------
    q_head_scores  : Dict[(layer, q_head), float] — from extract_node_head_scores.
    mlp_scores     : Dict[layer, float] (node-level) or Dict[layer, Tensor(d_mlp)]
                     (neuron-level) — from extract_node_mlp_scores or
                     extract_mlp_neuron_scores respectively.
    n_layers       : Total number of transformer layers in the model.
    aggregation    : How to reduce scores within a layer: "mean", "sum", "max".

    Returns
    -------
    (attn_layer_scores, mlp_layer_scores) — both Dict[int, float].
    Layers absent from the input dicts receive a score of 0.0.
    """
    _agg = {
        "mean": lambda vals: sum(vals) / len(vals) if vals else 0.0,
        "sum": sum,
        "max": max,
    }
    if aggregation not in _agg:
        raise ValueError(f"aggregation must be one of {list(_agg)}; got {aggregation!r}")
    reduce = _agg[aggregation]

    # --- Attention: group Q-head scores by layer ---
    attn_by_layer: Dict[int, List[float]] = {i: [] for i in range(n_layers)}
    for (layer, _head), score in q_head_scores.items():
        if layer in attn_by_layer:
            attn_by_layer[layer].append(abs(score))

    attn_layer_scores: Dict[int, float] = {}
    for layer in range(n_layers):
        vals = attn_by_layer[layer]
        attn_layer_scores[layer] = reduce(vals) if vals else 0.0

    # --- MLP: node-level float or neuron-level tensor ---
    mlp_layer_scores: Dict[int, float] = {}
    for layer in range(n_layers):
        if layer in mlp_scores:
            val = mlp_scores[layer]
            if isinstance(val, torch.Tensor):
                # Neuron-level: aggregate tensor to scalar
                vals = val.float().abs().tolist()
                mlp_layer_scores[layer] = reduce(vals) if vals else 0.0
            else:
                # Node-level: already a scalar
                mlp_layer_scores[layer] = abs(float(val))
        else:
            mlp_layer_scores[layer] = 0.0

    return attn_layer_scores, mlp_layer_scores


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------


def build_quantization_plan(
    attn_layer_scores: Dict[int, float],
    mlp_layer_scores: Dict[int, float],
    high_fraction: float = 0.3,
    mid_fraction: float = 0.0,
) -> Dict[str, Dict[int, str]]:
    """
    Assign each layer to a quantization tier based on its importance score.

    Layers are ranked independently for attention and MLP.  The top
    ``high_fraction`` (by score) are labelled ``"high"``, the next
    ``mid_fraction`` are ``"mid"``, and the remainder are ``"low"``.

    Parameters
    ----------
    attn_layer_scores : Dict[int, float] — from compute_layer_scores.
    mlp_layer_scores  : Dict[int, float] — from compute_layer_scores.
    high_fraction     : Fraction of layers to place in the high tier [0, 1].
    mid_fraction      : Fraction of layers to place in the mid tier [0, 1].
                        high_fraction + mid_fraction must be ≤ 1.

    Returns
    -------
    Dict with keys "attn" and "mlp", each mapping layer_idx → tier string.
    """
    if high_fraction + mid_fraction > 1.0 + 1e-6:
        raise ValueError(
            f"high_fraction ({high_fraction}) + mid_fraction ({mid_fraction}) must be ≤ 1"
        )

    def _assign(scores: Dict[int, float]) -> Dict[int, str]:
        layers = sorted(scores.keys())
        n = len(layers)
        if n == 0:
            return {}
        # Sort descending by score (higher = more important)
        ranked = sorted(layers, key=lambda i: scores[i], reverse=True)
        # ``round(fraction * n)`` uses Python's banker's-rounding, so a
        # request like ``high_fraction = 0.05`` on n = 28 layers (0.05 * 28
        # = 1.4) snaps to 1, while ``high_fraction = 0.15`` on n = 34
        # (0.15 * 34 = 5.1) snaps to 5. The actual ``bits_per_weight``
        # report uses the requested ``high_fraction`` (the IDEAL fraction)
        # — the realised tier size can differ by ±1 layer due to this
        # rounding. Acceptable noise for the audit since the budget tag
        # ``b{base}p{protect}f{frac}`` is the ground-truth experimental
        # identifier; the realised count is observable in the
        # ``effective_sparsity`` field of each cell's JSON. A
        # rounding-tied-down ``int(...)`` would be strictly conservative
        # but would also push every "5.1" case to 5 (same here) while
        # pushing "1.4" down to 1 (same here) — i.e. no behaviour change
        # in practice for our budgets. We keep ``round`` for symmetry
        # with the paper's "X% of layers protected" framing.
        n_high = max(0, round(high_fraction * n))
        n_mid = max(0, round(mid_fraction * n))
        high_set = set(ranked[:n_high])
        mid_set = set(ranked[n_high : n_high + n_mid])
        tier_map: Dict[int, str] = {}
        for layer in layers:
            if layer in high_set:
                tier_map[layer] = "high"
            elif layer in mid_set:
                tier_map[layer] = "mid"
            else:
                tier_map[layer] = "low"
        return tier_map

    return {
        "attn": _assign(attn_layer_scores),
        "mlp": _assign(mlp_layer_scores),
    }


# ---------------------------------------------------------------------------
# Random tier assignment (baseline)
# ---------------------------------------------------------------------------


def build_random_quantization_plan(
    n_layers: int,
    high_fraction: float = 0.3,
    mid_fraction: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, Dict[int, str]]:
    """
    Assign each layer to a quantization tier **randomly**.

    Attention and MLP tiers are shuffled independently (separate RNG draws)
    so the random baseline is a fair comparison to the circuit-guided plan,
    which also ranks them independently.  The same high/mid fractions are
    used so the total number of layers in each tier is identical.

    Parameters
    ----------
    n_layers      : Total number of transformer layers.
    high_fraction : Fraction of layers to place in the high tier.
    mid_fraction  : Fraction of layers to place in the mid tier.
    seed          : Optional random seed for reproducibility.

    Returns
    -------
    Dict with keys "attn" and "mlp", each mapping layer_idx → tier string.
    """
    import random

    if high_fraction + mid_fraction > 1.0 + 1e-6:
        raise ValueError(
            f"high_fraction ({high_fraction}) + mid_fraction ({mid_fraction}) must be ≤ 1"
        )

    def _random_assign(rng: random.Random) -> Dict[int, str]:
        layers = list(range(n_layers))
        n_high = max(0, round(high_fraction * n_layers))
        n_mid = max(0, round(mid_fraction * n_layers))
        shuffled = layers.copy()
        rng.shuffle(shuffled)
        high_set = set(shuffled[:n_high])
        mid_set = set(shuffled[n_high : n_high + n_mid])
        return {
            layer: ("high" if layer in high_set else "mid" if layer in mid_set else "low")
            for layer in layers
        }

    rng_attn = random.Random(seed)
    rng_mlp = random.Random(None if seed is None else seed + 1)

    return {
        "attn": _random_assign(rng_attn),
        "mlp": _random_assign(rng_mlp),
    }


# ---------------------------------------------------------------------------
# Random quantization entry point
# ---------------------------------------------------------------------------


def random_quantize(
    model: nn.Module,
    n_layers: int,
    low_weights=None,
    high_weights=None,
    mid_weights=None,
    activations=None,
    high_fraction: float = 0.3,
    mid_fraction: float = 0.0,
    exclude_lm_head: bool = True,
    model_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Dict[int, str]]:
    """
    Apply mixed-precision quantization with **randomly assigned** tiers.

    Identical tier fractions and qtypes as ``circuit_quantize``, but layers
    are assigned to tiers randomly rather than by circuit importance score.
    Used as a fair comparison baseline: the only difference is the ordering
    of layers into tiers.

    Supports all transformer architectures through automatic detection.

    Parameters
    ----------
    model         : HuggingFace CausalLM (in-place modified).
    n_layers      : Number of transformer layers.
    low_weights   : qtype for randomly-chosen "low" layers.
    high_weights  : qtype for randomly-chosen "high" layers (None = native).
    mid_weights   : qtype for randomly-chosen "mid" layers.
    activations   : Activation qtype, or None for weights-only.
    high_fraction : Fraction of layers randomly assigned to high tier.
    mid_fraction  : Fraction of layers randomly assigned to mid tier.
    exclude_lm_head : Whether to always exclude lm_head.
    model_type    : (Deprecated) Auto-detected from model.config.model_type.
                    Ignored if provided. Kept for backward compatibility.
    seed          : Random seed (for reproducibility).

    Returns
    -------
    The random tier assignment plan dict for inspection and logging.

    Raises
    ------
    UnsupportedArchitectureError: If model architecture not supported.
    ArchitectureValidationError: If model structure doesn't match expected paths.
    """
    from circuitkit.applications import (
        detect_model_architecture,
        get_arch_config,
        validate_model_paths,
    )

    if low_weights is None:
        # Resolve the documented default lazily so importing this module
        # doesn't hard-require optimum-quanto (an optional extra) — and guard
        # it with an actionable message, per pyproject's contract for this
        # module ("imported lazily with an ImportError guard").
        try:
            from optimum.quanto import qint4
        except ImportError as exc:
            raise ImportError(
                "The quanto quantization backend requires the optional "
                "dependency optimum-quanto. Install it with: "
                "pip install 'circuitkit[quantization]' "
                "(or: pip install 'optimum-quanto>=0.2')."
            ) from exc

        low_weights = qint4

    # Auto-detect architecture (ignore model_type parameter if provided)
    detected_type = detect_model_architecture(model)
    arch_cfg = get_arch_config(detected_type)
    validate_model_paths(model, arch_cfg)

    plan = build_random_quantization_plan(
        n_layers,
        high_fraction=high_fraction,
        mid_fraction=mid_fraction,
        seed=seed,
    )

    _apply_plan_to_model(
        model,
        plan,
        low_weights,
        mid_weights,
        high_weights,
        activations,
        exclude_lm_head,
        arch_cfg,
    )

    return plan


# ---------------------------------------------------------------------------
# Pattern building
# ---------------------------------------------------------------------------


def build_patterns(
    layer_indices: List[int],
    component: str,
    arch_cfg: Dict[str, Any],
) -> List[str]:
    """
    Convert layer indices to fnmatch patterns for optimum-quanto's
    include/exclude parameters.

    Uses architecture config to support multiple model types (LLaMA, Qwen, GPT-2,
    Falcon, etc.) with different layer paths and module names.

    Parameters
    ----------
    layer_indices : Layer indices to include.
    component     : "attn" or "mlp".
    arch_cfg      : Architecture config dict from MODEL_ARCH_REGISTRY.

    Returns
    -------
    List of fnmatch pattern strings, e.g.
    ["model.layers.5.self_attn.*", "transformer.h.12.attn.*"].
    """
    if component == "attn":
        submodule = arch_cfg["attn"]["module"]
    elif component == "mlp":
        # Prefer the per-arch MLP submodule name if the config provides it;
        # fall back to the literal ``"mlp"`` which works for Llama / Gemma /
        # Qwen / Mistral (every model in the audit panel). A future arch
        # that named its MLP block differently would otherwise generate
        # patterns that match nothing.
        mlp_cfg = arch_cfg.get("mlp", {})
        submodule = mlp_cfg.get("module", "mlp") if isinstance(mlp_cfg, dict) else "mlp"
    else:
        raise ValueError(f"component must be 'attn' or 'mlp'; got {component!r}")

    # Get the layers path (first one from the list, e.g., "model.layers" or "transformer.h")
    layers_path = arch_cfg["layers_path"][0]

    return [f"{layers_path}.{i}.{submodule}.*" for i in layer_indices]


# ---------------------------------------------------------------------------
# Main quantization entry point
# ---------------------------------------------------------------------------


def circuit_quantize(
    model: nn.Module,
    q_head_scores: Dict[Tuple[int, int], float],
    mlp_scores: Dict[int, torch.Tensor],
    n_layers: int,
    low_weights=None,  # qtype — most aggressive quantization for low tier; None -> qint4
    high_weights=None,  # qtype or None — None keeps native precision
    mid_weights=None,  # qtype or None — only used when mid_fraction > 0
    activations=None,  # qtype or None
    high_fraction: float = 0.3,
    mid_fraction: float = 0.0,
    score_aggregation: str = "mean",
    exclude_lm_head: bool = True,
    model_type: Optional[str] = None,
    protect_layers: Optional[List[int]] = None,
) -> Dict[str, Dict[int, str]]:
    """
    Apply mixed-precision quantization guided by circuit importance scores.

    Layers are ranked by importance and assigned to tiers.  Each tier is
    quantized with a different weight qtype by calling optimum-quanto's
    ``quantize()`` once per tier with non-overlapping ``include=`` patterns.
    Because the patterns are disjoint no layer is quantized twice.

    Supports all transformer architectures (LLaMA, Qwen, Gemma, GPT-2, Falcon,
    Mistral, Phi, etc.) through automatic architecture detection.

    Parameters
    ----------
    model           : HuggingFace CausalLM (in-place modified).
    q_head_scores   : Per-Q-head scores, Dict[(layer, head), float].
    mlp_scores      : Per-neuron MLP scores, Dict[layer, Tensor].
    n_layers        : Number of transformer layers.
    low_weights     : qtype for least-important layers (default qint4).
    high_weights    : qtype for most-important layers, or None to leave them
                      unquantized at native precision.
    mid_weights     : qtype for mid-importance layers (only used when
                      mid_fraction > 0).
    activations     : Activation qtype applied uniformly to all quantized
                      layers, or None for weights-only quantization.
    high_fraction   : Fraction of layers protected as high-importance.
    mid_fraction    : Fraction of layers in the mid tier.
    score_aggregation : "mean", "sum", or "max".
    exclude_lm_head : Whether to always exclude lm_head from quantization.
    model_type      : (Deprecated) Auto-detected from model.config.model_type.
                      Ignored if provided. Kept for backward compatibility.
    protect_layers  : Layer indices to leave at native precision — both their
                      attention and MLP submodules are excluded from every
                      quantization tier. None protects nothing.

    Returns
    -------
    The tier assignment plan dict ({"attn": {layer: tier}, "mlp": {layer: tier}})
    for inspection and logging.

    Raises
    ------
    UnsupportedArchitectureError: If model architecture not supported.
    ArchitectureValidationError: If model structure doesn't match expected paths.
    """

    from circuitkit.applications import (
        detect_model_architecture,
        get_arch_config,
        validate_model_paths,
    )

    if low_weights is None:
        # Resolve the documented default lazily so importing this module
        # doesn't hard-require optimum-quanto (an optional extra) — and guard
        # it with an actionable message, per pyproject's contract for this
        # module ("imported lazily with an ImportError guard").
        try:
            from optimum.quanto import qint4
        except ImportError as exc:
            raise ImportError(
                "The quanto quantization backend requires the optional "
                "dependency optimum-quanto. Install it with: "
                "pip install 'circuitkit[quantization]' "
                "(or: pip install 'optimum-quanto>=0.2')."
            ) from exc

        low_weights = qint4

    # Auto-detect architecture (ignore model_type parameter if provided)
    detected_type = detect_model_architecture(model)
    arch_cfg = get_arch_config(detected_type)
    validate_model_paths(model, arch_cfg)

    # 1. Aggregate scores to per-layer scalars
    attn_scores, mlp_scores_per_layer = compute_layer_scores(
        q_head_scores, mlp_scores, n_layers, aggregation=score_aggregation
    )

    # 2. Assign tiers
    plan = build_quantization_plan(
        attn_scores,
        mlp_scores_per_layer,
        high_fraction=high_fraction,
        mid_fraction=mid_fraction,
    )

    # 3. Apply the plan to the model
    _apply_plan_to_model(
        model,
        plan,
        low_weights,
        mid_weights,
        high_weights,
        activations,
        exclude_lm_head,
        arch_cfg,
        protect_layers=protect_layers,
    )

    return plan


def _apply_plan_to_model(
    model: nn.Module,
    plan: Dict[str, Dict[int, str]],
    low_weights,
    mid_weights,
    high_weights,
    activations,
    exclude_lm_head: bool,
    arch_cfg: Dict[str, Any],
    protect_layers: Optional[List[int]] = None,
) -> None:
    """Apply a quantization plan (tier assignment) to a model in-place.

    Parameters
    ----------
    model           : HuggingFace model
    plan            : Tier assignment from build_quantization_plan()
    low_weights     : qtype for low tier
    mid_weights     : qtype for mid tier
    high_weights    : qtype for high tier
    activations     : Activation qtype or None
    exclude_lm_head : Whether to exclude lm_head
    arch_cfg        : Architecture config from MODEL_ARCH_REGISTRY
    protect_layers  : Layer indices left at native precision (their attn + mlp
                      submodules are added to the quanto exclude list).
    """
    from optimum.quanto import quantize as quanto_quantize
    from transformers.pytorch_utils import Conv1D

    from ..arch_utils import UnsupportedArchitectureError

    # optimum-quanto only swaps nn.Linear -> QLinear. GPT-2 (and other Conv1D
    # architectures) store their attn/mlp projections as transformers Conv1D,
    # which quanto leaves untouched — so quantization would be a silent no-op
    # (later caught as "no QModule present"). Fail early with an actionable
    # message instead.
    if any(isinstance(m, Conv1D) for m in model.modules()):
        raise UnsupportedArchitectureError(
            "circuit_quantize: this model uses transformers Conv1D layers "
            "(e.g. GPT-2), which the optimum-quanto backend cannot quantize "
            "(it only converts nn.Linear). Quantization is unsupported for this "
            "architecture — use an nn.Linear-based model (Llama / Qwen / Gemma "
            "/ Mistral). Pruning, discovery and evaluation still work on GPT-2."
        )

    # Group layer indices by tier × component
    tiers: Dict[str, Dict[str, List[int]]] = {"high": {}, "mid": {}, "low": {}}
    for component in ("attn", "mlp"):
        for layer, tier in plan[component].items():
            tiers[tier].setdefault(component, []).append(layer)

    exclude_always = ["lm_head"] if exclude_lm_head else []
    # Exclude protected layers from quantization entirely: build attn+mlp
    # patterns for them and add them to quanto's exclude list.
    if protect_layers:
        protected = sorted(set(protect_layers))
        for component in ("attn", "mlp"):
            exclude_always.extend(build_patterns(protected, component, arch_cfg))

    def _quantize_tier(tier_name: str, qtype) -> None:
        if qtype is None:
            return
        patterns: List[str] = []
        for component in ("attn", "mlp"):
            layers_in_tier = tiers[tier_name].get(component, [])
            patterns.extend(build_patterns(layers_in_tier, component, arch_cfg))
        if not patterns:
            return
        logger.info(
            f"[quant] Quantizing {tier_name}-tier layers to {qtype} "
            f"({len(patterns)} pattern(s)) …"
        )
        quanto_quantize(
            model,
            weights=qtype,
            activations=activations,
            include=patterns,
            exclude=exclude_always,
        )

    _quantize_tier("low", low_weights)
    _quantize_tier("mid", mid_weights)
    _quantize_tier("high", high_weights)

    # optimum.quanto.quantize() only swaps nn.Linear -> QLinear; the weights
    # stay as lazy float QTensors that dequantize() back to the ORIGINAL
    # float values. freeze() is what actually converts them to packed low-bit
    # integers. Without this call the saved checkpoint is byte-identical to the
    # full-precision model and quantization has no effect. (This is weight-only
    # quantization — no activation calibration — so freezing here is correct.)
    freeze_model(model)

    # Guard: a silent no-op (the historical missing-freeze() bug, which made
    # every quant cell benchmark an un-quantized model) must fail loudly here.
    # After freeze() every quantized QModule must carry a frozen QTensor
    # weight (QModuleMixin.frozen == isinstance(weight, QTensor)).
    from optimum.quanto.nn.qmodule import QModuleMixin

    qmodules = [m for m in model.modules() if isinstance(m, QModuleMixin)]
    if not qmodules:
        raise RuntimeError(
            "_apply_plan_to_model: no quanto QModule present after quantize() — "
            "quantization did not apply (check include/exclude patterns)."
        )
    n_frozen = sum(1 for m in qmodules if m.frozen)
    if n_frozen == 0:
        raise RuntimeError(
            f"_apply_plan_to_model: {len(qmodules)} QModule(s) present but none "
            "are frozen — freeze() was a no-op; the exported checkpoint would be "
            "byte-identical to the fp model (the historical quant bug)."
        )
    logger.info(
        f"[quant] Guard OK: {n_frozen}/{len(qmodules)} QModule(s) frozen "
        "(packed low-bit weights)."
    )

    # NOTE: a plain hf_model.save_pretrained() does NOT persist quanto
    # quantization (no quantization_config / quantization map is written), so a
    # checkpoint must be saved via
    # circuitkit.evaluation.hf_checkpoint.save_quantized_checkpoint (which uses
    # optimum.quanto.QuantizedModelForCausalLM) and reloaded with
    # load_quantized_checkpoint — otherwise reload silently yields a plain fp
    # model.


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def freeze_model(model: nn.Module) -> None:
    """
    Freeze all QModule layers: convert dynamic float weights to static QTensor.

    Call after optional calibration and before saving / inference.
    """
    from optimum.quanto import freeze

    logger.info("[quant] Freezing quantized weights …")
    freeze(model)
    logger.info("[quant] Freeze complete.")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_quantized_model(
    model: nn.Module,
    tokenizer,
    eval_data: list,
    device: str = "cuda",
    n_samples: int = 64,
    batch_size: int = 8,
) -> None:
    """
    Calibrate activation quantization scales using circuit discovery task data.

    Runs ``n_samples`` forward passes inside an optimum-quanto ``Calibration``
    context manager to collect per-layer activation statistics (input/output
    scales).  Only useful when ``activations`` qtype was specified in
    ``circuit_quantize``.

    Parameters
    ----------
    model      : Quantized HuggingFace CausalLM.
    tokenizer  : Matching HuggingFace tokenizer.
    eval_data  : List of dicts with "clean" key (prompt strings).  Produced by
                 ``score_extractor.collect_eval_data``.
    device     : "cuda" or "cpu".
    n_samples  : Number of examples to run for calibration.
    batch_size : Forward-pass batch size.
    """
    from optimum.quanto import Calibration

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample = eval_data[:n_samples]
    logger.info(f"[quant] Calibrating with {len(sample)} examples …")

    with Calibration(momentum=0.9):
        for start in range(0, len(sample), batch_size):
            batch = sample[start : start + batch_size]
            texts = [ex["clean"] for ex in batch]
            # Prompts rendered through a chat template already carry their own
            # BOS; tokenizing them with add_special_tokens=True double-prepends
            # it.  collect_eval_data records this in each row's "templated" key.
            templated = bool(batch[0].get("templated", False))
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=not templated,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

    logger.info("[quant] Calibration complete.")


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------


def compute_ppl(
    model: nn.Module,
    tokenizer,
    device: str = "cuda",
    n_samples: int = 128,
    seq_len: int = 512,
) -> float:
    """
    Compute perplexity on wikitext-2 test set.

    Loads the wikitext-2-raw-v1 test split, tokenises it, and computes mean
    cross-entropy loss over ``n_samples`` non-overlapping windows of length
    ``seq_len``.  Returns perplexity (exp of mean loss).

    Parameters
    ----------
    model     : HuggingFace CausalLM (may be quantized).
    tokenizer : Matching HuggingFace tokenizer.
    device    : "cuda" or "cpu".
    n_samples : Number of seq_len-length windows to evaluate.
    seq_len   : Context window length in tokens.

    Returns
    -------
    Perplexity (float).
    """
    from datasets import load_dataset

    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])

    encodings = tokenizer(text, return_tensors="pt")
    all_ids = encodings["input_ids"][0]  # (total_tokens,)

    total_loss = 0.0
    count = 0

    for i in range(n_samples):
        start = i * seq_len
        end = start + seq_len + 1
        if end > len(all_ids):
            break
        chunk = all_ids[start:end].unsqueeze(0).to(device)  # (1, seq_len+1)
        input_ids = chunk[:, :-1]
        labels = chunk[:, 1:]
        with torch.no_grad():
            out = model(input_ids=input_ids)
        logits = out.logits  # (1, seq_len, vocab)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="mean",
        )
        total_loss += loss.item()
        count += 1

    if count == 0:
        return float("nan")

    mean_loss = total_loss / count
    return math.exp(mean_loss)


# ---------------------------------------------------------------------------
# Plan display
# ---------------------------------------------------------------------------


def print_quantization_plan(
    plan: Dict[str, Dict[int, str]],
    low_weights,
    high_weights,
    mid_weights=None,
) -> None:
    """
    Print a human-readable summary of the per-layer quantization assignment.
    """

    def _qname(qtype) -> str:
        if qtype is None:
            return "native"
        name = getattr(qtype, "name", None) or str(qtype)
        return name

    tier_labels = {
        "high": f"high  ({_qname(high_weights)})",
        "mid": f"mid   ({_qname(mid_weights)})",
        "low": f"low   ({_qname(low_weights)})",
    }

    n_layers = len(plan.get("attn", {}))
    logger.info(f"\n{'='*60}")
    logger.info("  QUANTIZATION PLAN")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Layer':<8}  {'Attn tier':<22}  {'MLP tier':<22}")
    logger.info(f"  {'-'*8}  {'-'*22}  {'-'*22}")
    for i in range(n_layers):
        attn_tier = plan["attn"].get(i, "low")
        mlp_tier = plan["mlp"].get(i, "low")
        logger.info(f"  {i:<8}  {tier_labels[attn_tier]:<22}  {tier_labels[mlp_tier]:<22}")
    logger.info(f"{'='*60}\n")
