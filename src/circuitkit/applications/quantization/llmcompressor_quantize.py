"""
llmcompressor_quantize.py — Circuit-guided true low-bit quantization via
``llm-compressor`` + ``compressed-tensors``.

Why a second backend
--------------------
``quant_utils.circuit_quantize`` uses ``optimum-quanto``, which only ships
``qint2`` / ``qint4`` / ``qint8`` — there is **no native 3-bit qtype**. The
EMNLP paper's §9.2 needs a *true* 3-bit stress budget. ``llm-compressor`` (the
vLLM-ecosystem quantizer, actively maintained, no CUDA-kernel compile step)
produces ``compressed-tensors`` checkpoints that vLLM loads natively, and its
``QuantizationArgs`` accepts an arbitrary ``num_bits`` — so a genuine 3-bit
weight type is expressible.

Circuit-aware mixed precision
-----------------------------
The discovered circuit's most-important layers (top ``high_fraction`` by the
exact same per-layer score aggregation the quanto path uses) plus any explicit
``protect_layers`` are kept at native (high) precision. They are passed to
``llm-compressor`` as ``ignore=[...]`` regex patterns; every other linear is
quantized to ``bits`` (3 by default). This mirrors ``circuit_quantize``'s
protected-vs-quantized split — the tier logic itself is reused from
``quant_utils`` (``compute_layer_scores`` + ``build_quantization_plan``).

API note (llm-compressor 0.10.x)
--------------------------------
We use ``llmcompressor.oneshot`` with a ``GPTQModifier`` whose ``scheme`` is a
``compressed_tensors.quantization.QuantizationScheme`` carrying the requested
``num_bits``. ``oneshot`` runs GPTQ calibration in-process and leaves the model
holding ``compressed-tensors`` quantized modules; ``save_pretrained`` then
writes a ``quantization_config`` so the checkpoint round-trips.

wandb / protobuf note
---------------------
``llmcompressor.metrics.logger`` does an unguarded ``import wandb``; the host
env ships protobuf 7.34.1, on which ``wandb`` fails to import. This module
installs a minimal ``sys.modules['wandb']`` stub *before* importing
llm-compressor (we never use wandb logging here), so ``import llmcompressor``
succeeds without downgrading the shared protobuf.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

from .quant_utils import build_quantization_plan, compute_layer_scores


import logging

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_BITS",
    "llmcompressor_circuit_quantize",
    "build_ignore_patterns",
]

# llm-compressor / compressed-tensors can express any int width; we expose the
# ones relevant to the paper (3-bit primary, plus 4/8).
SUPPORTED_BITS = (3, 4, 8)


# --------------------------------------------------------------------------- #
# wandb stub — must run before importing llmcompressor                        #
# --------------------------------------------------------------------------- #
def _install_wandb_stub() -> None:
    """Pre-empt llm-compressor's unguarded ``import wandb``.

    ``wandb`` is installed in the host env but crashes on protobuf 7.x.
    llm-compressor's logger imports it unconditionally. We register a harmless
    stub module (with a valid ``__spec__`` so ``importlib.util.find_spec`` is
    happy) so the import succeeds; we never use wandb logging here.
    """
    if "wandb" in sys.modules and not getattr(
        sys.modules["wandb"], "_circuitkit_stub", False
    ):
        return  # a real wandb already imported fine — leave it alone
    if "wandb" in sys.modules:
        return
    stub = types.ModuleType("wandb")
    stub.__version__ = "0.0.0-circuitkit-stub"
    stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    stub._circuitkit_stub = True
    sys.modules["wandb"] = stub


# --------------------------------------------------------------------------- #
# Protected-layer -> ignore pattern mapping                                   #
# --------------------------------------------------------------------------- #
def build_ignore_patterns(
    protected_layers: List[int],
    arch_cfg: Dict[str, Any],
    exclude_lm_head: bool = True,
) -> List[str]:
    """Translate a set of protected layer indices to llm-compressor ``ignore``.

    ``compressed-tensors`` matches an ``ignore`` entry against a module's
    *name* (a ``re:``-prefixed entry is a regex) or its class name. For a
    protected layer ``N`` we emit one regex covering **both** its attention and
    MLP linears — i.e. every linear under ``model.layers.N.`` — so that layer is
    left entirely at native precision, exactly as the quanto path adds
    ``model.layers.N.self_attn.*`` + ``model.layers.N.mlp.*`` to quanto's
    ``exclude`` list.

    Parameters
    ----------
    protected_layers : Layer indices to keep at native precision.
    arch_cfg         : Architecture config dict from MODEL_ARCH_REGISTRY.
    exclude_lm_head  : Whether to also leave ``lm_head`` unquantized.

    Returns
    -------
    List of ``ignore`` pattern strings for ``GPTQModifier(ignore=...)``.
    """
    patterns: List[str] = []
    if exclude_lm_head:
        patterns.append("lm_head")
    layers_path = arch_cfg["layers_path"][0]  # e.g. "model.layers"
    escaped = layers_path.replace(".", r"\.")
    for layer in sorted(set(protected_layers)):
        # Match every linear under this transformer layer (attn + mlp).
        patterns.append(rf"re:{escaped}\.{layer}\..*")
    return patterns


def _resolve_protected_layers(
    q_head_scores: Dict[Tuple[int, int], float],
    mlp_scores: Dict[int, Any],
    n_layers: int,
    high_fraction: float,
    score_aggregation: str,
    protect_layers: Optional[List[int]],
) -> List[int]:
    """Compute the protected (native-precision) layer set.

    Reuses the quanto path's tier logic verbatim: a layer is protected iff it
    is in the explicit ``protect_layers`` list OR its attention *or* MLP lands
    in the ``"high"`` tier of ``build_quantization_plan``. This is the direct
    analogue of ``_apply_plan_to_model`` leaving the ``high`` tier + protected
    layers unquantized — collapsed to a single per-layer set because
    llm-compressor's ``ignore`` is layer-granular, not tier-granular.
    """
    attn_scores, mlp_per_layer = compute_layer_scores(
        q_head_scores, mlp_scores, n_layers, aggregation=score_aggregation
    )
    plan = build_quantization_plan(
        attn_scores, mlp_per_layer, high_fraction=high_fraction, mid_fraction=0.0
    )
    protected = set(protect_layers or [])
    for layer in range(n_layers):
        if plan["attn"].get(layer) == "high" or plan["mlp"].get(layer) == "high":
            protected.add(layer)
    return sorted(protected)


# --------------------------------------------------------------------------- #
# Calibration data                                                            #
# --------------------------------------------------------------------------- #
def _build_calibration_dataset(
    tokenizer: Any,
    n_samples: int,
    max_seq_length: int,
    calibration_texts: Optional[List[str]],
) -> Any:
    """Build a small tokenized calibration ``datasets.Dataset`` for GPTQ.

    GPTQ needs a handful of forward passes to estimate the Hessian. We use the
    caller-supplied ``calibration_texts`` when given (e.g. the circuit
    discovery task data), else fall back to wikitext-2 — the same held-out
    corpus the quanto path's ``compute_ppl`` uses.
    """
    from datasets import Dataset

    texts: List[str]
    if calibration_texts:
        texts = [t for t in calibration_texts if t and t.strip()][:n_samples]
    else:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if t and len(t.strip()) > 64][:n_samples]

    if not texts:
        raise ValueError("No non-empty calibration texts available for GPTQ.")

    def _tokenize(example: Dict[str, str]) -> Dict[str, Any]:
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    raw = Dataset.from_dict({"text": texts})
    return raw.map(_tokenize, remove_columns=["text"])


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #
def llmcompressor_circuit_quantize(
    model: nn.Module,
    tokenizer: Any,
    q_head_scores: Dict[Tuple[int, int], float],
    mlp_scores: Dict[int, Any],
    n_layers: int,
    *,
    bits: int = 3,
    high_fraction: float = 0.3,
    protect_layers: Optional[List[int]] = None,
    score_aggregation: str = "mean",
    exclude_lm_head: bool = True,
    group_size: int = 128,
    symmetric: bool = True,
    n_calibration_samples: int = 128,
    max_seq_length: int = 512,
    calibration_texts: Optional[List[str]] = None,
    dampening_frac: float = 0.01,
) -> Dict[str, Any]:
    """Apply circuit-guided **true low-bit** quantization with llm-compressor.

    Quantizes every linear weight matrix to ``bits`` (3 by default — a genuine
    3-bit type, unavailable in optimum-quanto) **except** the linears of the
    circuit's most-important layers and any explicit ``protect_layers``, which
    are left at native precision. Calibration is one-shot GPTQ.

    The model is modified in place: afterwards its quantized modules carry
    ``compressed-tensors`` quantization state, and ``model.save_pretrained``
    writes a ``quantization_config`` so the checkpoint reloads via
    ``AutoModelForCausalLM.from_pretrained`` and is served natively by vLLM
    (no dequantization step).

    Parameters
    ----------
    model            : HuggingFace ``AutoModelForCausalLM`` (in-place modified).
    tokenizer        : Matching HuggingFace tokenizer (for calibration).
    q_head_scores    : Per-Q-head circuit scores, ``Dict[(layer, head), float]``.
    mlp_scores       : Per-layer / per-neuron MLP circuit scores.
    n_layers         : Number of transformer layers.
    bits             : Weight bit-width — 3 (primary), 4 or 8.
    high_fraction    : Fraction of layers protected as high-importance (same
                       semantics as ``circuit_quantize``).
    protect_layers   : Extra layer indices always kept at native precision.
    score_aggregation: "mean" / "sum" / "max" — passed to ``compute_layer_scores``.
    exclude_lm_head  : Leave ``lm_head`` unquantized.
    group_size       : Per-group quantization block size (-1 = per-channel).
    symmetric        : Symmetric (vs asymmetric) integer quantization.
    n_calibration_samples : Number of GPTQ calibration sequences.
    max_seq_length   : Calibration sequence length cap.
    calibration_texts: Optional list of raw calibration strings; falls back to
                       wikitext-2 train when omitted.
    dampening_frac   : GPTQ Hessian dampening fraction.

    Returns
    -------
    A dict describing what happened: ``backend``, ``bits``, ``protected_layers``,
    ``ignore_patterns``, ``quantized_layers`` — for logging / inspection.

    Raises
    ------
    ValueError: If ``bits`` is not one of :data:`SUPPORTED_BITS`.
    UnsupportedArchitectureError: If the model architecture is not supported.
    """
    if int(bits) not in SUPPORTED_BITS:
        raise ValueError(
            f"bits must be one of {SUPPORTED_BITS} for the llmcompressor "
            f"backend; got {bits!r}."
        )

    _install_wandb_stub()

    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStrategy,
    )
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    from circuitkit.applications import (

        detect_model_architecture,
        get_arch_config,
        validate_model_paths,
    )

    detected_type = detect_model_architecture(model)
    arch_cfg = get_arch_config(detected_type)
    validate_model_paths(model, arch_cfg)

    # 1. Circuit -> protected layer set (reuses the quanto tier logic).
    protected = _resolve_protected_layers(
        q_head_scores,
        mlp_scores,
        n_layers,
        high_fraction,
        score_aggregation,
        protect_layers,
    )

    # 2. Protected layers -> llm-compressor ``ignore`` regex patterns.
    ignore = build_ignore_patterns(protected, arch_cfg, exclude_lm_head=exclude_lm_head)

    # 3. Build the (true low-bit) weight-only quantization scheme.
    strategy = (
        QuantizationStrategy.GROUP if group_size and group_size > 0
        else QuantizationStrategy.CHANNEL
    )
    weight_args = QuantizationArgs(
        num_bits=int(bits),
        type="int",
        symmetric=bool(symmetric),
        strategy=strategy,
        group_size=int(group_size) if strategy == QuantizationStrategy.GROUP else None,
    )
    # GPTQModifier's ``scheme`` only accepts *preset* scheme names; a true
    # 3-bit width is not a preset, so we pass an explicit ``config_groups``
    # entry built from a custom QuantizationScheme instead.
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=weight_args,
    )

    modifier = GPTQModifier(
        config_groups={f"group_w{bits}a16": scheme},
        targets=["Linear"],
        ignore=ignore,
        dampening_frac=dampening_frac,
    )

    # 4. Build a small calibration set and run one-shot GPTQ.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    calib = _build_calibration_dataset(
        tokenizer, n_calibration_samples, max_seq_length, calibration_texts
    )

    logger.info(
        f"[llmcompressor] {bits}-bit GPTQ quantization — "
        f"{len(protected)} protected layer(s) {protected} kept native; "
        f"{n_calibration_samples} calibration samples."
    )

    oneshot(
        model=model,
        dataset=calib,
        recipe=modifier,
        tokenizer=tokenizer,
        num_calibration_samples=min(n_calibration_samples, len(calib)),
        max_seq_length=max_seq_length,
        pipeline="sequential",
        output_dir=None,
    )

    quantized = _count_quantized_linears(model)
    logger.info(f"[llmcompressor] Quantization complete — {quantized} linear module(s) quantized.")

    return {
        "backend": "llmcompressor",
        "bits": int(bits),
        "protected_layers": protected,
        "ignore_patterns": ignore,
        "quantized_layers": quantized,
        "n_layers": int(n_layers),
        "group_size": int(group_size),
    }


def _count_quantized_linears(model: nn.Module) -> int:
    """Count modules carrying a compressed-tensors quantization scheme."""
    n = 0
    for mod in model.modules():
        if hasattr(mod, "quantization_scheme") and getattr(
            mod, "quantization_scheme", None
        ) is not None:
            n += 1
    return n


def is_llmcompressor_quantized(model: nn.Module) -> bool:
    """True iff ``model`` carries compressed-tensors quantized modules."""
    return _count_quantized_linears(model) > 0
