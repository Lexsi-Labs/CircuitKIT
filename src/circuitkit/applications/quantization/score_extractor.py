"""
score_extractor.py — Node-level circuit discovery for quantization.

Three-phase pipeline
--------------------
Phase A  run_discovery()          Load TransformerLens model, run EAP-IG node-level
                                   attribution, return populated Graph + model reference.

Phase B  extract_node_head_scores()  One float per attention head from graph.nodes_scores.
         extract_node_mlp_scores()   One float per MLP layer from graph.nodes_scores.
         aggregate_attn_layer_scores()  Collapse per-head scores → one score per layer.
         save_scores() / load_scores()  Persist/restore scores between runs.

Phase C  (done in quant_utils.py)  Tier assignment + quantize() calls.

Key differences from pruning/score_extractor.py
------------------------------------------------
* neuron_level=False  →  graph.nodes_scores is 1-D (n_forward,), one float per node.
* mlp_hook="mlp_out"  →  mlp2 hook (output of down-proj, d_model space).  At node
  level the hook choice only affects WHERE in the MLP the activation difference is
  measured; the score itself is a single scalar per layer regardless.
* No per-neuron tensors — MLP scores are already scalars, requiring no further
  aggregation before quantization tier assignment.

Typical usage
-------------
    # --- one-time discovery (expensive) ---
    graph, tl_model = run_discovery("meta-llama/Llama-3.2-1B", "ioi", ...)
    head_scores = extract_node_head_scores(graph)   # {(layer, head): float}
    mlp_scores  = extract_node_mlp_scores(graph)    # {layer: float}
    save_scores(head_scores, mlp_scores, "scores/llama_ioi_node.pt")

    del tl_model, graph
    import torch; torch.cuda.empty_cache()

    # --- reusable quantization (cheap) ---
    head_scores, mlp_scores = load_scores("scores/llama_ioi_node.pt")
    # Pass directly to quant_utils.circuit_quantize()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# circuitkit imports (discovered at call-time to avoid circular deps)
# ---------------------------------------------------------------------------


def _bootstrap_tasks():
    """Register built-in tasks if they haven't been registered yet."""
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks

    _bootstrap_builtin_tasks()


# ---------------------------------------------------------------------------
# Phase A: run circuit discovery at node level
# ---------------------------------------------------------------------------


def run_discovery(
    model_name: str,
    task: str,
    ig_steps: int = 5,
    num_examples: int = 200,
    batch_size: int = 4,
    device: Optional[str] = None,
    mlp_hook: str = "mlp_out",
    precision: str = "bfloat16",
    data_params: Optional[Dict] = None,
):
    """
    Load a TransformerLens model and run EAP-IG node-level attribution.

    Each node (attention head or MLP block) receives a single scalar score
    stored in ``graph.nodes_scores``.  This is cheaper and sufficient for
    block-level quantization decisions — no per-neuron granularity is needed.

    Parameters
    ----------
    model_name  : HuggingFace / TransformerLens model identifier.
    task        : Registered circuitkit task name (e.g. "ioi", "mmlu").
    ig_steps    : Number of Integrated Gradients steps.
    num_examples: Number of task examples for computing attribution.
    batch_size  : Dataloader batch size.
    device      : "cuda" or "cpu".  Auto-detects if None.
    mlp_hook    : "mlp_out" (mlp2 — output of full MLP block, d_model space,
                  default) or "post_act" (mlp1 — post-activation, d_mlp space).
                  At node level both yield one scalar per MLP block.
    precision   : Model dtype string ("bfloat16", "float16", "float32").
    data_params : Extra kwargs forwarded to task_spec.build_dataloader()
                  (e.g. {"subjects": ["anatomy"]} for MMLU).

    Returns
    -------
    graph     : circuitkit Graph with populated nodes_scores tensor.
    tl_model  : The HookedTransformer (caller should delete + empty cache).
    """
    
    import warnings
    warnings.warn(
        "score_extractor.run_discovery() is deprecated and will be removed in v1.1. "
        "Use circuitkit.discover() or Pipeline.discover() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    
    from transformer_lens import HookedTransformer

    from circuitkit.backends.eap.attribute_node import attribute_node
    from circuitkit.backends.eap.graph import Graph
    from circuitkit.tasks.registry import get_task

    _bootstrap_tasks()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, precision)

    logger.info(f"[discovery] Loading {model_name} via TransformerLens …")
    tl_model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_hook_mlp_in = True
    if hasattr(tl_model.cfg, "ungroup_grouped_query_attention"):
        tl_model.cfg.ungroup_grouped_query_attention = True

    task_spec = get_task(task)
    discovery_cfg = {
        "algorithm": "eap-ig",
        "task": task,
        "level": "node",
        "mlp_hook": mlp_hook,
        "batch_size": batch_size,
        "ig_steps": ig_steps,
        "model_name": model_name,
        "data_params": {
            "num_examples": num_examples,
            **(data_params or {}),
        },
        # Flatten data_params to top-level so task specs that read keys
        # directly (e.g. MMLU reading 'subjects') see them too.
        **(data_params or {}),
    }
    task_spec.validate_discovery_config(discovery_cfg)

    logger.info(f"[discovery] Building dataloader for task='{task}' …")
    dataloader = task_spec.build_dataloader(tl_model, discovery_cfg, device)
    metric = task_spec.metric_fn()

    logger.info(f"[discovery] Building graph (mlp_hook={mlp_hook}, neuron_level=False) …")
    graph = Graph.from_model(
        tl_model,
        node_scores=True,
        neuron_level=False,
        mlp_hook=mlp_hook,
    )

    logger.info(f"[discovery] Running EAP-IG node-level attribution (ig_steps={ig_steps}) …")
    attribute_node(
        tl_model,
        graph,
        dataloader,
        metric,
        method="EAP-IG-inputs",
        ig_steps=ig_steps,
        neuron=False,
    )

    logger.info("[discovery] Attribution complete.")
    return graph, tl_model


# ---------------------------------------------------------------------------
# Phase B: extract scores from a populated graph
# ---------------------------------------------------------------------------


def extract_node_head_scores(graph) -> Dict[Tuple[int, int], float]:
    """
    Extract one importance score per attention head from a node-level graph.

    At node level, ``graph.nodes_scores[fwd_idx]`` is a single scalar
    representing the total attributional importance of that head.

    Returns
    -------
    Dict mapping (layer_idx, head_idx) → float importance score.
    Higher = more important.
    """
    from circuitkit.backends.eap.graph import AttentionNode

    scores: Dict[Tuple[int, int], float] = {}
    for node in graph.nodes.values():
        if not isinstance(node, AttentionNode):
            continue
        fwd_idx = graph.forward_index(node, attn_slice=False)
        score = float(graph.nodes_scores[fwd_idx].item())
        scores[(node.layer, node.head)] = score
    return scores


def extract_node_mlp_scores(graph) -> Dict[int, float]:
    """
    Extract one importance score per MLP block from a node-level graph.

    At node level each MLP block has a single scalar score — no per-neuron
    breakdown.  The ``mlp_hook`` choice (mlp_out vs post_act) in
    ``run_discovery`` affects how the activation difference is measured but
    the output here is always one float per layer.

    Returns
    -------
    Dict mapping layer_idx → float importance score.
    Higher = more important.
    """
    from circuitkit.backends.eap.graph import MLPNode

    scores: Dict[int, float] = {}
    for node in graph.nodes.values():
        if not isinstance(node, MLPNode):
            continue
        fwd_idx = graph.forward_index(node, attn_slice=False)
        score = float(graph.nodes_scores[fwd_idx].item())
        scores[node.layer] = score
    return scores


# ---------------------------------------------------------------------------
# Score-format shim: bridge between the two parallel score conventions in the
# library.  Pruning's selector registry uses string keys like ``"A0.5"`` and
# ``"MLP 12"`` (because run_grid.py serialises pruned-node lists as strings
# for JSON); quantization's ``score_extractor`` returns tuple/int keys
# (``Dict[(layer, head), float]`` for heads, ``Dict[layer, float]`` for
# MLPs) because the tier-assignment logic indexes by layer/head numerics.
# The two pipelines were designed independently and never have to interop in
# the audit grid, but a user who wants to feed a *pruning* selector's score
# dict into the *quantization* tier-builder (or vice versa) needs a
# conversion. The two helpers below are the canonical converters.
# ---------------------------------------------------------------------------


def head_scores_str_to_tuple(
    str_scores: Dict[str, float],
) -> Dict[Tuple[int, int], float]:
    """Convert ``{"A{layer}.{head}": float}`` → ``{(layer, head): float}``.

    Filters out MLP-style keys (``"MLP {layer}"``); only attention-head
    entries are returned. Use this to feed a pruning-selector's flat score
    dict into the quantization tier-assignment helpers.
    """
    import re

    pat = re.compile(r"A(\d+)\.(\d+)")
    out: Dict[Tuple[int, int], float] = {}
    for k, v in str_scores.items():
        m = pat.match(k)
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = float(v)
    return out


def mlp_scores_str_to_int(str_scores: Dict[str, float]) -> Dict[int, float]:
    """Convert ``{"MLP {layer}": float}`` → ``{layer: float}``.

    Filters out attention-head keys (``"A{layer}.{head}"``); only MLP
    entries are returned. Mirror of ``head_scores_str_to_tuple``.
    """
    import re

    pat = re.compile(r"MLP (\d+)")
    out: Dict[int, float] = {}
    for k, v in str_scores.items():
        m = pat.match(k)
        if m:
            out[int(m.group(1))] = float(v)
    return out


def head_scores_tuple_to_str(
    tup_scores: Dict[Tuple[int, int], float],
) -> Dict[str, float]:
    """Inverse of ``head_scores_str_to_tuple``."""
    return {f"A{layer}.{head}": float(v) for (layer, head), v in tup_scores.items()}


def mlp_scores_int_to_str(int_scores: Dict[int, float]) -> Dict[str, float]:
    """Inverse of ``mlp_scores_str_to_int``."""
    return {f"MLP {layer}": float(v) for layer, v in int_scores.items()}


def aggregate_attn_layer_scores(
    head_scores: Dict[Tuple[int, int], float],
    reduction: str = "mean",
) -> Dict[int, float]:
    """
    Aggregate per-head attention scores to a single score per layer.

    Parameters
    ----------
    head_scores : Dict[(layer, head), float] from extract_node_head_scores().
    reduction   : "mean" | "sum" | "max" — how to combine head scores.

    Returns
    -------
    Dict mapping layer_idx → aggregated float importance score.
    """
    from collections import defaultdict

    layer_groups: Dict[int, List[float]] = defaultdict(list)
    for (layer, _head), score in head_scores.items():
        layer_groups[layer].append(abs(score))

    layer_scores: Dict[int, float] = {}
    for layer, vals in layer_groups.items():
        if not vals:
            layer_scores[layer] = 0.0
        elif reduction == "mean":
            layer_scores[layer] = sum(vals) / len(vals)
        elif reduction == "sum":
            layer_scores[layer] = sum(vals)
        elif reduction == "max":
            layer_scores[layer] = max(vals)
        else:
            raise ValueError(f"Unknown reduction '{reduction}'. Use 'mean', 'sum', or 'max'.")
    return layer_scores


# ---------------------------------------------------------------------------
# Score persistence
# ---------------------------------------------------------------------------


def save_scores(
    head_scores: Dict[Tuple[int, int], float],
    mlp_scores: Dict[int, float],
    path: str,
) -> None:
    """
    Save node-level circuit-discovery scores to disk.

    Parameters
    ----------
    head_scores : Dict[(layer, head), float] — from extract_node_head_scores()
    mlp_scores  : Dict[layer, float]          — from extract_node_mlp_scores()
    path        : File path (suggested extension: .pt)
    """
    os.makedirs(Path(path).parent, exist_ok=True)
    payload = {
        "head_scores": {str(k): v for k, v in head_scores.items()},
        "mlp_scores": {int(k): float(v) for k, v in mlp_scores.items()},
        "format": "node",  # sentinel so load_scores can detect format
    }
    torch.save(payload, path)
    logger.info(f"[scores] Saved to {path}")


def load_scores(
    path: str,
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, float]]:
    """
    Load scores previously saved with save_scores().

    Returns
    -------
    head_scores : Dict[(layer, head), float]
    mlp_scores  : Dict[layer, float]
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)

    raw_h = payload["head_scores"]
    head_scores: Dict[Tuple[int, int], float] = {}
    for k, v in raw_h.items():
        k = k.strip("() ")
        parts = [p.strip() for p in k.split(",")]
        head_scores[(int(parts[0]), int(parts[1]))] = float(v)

    mlp_scores: Dict[int, float] = {int(k): float(v) for k, v in payload["mlp_scores"].items()}

    logger.info(f"[scores] Loaded from {path}")
    return head_scores, mlp_scores


# ---------------------------------------------------------------------------
# Task evaluation data collection (call while TL model is still loaded)
# ---------------------------------------------------------------------------


def collect_eval_data(
    tl_model,
    task: str,
    discovery_cfg: dict,
    device: str,
    max_examples: Optional[int] = None,
) -> List[Dict]:
    """
    Collect evaluation data from a task's dataloader as raw strings + token IDs.

    Must be called while ``tl_model`` is still alive.  After calling this
    function you can safely ``del tl_model`` and load the HuggingFace model.

    Returns
    -------
    List of dicts with keys ``clean``, ``correct_idx``, ``incorrect_idx``,
    and ``templated`` (whether the dataloader is producing chat-templated
    strings — read downstream by ``calibrate_quantized_model`` to decide
    whether ``add_special_tokens`` should re-prepend BOS, since
    chat-templated text already carries its own beginning-of-text token).

    The ``templated`` key was originally omitted, which caused the
    downstream calibrator to silently default to ``templated=False`` and
    double-prepend BOS on chat-templated calibration data — dormant in
    our audit grid (which uses raw text for quant calibration) but a real
    bug on any future templated-calibration run.
    """
    from circuitkit.tasks.registry import get_task


    _bootstrap_tasks()
    task_spec = get_task(task)
    dataloader = task_spec.build_dataloader(tl_model, discovery_cfg, device)
    # The dataloader carries the templated flag as an attribute when its
    # task spec decides to wrap prompts in a chat template; default to
    # False for non-chat tasks / older task specs.
    templated = bool(getattr(dataloader, "templated", False))

    eval_data: List[Dict] = []
    for batch in dataloader:
        clean_texts, _, labels = batch
        for i, text in enumerate(clean_texts):
            row = {
                "clean": text,
                "correct_idx": int(labels[i, 0].item()),
                "incorrect_idx": labels[i, 1:].tolist(),
                "templated": templated,
            }
            eval_data.append(row)
            if max_examples is not None and len(eval_data) >= max_examples:
                return eval_data

    return eval_data


def save_eval_data(eval_data: List[Dict], path: str) -> None:
    """Save collected eval data to disk (torch .pt format)."""
    os.makedirs(Path(path).parent, exist_ok=True)
    torch.save(eval_data, path)
    logger.info(f"[scores] Eval data ({len(eval_data)} examples) saved to {path}")


def load_eval_data(path: str) -> List[Dict]:
    """Load eval data previously saved with save_eval_data()."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    logger.info(f"[scores] Loaded {len(data)} eval examples from {path}")
    return data
