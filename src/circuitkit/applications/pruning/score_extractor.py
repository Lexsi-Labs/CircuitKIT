"""
score_extractor.py — Circuit-guided pruning score extraction.

Three-phase pipeline
--------------------
Phase A  run_discovery()          Load TransformerLens model, run EAP-IG attribution,
                                   return populated Graph + model reference.

Phase B  extract_*()              Pull raw scores from graph.neurons_scores.
         aggregate_to_kv_heads()  For GQA: collapse Q-head scores → KV-head scores.
         save_scores() / load_scores()  Persist/restore scores between runs.

Phase C  build_importance_dict()  Map layer-indexed scores → HF nn.Module instances,
                                   producing the dict consumed by CircuitKitImportance.

Typical usage
-------------
    # --- one-time discovery (expensive) ---
    graph, tl_model = run_discovery("meta-llama/Llama-2-7b-hf", "ioi", ...)
    q_scores   = extract_q_head_scores(graph)
    mlp_scores = extract_mlp_neuron_scores(graph)
    save_scores(q_scores, mlp_scores, "scores/llama_ioi.pt")

    del tl_model, graph
    import torch; torch.cuda.empty_cache()

    # --- reusable pruning (cheap) ---
    q_scores, mlp_scores = load_scores("scores/llama_ioi.pt")
    kv_scores = aggregate_to_kv_heads(q_scores, n_q_heads=32, n_kv_heads=8)
    scores_dict = build_importance_dict(hf_model, kv_scores, mlp_scores,
                                        attn_layers=range(32), mlp_layers=range(32))
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# circuitkit imports (discovered at call-time to avoid circular deps)
# ---------------------------------------------------------------------------


def _bootstrap_tasks():
    """Register built-in tasks if they haven't been registered yet."""
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks

    _bootstrap_builtin_tasks()


# ---------------------------------------------------------------------------
# Phase A: run circuit discovery
# ---------------------------------------------------------------------------


def run_discovery(
    model_name: str,
    task: str,
    ig_steps: int = 5,
    num_examples: int = 200,
    batch_size: int = 4,
    device: Optional[str] = None,
    mlp_hook: str = "post_act",
    precision: str = "bfloat16",
    data_params: Optional[Dict] = None,
):
    """
    Load a TransformerLens model and run EAP-IG neuron-level attribution.

    Returns the populated ``Graph`` and the ``HookedTransformer`` model so the
    caller can access ``graph.neurons_scores`` and then delete both objects to
    free GPU memory before loading the HuggingFace model.

    Parameters
    ----------
    model_name   : HuggingFace / TransformerLens model identifier.
    task         : Registered circuitkit task name (e.g. "ioi", "mmlu").
    ig_steps     : Number of Integrated Gradients steps (higher = more accurate,
                   slower).  5 is a reasonable default; use 2 for quick tests.
    num_examples : Number of task examples for computing attribution.
    batch_size   : Dataloader batch size.
    device       : "cuda" or "cpu".  Auto-detects if None.
    mlp_hook     : "post_act" to score neurons in d_mlp space (recommended);
                   "mlp_out" scores in d_model space.
    precision    : Model dtype string ("bfloat16", "float16", "float32").
    data_params  : Extra kwargs forwarded to task_spec.build_dataloader()
                   (e.g. {"subjects": ["anatomy"]} for MMLU).

    Returns
    -------
    graph     : circuitkit Graph with populated neurons_scores tensor.
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
        "level": "neuron",
        "mlp_hook": mlp_hook,
        "batch_size": batch_size,
        "ig_steps": ig_steps,
        "model_name": model_name,
        "data_params": {
            "num_examples": num_examples,
            **(data_params or {}),
        },
        # Flatten data_params into top-level so task specs that read keys
        # directly (e.g. MMLU reading 'subjects') see them too.
        **(data_params or {}),
    }
    task_spec.validate_discovery_config(discovery_cfg)

    logger.info(f"[discovery] Building dataloader for task='{task}' …")
    dataloader = task_spec.build_dataloader(tl_model, discovery_cfg, device)
    metric = task_spec.metric_fn()

    logger.info(f"[discovery] Building graph (mlp_hook={mlp_hook}, neuron_level=True) …")
    graph = Graph.from_model(
        tl_model,
        node_scores=True,
        neuron_level=True,
        mlp_hook=mlp_hook,
    )

    logger.info(f"[discovery] Running EAP-IG attribution (ig_steps={ig_steps}) …")
    attribute_node(
        tl_model,
        graph,
        dataloader,
        metric,
        method="EAP-IG-inputs",
        ig_steps=ig_steps,
        neuron=True,
    )

    logger.info("[discovery] Attribution complete.")
    return graph, tl_model


# ---------------------------------------------------------------------------
# Phase B: extract scores from a populated graph
# ---------------------------------------------------------------------------


def extract_q_head_scores(graph) -> Dict[Tuple[int, int], float]:
    """
    Extract one importance score per Q-head from a neuron-level graph.

    Each attention node's neurons_scores slice has shape ``(d_model,)`` and
    captures the attributional signal from that head's output directions in the
    residual stream.  We aggregate to a single float by summing absolute values,
    which is equivalent to the L1 norm of the attribution vector.

    Returns
    -------
    Dict mapping (layer_idx, q_head_idx) → float importance score.
    Higher = more important.
    """
    from circuitkit.backends.eap.graph import AttentionNode

    scores: Dict[Tuple[int, int], float] = {}
    for node in graph.nodes.values():
        if not isinstance(node, AttentionNode):
            continue
        fwd_idx = graph.forward_index(node, attn_slice=False)
        raw = graph.neurons_scores[fwd_idx, : node.d_neuron]  # (d_model,)
        scores[(node.layer, node.head)] = float(raw.abs().sum().item())
    return scores


def extract_mlp_neuron_scores(graph) -> Dict[int, torch.Tensor]:
    """
    Extract per-neuron attribution scores for each MLP layer.

    When ``mlp_hook="post_act"`` the scores live in ``d_mlp`` space, which
    directly corresponds to gate_proj / up_proj output channels in HuggingFace
    models.

    Returns
    -------
    Dict mapping layer_idx → 1-D float tensor of shape ``(d_mlp,)`` (or
    ``(d_model,)`` if mlp_hook="mlp_out").  Higher = more important.
    """
    from circuitkit.backends.eap.graph import MLPNode

    scores: Dict[int, torch.Tensor] = {}
    for node in graph.nodes.values():
        if not isinstance(node, MLPNode):
            continue
        fwd_idx = graph.forward_index(node, attn_slice=False)
        raw = graph.neurons_scores[fwd_idx, : node.d_neuron].clone().cpu()  # (d_mlp,)
        scores[node.layer] = raw
    return scores


def aggregate_to_kv_heads(
    q_head_scores: Dict[Tuple[int, int], float],
    n_q_heads: int,
    n_kv_heads: int,
    reduction: str = "sum",
) -> Dict[Tuple[int, int], float]:
    """
    Aggregate Q-head importance scores to KV-head granularity for GQA models.

    In a GQA model each KV head is shared by ``n_q_heads // n_kv_heads`` Q
    heads.  We combine the Q-head scores within each group to produce one score
    per KV head.  The resulting scores determine which *entire* KV heads (and
    the Q heads that attend to them) are pruned.

    Example: 16 Q heads, 4 KV heads → group_size = 4.
      KV head 0 aggregates Q heads 0-3,
      KV head 1 aggregates Q heads 4-7, …

    Parameters
    ----------
    q_head_scores : Dict[(layer, q_head), float]
    n_q_heads     : Total number of Q heads in the model.
    n_kv_heads    : Total number of K/V heads (determines group size).
    reduction     : "sum" | "mean" | "max" — how to combine Q scores within a
                    group.

    Returns
    -------
    Dict mapping (layer_idx, kv_head_idx) → aggregated float importance score.
    """
    if n_kv_heads == n_q_heads:
        # MHA: identity mapping, just re-key as kv_head indices
        return {(lyr, h): s for (lyr, h), s in q_head_scores.items()}

    group_size = n_q_heads // n_kv_heads
    layers = sorted(set(lyr for lyr, _ in q_head_scores))
    kv_scores: Dict[Tuple[int, int], float] = {}

    for layer in layers:
        for kv_h in range(n_kv_heads):
            q_range = range(kv_h * group_size, (kv_h + 1) * group_size)
            group_vals = [q_head_scores.get((layer, qh), 0.0) for qh in q_range]
            if reduction == "sum":
                agg = sum(group_vals)
            elif reduction == "mean":
                agg = sum(group_vals) / len(group_vals)
            elif reduction == "max":
                agg = max(group_vals)
            else:
                raise ValueError(f"Unknown reduction '{reduction}'. Use 'sum', 'mean', or 'max'.")
            kv_scores[(layer, kv_h)] = agg

    return kv_scores


# ---------------------------------------------------------------------------
# Score persistence
# ---------------------------------------------------------------------------


def save_scores(
    q_head_scores: Dict[Tuple[int, int], float],
    mlp_scores: Dict[int, torch.Tensor],
    path: str,
) -> None:
    """
    Save raw circuit-discovery scores to disk.

    The saved file stores Q-head scores (pre-aggregation) so you can later
    re-aggregate for different n_kv_heads values without re-running discovery.

    Parameters
    ----------
    q_head_scores : Dict[(layer, q_head), float]  — output of extract_q_head_scores()
    mlp_scores    : Dict[layer, Tensor(d_mlp)]     — output of extract_mlp_neuron_scores()
    path          : File path (suggested extension: .pt)
    """
    os.makedirs(Path(path).parent, exist_ok=True)
    payload = {
        "q_head_scores": {str(k): v for k, v in q_head_scores.items()},
        "mlp_scores": mlp_scores,
    }
    torch.save(payload, path)
    logger.info(f"[scores] Saved to {path}")


def load_scores(
    path: str,
) -> Tuple[Dict[Tuple[int, int], float], Dict[int, torch.Tensor]]:
    """
    Load scores previously saved with save_scores().

    Returns
    -------
    q_head_scores : Dict[(layer, q_head), float]
    mlp_scores    : Dict[layer, Tensor(d_mlp)]
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    # Restore tuple keys (they were serialized as strings "layer,head")
    raw_q = payload["q_head_scores"]
    q_head_scores: Dict[Tuple[int, int], float] = {}
    for k, v in raw_q.items():
        # key format: "(layer, head)"  or  "layer,head"
        k = k.strip("() ")
        parts = [p.strip() for p in k.split(",")]
        q_head_scores[(int(parts[0]), int(parts[1]))] = float(v)

    mlp_scores: Dict[int, torch.Tensor] = {int(k): v for k, v in payload["mlp_scores"].items()}
    logger.info(f"[scores] Loaded from {path}")
    return q_head_scores, mlp_scores


# ---------------------------------------------------------------------------
# Phase C: build the module → tensor importance dict for LLM-Pruner
# ---------------------------------------------------------------------------
# Architecture-aware implementation supporting multiple model types


def build_importance_dict(
    hf_model: nn.Module,
    kv_head_scores: Dict[Tuple[int, int], float],
    mlp_scores: Dict[int, torch.Tensor],
    attn_layers: Iterable[int],
    mlp_layers: Iterable[int],
) -> Dict[nn.Module, torch.Tensor]:
    """
    Produce a ``{nn.Module: Tensor}`` map consumable by CircuitKitImportance.

    Architecture-aware version supporting multiple model types (LLaMA, Gemma, GPT-2, etc.)

    Attention heads (k_proj root)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For each attention layer i, k_proj has shape ``(n_kv * head_dim, hidden)``.
    We build a 1-D tensor of length ``n_kv * head_dim`` where the head_dim
    channels belonging to KV head *h* all receive the score for (i, h).
    Because MetaPruner groups consecutive channels by ``head_dim`` and sums
    within each group, uniform values within a group preserve relative ranking.

    MLP neurons (gate_proj root)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gate_proj output channels correspond 1-to-1 with post-activation neurons.
    We directly use the absolute circuit-discovery scores as per-channel
    importance.  The dependency graph then cascades pruning to up_proj
    (same output channels) and down_proj (input channels) automatically.

    Parameters
    ----------
    hf_model       : HuggingFace model (supports LLaMA, Gemma, Qwen, GPT-2, etc.)
    kv_head_scores : Dict[(layer, kv_head), float] from aggregate_to_kv_heads()
    mlp_scores     : Dict[layer, Tensor(d_mlp)] from extract_mlp_neuron_scores()
    attn_layers    : Iterable of layer indices to include for attention pruning.
    mlp_layers     : Iterable of layer indices to include for MLP pruning.

    Returns
    -------
    Dict[nn.Module, torch.Tensor]  ready to pass to CircuitKitImportance().

    Raises
    ------
    UnsupportedArchitectureError: If model type not supported
    ArchitectureValidationError: If layer structure doesn't match expected paths
    """
    # Import here to avoid circular dependencies
    from circuitkit.applications.arch_utils import (
        detect_model_architecture,
        get_arch_config,
        get_attn_proj,
        get_head_dim,
        get_layers,
        get_mlp_proj,
        validate_model_paths,
    )

    # Detect and validate architecture
    model_type = detect_model_architecture(hf_model)
    arch_cfg = get_arch_config(model_type)
    validate_model_paths(hf_model, arch_cfg)

    # Get the layers module
    layers = get_layers(hf_model, arch_cfg)

    scores_dict: Dict[nn.Module, torch.Tensor] = {}

    # Process attention heads
    for i in attn_layers:
        try:
            layer = layers[i]
            k_proj = get_attn_proj(layer, arch_cfg, "k_proj")
            head_dim = get_head_dim(layer, arch_cfg, hf_model.config.num_attention_heads)
            n_kv = k_proj.weight.shape[0] // head_dim

            imp = torch.zeros(n_kv * head_dim, dtype=torch.float32)
            for kv_h in range(n_kv):
                score = kv_head_scores.get((i, kv_h), 0.0)
                imp[kv_h * head_dim : (kv_h + 1) * head_dim] = score

            scores_dict[k_proj] = imp
        except Exception as e:
            logger.warning(f"Failed to extract attention importance for layer {i}: {e}")
            continue

    # Process MLP neurons
    for i in mlp_layers:
        try:
            layer = layers[i]
            gate_proj = get_mlp_proj(layer, arch_cfg, "gate_proj")

            if gate_proj is None:
                logger.warning(f"No gate_proj found for layer {i}, skipping MLP pruning")
                continue

            d_mlp = gate_proj.weight.shape[0]

            if i in mlp_scores:
                raw = mlp_scores[i].float()
                if len(raw) != d_mlp:
                    logger.warning(
                        f"Layer {i}: mlp_scores length {len(raw)} ≠ "
                        f"gate_proj out {d_mlp}. Falling back to magnitude."
                    )
                    continue
                scores_dict[gate_proj] = raw.abs()
        except Exception as e:
            logger.warning(f"Failed to extract MLP importance for layer {i}: {e}")
            continue

    return scores_dict


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

    Must be called while the ``tl_model`` (HookedTransformer) is still alive,
    since the task spec uses it to generate / tokenize data.  After calling
    this function you can safely ``del tl_model`` and load the HuggingFace
    counterpart for pruning.

    The eval data can be used with ``eval_utils.eval_hf_model_on_task`` to
    measure accuracy and loss on a pruned HuggingFace model.

    Parameters
    ----------
    tl_model      : Loaded HookedTransformer (used for tokenization).
    task          : circuitkit task name (e.g. "ioi", "mmlu").
    discovery_cfg : Same config dict passed to ``run_discovery``.
    device        : "cuda" or "cpu".
    max_examples  : Cap the number of examples collected (None = all).

    Returns
    -------
    List of dicts, each with keys:
        ``clean``        : the clean prompt string.
        ``correct_idx``  : int, token ID of the correct completion.
        ``incorrect_idx``: list of ints, token IDs of wrong completions.
        ``templated``    : bool, whether ``clean`` was rendered through the
                           model's chat template (so it already carries its
                           own BOS).  Downstream consumers must tokenize with
                           ``add_special_tokens=not templated`` to avoid a
                           double-BOS.
    """
    from circuitkit.tasks.registry import get_task

    _bootstrap_tasks()
    task_spec = get_task(task)
    dataloader = task_spec.build_dataloader(tl_model, discovery_cfg, device)
    templated = getattr(dataloader, "templated", False)

    eval_data: List[Dict] = []
    for batch in dataloader:
        # collate_EAP returns (clean: list[str], corrupted: list[str], labels: Tensor)
        clean_texts, _, labels = batch
        # labels shape: (batch, n_labels)  col-0 = correct, rest = incorrect
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
