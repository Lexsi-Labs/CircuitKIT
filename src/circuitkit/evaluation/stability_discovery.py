"""
Internal helper for Pillar 3: re-runs circuit discovery across seeds/resamples.

The model is loaded once by the caller and passed in — this module never
loads weights. Each call to `rediscover()` returns a list of node-score
dicts (one per run) that Pillar3_Stability uses to compute overlap metrics.

Supported algorithms: eap, eap-ig, acdc, ibcircuit.
"""

import copy
import gc
import logging
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _log_gpu_mem(label: str, logger):
    """Log GPU memory stats at DEBUG level. No-op if CUDA unavailable."""
    import torch

    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free_reserved = reserved - allocated
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_total = total - reserved
    logger.debug(
        f"[GPU-MEM] {label}: "
        f"alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, "
        f"free_in_reserved={free_reserved:.2f}GB, free_total={free_total:.2f}GB"
    )


def rediscover(
    model,
    task_spec,
    discovery_cfg: dict,
    n_runs: int = 5,
    seed_start: int = 42,
    device: str = "auto",
) -> List[Dict[str, float]]:
    """
    Re-run circuit discovery n_runs times with different data seeds.

    Each run clones discovery_cfg, injects a fresh seed into data_params,
    and re-runs attribution from scratch on the same loaded model.
    Different seeds produce different data subsets, yielding genuinely
    distinct circuits whose overlap is a meaningful stability metric.

    Args:
        model: Already-loaded HookedTransformer (not reloaded per run).
        task_spec: TaskSpec with build_dataloader() and metric_fn().
        discovery_cfg: Original discovery config dict (not mutated).
        n_runs: Number of independent re-discoveries (default 5).
        seed_start: First seed; run i uses seed seed_start + i.
        device: 'cuda' or 'cpu'.

    Returns:
        List of length n_runs, each a Dict[node_name, float] of raw scores.
    """
    algo = discovery_cfg.get("algorithm", "").lower()

    # All algorithms that go through the EAP attribute_node() backend.
    # Mirrors _ALGO_METHOD_MAP in api.py exactly.
    _EAP_FAMILY = frozenset(
        {
            "eap",
            "eap-ig",
            "eap-ig-activations",
            "eap-clean-corrupted",
            "eap-exact",
            "atp-gd",
            "eap-gp",
            "relp",
            "peap",
            "eap-ifr",
        }
    )
    _ALL_SUPPORTED = _EAP_FAMILY | {"acdc", "ibcircuit", "cdt"}

    if algo not in _ALL_SUPPORTED:
        raise ValueError(
            f"stability_discovery: unsupported algorithm {algo!r}. "
            f"Supported: {', '.join(sorted(_ALL_SUPPORTED))}."
        )

    results: List[Dict[str, float]] = []

    for i in range(n_runs):
        seed = seed_start + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        cfg = copy.deepcopy(discovery_cfg)
        cfg.setdefault("data_params", {})["seed"] = seed
        cfg["seed"] = seed

        logger.info(f"Stability re-discovery run {i + 1}/{n_runs} (seed={seed}, algo={algo})")

        _log_gpu_mem(f"rediscover: before run {i+1}/{n_runs} (algo={algo})", logger)

        if algo in _EAP_FAMILY:
            scores = _rediscover_eap(model, task_spec, cfg, algo, device)
        elif algo == "acdc":
            scores = _rediscover_acdc(model, task_spec, cfg, device)
        elif algo == "cdt":
            scores = _rediscover_cdt(model, task_spec, cfg, device)
        else:  # ibcircuit
            scores = _rediscover_ibcircuit(model, task_spec, cfg, device)

        logger.info(f"  Run {i + 1}: {len(scores)} scored nodes")
        results.append(scores)

        # --- MEMORY OPTIMIZATION: Flush stale computation graphs ---
        model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _log_gpu_mem(f"rediscover: after run {i+1}/{n_runs}, after cleanup", logger)

    return results


def node_scores_to_circuit(
    node_scores: Dict[str, float],
    sparsity: float,
) -> Dict[str, float]:
    """
    Apply a sparsity threshold to raw node scores.

    Sorts nodes by ascending score (lowest = least important), discards
    the bottom `sparsity` fraction, and returns the surviving nodes.
    This mirrors the pruning logic in api.discover_circuit so that
    stability is measured on comparable circuit sets.

    Args:
        node_scores: {node_name: score} from rediscover().
        sparsity: Fraction of nodes to discard (0.0–1.0).

    Returns:
        Dict of kept {node_name: score}.
    """
    if not node_scores:
        return {}

    # Coerce values to plain Python floats defensively — backends (e.g. ibcircuit)
    # may return Tensor scalars or multi-element Tensors that break sorted().
    def _to_float(v) -> float:
        if isinstance(v, torch.Tensor):
            return float(v.detach().abs().sum().item())
        return float(v)

    sorted_nodes = sorted(
        ((k, _to_float(v)) for k, v in node_scores.items()),
        key=lambda x: x[1],
    )
    n_prune = int(len(sorted_nodes) * sparsity)
    return dict(sorted_nodes[n_prune:])


def spearman_rank_correlation(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
) -> float:
    """
    Spearman rank correlation of node scores across two runs.

    Only nodes present in both dicts are compared. Returns 0.0 if fewer
    than 2 shared nodes exist (undefined correlation).

    Args:
        scores_a: {node_name: score} from run A.
        scores_b: {node_name: score} from run B.

    Returns:
        Spearman rho in [-1, 1].
    """
    shared = sorted(set(scores_a) & set(scores_b))
    if len(shared) < 2:
        return 0.0

    from scipy.stats import spearmanr  # soft dependency; scipy is already in env

    a_vals = [scores_a[n] for n in shared]
    b_vals = [scores_b[n] for n in shared]
    rho, _ = spearmanr(a_vals, b_vals)
    return float(rho) if not np.isnan(rho) else 0.0


# ---------------------------------------------------------------------------
# Per-algorithm re-discovery helpers
# ---------------------------------------------------------------------------


def _graph_to_node_scores(graph) -> Dict[str, float]:
    """
    Extract {node_name: abs(score)} from an EAP graph post-attribution.
    For neuron-level graphs, aggregates neuron scores to node level (sum of abs).
    """
    from ..backends.eap.graph import AttentionNode, MLPNode

    scores: Dict[str, float] = {}
    for name, node in graph.nodes.items():
        if not isinstance(node, (AttentionNode, MLPNode)):
            continue

        # Neuron-level: aggregate from neurons_scores
        if graph.neurons_scores is not None:
            fwd_idx = graph.forward_index(node, attn_slice=False)
            neuron_scores = graph.neurons_scores[fwd_idx]
            valid = neuron_scores[~torch.isnan(neuron_scores)]
            val = float(valid.abs().sum().item()) if len(valid) > 0 else 0.0

        else:
            # Node-level: use node.score directly
            if node.score is None:
                continue
            raw = node.score
            val = abs(float(raw.item()) if hasattr(raw, "item") else raw)
            if torch.isnan(torch.tensor(val)):
                continue

        scores[name] = val

    return scores


def _rediscover_eap(model, task_spec, cfg: dict, algo: str, device: str) -> Dict[str, float]:
    from ..backends.eap.attribute_node import attribute_node
    from ..backends.eap.graph import Graph

    try:
        from ..utils.config import DEFAULT_CONFIG

        default = DEFAULT_CONFIG.get("discovery", {})
    except Exception:
        default = {}

    # Mirrors _ALGO_METHOD_MAP in api.py exactly.
    _ALGO_METHOD_MAP = {
        "eap": "EAP",
        "eap-ig": "EAP-IG-inputs",
        "eap-ig-activations": "EAP-IG-activations",
        "eap-clean-corrupted": "clean-corrupted",
        "eap-exact": "exact",
        "atp-gd": "atp-gd",
        "eap-gp": "eap-gp",
        "relp": "relp",
        "peap": "peap",
        "eap-ifr": "ifr",
    }

    # eap-ig honours an explicit method override (legacy behaviour from api.py).
    if algo == "eap-ig":
        _valid_node_methods = (
            "EAP",
            "EAP-IG-inputs",
            "EAP-IG-activations",
            "exact",
            "clean-corrupted",
        )
        method = cfg.get("method", default.get("method", "EAP-IG-inputs"))
        if method not in _valid_node_methods:
            method = "EAP-IG-inputs"
    else:
        method = _ALGO_METHOD_MAP[algo]

    # Methods that hard-require intervention='patching' in attribute_node()
    # (raise ValueError otherwise): all IG-based methods plus clean-corrupted.
    _patching_only_methods = {
        "EAP-IG-inputs",
        "EAP-IG-activations",
        "eap-gp",
        "clean-corrupted",
    }
    if method in _patching_only_methods:
        intervention = "patching"
    else:
        intervention = cfg.get("intervention", default.get("intervention", "patching"))

    is_neuron = cfg.get("level", "node") == "neuron"
    mlp_hook = cfg.get("mlp_hook", default.get("mlp_hook", "mlp_out"))

    graph = Graph.from_model(model, node_scores=True, neuron_level=is_neuron, mlp_hook=mlp_hook)
    dataloader = task_spec.build_dataloader(model, cfg, device)
    metric = task_spec.metric_fn()

    attribute_node(
        model,
        graph,
        dataloader,
        metric,
        method=method,
        ig_steps=cfg.get("ig_steps", default.get("ig_steps", 20)),
        neuron=is_neuron,
        intervention=intervention,
    )

    return _graph_to_node_scores(graph)


def _rediscover_acdc(model, task_spec, cfg: dict, device: str) -> Dict[str, float]:
    from ..backends.acdc.prune import calculate_node_scores_from_edges
    from ..backends.acdc.prune_algos.ACDC import acdc_prune_scores
    from ..backends.acdc.utils.graph_utils import patchable_model

    p_model = patchable_model(
        model, factorized=True, slice_output="last_seq", separate_qkv=True, device=device
    )
    train_loader = task_spec.build_dataloader(model, cfg, device)

    _acdc_kwargs = {}
    if "tao_exps" in cfg:
        _acdc_kwargs["tao_exps"] = list(cfg["tao_exps"])
    if "tao_bases" in cfg:
        _acdc_kwargs["tao_bases"] = list(cfg["tao_bases"])
    if "faithfulness_target" in cfg:
        _acdc_kwargs["faithfulness_target"] = cfg["faithfulness_target"]

    edge_scores = acdc_prune_scores(p_model, train_loader, official_edges=None, **_acdc_kwargs)
    logger.warning(
        "ACDC re-discovery returns scores keyed by module names, not EAP graph node "
        "names. Pairwise Jaccard/Dice stability metrics will be correct across ACDC "
        "runs, but layer-wise overlap breakdown will be empty. This is expected."
    )
    return calculate_node_scores_from_edges(p_model, edge_scores)


def _rediscover_ibcircuit(model, task_spec, cfg: dict, device: str) -> Dict[str, float]:
    from ..backends.ibcircuit.trainer import run_ib_discovery

    try:
        from ..utils.config import DEFAULT_CONFIG

        default = DEFAULT_CONFIG.get("discovery", {})
    except Exception:
        default = {}

    ib_config = {
        "num_epochs": cfg.get("num_epochs", default.get("num_epochs", 10)),
        "learning_rate": cfg.get("learning_rate", default.get("learning_rate", 1e-3)),
        "alpha": cfg.get("alpha", default.get("alpha", 1.0)),
        "beta": cfg.get("beta", default.get("beta", 1.0)),
        "alpha_loss": cfg.get("alpha_loss", default.get("alpha_loss", 1.0)),
        "scope": cfg.get("scope", default.get("scope", "both")),
        "mask_type": cfg.get("mask_type", default.get("mask_type", "hard")),
        "level": cfg.get("level", "node"),
        "mlp_hook": cfg.get("mlp_hook", default.get("mlp_hook", "mlp_out")),
    }

    # evaluate_circuit() sets use_attn_result/use_split_qkv_input/use_hook_mlp_in=True
    # on the model unconditionally for graph-based pillar evaluation. These flags cause
    # TransformerLens to store hook_result, hook_q/k/v_input, and hook_mlp_in activations
    # on every forward pass. IB training does not use any of these — it only hooks hook_z
    # and hook_mlp_out — but with the flags on, their activations still enter the autograd
    # graph, causing 3.5x memory blowup (21.7GB → 74GB) vs. initial discovery.
    # Disable them for IB training only, and restore unconditionally via finally.
    _eap_flags = ("use_attn_result", "use_split_qkv_input", "use_hook_mlp_in")
    saved_flags = {f: getattr(model.cfg, f, False) for f in _eap_flags}
    for f in _eap_flags:
        setattr(model.cfg, f, False)

    _log_gpu_mem("_rediscover_ibcircuit: before build_dataloader", logger)

    # Initialise to None so the finally block's del is always safe, even if
    # build_dataloader or run_ib_discovery raise before assignment.
    dataloader = None
    ib_model = None
    node_scores = None

    try:
        dataloader = task_spec.build_dataloader(model, cfg, device)

        _log_gpu_mem(
            "_rediscover_ibcircuit: after build_dataloader, before run_ib_discovery", logger
        )

        node_scores, ib_model = run_ib_discovery(model, dataloader, ib_config, device)

        # run_ib_discovery may return per-neuron Tensors (shape [d_model] or similar)
        # rather than scalar floats. Aggregate each value to a single Python float
        # (sum of abs) so that node_scores_to_circuit can sort and downstream
        # consumers (e.g. spearman_rank_correlation) never receive Tensor values.
        node_scores = {
            k: float(v.detach().abs().sum().item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in node_scores.items()
        }

    finally:
        # Restore flags so subsequent graph-based pillars (if any) still work.
        for f, v in saved_flags.items():
            setattr(model.cfg, f, v)
        if ib_model is not None:
            del ib_model
        if dataloader is not None:
            del dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return node_scores


def _rediscover_cdt(model, task_spec, cfg: dict, device: str) -> Dict[str, float]:
    """
    CD-T re-discovery for stability evaluation.

    CD-T returns node-level scores directly (no EAP graph, no edge aggregation).
    The adapter's output is already a Dict[str, float] compatible with
    node_scores_to_circuit and spearman_rank_correlation.

    Note: CD-T is node-level only; the 'level' key in cfg is ignored.
    Pairwise Jaccard/Dice stability metrics are meaningful across CD-T runs.
    Layer-wise overlap breakdown works correctly because CD-T node names
    share the EAP graph naming convention ('A{L}.{H}', 'MLP {L}').
    """
    from ..backends.cdt.adapter import run_cdt_discovery

    dataloader = task_spec.build_dataloader(model, cfg, device)
    node_scores = run_cdt_discovery(
        tl_model=model,
        dataloader=dataloader,
        device=device,
        n_examples=cfg.get("data_params", {}).get("num_examples", 16),
    )
    # run_cdt_discovery returns Dict[str, float]; coerce defensively.
    return {
        k: float(v.detach().abs().sum().item()) if isinstance(v, torch.Tensor) else float(v)
        for k, v in node_scores.items()
    }
