"""Pillar 7 - Intervention Reliability.

Measures how consistently a circuit-guided intervention produces its
intended effect across re-runs, random seeds, and minor prompt
variations. Three sub-scores are combined into a single reliability
index:

  R1 (seed consistency): Spearman rho of per-node attribution scores
     across n_seeds random dataloader shuffles of the same task.
     High rho => the top-K circuit is reproducible across seeds.

  R2 (effect magnitude): Mean normalized intervention effect
     (circuit score - baseline score) / baseline score across seeds.
     Positive => the intervention meaningfully changes model behavior.

  R3 (effect variance): 1 - CV of intervention effect across seeds,
     where CV = std / |mean|. Low variance => reliable effect size.

Overall reliability = harmonic mean of (R1 + 1) / 2, R2_clamped, R3.
Ranges in [0, 1]; higher is better.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation of two score lists."""
    n = len(x)
    if n < 3:
        return 0.0
    rank_x = _ranks(x)
    rank_y = _ranks(y)
    d2 = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    denom = n * (n**2 - 1)
    return 1.0 - 6.0 * d2 / denom if denom > 0 else 0.0


def _ranks(scores: List[float]) -> List[float]:
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    for rank, idx in enumerate(sorted_idx, 1):
        ranks[idx] = float(rank)
    return ranks


def _score_circuit_vs_baseline(
    model,
    graph,
    dataloader: DataLoader,
    metric_fn,
    device: str,
) -> Tuple[float, float]:
    """Return (circuit_score, baseline_score) averaged over the dataloader."""
    from circuitkit.evaluation.evaluate import evaluate_baseline, evaluate_graph

    baseline = float(torch.mean(evaluate_baseline(model, dataloader, metric_fn)).item())
    circuit = float(torch.mean(evaluate_graph(model, graph, dataloader, metric_fn)).item())
    return circuit, baseline


def _node_score_vector(graph) -> List[float]:
    """Extract a flat node-score vector from a Graph (for rank correlation)."""
    from circuitkit.backends.eap.graph import AttentionNode, MLPNode

    scores = []
    for node in graph.nodes.values():
        if isinstance(node, (AttentionNode, MLPNode)):
            s = getattr(node, "score", None)
            if s is not None:
                scores.append(float(s.item()) if hasattr(s, "item") else float(s))
            else:
                scores.append(0.0)
    return scores


def _rebuild_graph_with_seed(
    model,
    task_spec,
    discovery_cfg: Dict[str, Any],
    pruning_cfg: Dict[str, Any],
    device: str,
    seed: int,
):
    """Re-run discovery with a different data seed and return the scored graph."""
    from circuitkit.api import _compute_n_topn, _convert_eap_scores_to_ck_format
    from circuitkit.backends.eap.graph import Graph

    # Override data seed at both levels (top-level for WMDP/MMLU/Generic/GLUE/IOI-ACDC,
    # data_params level for IOI/BoolQ/SVA/GreaterThan and the EAP family).
    seeded_cfg = {
        **discovery_cfg,
        "seed": seed,
        "data_params": {**discovery_cfg.get("data_params", {}), "seed": seed},
    }
    seeded_dataloader = task_spec.build_dataloader(model, seeded_cfg, device)

    algo = discovery_cfg.get("algorithm", "eap").lower()

    # Run attribution.

    from circuitkit.backends.eap.attribute_node import attribute_node

    metric = task_spec.metric_fn()
    mlp_hook = discovery_cfg.get("mlp_hook", "mlp_out")
    graph = Graph.from_model(model, node_scores=True, neuron_level=False, mlp_hook=mlp_hook)

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
    method = _ALGO_METHOD_MAP.get(algo, "EAP")

    attribute_node(
        model,
        graph,
        seeded_dataloader,
        metric,
        method=method,
        ig_steps=discovery_cfg.get("ig_steps", 3),
        neuron=False,
        intervention=discovery_cfg.get("intervention", "patching"),
    )

    node_scores = _convert_eap_scores_to_ck_format(graph)
    n_topn, _ = _compute_n_topn(
        graph,
        pruning_cfg.get("scope", "heads"),
        pruning_cfg.get("target_sparsity", 0.1),
    )
    graph.apply_topn(n_topn, level="node", prune=True)
    return graph, node_scores


_R2_BASELINE_EPS = 1e-6


def _r2_effect_magnitude(deltas: List[float], baselines: List[float]) -> float:
    """Mean normalized effect magnitude, mapped to [0, 1].

    Seeds whose baseline is ~0 are SKIPPED rather than normalized: the old
    ``d / (abs(b) + 1e-8)`` with ``b ~ 0`` produced an astronomically large
    norm_delta that saturated the clamp to a maximal r2 = 1.0 — reporting a
    top "effect magnitude" for an undefined normalized effect and inflating
    the reliability_index. If every baseline is degenerate there is no
    defined effect magnitude, so the neutral midpoint 0.5 is returned rather
    than a saturated extreme.
    """
    valid_pairs = [(d, b) for d, b in zip(deltas, baselines) if abs(b) >= _R2_BASELINE_EPS]
    n_skipped = len(deltas) - len(valid_pairs)
    if n_skipped:
        logger.warning(
            f"Pillar 7 R2: skipped {n_skipped} of {len(deltas)} seed(s) with "
            f"|baseline| < {_R2_BASELINE_EPS} — normalized effect undefined there."
        )
    if not valid_pairs:
        return 0.5
    norm_deltas = [d / abs(b) for d, b in valid_pairs]
    r2_raw = float(sum(norm_deltas) / len(norm_deltas))
    return max(0.0, min(1.0, (r2_raw + 1) / 2))  # map [-1,1] -> [0,1]


def run_intervention_reliability(
    model,
    graph,
    task_spec,
    discovery_cfg: Dict[str, Any],
    pruning_cfg: Dict[str, Any],
    device: str,
    metric_fn,
    dataloader: DataLoader,
    n_seeds: int = 3,
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run Pillar 7 (Intervention Reliability).

    Args:
        model: HookedTransformer model.
        graph: Circuit graph from the primary discovery run.
        task_spec: Task specification with build_dataloader().
        discovery_cfg: Discovery configuration dict (algorithm, task, etc.).
        pruning_cfg: Pruning configuration dict.
        device: Torch device string.
        metric_fn: Metric callable used for evaluation.
        dataloader: Evaluation dataloader (primary seed).
        n_seeds: Number of re-run seeds for stability measurement.
        seeds: Explicit seed list; overrides n_seeds if provided.

    Returns:
        Dict with keys:
            r1_seed_consistency  : Spearman rho averaged across seed pairs [0,1].
            r2_effect_magnitude  : Mean (circuit - baseline) / baseline.
            r3_effect_variance   : 1 - CV across seeds, clamped to [0,1].
            reliability_index    : Harmonic mean of the three sub-scores.
            per_seed             : List of per-seed dicts with circuit/baseline/delta.
    """
    if seeds is None:
        seeds = [42 + i * 17 for i in range(n_seeds)]

    logger.info(f"Pillar 7 — Intervention Reliability: seeds={seeds}")

    # Score on the primary (already-discovered) circuit.
    try:
        circuit_score_0, baseline_score_0 = _score_circuit_vs_baseline(
            model, graph, dataloader, metric_fn, device
        )
    except Exception as exc:
        logger.warning(f"Pillar 7: primary circuit score failed: {exc}")
        return _empty_result()

    per_seed_scores: List[Dict[str, float]] = [
        {
            "seed": seeds[0] if seeds else 42,
            "circuit": circuit_score_0,
            "baseline": baseline_score_0,
            "delta": circuit_score_0 - baseline_score_0,
        }
    ]

    all_node_vecs: List[List[float]] = [_node_score_vector(graph)]

    for seed in seeds[1:]:
        try:
            seeded_graph, _ = _rebuild_graph_with_seed(
                model, task_spec, discovery_cfg, pruning_cfg, device, seed
            )
            seeded_dl = task_spec.build_dataloader(
                model,
                {
                    **discovery_cfg,
                    "seed": seed,
                    "data_params": {**discovery_cfg.get("data_params", {}), "seed": seed},
                },
                device,
            )
            cs, bs = _score_circuit_vs_baseline(model, seeded_graph, seeded_dl, metric_fn, device)
            per_seed_scores.append(
                {
                    "seed": seed,
                    "circuit": cs,
                    "baseline": bs,
                    "delta": cs - bs,
                }
            )
            all_node_vecs.append(_node_score_vector(seeded_graph))
        except Exception as exc:
            logger.warning(f"Pillar 7: seed {seed} failed: {exc}")

    if len(per_seed_scores) < 2:
        return _empty_result()

    # R1: mean Spearman rho across all seed pairs.
    n = len(all_node_vecs)
    rho_pairs: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if len(all_node_vecs[i]) == len(all_node_vecs[j]):
                rho_pairs.append(_spearman_rho(all_node_vecs[i], all_node_vecs[j]))
    r1 = float(sum(rho_pairs) / len(rho_pairs)) if rho_pairs else 0.0

    # R2: mean normalized effect magnitude (see _r2_effect_magnitude for the
    # near-zero-baseline handling).
    deltas = [e["delta"] for e in per_seed_scores]
    baselines = [e["baseline"] for e in per_seed_scores]
    r2 = _r2_effect_magnitude(deltas, baselines)

    # R3: 1 - CV of raw deltas; CV = std / |mean|.
    mean_delta = sum(deltas) / len(deltas)
    if abs(mean_delta) < 1e-8:
        r3 = 1.0 if all(abs(d) < 1e-6 for d in deltas) else 0.0
    else:
        std_delta = math.sqrt(sum((d - mean_delta) ** 2 for d in deltas) / len(deltas))
        cv = std_delta / abs(mean_delta)
        r3 = max(0.0, min(1.0, 1.0 - cv))

    # Harmonic mean of R1_norm, R2, R3 (all in [0,1]).
    r1_norm = (r1 + 1.0) / 2.0  # Spearman in [-1,1] -> [0,1]
    components = [r1_norm, r2, r3]
    denom = sum(1.0 / (c + 1e-8) for c in components)
    reliability_index = len(components) / denom

    result: Dict[str, Any] = {
        "r1_seed_consistency": round(r1, 4),
        # Report the clamped, [0,1]-mapped r2 that actually feeds the
        # reliability index — not r2_raw, which is unbounded and would put an
        # out-of-range value under a field the composite treats as [0,1].
        "r2_effect_magnitude": round(r2, 4),
        "r3_effect_variance": round(r3, 4),
        "reliability_index": round(reliability_index, 4),
        "n_seeds": len(per_seed_scores),
        "per_seed": per_seed_scores,
    }
    logger.info(
        f"Pillar 7 done: R1={r1:.3f} R2={r2:.3f} R3={r3:.3f} "
        f"reliability={reliability_index:.3f}"
    )
    return result


def _empty_result() -> Dict[str, Any]:
    return {
        "r1_seed_consistency": None,
        "r2_effect_magnitude": None,
        "r3_effect_variance": None,
        "reliability_index": None,
        "n_seeds": 0,
        "per_seed": [],
        "error": "insufficient seed results",
    }
