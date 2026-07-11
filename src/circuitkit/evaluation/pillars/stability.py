"""
Pillar 3: Stability Under Resampling

Evaluates whether the discovered circuit is stable and reproducible across
different random seeds and data resampling. Measures circuit consistency by
running discovery N times with different seeds and computing overlap metrics.

Core Concept:
- Run circuit discovery multiple times with different random seeds
- Compute pairwise overlap (Jaccard, Dice) between discovered circuits
- Score: Mean overlap and standard deviation across all pairs
- High score (near 1.0): Circuit is stable and consistent
- Low score (near 0.0): Circuit changes significantly with seed/data variations

This pillar answers: "Is the discovered circuit robust to initialization and
data variations? How consistent is the circuit across multiple runs?"
"""

import logging
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from ...backends.eap.graph import Graph
from ..evaluate import evaluate_graph
from ..stability_discovery import node_scores_to_circuit, rediscover, spearman_rank_correlation
from circuitkit.utils.device import get_device, empty_cache

logger = logging.getLogger(__name__)


class Pillar3_Stability:
    """
    Pillar 3: Stability Under Resampling.

    Measures whether the circuit is stable and reproducible by running
    discovery multiple times with different seeds and computing overlap
    metrics between discovered circuits.

    This pillar answers: "Is the discovered circuit robust to random
    variations in initialization and data?"

    Metrics:
    - Jaccard: |A ∩ B| / |A ∪ B|
    - Dice: 2|A ∩ B| / (|A| + |B|)
    - Layer-wise overlap breakdown
    """

    @staticmethod
    def run(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        n_runs: int = 5,
        seed_start: int = 42,
        device: str = "auto",
        quiet: bool = False,
        task_spec=None,
        discovery_cfg: Optional[Dict] = None,
        sparsity: float = 0.3,
    ) -> Dict:
        """
        Run circuit stability evaluation across multiple discovery runs.

        Discovers the circuit N times with different random seeds and
        computes pairwise overlap metrics between all discovered circuits.

        Args:
            model: HookedTransformer model with use_attn_result=True.
            graph: Circuit graph with in_graph flags set on edges/nodes.
            dataloader: Evaluation dataset yielding (clean, corrupted, label) batches.
            metric_fn: Metric function with signature
                (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
            n_runs: Number of discovery runs with different seeds (default 5).
            seed_start: Starting seed for determinism (default 42).
            device: Target device ("cuda" or "cpu"). Defaults to "cuda".
            quiet: Suppress progress bar. Defaults to False.
            task_spec: TaskSpec used to rebuild dataloaders for each re-discovery run.
                Required for real stability evaluation. If None, falls back to
                comparing the baseline circuit against itself (meaningless).
            discovery_cfg: Discovery configuration dict passed to stability_discovery.
                Required for real stability evaluation alongside task_spec.
            sparsity: Fraction of lowest-scoring nodes to discard when converting
                raw re-discovery scores to circuit node sets. Should match the
                sparsity used in the original circuit discovery (default 0.3).

        Returns:
            Dict with keys:
            - 'mean_jaccard': Mean Jaccard similarity across all pairs
            - 'std_jaccard': Standard deviation of Jaccard similarity
            - 'mean_dice': Mean Dice coefficient across all pairs
            - 'std_dice': Standard deviation of Dice coefficient
            - 'mean_spearman': Mean Spearman rank correlation across all pairs
            - 'std_spearman': Std of Spearman rank correlation
            - 'jaccard_matrix': ndarray (n_runs, n_runs) with pairwise Jaccard
            - 'dice_matrix': ndarray (n_runs, n_runs) with pairwise Dice
            - 'spearman_matrix': ndarray (n_runs, n_runs) with pairwise Spearman
            - 'n_stable_nodes': Number of nodes present in all runs
            - 'overlap_per_layer': Dict mapping layer index to mean layer-wise Jaccard
            - 'circuits': List of circuit node dicts (post-sparsity, for inspection)
            - 'raw_scores': List of raw score dicts (pre-sparsity, for further analysis)

        Raises:
            AssertionError: If model.cfg.use_attn_result is False.
            ValueError: If graph is None or invalid.
        """
        if graph is None:
            raise ValueError("Graph cannot be None")

        if not hasattr(model.cfg, "use_attn_result") or not model.cfg.use_attn_result:
            raise AssertionError(
                "Model must be configured with use_attn_result=True. "
                "Configure model with: model.cfg.use_attn_result = True"
            )

        logger.info(
            f"Pillar 3: Running stability evaluation with {n_runs} runs "
            f"(seeds {seed_start}-{seed_start + n_runs - 1})..."
        )

        # Extract current circuit as baseline (run 0)
        baseline_circuit = Pillar3_Stability._extract_circuit_nodes(graph)

        logger.info(
            f"Baseline circuit has {len(baseline_circuit)} nodes "
            f"({Pillar3_Stability._count_nodes_by_type(graph, baseline_circuit)})"
        )

        # Real re-discovery: run discovery n_runs times with different seeds.
        # If task_spec / discovery_cfg are provided, use stability_discovery.rediscover().
        # Otherwise fall back to bootstrap-only mode (no re-discovery — warns loudly).
        if task_spec is not None and discovery_cfg is not None:
            raw_scores_list = rediscover(
                model=model,
                task_spec=task_spec,
                discovery_cfg=discovery_cfg,
                n_runs=n_runs,
                seed_start=seed_start,
                device=device,
            )
            # Apply sparsity threshold to each run's raw scores to get circuit node sets
            circuits = [node_scores_to_circuit(rs, sparsity) for rs in raw_scores_list]
        else:
            logger.warning(
                "Pillar3_Stability.run() called without task_spec/discovery_cfg — "
                "stability scores will be meaningless (all circuits identical). "
                "Pass task_spec and discovery_cfg for real re-discovery."
            )
            raw_scores_list = [baseline_circuit] * n_runs
            circuits = [baseline_circuit.copy() for _ in range(n_runs)]

        # Compute pairwise overlap metrics
        jaccard_matrix = np.zeros((n_runs, n_runs))
        dice_matrix = np.zeros((n_runs, n_runs))
        spearman_matrix = np.zeros((n_runs, n_runs))

        for i in range(n_runs):
            for j in range(n_runs):
                jaccard_matrix[i, j] = Pillar3_Stability.compute_jaccard(circuits[i], circuits[j])
                dice_matrix[i, j] = Pillar3_Stability.compute_dice(circuits[i], circuits[j])
                # Spearman on raw scores (pre-sparsity) — threshold-free stability measure
                spearman_matrix[i, j] = spearman_rank_correlation(
                    raw_scores_list[i] if isinstance(raw_scores_list[i], dict) else circuits[i],
                    raw_scores_list[j] if isinstance(raw_scores_list[j], dict) else circuits[j],
                )

        # Compute statistics (excluding diagonal)
        if n_runs <= 1:
            mean_jaccard, std_jaccard = 1.0, 0.0
            mean_dice, std_dice = 1.0, 0.0
            mean_spearman, std_spearman = 1.0, 0.0
        else:
            mask = ~np.eye(n_runs, dtype=bool)
            jaccard_values = jaccard_matrix[mask]
            dice_values = dice_matrix[mask]
            spearman_values = spearman_matrix[mask]

            mean_jaccard = float(np.mean(jaccard_values))
            std_jaccard = float(np.std(jaccard_values))
            mean_dice = float(np.mean(dice_values))
            std_dice = float(np.std(dice_values))
            mean_spearman = float(np.mean(spearman_values))
            std_spearman = float(np.std(spearman_values))

        # Count nodes present in every run
        n_stable_nodes = len(set.intersection(*[set(c.keys()) for c in circuits]))

        # Compute layer-wise overlap
        overlap_per_layer = Pillar3_Stability._compute_layer_wise_overlap(circuits, graph)

        logger.info("Pillar 3 Stability Metrics:")
        logger.info(f"  Mean Jaccard:     {mean_jaccard:.4f} ± {std_jaccard:.4f}")
        logger.info(f"  Mean Dice:        {mean_dice:.4f} ± {std_dice:.4f}")
        logger.info(f"  Mean Spearman ρ:  {mean_spearman:.4f} ± {std_spearman:.4f}")
        logger.info(f"  Stable nodes:     {n_stable_nodes}")
        logger.info(f"  Layer breakdown:  {overlap_per_layer}")

        return {
            "mean_jaccard": mean_jaccard,
            "std_jaccard": std_jaccard,
            "mean_dice": mean_dice,
            "std_dice": std_dice,
            "mean_spearman": mean_spearman,
            "std_spearman": std_spearman,
            "jaccard_matrix": jaccard_matrix,
            "dice_matrix": dice_matrix,
            "spearman_matrix": spearman_matrix,
            "n_stable_nodes": n_stable_nodes,
            "overlap_per_layer": overlap_per_layer,
            "circuits": circuits,
            "raw_scores": raw_scores_list,
            "n_runs": n_runs,
        }

    @staticmethod
    def compute_jaccard(circuit1: Dict, circuit2: Dict) -> float:
        """
        Compute Jaccard similarity between two circuits.

        Jaccard = |A ∩ B| / |A ∪ B|

        Args:
            circuit1: Dict of node name -> node info
            circuit2: Dict of node name -> node info

        Returns:
            float: Jaccard similarity in [0, 1]
        """
        set1 = set(circuit1.keys())
        set2 = set(circuit2.keys())

        if len(set1 | set2) == 0:
            return 1.0  # Both empty circuits

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def compute_dice(circuit1: Dict, circuit2: Dict) -> float:
        """
        Compute Dice coefficient between two circuits.

        Dice = 2|A ∩ B| / (|A| + |B|)

        Args:
            circuit1: Dict of node name -> node info
            circuit2: Dict of node name -> node info

        Returns:
            float: Dice coefficient in [0, 1]
        """
        set1 = set(circuit1.keys())
        set2 = set(circuit2.keys())

        if len(set1) + len(set2) == 0:
            return 1.0  # Both empty circuits

        intersection = len(set1 & set2)
        total = len(set1) + len(set2)

        return float(2.0 * intersection / total) if total > 0 else 0.0

    @staticmethod
    def _extract_circuit_nodes(graph: Graph) -> Dict:
        """
        Extract in-graph nodes with derived scores (neuron-aware).

        Args:
            graph: Circuit graph

        Returns:
            Dict mapping node name to node metadata
        """
        circuit = {}
        for node_name, node in graph.nodes.items():
            if node_name == "logits":
                continue
            if graph.neurons_in_graph is not None:
                # ibcircuit / neuron-level: node is in circuit if any of its neurons
                # survive pruning (i.e. neurons_in_graph row is not all-zero)
                fwd_idx = graph.forward_index(node, attn_slice=False)
                if not graph.neurons_in_graph[fwd_idx].any():
                    continue
            else:
                # EAP / node-level: use the standard in_graph flag
                if not node.in_graph:
                    continue

            # Derive score: aggregate neurons if neuron-level
            if graph.neurons_scores is not None:
                fwd_idx = graph.forward_index(node, attn_slice=False)
                neuron_scores = graph.neurons_scores[fwd_idx]
                valid = neuron_scores[~torch.isnan(neuron_scores)]
                score = float(valid.abs().sum().item()) if len(valid) > 0 else 0.0
            else:
                s = node.score
                score = float(s.item()) if s is not None and not torch.isnan(s) else None

            circuit[node_name] = {
                "layer": node.layer,
                "type": type(node).__name__,
                "score": score,
            }

        return circuit

    @staticmethod
    def _count_nodes_by_type(graph: Graph, circuit: Dict) -> Dict[str, int]:
        """
        Count circuit nodes by type (AttentionNode, MLPNode, etc.).

        Args:
            graph: Circuit graph
            circuit: Circuit nodes dict

        Returns:
            Dict mapping type name to count
        """
        counts = {}
        for node_name in circuit.keys():
            if node_name in graph.nodes:
                node = graph.nodes[node_name]
                type_name = type(node).__name__
                counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    # Convention-agnostic node-name parsers. Covers the node-name forms used
    # across the codebase:
    #   - EAP attention heads: 'a{L}.h{H}'  (graph.py AttentionNode)
    #   - EAP MLP nodes:       'm{L}'       (graph.py MLPNode)
    #   - IBCircuit / ACDC / CD-T attention heads: 'A{L}.{H}' (also 'a{L}.{H}')
    #   - IBCircuit / ACDC / CD-T MLP nodes:       'MLP {L}', 'mlp{L}', 'm{L}'
    _ATTN_NAME_RE = re.compile(r"^[Aa]\s*(\d+)\s*\.\s*h?(\d+)$")
    _MLP_NAME_RE = re.compile(r"^(?:MLP|mlp|m)\s*(\d+)$", re.IGNORECASE)

    @staticmethod
    def _node_layer(node_name: str, graph: Optional[Graph] = None) -> Optional[int]:
        """
        Derive the transformer layer index of a circuit node.

        Uses ``graph.nodes[node_name].layer`` when the name is a key in the
        EAP graph (most accurate), and otherwise falls back to parsing the
        node-name string convention-agnostically. Handles attention-head names
        (``a{L}.h{H}`` / ``A{L}.{H}`` / ``a{L}.{H}``) and MLP names
        (``MLP {L}`` / ``m{L}`` / ``mlp{L}``).

        Args:
            node_name: Circuit node name.
            graph: Optional EAP graph; consulted first when the name is a key.

        Returns:
            int layer index, or None if the layer cannot be determined.
        """
        if node_name == "logits" or node_name == "input":
            return None

        # Most accurate: the EAP graph knows the layer directly.
        if graph is not None and node_name in graph.nodes:
            layer = getattr(graph.nodes[node_name], "layer", None)
            if layer is not None:
                return int(layer)

        # Fall back to parsing the name string.
        m = Pillar3_Stability._ATTN_NAME_RE.match(node_name.strip())
        if m:
            return int(m.group(1))
        m = Pillar3_Stability._MLP_NAME_RE.match(node_name.strip())
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _compute_layer_wise_overlap(circuits: List[Dict], graph: Graph) -> Dict[int, float]:
        """
        Compute mean overlap per layer across all circuit pairs.

        The layer of each node is derived by :meth:`_node_layer`, which uses
        the EAP graph when the node name is a graph key and otherwise parses
        the node-name string. This keeps the breakdown non-empty for circuits
        (IBCircuit / ACDC / CD-T) whose node names do not match EAP graph keys.

        Args:
            circuits: List of circuit node dicts
            graph: Circuit graph

        Returns:
            Dict mapping layer index to mean Jaccard overlap
        """
        overlap_per_layer = {}

        # Map every node in every circuit to its layer (parse-based, graph-aware).
        node_layers: Dict[str, Optional[int]] = {}
        for circuit in circuits:
            for node_name in circuit.keys():
                if node_name not in node_layers:
                    node_layers[node_name] = Pillar3_Stability._node_layer(node_name, graph)

        all_layers = {layer for layer in node_layers.values() if layer is not None}

        if not all_layers:
            logger.warning(
                "_compute_layer_wise_overlap: could not derive a layer for any circuit "
                "node. Layer-wise overlap will be empty. This indicates a node-name "
                "convention mismatch — node names matched neither the EAP graph keys nor "
                "any known attention/MLP naming convention."
            )
            return {}

        # Compute layer-wise Jaccard for each pair
        for layer in sorted(all_layers):
            layer_jaccard_values = []

            for i in range(len(circuits)):
                for j in range(i + 1, len(circuits)):
                    # Extract nodes for this layer from each circuit
                    circuit1_layer = {
                        n: v for n, v in circuits[i].items() if node_layers.get(n) == layer
                    }
                    circuit2_layer = {
                        n: v for n, v in circuits[j].items() if node_layers.get(n) == layer
                    }

                    jaccard = Pillar3_Stability.compute_jaccard(circuit1_layer, circuit2_layer)
                    layer_jaccard_values.append(jaccard)

            if layer_jaccard_values:
                overlap_per_layer[int(layer)] = float(np.mean(layer_jaccard_values))
            else:
                overlap_per_layer[int(layer)] = 1.0  # No nodes in layer, perfect overlap

        return overlap_per_layer

    @staticmethod
    def compute_bootstrap_stability(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        n_bootstrap: int = 10,
        bootstrap_fraction: float = 0.8,
        device: str = "auto",
        quiet: bool = False,
    ) -> Dict:
        """
        Compute circuit stability under data resampling (bootstrap).

        Resamples the dataloader N times and evaluates the circuit on each
        resample, measuring performance variance.

        Args:
            model: HookedTransformer model
            graph: Circuit graph
            dataloader: Evaluation dataset
            metric_fn: Metric function
            n_bootstrap: Number of bootstrap samples (default 10)
            bootstrap_fraction: Fraction of data to sample per bootstrap (default 0.8)
            device: Target device
            quiet: Suppress progress bar

        Returns:
            Dict with keys:
            - 'scores': List of per-bootstrap circuit scores
            - 'mean_score': Mean score across bootstraps
            - 'std_score': Standard deviation of scores
            - 'performance_stability': Coefficient of variation std/|mean|
              (lower is more stable; unclamped, so values above 1 mean the
              spread exceeds the mean). ``None`` with ``status='invalid'``
              when |mean| ~ 0 (signed metric collapsed or straddling zero),
              where the CV is undefined.
        """
        logger.info(
            f"Computing bootstrap stability with {n_bootstrap} resamples "
            f"(fraction: {bootstrap_fraction})..."
        )

        scores = []

        # Collect all batches once, then resample with replacement per bootstrap
        all_batches = list(dataloader)
        rng = np.random.RandomState(42)

        sample_size = max(1, int(len(all_batches) * bootstrap_fraction))

        for i in range(n_bootstrap):
            # Sample len(all_batches) batches with replacement
            indices = rng.choice(len(all_batches), size=sample_size, replace=True)
            bootstrap_batches = [all_batches[idx] for idx in indices]

            try:
                circuit_score = evaluate_graph(
                    model=model,
                    graph=graph,
                    dataloader=bootstrap_batches,
                    metrics=metric_fn,
                    quiet=True,
                    intervention="patching",
                    skip_clean=True,
                )

                if isinstance(circuit_score, list):
                    circuit_score = circuit_score[0]

                if hasattr(circuit_score, "cpu"):
                    circuit_score = circuit_score.cpu()

                if circuit_score.ndim == 0:
                    score = float(circuit_score.item())
                else:
                    score = float(circuit_score.mean().item())

                scores.append(score)

            except Exception as e:
                logger.warning(f"Bootstrap sample {i} failed: {e}")
                continue

        if not scores:
            raise RuntimeError("All bootstrap samples failed")

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        # The coefficient of variation must use |mean|: the old
        # `if mean_score > 0 else 0.0` returned a 0.0 SENTINEL for an
        # all-negative signed metric (e.g. scores [-2.1, -1.9, -2.0]), which
        # per the "lower is more stable" convention reads as PERFECTLY STABLE
        # — a confident best-case verdict for an undefined case. And a
        # sign-mixed sample with a near-zero mean produced a huge ratio that
        # min(..., 1.0) silently clamped, hiding the degenerate denominator.
        if abs(mean_score) < 1e-6:
            logger.warning(
                f"Bootstrap stability: |mean score| ~ 0 ({mean_score:.6f}) — the "
                "coefficient of variation is undefined (scores straddle zero or the "
                "metric collapsed). Reporting performance_stability=None."
            )
            return {
                "scores": scores,
                "mean_score": mean_score,
                "std_score": std_score,
                "performance_stability": None,
                "status": "invalid",
                "reason": (
                    "performance_stability (std/|mean|) undefined: the mean bootstrap "
                    "score is ~0 for this signed metric, so the coefficient of "
                    "variation explodes. Inspect 'scores' directly."
                ),
                "n_bootstrap": n_bootstrap,
            }

        stability_ratio = std_score / abs(mean_score)

        logger.info("Bootstrap Stability:")
        logger.info(f"  Mean score:        {mean_score:.4f} ± {std_score:.4f}")
        logger.info(f"  Stability ratio:   {stability_ratio:.4f} (lower is better)")

        return {
            "scores": scores,
            "mean_score": mean_score,
            "std_score": std_score,
            # Unclamped: a CV above 1 is real information (std exceeds |mean|),
            # and the old min(..., 1.0) hid exactly the degenerate cases.
            "performance_stability": stability_ratio,
            "n_bootstrap": n_bootstrap,
        }
