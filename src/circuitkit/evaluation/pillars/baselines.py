"""
Pillar 5: Baselines Comparison

Evaluates circuit quality by comparing it against simple heuristic baselines.
Provides quantitative evidence that the discovered circuit is more effective
than random selection, magnitude-based selection, or other baselines.

Core Concept:
- Circuit: Discovered via attribution algorithm
- Random Baseline: Same sparsity, random nodes
- Magnitude Baseline: Top-k nodes by |score|
- Wanda Baseline: Weight * Activation product (LLM pruning heuristic)
- Score: How much the circuit outperforms each baseline

High score (circuit >> baselines): Circuit captures meaningful causal structure
Low score (circuit ~ baselines): Circuit is no better than heuristics

This pillar answers: "Is the discovered circuit better than simple heuristics?
How much do we gain from using the attribution algorithm?"
"""

import logging
from circuitkit.utils.device import get_device, empty_cache
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from ...backends.eap.graph import AttentionNode, Graph, MLPNode
from ..evaluate import evaluate_graph

logger = logging.getLogger(__name__)


class Pillar5_Baselines:
    """
    Pillar 5: Baselines Comparison.

    Evaluates circuit quality by comparing against simple heuristic baselines.
    Ensures the discovered circuit provides meaningful advantage over random
    selection or magnitude-based pruning.

    This pillar answers: "Is the circuit better than simple heuristics?
    How much does attribution help vs random/magnitude selection?"

    Supported baselines:
    - 'random': Same sparsity, random node selection
    - 'magnitude': Top-k nodes by |score|
    - 'wanda': Weight*Activation product (LLM pruning heuristic)
    """

    @staticmethod
    def _is_neuron_level(graph: Graph) -> bool:
        return graph.neurons_in_graph is not None

    @staticmethod
    def _get_neuron_level_sparsity(graph: Graph) -> float:
        """Compute fraction of neuron slots that are active (in-circuit)."""
        total = graph.neurons_in_graph.numel()
        active = int(graph.neurons_in_graph.sum().item())
        return active / total if total > 0 else 0.0

    @staticmethod
    def _get_node_summary_scores(graph: Graph) -> Dict[str, float]:
        """
        Derive per-node scores usable for magnitude ranking.
        For neuron-level: sum of absolute neuron scores per node.
        For node-level: node.score as before.
        """

        scores = {}
        for node_name, node in graph.nodes.items():
            if node_name == "logits":
                continue
            if graph.neurons_scores is not None:
                fwd_idx = graph.forward_index(node, attn_slice=False)
                neuron_scores = graph.neurons_scores[fwd_idx]
                valid = neuron_scores[~torch.isnan(neuron_scores)]
                scores[node_name] = float(valid.abs().sum().item()) if len(valid) > 0 else 0.0
            else:
                s = node.score
                scores[node_name] = (
                    abs(float(s.item())) if s is not None and not torch.isnan(s) else 0.0
                )
        return scores

    @staticmethod
    def _compute_weight_norm_scores(model: HookedTransformer, graph: Graph) -> torch.Tensor:
        """
        Compute per-neuron weight-norm scores as an independent magnitude criterion.

        Returns a tensor of the same shape as graph.neurons_scores [n_forward, max_d],
        with NaN in padding slots and L2 weight norms in valid neuron slots.
        Used to build a truly independent magnitude baseline.

        model.W_O and model.W_out are TransformerLens @property accessors that
        call torch.stack across all layers on every access, allocating a fresh
        CUDA tensor each time. Calling them inside the node loop on a 26-layer
        model creates ~51 GB of transient CUDA allocations across 208 attention
        nodes, fragmenting VRAM to the point of OOM. Both matrices are cached on
        CPU once before the loop; per-node access is then a free CPU index op.
        float32 is used for norm accuracy; results are cast back to the target
        dtype at assignment. CPU RAM cost is ~2.7 GB for Gemma-2-2B scale.
        """
        weight_scores = torch.full_like(graph.neurons_scores, float("nan"))

        # Cache weight matrices on CPU once to avoid O(n_nodes) CUDA re-stacks.
        with torch.no_grad():
            W_O_cpu = (
                torch.stack([block.attn.W_O for block in model.blocks], dim=0).cpu().float()
            )  # [n_layers, n_heads, d_head, d_model]
            W_out_cpu = (
                torch.stack([block.mlp.W_out for block in model.blocks], dim=0).cpu().float()
            )  # [n_layers, d_mlp, d_model]

        for node_name, node in graph.nodes.items():
            if node_name in ("logits", "input"):
                continue

            fwd_idx = graph.forward_index(node, attn_slice=False)
            n = node.d_neuron  # authoritative count of actual neuron slots

            if isinstance(node, AttentionNode):
                # W_O: per-head [d_head, d_model]; norm across d_head → [d_model]
                W = W_O_cpu[node.layer, node.head]  # [d_head, d_model]
                norms = W.norm(dim=0)  # [d_model]
            elif isinstance(node, MLPNode):
                # W_out: [d_mlp, d_model]
                if graph.cfg.get("mlp_hook") == "post_act":
                    W = W_out_cpu[node.layer]  # [d_mlp, d_model]
                    norms = W.norm(dim=1)  # [d_mlp]
                else:
                    W = W_out_cpu[node.layer]  # [d_mlp, d_model]
                    norms = W.norm(dim=0)  # [d_model]
            else:
                continue

            weight_scores[fwd_idx, :n] = norms[:n].to(
                device=weight_scores.device, dtype=weight_scores.dtype
            )

        del W_O_cpu, W_out_cpu  # release CPU RAM immediately
        return weight_scores

    @staticmethod
    def _compute_node_weight_norm_scores(model: HookedTransformer, graph: Graph) -> dict:
        """
        Compute per-node weight-norm scores for node-level graphs.

        Uses RMS weight magnitude (Frobenius norm / sqrt(numel)) so that attention
        heads and MLP nodes are on a size-invariant scale and can be ranked together.
        Nodes without weight matrices (input, logits, and any unrecognised type)
        receive inf so they are always retained.

        Weight matrices are cached on CPU once before the loop to avoid repeated
        O(n_nodes) CUDA stacks — see _compute_weight_norm_scores for the rationale.

        Returns:
            dict mapping node_name -> float score (higher = larger RMS weight norm)
        """
        node_scores: dict = {}

        with torch.no_grad():
            W_O_cpu = (
                torch.stack([block.attn.W_O for block in model.blocks], dim=0).cpu().float()
            )  # [n_layers, n_heads, d_head, d_model]
            W_out_cpu = (
                torch.stack([block.mlp.W_out for block in model.blocks], dim=0).cpu().float()
            )  # [n_layers, d_mlp, d_model]

        for node_name, node in graph.nodes.items():
            if node_name in ("logits", "input"):
                node_scores[node_name] = float("inf")
                continue

            if isinstance(node, AttentionNode):
                W = W_O_cpu[node.layer, node.head]  # [d_head, d_model]
            elif isinstance(node, MLPNode):
                W = W_out_cpu[node.layer]  # [d_mlp, d_model]
            else:
                node_scores[node_name] = float("inf")
                continue

            node_scores[node_name] = float((W.norm() / (W.numel() ** 0.5)).item())

        del W_O_cpu, W_out_cpu
        return node_scores

    @staticmethod
    def _sync_edges_to_nodes(graph) -> None:
        """
        After setting node.in_graph flags, rebuild the edge matrix from them.

        CONTRACT: callers MUST have already set ``graph.nodes_in_graph`` to the
        final baseline node mask before calling — this rebuilds the edge matrix
        purely from the current node mask, so any stale/unreset node flag yields
        a stale edge. All existing builders (`_build_random_circuit`,
        `_build_magnitude_circuit`, `_build_wanda_circuit`) reset the mask first.

        Node.in_graph writes to graph.nodes_in_graph (1D tensor).
        Edge.in_graph reads/writes graph.in_graph (2D edge matrix).
        These are independent — the baseline builders only touch nodes_in_graph,
        leaving the edge matrix stale. This syncs them before evaluate_graph
        calls graph.prune(), which reads the edge matrix to determine connectivity.

        The edge set is REBUILT as the node-induced subgraph (all structurally
        valid edges between selected nodes), not intersected with the stale
        matrix. The cloned baseline graph carries the *discovered circuit's*
        edges, so the previous in-place multiply (`graph.in_graph *= edge_mask`)
        could only ever REMOVE edges — every random/magnitude/wanda baseline
        collapsed to a subset of the circuit's own edges instead of a true
        same-sparsity baseline, systematically understating baseline scores and
        overstating circuit_advantage / the z-score.
        """
        from einops import einsum

        forward_in_graph = graph.nodes_in_graph.float()
        backward_in_graph = graph.nodes_in_graph.float() @ graph.forward_to_backward.float()
        backward_in_graph[-1] = 1  # logits node is always a valid destination
        edge_mask = (
            einsum(forward_in_graph, backward_in_graph, "forward, backward -> forward backward") > 0
        )
        graph.in_graph.copy_(edge_mask & graph.real_edge_mask)

    @staticmethod
    def run(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        baseline_types: List[str] = None,
        n_random_draws: int = 5,
        device: str = "auto",
        quiet: bool = False,
    ) -> Dict:
        """
        Run baseline comparison evaluation on a circuit.

        Compares the discovered circuit against multiple baseline methods,
        all evaluated with the same evaluation procedure.

        Args:
            model: HookedTransformer model with use_attn_result=True.
            graph: Circuit graph with in_graph flags set on edges/nodes.
            dataloader: Evaluation dataset yielding (clean, corrupted, label) batches.
            metric_fn: Metric function with signature
                (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
            baseline_types: List of baseline methods to compute. Options:
                - 'random': Random baseline (same sparsity)
                - 'magnitude': Magnitude-based baseline
                - 'wanda': WANDA baseline (weight*activation)
                Default: ['random', 'magnitude']
            device: Target device ("cuda" or "cpu"). Defaults to "cuda".
            quiet: Suppress progress bar. Defaults to False.

        Returns:
            Dict with keys:
            - 'circuit_score': Score of the discovered circuit
            - 'baselines': Dict with results for each baseline:
                - '{baseline}_score': Baseline performance
                - '{baseline}_percentage': Circuit rank vs baseline
                - '{baseline}_improvement': Ratio (circuit / baseline), values > 1.0 indicate the circuit outperforms the baseline. None when the baseline score is <= 0 (ratio undefined / not comparable).
                - '{baseline}_improvement_valid': bool, True when baseline score > 0 so 'improvement' is a meaningful ratio.
            - 'best_baseline_score': Score of the best-performing baseline
            - 'circuit_advantage': How much circuit outperforms best baseline
            - 'summary': Human-readable summary of comparison

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

        if baseline_types is None:
            baseline_types = ["random", "magnitude"]

        logger.info(f"Pillar 5: Running baselines comparison " f"({', '.join(baseline_types)})...")

        # Extract circuit sparsity for baseline matching
        if Pillar5_Baselines._is_neuron_level(graph):
            circuit_sparsity = Pillar5_Baselines._get_neuron_level_sparsity(graph)
        else:
            circuit_nodes = Pillar5_Baselines._extract_circuit_nodes(graph)
            total_nodes = len(graph.nodes)
            circuit_sparsity = len(circuit_nodes) / total_nodes if total_nodes > 0 else 0.0

        logger.info(f"  Circuit sparsity: {circuit_sparsity:.2%}")

        # Evaluate the discovered circuit
        logger.info("Evaluating discovered circuit...")
        circuit_score = Pillar5_Baselines._evaluate_circuit(
            model, graph, dataloader, metric_fn, quiet=quiet
        )
        logger.info(f"  Circuit score: {circuit_score:.4f}")

        # Evaluate baselines
        baselines_results = {}

        for baseline_type in baseline_types:
            logger.info(f"Evaluating {baseline_type} baseline...")
            random_scores = None

            if baseline_type == "random":
                # P5: draw the random baseline n_random_draws times with distinct
                # seeds so "better than chance" is estimated from a distribution,
                # not a single sample (a single draw has no notion of chance spread).
                random_scores = [
                    Pillar5_Baselines._evaluate_random_baseline(
                        model,
                        graph,
                        dataloader,
                        metric_fn,
                        circuit_sparsity=circuit_sparsity,
                        device=device,
                        quiet=quiet,
                        seed=42 + i,
                    )
                    for i in range(max(1, n_random_draws))
                ]
                baseline_score = float(np.mean(random_scores))

            elif baseline_type == "magnitude":
                baseline_score = Pillar5_Baselines._evaluate_magnitude_baseline(
                    model,
                    graph,
                    dataloader,
                    metric_fn,
                    circuit_sparsity=circuit_sparsity,
                    device=device,
                    quiet=quiet,
                )

            elif baseline_type == "wanda":
                baseline_score = Pillar5_Baselines._evaluate_wanda_baseline(
                    model,
                    graph,
                    dataloader,
                    metric_fn,
                    circuit_sparsity=circuit_sparsity,
                    device=device,
                    quiet=quiet,
                )

            else:
                logger.warning(f"Unknown baseline type: {baseline_type}, skipping")
                continue

            logger.info(f"  {baseline_type.capitalize()} baseline score: {baseline_score:.4f}")

            # improvement = circuit/baseline is only a meaningful ratio when the
            # baseline metric is positive. For baseline_score <= 0 the ratio of
            # signed metrics is meaningless, so report improvement=None with
            # improvement_valid=False rather than fabricating a "tied" 1.0.
            improvement_valid = baseline_score > 0
            baselines_results[baseline_type] = {
                "score": baseline_score,
                "percentage": Pillar5_Baselines._compute_percentage_of_baseline(
                    circuit_score, baseline_score
                ),
                # percentage is None exactly when improvement is (baseline <= 0):
                # both quotients share the same validity condition.
                "percentage_valid": improvement_valid,
                "improvement": (circuit_score / baseline_score if improvement_valid else None),
                "improvement_valid": improvement_valid,
            }

            if random_scores is not None:
                # Report the random-baseline distribution: how far above chance the
                # circuit sits, in standard deviations (z-score), plus the raw draws.
                random_std = float(np.std(random_scores)) if len(random_scores) > 1 else 0.0
                baselines_results[baseline_type].update(
                    {
                        "score_std": random_std,
                        "n_draws": len(random_scores),
                        "random_scores": [float(s) for s in random_scores],
                        "z_score": (
                            float((circuit_score - baseline_score) / random_std)
                            if random_std > 0
                            else None
                        ),
                    }
                )

        # Compute summary statistics
        best_baseline_score = max([v["score"] for v in baselines_results.values()], default=0.0)
        circuit_advantage = circuit_score - best_baseline_score

        # Generate summary
        summary = Pillar5_Baselines._generate_summary(
            circuit_score, baselines_results, baseline_types
        )

        logger.info("\nBaseline Comparison Summary:")
        logger.info(f"  Circuit:          {circuit_score:.4f}")
        for baseline_type, result in baselines_results.items():
            improvement_str = (
                f"{result['improvement']:.2f}x"
                if result["improvement"] is not None
                else "n/a (baseline <= 0)"
            )
            percentage_str = (
                f"{result['percentage']:.1f}"
                if result["percentage"] is not None
                else "n/a (baseline <= 0)"
            )
            logger.info(
                f"  {baseline_type.capitalize()}: {result['score']:.4f} "
                f"(improvement: {improvement_str}, "
                f"percentage: {percentage_str})"
            )
        logger.info(f"  Advantage:        {circuit_advantage:.4f}")
        logger.info(f"\nSummary: {summary}")

        return {
            "circuit_score": circuit_score,
            "baselines": baselines_results,
            "best_baseline_score": best_baseline_score,
            "circuit_advantage": circuit_advantage,
            "summary": summary,
            "circuit_sparsity": circuit_sparsity,
            "baseline_types": baseline_types,
        }

    @staticmethod
    def _evaluate_circuit(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        quiet: bool = False,
    ) -> float:
        """
        Evaluate the discovered circuit.

        Args:
            model: HookedTransformer model
            graph: Circuit graph
            dataloader: Evaluation dataset
            metric_fn: Metric function
            quiet: Suppress progress bar

        Returns:
            float: Circuit performance score
        """
        scores = evaluate_graph(
            model=model,
            graph=graph,
            dataloader=dataloader,
            metrics=metric_fn,
            quiet=quiet,
            intervention="patching",
            skip_clean=True,
        )

        # Convert to scalar
        if isinstance(scores, list):
            scores = scores[0]

        if hasattr(scores, "cpu"):
            scores = scores.cpu()

        if scores.ndim == 0:
            return float(scores.item())
        else:
            return float(scores.mean().item())

    @staticmethod
    def _evaluate_random_baseline(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        circuit_sparsity: float,
        device: str = "auto",
        quiet: bool = False,
        seed: int = 42,
    ) -> float:
        """
        Evaluate random baseline: randomly select nodes with same sparsity.

        Args:
            model: HookedTransformer model
            graph: Circuit graph
            dataloader: Evaluation dataset
            metric_fn: Metric function
            circuit_sparsity: Fraction of nodes to keep
            device: Target device
            quiet: Suppress progress bar

        Returns:
            float: Baseline score
        """
        # Create a copy of the graph
        baseline_graph = Pillar5_Baselines._build_random_circuit(
            graph, sparsity=circuit_sparsity, seed=seed
        )

        return Pillar5_Baselines._evaluate_circuit(
            model, baseline_graph, dataloader, metric_fn, quiet=quiet
        )

    @staticmethod
    def _evaluate_magnitude_baseline(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        circuit_sparsity: float,
        device: str = "auto",
        quiet: bool = False,
    ) -> float:
        """
        Evaluate magnitude baseline: select top-k nodes by |score|.

        Args:
            model: HookedTransformer model
            graph: Circuit graph
            dataloader: Evaluation dataset
            metric_fn: Metric function
            circuit_sparsity: Fraction of nodes to keep
            device: Target device
            quiet: Suppress progress bar

        Returns:
            float: Baseline score
        """
        baseline_graph = Pillar5_Baselines._build_magnitude_circuit(
            model, graph, sparsity=circuit_sparsity
        )

        return Pillar5_Baselines._evaluate_circuit(
            model, baseline_graph, dataloader, metric_fn, quiet=quiet
        )

    @staticmethod
    def _evaluate_wanda_baseline(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        circuit_sparsity: float,
        device: str = "auto",
        quiet: bool = False,
    ) -> float:
        """
        Evaluate WANDA baseline: select top-k by weight*activation product.

        WANDA (https://arxiv.org/abs/2306.11695) is a structured pruning
        heuristic for LLMs based on weight-activation importance.

        Args:
            model: HookedTransformer model
            graph: Circuit graph
            dataloader: Evaluation dataset
            metric_fn: Metric function
            circuit_sparsity: Fraction of nodes to keep
            device: Target device
            quiet: Suppress progress bar

        Returns:
            float: Baseline score
        """
        baseline_graph = Pillar5_Baselines._build_wanda_circuit(
            model, graph, dataloader, sparsity=circuit_sparsity
        )

        return Pillar5_Baselines._evaluate_circuit(
            model, baseline_graph, dataloader, metric_fn, quiet=quiet
        )

    @staticmethod
    def _build_random_circuit(graph: Graph, sparsity: float, seed: int = 42) -> Graph:
        """
        Build a random circuit by selecting nodes uniformly at random.

        Args:
            graph: Original circuit graph
            sparsity: Fraction of nodes to keep (0 < sparsity <= 1)
            seed: Random seed for reproducibility

        Returns:
            Graph: New graph with random circuit nodes marked in_graph
        """

        baseline_graph = Pillar5_Baselines._clone_graph(graph)

        if Pillar5_Baselines._is_neuron_level(baseline_graph):
            # Neuron-level: use graph's own apply_random at neuron level
            # sparsity here = fraction ACTIVE, so n_keep = sparsity * total_scored
            scored_neurons = ~torch.isnan(baseline_graph.neurons_scores)
            n_scored = int(scored_neurons.sum().item())
            n_keep = max(1, int(n_scored * sparsity))
            baseline_graph.reset(empty=True)
            baseline_graph.apply_random(n_keep, level="neuron", prune=True, seed=seed)

        else:
            rng = np.random.RandomState(seed)

            # Reset all nodes to not in_graph — SKIP logits (no forward index slot)
            for node in baseline_graph.nodes.values():
                if node.name != "logits":
                    node.in_graph = False

            # Get all node names — SKIP logits
            all_nodes = [n for n in baseline_graph.nodes.keys() if n != "logits"]
            n_keep = max(1, int(len(all_nodes) * sparsity))

            # Randomly select nodes
            selected_nodes = rng.choice(all_nodes, size=n_keep, replace=False)

            # Mark selected nodes in_graph
            for node_name in selected_nodes:
                baseline_graph.nodes[node_name].in_graph = True

            Pillar5_Baselines._sync_edges_to_nodes(baseline_graph)

        return baseline_graph

    @staticmethod
    def _build_magnitude_circuit(model: HookedTransformer, graph: Graph, sparsity: float) -> Graph:
        """
        Build a magnitude baseline by selecting top-k nodes by |score|.

        Args:
            graph: Original circuit graph
            sparsity: Fraction of nodes to keep (0 < sparsity <= 1)

        Returns:
            Graph: New graph with magnitude-selected nodes marked in_graph
        """

        baseline_graph = Pillar5_Baselines._clone_graph(graph)

        if Pillar5_Baselines._is_neuron_level(baseline_graph):
            weight_scores = Pillar5_Baselines._compute_weight_norm_scores(model, baseline_graph)
            scored_mask = ~torch.isnan(weight_scores)
            n_scored = int(scored_mask.sum().item())
            n_keep = max(1, int(n_scored * sparsity))

            scores_copy = weight_scores.clone()
            scores_copy[~scored_mask] = -float("inf")
            flat_sorted = torch.argsort(scores_copy.view(-1), descending=True)

            baseline_graph.reset(empty=True)
            baseline_graph.neurons_in_graph.view(-1)[flat_sorted[:n_keep]] = True
            # Keep padding/unscored slots (NaN in weight_scores = no weight norm = always keep)
            baseline_graph.neurons_in_graph.view(-1)[~scored_mask.view(-1)] = True
            baseline_graph.nodes_in_graph += baseline_graph.neurons_in_graph.any(dim=1)
            baseline_graph.in_graph += baseline_graph.nodes_in_graph.view(-1, 1)
            baseline_graph.prune()

        else:
            # Reset all nodes to not in_graph — SKIP logits
            for node in baseline_graph.nodes.values():
                if node.name != "logits":
                    node.in_graph = False

            # Use RMS weight norms as an independent magnitude criterion —
            # independent of the attribution scores used to build the circuit.
            weight_norm_scores = Pillar5_Baselines._compute_node_weight_norm_scores(
                model, baseline_graph
            )

            node_scores = []
            for node_name in baseline_graph.nodes:
                if node_name == "logits":
                    continue
                node_scores.append((node_name, weight_norm_scores.get(node_name, float("inf"))))

            node_scores.sort(key=lambda x: x[1], reverse=True)
            n_keep = max(1, int(len(node_scores) * sparsity))

            for i in range(n_keep):
                baseline_graph.nodes[node_scores[i][0]].in_graph = True

            Pillar5_Baselines._sync_edges_to_nodes(baseline_graph)

        return baseline_graph

    @staticmethod
    def _build_wanda_circuit(
        model: HookedTransformer, graph: Graph, dataloader: DataLoader, sparsity: float
    ) -> Graph:
        """
        Build a WANDA baseline by weight*activation importance.

        WANDA computes importance as the product of weight magnitude and
        activation magnitude across the dataset.

        Args:
            model: HookedTransformer model
            graph: Original circuit graph
            dataloader: Evaluation dataset
            sparsity: Fraction of nodes to keep (0 < sparsity <= 1)

        Returns:
            Graph: New graph with WANDA-selected nodes marked in_graph
        """

        baseline_graph = Pillar5_Baselines._clone_graph(graph)
        # Reset all nodes to not in_graph — SKIP terminal/input nodes whose
        # in_graph flag is fixed (logits is always True; input has no slot).
        for node_name, node in baseline_graph.nodes.items():
            if node_name not in ("logits", "input"):
                node.in_graph = False

        # Accumulate mean activation L2-norm per node across the dataloader.
        device = next(model.parameters()).device
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads

        # hook_result: [batch, pos, n_heads, d_model] per layer
        # hook_mlp_out: [batch, pos, d_model] per layer
        act_accum: Dict[str, float] = {}
        count = 0
        for clean, corrupted, _ in dataloader:
            clean_tokens = (
                model.to_tokens(clean, prepend_bos=True)
                if isinstance(clean[0], str)
                else clean.to(device)
            )
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    clean_tokens,
                    names_filter=lambda n: (
                        n.endswith("hook_result") or n.endswith("hook_mlp_out")
                    ),
                )
            for layer in range(n_layers):
                key_result = f"blocks.{layer}.attn.hook_result"
                if key_result in cache:
                    result = cache[key_result]  # [batch, pos, n_heads, d_model]
                    for h in range(n_heads):
                        node_key = f"a{layer}.h{h}"
                        mag = float(result[:, :, h, :].norm(dim=-1).mean().item())
                        act_accum[node_key] = act_accum.get(node_key, 0.0) + mag
                key_mlp = f"blocks.{layer}.hook_mlp_out"
                if key_mlp in cache:
                    mlp_out = cache[key_mlp]  # [batch, pos, d_model]
                    node_key = f"m{layer}"
                    mag = float(mlp_out.norm(dim=-1).mean().item())
                    act_accum[node_key] = act_accum.get(node_key, 0.0) + mag
            count += 1
        if count > 0:
            act_accum = {k: v / count for k, v in act_accum.items()}

        # WANDA score = |node_score| * activation_magnitude.
        wanda_scores = []
        for node_name, node in baseline_graph.nodes.items():
            if node_name in ("logits", "input"):
                continue
            weight_mag = abs(float(node.score)) if getattr(node, "score", None) is not None else 0.0
            activation_mag = act_accum.get(node_name, 1.0)
            wanda_scores.append((node_name, weight_mag * activation_mag))

        wanda_scores.sort(key=lambda x: x[1], reverse=True)
        n_keep = max(1, int(len(wanda_scores) * sparsity))
        for node_name, _ in wanda_scores[:n_keep]:
            baseline_graph.nodes[node_name].in_graph = True

        Pillar5_Baselines._sync_edges_to_nodes(baseline_graph)
        return baseline_graph

    @staticmethod
    def _extract_circuit_nodes(graph: Graph) -> Dict:
        """
        Extract all nodes marked as in_graph from the circuit.

        Args:
            graph: Circuit graph

        Returns:
            Dict of node name -> node metadata
        """
        summary_scores = Pillar5_Baselines._get_node_summary_scores(graph)
        circuit = {}
        for node_name, node in graph.nodes.items():
            # ADD THIS GUARD: Skip the terminal node
            if node_name == "logits":
                continue

            if node.in_graph:
                circuit[node_name] = {
                    "layer": node.layer,
                    "type": type(node).__name__,
                    "score": summary_scores.get(node_name, 0.0),
                }
        return circuit

    @staticmethod
    def _clone_graph(graph: Graph) -> Graph:
        """
        Safely clone a graph by recreating its static topology and copying its state tensors.
        Avoids copy.deepcopy() which fails on cyclic sets and is computationally expensive.
        """
        # 1. Recreate the static structural topology
        new_graph = Graph.from_model(
            graph.cfg,
            neuron_level=(graph.neurons_in_graph is not None),
            node_scores=(graph.nodes_scores is not None),
            mlp_hook=graph.cfg.get("mlp_hook", "mlp_out"),
        )

        # 2. Deepcopy the dynamic state tensors
        new_graph.in_graph.copy_(graph.in_graph)
        new_graph.scores.copy_(graph.scores)
        new_graph.nodes_in_graph.copy_(graph.nodes_in_graph)

        if graph.nodes_scores is not None:
            new_graph.nodes_scores.copy_(graph.nodes_scores)

        if graph.neurons_in_graph is not None:
            new_graph.neurons_in_graph.copy_(graph.neurons_in_graph)
        if graph.neurons_scores is not None:
            new_graph.neurons_scores.copy_(graph.neurons_scores)

        return new_graph

    @staticmethod
    def _compute_percentage_of_baseline(circuit_score: float, baseline_score: float):
        """
        Compute the circuit performance as a percentage of baseline performance.

        Returns ``None`` when the percentage is undefined: for a signed metric
        a non-positive (or negligible) baseline flips or explodes the quotient
        — e.g. circuit=0.5 vs baseline=-0.5 gave -100.0 (direction inverted),
        circuit=-0.5 vs baseline=-1.0 gave 50.0 ("half of baseline") when the
        circuit is actually BETTER (less negative), and the old exact `== 0`
        float check let baseline=1e-12 produce a ~1e14 percentage. Mirrors the
        sibling ``improvement``/``improvement_valid`` convention.
        """
        if baseline_score <= 1e-6:
            return None

        return (circuit_score / baseline_score) * 100.0

    @staticmethod
    def _generate_summary(
        circuit_score: float, baselines_results: Dict, baseline_types: List[str]
    ) -> str:
        """
        Generate human-readable summary of baseline comparison.

        Args:
            circuit_score: Circuit performance
            baselines_results: Results dict for each baseline
            baseline_types: List of baseline types evaluated

        Returns:
            str: Summary statement
        """
        if not baseline_types:
            return "No baselines evaluated"

        if "random" in baselines_results:
            random_improvement = baselines_results["random"]["improvement"]

            # improvement is None when the random baseline metric was <= 0, so
            # the ratio is not comparable — do not count it as a pass/fail.
            if random_improvement is None:
                return (
                    "Circuit vs random baseline not comparable "
                    "(random baseline score <= 0, so the improvement ratio is undefined)"
                )

            if random_improvement > 1.5:
                return (
                    f"Circuit substantially outperforms random baseline "
                    f"({random_improvement:.2f}x improvement)"
                )
            elif random_improvement > 1.1:
                return (
                    f"Circuit meaningfully outperforms random baseline "
                    f"({random_improvement:.2f}x improvement)"
                )
            elif random_improvement >= 1.0:
                return (
                    f"Circuit only marginally outperforms random baseline "
                    f"({random_improvement:.2f}x improvement)"
                )
            else:
                # improvement < 1.0 means the circuit scored BELOW the random
                # baseline (including negative circuit scores against a
                # positive baseline). The old final `else` described every
                # such case as "marginally outperforms" — e.g. a -0.40x or
                # 0.60x improvement was reported as outperforming chance.
                return (
                    f"Circuit UNDERPERFORMS the random baseline "
                    f"({random_improvement:.2f}x improvement) — the discovered "
                    f"circuit scores below chance on this task/metric"
                )

        return "Baseline comparison completed"
