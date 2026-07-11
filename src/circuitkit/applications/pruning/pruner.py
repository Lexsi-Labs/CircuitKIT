"""
Structured *masking* of attention heads and MLP layers.

This module does NOT physically remove parameters or change tensor shapes.
It performs structured masking: whole attention heads and whole MLP output
projections are zeroed *in place*. The pruned head/MLP contributes nothing to
the residual stream, but every weight tensor keeps its original shape, so the
``HookedTransformer`` config (``cfg.n_heads``, ``cfg.d_mlp``, ...) stays valid
and the model still runs a forward pass.

Consequently this step does NOT reduce FLOPs, memory usage or parameter count
on its own — it only drives a structured subset of weights to exactly zero.
Genuine *physical* parameter removal (smaller weight matrices, smaller
checkpoints on disk) happens later, at checkpoint export, in
:mod:`circuitkit.evaluation.hf_checkpoint` (``save_pruned_checkpoint``), which
applies the same head/MLP selection to a real HuggingFace model.

This module handles:
- Masking attention heads by zeroing their Q/K/V/O weights and biases
- Masking MLP layers by zeroing their output projection
- Measuring the *effective sparsity* (fraction of weights driven to zero)
"""

import copy
import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch as t
from transformer_lens import HookedTransformer

from circuitkit.artifacts.scores import CircuitScores
from circuitkit.utils.logging import get_logger


class StructuralPruner:
    """
    Structured masking of attention heads and MLP layers.

    This pruner does NOT physically remove parameters or resize any tensor.
    It masks whole heads / MLP layers by zeroing their weights in place:

    1. Identifying pruning targets from CircuitScores
    2. Zeroing the Q/K/V/O weights and biases of pruned attention heads
    3. Zeroing the output projection (``W_out`` / ``c_proj`` / ``down_proj``)
       of pruned MLP layers

    Tensor shapes — and therefore the ``HookedTransformer`` config — are left
    unchanged, so the masked model still runs a forward pass. Because nothing
    is resized, this step alone does not reduce FLOPs, memory or parameter
    count; it only produces a structured zero pattern.

    Genuine physical parameter removal and smaller on-disk checkpoints are
    produced separately at export time by
    :func:`circuitkit.evaluation.hf_checkpoint.save_pruned_checkpoint`.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the structural pruner.

        Args:
            logger: Logger instance (default: circuitkit.applications.pruning.pruner)
        """
        self.logger = logger or get_logger("circuitkit.applications.pruning.pruner")

    def prune(
        self,
        model: HookedTransformer,
        circuit_scores: CircuitScores,
        sparsity: float = 0.3,
        inplace: Optional[bool] = None,
        dry_run: Optional[bool] = None,
        scope: str = "both",
        protect_layers: Optional[List[int]] = None,
    ) -> HookedTransformer:
        """
        Mask heads/MLPs in a model based on circuit scores.

        Selects the lowest-scoring node-level components (attention heads and
        MLP layers) up to the target ``sparsity`` and zeroes their weights.
        This is structured *masking*, not physical removal: tensor shapes are
        unchanged and the returned model still runs (see the class docstring).

        Args:
            model (HookedTransformer): Model to mask.
            circuit_scores (CircuitScores): Node scores from discovery.
            sparsity (float): Target fraction of nodes to mask (0.0 to 1.0).
                Default 0.3 (mask 30% of nodes).
            scope (str): Which component type may be masked — ``"heads"``
                (only attention-head nodes), ``"mlp"`` (only MLP nodes) or
                ``"both"`` (default). The candidate node set is filtered to the
                chosen component type *before* the bottom-k sparsity selection,
                so the sparsity budget is taken within that component type.
            protect_layers (Optional[List[int]]): Layer indices that must never
                be masked. Any attention head or MLP belonging to one of these
                layers is excluded from the candidate set. ``None`` (default)
                protects nothing.
            inplace (bool): Controls whether ``model`` itself is mutated.

                * ``inplace=True``  — mask ``model`` in place and return it
                  (the passed-in model object is modified).
                * ``inplace=False`` (default) — deep-copy ``model`` first,
                  mask the copy, and return the copy; the original ``model``
                  is left untouched.

            dry_run (bool): Deprecated alias for ``not inplace``, kept for
                back-compatibility. ``dry_run=True`` is equivalent to
                ``inplace=False`` (copy mode); ``dry_run=False`` is equivalent
                to ``inplace=True`` (in-place mode). Note the historical name
                is misleading — ``dry_run=True`` still performs masking, just
                on a copy. If both ``inplace`` and ``dry_run`` are given,
                ``inplace`` wins. Passing ``dry_run`` emits a
                ``DeprecationWarning``.

        Returns:
            HookedTransformer: The masked model (a copy unless ``inplace=True``).

        Raises:
            ValueError: If sparsity, scope or scores level is invalid.
            RuntimeError: If masking fails on a specific layer.
        """
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0.0, 1.0], got {sparsity}")

        # Validate scope with the same vocabulary as quick.build_discovery_config.
        if scope not in ("heads", "mlp", "both"):
            raise ValueError(f"scope must be 'heads', 'mlp' or 'both', got {scope!r}")

        if circuit_scores.level != "node":
            raise ValueError(
                f"StructuralPruner only supports node-level scores; "
                f"got level={circuit_scores.level!r}"
            )

        # Resolve inplace vs. the deprecated dry_run alias.
        # - inplace wins if both are passed.
        # - dry_run=True  <=> inplace=False (copy mode).
        # - dry_run=False <=> inplace=True  (in-place mode).
        if dry_run is not None:
            warnings.warn(
                "StructuralPruner.prune(dry_run=...) is deprecated and "
                "misleadingly named (dry_run=True still performs masking, "
                "just on a copy). Use inplace=... instead: "
                "inplace=False (default) for copy mode, inplace=True for "
                "in-place masking.",
                DeprecationWarning,
                stacklevel=2,
            )
        if inplace is None:
            inplace = (not dry_run) if dry_run is not None else False

        # Work on a copy unless the caller explicitly asked for in-place.
        model_to_prune = model if inplace else copy.deepcopy(model)

        self.logger.info(
            f"Starting structural masking: {circuit_scores.algorithm} "
            f"scores on {circuit_scores.model} (inplace={inplace})"
        )

        # Snapshot the nonzero-parameter count BEFORE any mutation. This is the
        # only reliable baseline: when inplace=True, model_to_prune IS model,
        # so we cannot compare against it after masking — it would already be
        # mutated and effective sparsity would read as 0.
        nonzero_before, total_params = self._param_counts(model_to_prune)

        # Parse pruning targets from scores
        nodes_to_prune = self._select_nodes_to_prune(
            circuit_scores, sparsity, scope=scope, protect_layers=protect_layers
        )

        if not nodes_to_prune:
            self.logger.warning(
                f"No nodes selected for masking at sparsity={sparsity}. "
                "Returning unchanged model."
            )
            return model_to_prune

        self.logger.info(
            f"Masking {len(nodes_to_prune)} nodes: " f"{self._format_nodes(nodes_to_prune)}"
        )

        # Apply structural masking
        self._prune_attention_heads(model_to_prune, nodes_to_prune)
        self._prune_mlp_neurons(model_to_prune, nodes_to_prune)

        sparsity_achieved = self.measure_sparsity(
            model_to_prune, nonzero_before=nonzero_before, total_params=total_params
        )
        self.logger.info(f"Masking complete. Effective sparsity: {sparsity_achieved:.2%}")

        return model_to_prune

    def _select_nodes_to_prune(
        self,
        circuit_scores: CircuitScores,
        sparsity: float,
        scope: str = "both",
        protect_layers: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Select bottom-k nodes to prune based on sparsity target.

        The candidate node set is filtered by ``scope`` (component type) and
        ``protect_layers`` *before* the bottom-k selection, so the sparsity
        budget is taken within the chosen component type and never touches a
        protected layer.

        Args:
            circuit_scores: Node scores artifact.
            sparsity: Target sparsity fraction.
            scope: ``"heads"``, ``"mlp"`` or ``"both"`` — which component type
                is eligible for masking.
            protect_layers: Layer indices excluded from the candidate set
                (both their heads and MLP). ``None`` protects nothing.

        Returns:
            Dict[str, float]: Nodes to prune with their scores.
        """
        protected = set(protect_layers or [])

        # Filter the candidate set by component type and protected layers.
        # Node-name conventions match node_pruner.get_nodes_to_prune:
        # attention heads 'A{layer}.{head}', MLP layers 'MLP {layer}'.
        candidates: Dict[str, float] = {}
        for name, score in circuit_scores.node_scores.items():
            attn_match = re.match(r"A(\d+)\.(\d+)", name)
            mlp_match = re.match(r"MLP (\d+)", name)
            if attn_match:
                if scope not in ("heads", "both"):
                    continue
                if int(attn_match.group(1)) in protected:
                    continue
            elif mlp_match:
                if scope not in ("mlp", "both"):
                    continue
                if int(mlp_match.group(1)) in protected:
                    continue
            else:
                # Non head/MLP nodes (e.g. 'Resid Start') are never masked.
                continue
            candidates[name] = score

        num_to_prune = int(len(candidates) * sparsity)
        if num_to_prune == 0:
            return {}

        # Get lowest-scoring candidates (bottom-k within the filtered set).
        sorted_nodes = sorted(candidates.items(), key=lambda kv: kv[1])
        return dict(sorted_nodes[:num_to_prune])

    def _prune_attention_heads(
        self, model: HookedTransformer, nodes_to_prune: Dict[str, float]
    ) -> None:
        """
        Remove attention heads from the model.

        Modifies weight matrices in-place:
        - W_Q, W_K, W_V: Remove rows corresponding to pruned heads
        - W_O: Remove columns corresponding to pruned heads
        - Biases: Remove entries for pruned heads

        Args:
            model: Model to modify in-place.
            nodes_to_prune: Dict of node names to scores.

        Raises:
            RuntimeError: If pruning fails on specific layer/head.
        """
        heads_by_layer: Dict[int, List[int]] = {}

        # Parse attention node names: 'A{layer}.{head}'
        for node_name in nodes_to_prune.keys():
            if node_name.startswith("A"):
                parts = node_name[1:].split(".")
                if len(parts) == 2:
                    try:
                        layer = int(parts[0])
                        head = int(parts[1])
                        if layer not in heads_by_layer:
                            heads_by_layer[layer] = []
                        heads_by_layer[layer].append(head)
                    except (ValueError, IndexError):
                        self.logger.warning(f"Could not parse attention node name: {node_name}")

        # Sort heads in descending order to remove from end first
        # (prevents index shifting)
        for layer in heads_by_layer:
            heads_by_layer[layer].sort(reverse=True)

        for layer_idx, heads_to_remove in heads_by_layer.items():
            try:
                self._remove_attention_heads_from_layer(model, layer_idx, heads_to_remove)
            except Exception as e:
                self.logger.error(
                    f"Failed to prune heads {heads_to_remove} from layer {layer_idx}: {e}"
                )
                raise RuntimeError(f"Structural pruning failed at layer {layer_idx}") from e

    def _remove_attention_heads_from_layer(
        self, model: HookedTransformer, layer_idx: int, heads_to_remove: List[int]
    ) -> None:
        """
        Remove specific heads from a single attention layer.

        ``HookedTransformer`` carries a single global ``cfg.n_heads`` shared by
        every block, so physically deleting head rows from one layer's weight
        tensors would desynchronise that layer from ``cfg`` and break the
        forward pass (the residual reshape ``[batch, seq, n_heads*d_head]``
        would no longer match). Circuit-derived pruning removes a *different*
        set of heads per layer, which a single global ``n_heads`` cannot
        represent.

        We therefore prune heads structurally-equivalently by zeroing the
        pruned heads' Q/K/V/O weights and biases in place: the head produces
        no output and contributes nothing to the residual stream, while tensor
        shapes (and hence ``cfg``) stay valid and the model still runs. This
        mirrors the MLP path in :meth:`_prune_mlp_neurons`, which zeroes
        ``W_out`` rather than resizing.

        Args:
            model: Model to modify in-place.
            layer_idx: Layer index (0-based).
            heads_to_remove: List of head indices to remove.
        """
        if not (0 <= layer_idx < len(model.blocks)):
            raise ValueError(f"layer_idx {layer_idx} out of bounds [0, {len(model.blocks)})")

        attn = model.blocks[layer_idx].attn
        n_heads = model.cfg.n_heads

        for head_idx in heads_to_remove:
            if not (0 <= head_idx < n_heads):
                raise ValueError(f"head_idx {head_idx} out of bounds [0, {n_heads})")

        # W_Q / W_O and b_Q are per-query-head Parameters of shape [n_heads, ...].
        # On Grouped-Query-Attention models (e.g. Llama-3), W_K/W_V/b_K/b_V are
        # NOT Parameters: they are read-only properties that *repeat* the
        # underlying _W_K/_W_V (one row per KV head, not per query head).
        # Writing through `attn.W_K.data[...]` mutates a transient repeated
        # tensor and is silently discarded. We therefore zero the backing
        # _W_K/_W_V Parameters directly, and only when every query head sharing
        # a KV group has been pruned (so we don't disable a KV head still in
        # use by a surviving query head).
        n_kv = getattr(model.cfg, "n_key_value_heads", None) or n_heads
        is_gqa = n_kv != n_heads
        group_size = n_heads // n_kv if is_gqa else 1

        for head_idx in heads_to_remove:
            # Per-query-head weights: always real Parameters.
            attn.W_Q.data[head_idx].zero_()
            attn.W_O.data[head_idx].zero_()
            if attn.b_Q is not None:
                attn.b_Q.data[head_idx].zero_()

            if not is_gqa:
                # MHA: W_K/W_V/b_K/b_V are per-head Parameters — zero directly.
                attn.W_K.data[head_idx].zero_()
                attn.W_V.data[head_idx].zero_()
                if attn.b_K is not None:
                    attn.b_K.data[head_idx].zero_()
                if attn.b_V is not None:
                    attn.b_V.data[head_idx].zero_()

        if is_gqa:
            # Zero a KV head only when ALL query heads in its group are pruned.
            removed = set(heads_to_remove)
            for kv_idx in range(n_kv):
                group = set(range(kv_idx * group_size, (kv_idx + 1) * group_size))
                if group.issubset(removed):
                    attn._W_K.data[kv_idx].zero_()
                    attn._W_V.data[kv_idx].zero_()
                    if getattr(attn, "_b_K", None) is not None:
                        attn._b_K.data[kv_idx].zero_()
                    if getattr(attn, "_b_V", None) is not None:
                        attn._b_V.data[kv_idx].zero_()

    def _prune_mlp_neurons(
        self, model: HookedTransformer, nodes_to_prune: Dict[str, float]
    ) -> None:
        """
        Remove MLP neurons from the model.

        Modifies weight matrices in-place:
        - W_in: Remove columns (each neuron is a column)
        - W_out: Remove rows (each neuron's output)
        - Biases: Remove entries for pruned neurons

        Args:
            model: Model to modify in-place.
            nodes_to_prune: Dict of node names to scores.

        Raises:
            RuntimeError: If pruning fails on specific layer.
        """
        mlp_layers_to_prune: List[int] = []

        # Parse MLP node names: 'MLP {layer}'
        for node_name in nodes_to_prune.keys():
            if node_name.startswith("MLP "):
                parts = node_name.split()
                if len(parts) == 2:
                    try:
                        mlp_layers_to_prune.append(int(parts[1]))
                    except ValueError:
                        self.logger.warning(f"Could not parse MLP node name: {node_name}")

        for layer_idx in sorted(set(mlp_layers_to_prune)):
            if not (0 <= layer_idx < len(model.blocks)):
                self.logger.warning(f"MLP layer {layer_idx} out of bounds; skipping")
                continue
            try:
                mlp = model.blocks[layer_idx].mlp
                # Zero out the output projection so the entire MLP layer is a no-op.
                # This is the node-level structural equivalent of removing the MLP:
                # it preserves tensor shapes (no dimension change) while eliminating
                # all contribution from this layer to the residual stream.
                for attr in ("W_out", "c_proj"):
                    if hasattr(mlp, attr):
                        getattr(mlp, attr).data.zero_()
                        self.logger.debug(f"Zeroed {attr} at MLP layer {layer_idx}")
                        break
                else:
                    # Llama / gated MLP: zero the down projection
                    if hasattr(mlp, "down_proj"):
                        mlp.down_proj.weight.data.zero_()
                        self.logger.debug(f"Zeroed down_proj at MLP layer {layer_idx}")
            except Exception as e:
                self.logger.error(f"Failed to prune MLP layer {layer_idx}: {e}")

        self.logger.debug(f"MLP layers pruned (output-zeroed): {sorted(set(mlp_layers_to_prune))}")

    @staticmethod
    def _param_counts(model: HookedTransformer) -> tuple:
        """Return ``(nonzero_param_count, total_param_count)`` for a model.

        Counting the nonzero count at a specific moment lets callers snapshot
        a pre-masking baseline that survives an in-place mutation of the same
        model object.
        """
        total = 0
        nonzero = 0
        for p in model.parameters():
            total += p.numel()
            nonzero += int((p != 0).sum().item())
        return nonzero, total

    def measure_sparsity(
        self,
        model_after: HookedTransformer,
        model_before: Optional[HookedTransformer] = None,
        *,
        nonzero_before: Optional[int] = None,
        total_params: Optional[int] = None,
    ) -> float:
        """
        Measure the effective sparsity produced by masking.

        Masking here zeroes the weights of selected heads/MLPs in place rather
        than physically resizing tensors (see the class docstring), so total
        parameter *count* is unchanged. Effective sparsity is therefore the
        fraction of weights driven to exactly zero by masking:

            (nonzero_before - nonzero_after) / total_params

        The pre-masking baseline must be captured *before* the model is
        mutated. Pass it as a snapshot via ``nonzero_before`` (and optionally
        ``total_params``) — this is the only correct route when masking is
        done in place, because then ``model_after`` *is* the original model
        object and there is no distinct unmasked model left to compare with.
        Captured counts come from :meth:`_param_counts`.

        Args:
            model_after: The masked model (counted now).
            model_before: Optional separate unmasked model. Only meaningful in
                copy mode, where it is a genuinely distinct object. Ignored
                when ``nonzero_before`` is supplied.
            nonzero_before: Pre-masking nonzero-parameter count snapshot. When
                given, this is used as the baseline regardless of
                ``model_before`` (required for correct in-place results).
            total_params: Total parameter count snapshot. Defaults to the
                count of ``model_after`` (unchanged by masking).

        Returns:
            float: Effective sparsity achieved (0.0 to 1.0).
        """
        nonzero_after, total_after = self._param_counts(model_after)

        total = total_params if total_params is not None else total_after
        if total == 0:
            return 0.0

        if nonzero_before is None:
            if model_before is None:
                raise ValueError(
                    "measure_sparsity needs a pre-masking baseline: pass "
                    "either nonzero_before (a snapshot taken before masking) "
                    "or model_before (a distinct unmasked model)."
                )
            nonzero_before, _ = self._param_counts(model_before)

        sparsity = (nonzero_before - nonzero_after) / total
        return max(0.0, min(1.0, sparsity))  # Clamp to [0, 1]

    @staticmethod
    def _format_nodes(nodes: Dict[str, float]) -> str:
        """Format node dict for logging."""
        if len(nodes) <= 5:
            return str(list(nodes.keys()))
        return f"{list(nodes.keys())[:5]} + {len(nodes) - 5} more"

    def save_pruned_model(self, model: HookedTransformer, path: Union[str, Path]) -> Path:
        """
        Save pruned model to disk.

        Args:
            model: Pruned HookedTransformer model.
            path: Output path (typically .pt).

        Returns:
            Path to saved model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        t.save(model.state_dict(), path)
        self.logger.info(f"Pruned model saved to {path}")
        return path
