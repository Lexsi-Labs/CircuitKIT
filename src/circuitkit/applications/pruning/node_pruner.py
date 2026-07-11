# FILE: circuitkit/applications/pruning/node_pruner.py

import re
from collections import defaultdict
from functools import partial

import torch as t

from circuitkit.backends.acdc.utils.patchable_model import PatchableModel
import logging



logger = logging.getLogger(__name__)

class NodePruner:
    """Small object-oriented wrapper around node-score pruning."""

    def prune(
        self,
        node_scores: dict[str, float],
        target_sparsity: float,
        scope: str = "both",
        protected_nodes: list[str] = ["Resid Start"],
    ) -> list[str]:
        return get_nodes_to_prune(
            node_scores=node_scores,
            target_sparsity=target_sparsity,
            protected_nodes=protected_nodes,
            pruning_scope=scope,
        )


def get_nodes_to_prune(
    node_scores: dict[str, float],
    target_sparsity: float,
    protected_nodes: list[str] = ["Resid Start"],
    pruning_scope: str = "both",  # <-- NEW PARAMETER
) -> list[str]:
    """
    Identifies which nodes to prune based on a categorized sparsity approach.

    Args:
        node_scores: A dictionary mapping node names to their importance scores.
        target_sparsity: The fraction of nodes of each type to prune.
        protected_nodes: A list of node names that should never be pruned.
        pruning_scope: Specifies what to prune. Can be 'heads', 'mlp', or 'both'.
    """
    if pruning_scope not in ["heads", "mlp", "both"]:
        raise ValueError(
            f"pruning_scope must be one of 'heads', 'mlp', or 'both', but got {pruning_scope}"
        )

    logger.info(f"\n--- Running Categorized Node Pruning (Scope: {pruning_scope.upper()}) ---")
    prunable_nodes = {
        name: score for name, score in node_scores.items() if name not in protected_nodes
    }

    # 1. Separate nodes into categories: Attention Heads and MLP layers
    attn_head_scores = {}
    mlp_scores = {}
    attn_heads_by_layer = defaultdict(dict)

    for name, score in prunable_nodes.items():
        attn_match = re.match(r"A(\d+)\.(\d+)", name)
        mlp_match = re.match(r"MLP (\d+)", name)

        if attn_match:
            layer_idx = int(attn_match.group(1))
            attn_head_scores[name] = score
            attn_heads_by_layer[layer_idx][name] = score
        elif mlp_match:
            mlp_scores[name] = score

    logger.info(
        f"Found {len(attn_head_scores)} Attention Heads and {len(mlp_scores)} MLP layers to consider for pruning."
    )

    nodes_to_prune = []

    # 2. Prune MLP layers if scope allows
    if pruning_scope in ["mlp", "both"] and mlp_scores:
        sorted_mlps = sorted(mlp_scores.items(), key=lambda item: item[1])
        num_mlps_to_prune = int(len(sorted_mlps) * target_sparsity)
        pruned_mlps = [name for name, score in sorted_mlps[:num_mlps_to_prune]]
        nodes_to_prune.extend(pruned_mlps)
        logger.info(
            f"Pruning {len(pruned_mlps)} out of {len(mlp_scores)} MLP layers ({target_sparsity:.0%})."
        )
    elif pruning_scope == "heads":
        logger.info("Skipping MLP layer pruning as per scope.")

    # 3. Prune Attention Heads if scope allows (with 50% per-layer constraint)
    if pruning_scope in ["heads", "both"] and attn_head_scores:
        total_heads_to_prune = int(len(attn_head_scores) * target_sparsity)
        logger.info(
            f"Targeting to prune {total_heads_to_prune} out of {len(attn_head_scores)} total Attention Heads ({target_sparsity:.0%})."
        )

        sorted_attn_heads = sorted(attn_head_scores.items(), key=lambda item: item[1])
        pruned_heads_count_by_layer = defaultdict(int)
        pruned_attn_heads = []

        for head_name, score in sorted_attn_heads:
            if len(pruned_attn_heads) >= total_heads_to_prune:
                break

            attn_match = re.match(r"A(\d+)\.(\d+)", head_name)
            layer_idx = int(attn_match.group(1))

            total_heads_in_layer = len(attn_heads_by_layer[layer_idx])
            max_prune_for_layer = int(total_heads_in_layer * 0.5)
            if target_sparsity > 0:
                max_prune_for_layer = max(1, max_prune_for_layer)

            if pruned_heads_count_by_layer[layer_idx] < max_prune_for_layer:
                pruned_attn_heads.append(head_name)
                pruned_heads_count_by_layer[layer_idx] += 1

        nodes_to_prune.extend(pruned_attn_heads)
        logger.info(f"Actually pruned {len(pruned_attn_heads)} Attention Heads.")
        for layer, count in sorted(pruned_heads_count_by_layer.items()):
            total_in_layer = len(attn_heads_by_layer[layer])
            logger.info(f"  - Layer {layer}: Pruned {count}/{total_in_layer} heads.")
    elif pruning_scope == "mlp":
        logger.info("Skipping Attention Head pruning as per scope.")

    logger.info("--- Categorized Pruning Finished ---")
    return nodes_to_prune


# ... (zero_head_hook and zero_mlp_hook functions remain the same) ...
def zero_head_hook(activation: t.Tensor, hook, head_index: int):
    activation[:, :, head_index, :] = 0.0
    # Don't return anything for in-place modifications


def zero_mlp_hook(activation: t.Tensor, hook):
    return t.zeros_like(activation)


def prune_nodes_and_evaluate(
    model: PatchableModel,
    dataloader,
    node_scores: dict[str, float],
    target_sparsity: float,
    metric_func,
    pruning_scope: str = "both",  # <-- NEW PARAMETER
):
    """
    Applies conceptual node pruning via hooks and evaluates the model's performance.
    """
    # <-- MODIFIED CALL to pass the new parameter -->
    nodes_to_prune = get_nodes_to_prune(node_scores, target_sparsity, pruning_scope=pruning_scope)

    logger.info(f"\nTotal nodes to prune after categorization: {len(nodes_to_prune)}")

    transformer_model = model.wrapped_model

    hooks_to_add = []
    for node_name in nodes_to_prune:
        attn_match = re.match(r"A(\d+)\.(\d+)", node_name)
        if attn_match:
            layer_idx, head_idx = int(attn_match.group(1)), int(attn_match.group(2))
            hook_point = f"blocks.{layer_idx}.attn.hook_result"
            hook_func = partial(zero_head_hook, head_index=head_idx)
            hooks_to_add.append((hook_point, hook_func))
            continue

        mlp_match = re.match(r"MLP (\d+)", node_name)
        if mlp_match:
            layer_idx = int(mlp_match.group(1))
            hook_point = f"blocks.{layer_idx}.hook_mlp_out"
            hooks_to_add.append((hook_point, zero_mlp_hook))
            continue

    pruned_perf = 0
    batch_count = 0

    with transformer_model.hooks(fwd_hooks=hooks_to_add):
        for batch in dataloader:
            model_output = transformer_model(batch.clean)
            pruned_perf += metric_func(model_output, batch)
            batch_count += 1

    return pruned_perf / batch_count
