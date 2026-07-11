# FILE: circuitkit/applications/pruning/weight_pruner.py

import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

from transformer_lens import HookedTransformer
import logging



logger = logging.getLogger(__name__)

def get_attention_architecture_info(model: HookedTransformer) -> Dict[str, Any]:
    """
    Get information about the attention architecture of the model.

    Args:
        model: The HookedTransformer model

    Returns:
        Dictionary with attention architecture information
    """
    n_heads = model.cfg.n_heads
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", n_heads)
    if n_kv_heads is None:
        n_kv_heads = n_heads
    d_head = model.cfg.d_head

    architecture_type = "Standard Multi-Head Attention"
    if n_kv_heads < n_heads:
        if n_kv_heads == 1:
            architecture_type = "Multi-Query Attention (MQA)"
        else:
            architecture_type = "Grouped Query Attention (GQA)"

    return {
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "architecture_type": architecture_type,
        "q_to_kv_ratio": n_heads / n_kv_heads if n_kv_heads > 0 else 1.0,
    }


def zero_attention_head_weights(model: HookedTransformer, layer_idx: int, head_idx: int):
    """
    Zero out the weights for a specific attention head by modifying the model weights directly.
    Handles different attention architectures including Grouped Query Attention (GQA).

    In GQA, we only prune Q heads and their output projections, NOT the shared K/V heads.

    Args:
        model: The HookedTransformer model
        layer_idx: Layer index (0-based)
        head_idx: Head index within the layer (0-based)

    Raises:
        ValueError: If layer_idx or head_idx are out of bounds
        RuntimeError: If weight matrix operations fail
    """
    # Input validation
    if not (0 <= layer_idx < len(model.blocks)):
        raise ValueError(f"Invalid layer_idx {layer_idx}, must be in [0, {len(model.blocks)})")

    if not (0 <= head_idx < model.cfg.n_heads):
        raise ValueError(f"Invalid head_idx {head_idx}, must be in [0, {model.cfg.n_heads})")

    # Get the attention module for this layer
    try:
        attn_module = model.blocks[layer_idx].attn
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"Failed to access attention module for layer {layer_idx}: {e}")

    # Get attention configuration
    n_heads = model.cfg.n_heads
    model.cfg.d_head
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", n_heads)  # For GQA/MQA
    if n_kv_heads is None:
        n_kv_heads = n_heads

    # Always zero Q weights (each Q head is independent)
    # W_Q shape is [n_heads, d_model, d_head]
    if hasattr(attn_module, "W_Q"):
        try:
            attn_module.W_Q.data[head_idx, :, :] = 0.0
        except IndexError as e:
            raise RuntimeError(
                f"Failed to prune Q weights for head {head_idx} in layer {layer_idx}: {e}"
            )
    else:
        warnings.warn(f"Layer {layer_idx} missing W_Q matrix, skipping Q pruning")

    # For K and V weights, only prune if it's standard multi-head attention
    # In GQA/MQA, K/V heads are shared, so we don't prune them
    if n_kv_heads == n_heads:
        # Standard multi-head attention - each Q head has its own K/V head
        # W_K and W_V shapes are [n_heads, d_model, d_head]
        if hasattr(attn_module, "W_K"):
            try:
                attn_module.W_K.data[head_idx, :, :] = 0.0
            except IndexError as e:
                raise RuntimeError(
                    f"Failed to prune K weights for head {head_idx} in layer {layer_idx}: {e}"
                )
        else:
            warnings.warn(f"Layer {layer_idx} missing W_K matrix, skipping K pruning")

        if hasattr(attn_module, "W_V"):
            try:
                attn_module.W_V.data[head_idx, :, :] = 0.0
            except IndexError as e:
                raise RuntimeError(
                    f"Failed to prune V weights for head {head_idx} in layer {layer_idx}: {e}"
                )
        else:
            warnings.warn(f"Layer {layer_idx} missing W_V matrix, skipping V pruning")
    else:
        # Grouped Query Attention - K/V heads are shared, so we don't prune them
        # This is correct because other Q heads might still need these K/V heads
        logger.info(
            f"  Note: Not pruning K/V weights for head {head_idx} in GQA (shared with other heads)"
        )

    # Always zero the output projection weights for this head
    # W_O shape is [n_heads, d_head, d_model]
    if hasattr(attn_module, "W_O"):
        try:
            attn_module.W_O.data[head_idx, :, :] = 0.0
        except IndexError as e:
            raise RuntimeError(
                f"Failed to prune output weights for head {head_idx} in layer {layer_idx}: {e}"
            )
    else:
        warnings.warn(f"Layer {layer_idx} missing W_O matrix, skipping output pruning")


def zero_attention_neuron_weights(
    model: HookedTransformer, layer_idx: int, head_idx: int, neuron_indices: List[int]
):
    """
    Zero the projection weights for specific residual-space neurons for a given attention head.

    This function implements weight-level neuron pruning for attention heads in the residual space
    (i.e., per-head post-W_O contribution). For each residual dimension k in neuron_indices, we set
    W_O[head_idx, :, k] = 0 so that this head no longer contributes to residual neuron k.

    Note: We DO NOT touch W_Q / W_K / W_V for neuron-level pruning. This is intentional so that
    pruning aligns with neuron scores produced in residual space by EAP when use_attn_result=True.

    Args:
        model: The HookedTransformer model
        layer_idx: Layer index (0-based)
        head_idx: Head index within the layer (0-based)
        neuron_indices: Residual-space neuron indices to prune (must be in [0, d_model))

    Raises:
        ValueError: If layer_idx, head_idx, or neuron_indices are out of bounds
        RuntimeError: If weight matrix operations fail
    """
    # Input validation
    if not (0 <= layer_idx < len(model.blocks)):
        raise ValueError(f"Invalid layer_idx {layer_idx}, must be in [0, {len(model.blocks)})")

    if not (0 <= head_idx < model.cfg.n_heads):
        raise ValueError(f"Invalid head_idx {head_idx}, must be in [0, {model.cfg.n_heads})")

    if not neuron_indices:
        warnings.warn(
            f"No neuron indices provided for pruning head {head_idx} in layer {layer_idx}"
        )
        return

    # Get the attention module for this layer
    try:
        attn_module = model.blocks[layer_idx].attn
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"Failed to access attention module for layer {layer_idx}: {e}")

    # Get dimensions
    model.cfg.n_heads
    d_model = model.cfg.d_model

    # Validate neuron indices - residual-space dimensions [0, d_model)
    invalid_neurons = [idx for idx in neuron_indices if not (0 <= idx < d_model)]
    if invalid_neurons:
        # Provide more helpful error message
        max_idx = max(neuron_indices) if neuron_indices else 0
        min_idx = min(neuron_indices) if neuron_indices else 0
        raise ValueError(
            f"Invalid residual neuron indices for head {head_idx} in layer {layer_idx}. "
            f"Indices must be in [0, {d_model}) but got range [{min_idx}, {max_idx}]. "
            f"Invalid indices: {invalid_neurons[:10]}{'...' if len(invalid_neurons) > 10 else ''}."
        )

    # Zero out the output projection weights for specific residual neurons in this head
    # W_O shape is [n_heads, d_head, d_model]
    if hasattr(attn_module, "W_O"):
        try:
            for resid_idx in neuron_indices:
                # Zero the column corresponding to this residual dimension
                attn_module.W_O.data[head_idx, :, resid_idx] = 0.0
        except IndexError as e:
            raise RuntimeError(
                f"Failed to prune W_O projection columns for neurons {neuron_indices} in head {head_idx}, layer {layer_idx}: {e}"
            )
    else:
        warnings.warn(
            f"Layer {layer_idx} missing W_O matrix, skipping attention neuron projection pruning"
        )


def zero_mlp_weights(model: HookedTransformer, layer_idx: int):
    """
    Zero out all MLP weights for a specific layer.

    Args:
        model: The HookedTransformer model
        layer_idx: Layer index (0-based)

    Raises:
        ValueError: If layer_idx is out of bounds
        RuntimeError: If weight matrix operations fail
    """
    # Input validation
    if not (0 <= layer_idx < len(model.blocks)):
        raise ValueError(f"Invalid layer_idx {layer_idx}, must be in [0, {len(model.blocks)})")

    # Get the MLP module for this layer
    try:
        mlp_module = model.blocks[layer_idx].mlp
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"Failed to access MLP module for layer {layer_idx}: {e}")

    # Zero out all MLP weights
    if hasattr(mlp_module, "W_in"):
        try:
            mlp_module.W_in.data.zero_()
        except RuntimeError as e:
            raise RuntimeError(f"Failed to zero W_in weights for layer {layer_idx}: {e}")
    else:
        warnings.warn(f"Layer {layer_idx} missing W_in matrix, skipping input weight pruning")

    if hasattr(mlp_module, "W_out"):
        try:
            mlp_module.W_out.data.zero_()
        except RuntimeError as e:
            raise RuntimeError(f"Failed to zero W_out weights for layer {layer_idx}: {e}")
    else:
        warnings.warn(f"Layer {layer_idx} missing W_out matrix, skipping output weight pruning")

    # Zero out the gate weights (for SwiGLU MLP)
    if hasattr(mlp_module, "W_gate"):
        try:
            mlp_module.W_gate.data.zero_()
        except RuntimeError as e:
            raise RuntimeError(f"Failed to zero W_gate weights for layer {layer_idx}: {e}")
    else:
        warnings.warn(f"Layer {layer_idx} missing W_gate matrix, skipping gate weight pruning")

    # Zero out biases
    if hasattr(mlp_module, "b_in"):
        try:
            mlp_module.b_in.data.zero_()
        except RuntimeError as e:
            raise RuntimeError(f"Failed to zero b_in bias for layer {layer_idx}: {e}")
    else:
        warnings.warn(f"Layer {layer_idx} missing b_in bias, skipping input bias pruning")

    if hasattr(mlp_module, "b_out"):
        try:
            mlp_module.b_out.data.zero_()
        except RuntimeError as e:
            raise RuntimeError(f"Failed to zero b_out bias for layer {layer_idx}: {e}")
    else:
        warnings.warn(f"Layer {layer_idx} missing b_out bias, skipping output bias pruning")


def zero_mlp_neuron_weights(model: HookedTransformer, layer_idx: int, neuron_indices: List[int]):
    """
    Zero out specific neuron weights in the MLP layer.

    TransformerLens MLP weight shape convention (assumed throughout):
        ``W_in``   has shape ``[d_model, d_mlp]`` — input projection, the
                   ``i``-th neuron's incoming weights are the ``i``-th *column*.
        ``W_out``  has shape ``[d_mlp, d_model]`` — output projection, the
                   ``i``-th neuron's outgoing weights are the ``i``-th *row*.
        ``W_gate`` has shape ``[d_model, d_mlp]`` (gated MLP) — same column
                   convention as ``W_in``.

    Earlier versions of this helper indexed the wrong axis on every matrix
    and read ``mlp_width`` from the ``d_model`` axis of ``W_in``; those
    bugs would silently zero ``d_model`` rows / columns rather than the
    requested neurons, and would reject valid neuron indices in models
    where ``d_mlp > d_model`` (e.g. Llama, Gemma). The code below indexes
    along the explicit *neuron* axis on each matrix.

    Args:
        model: The HookedTransformer model
        layer_idx: Layer index (0-based)
        neuron_indices: List of neuron indices to zero out (each index ∈ [0, d_mlp))

    Raises:
        ValueError: If layer_idx or neuron_indices are out of bounds
        RuntimeError: If weight matrix operations fail
    """
    # Input validation
    if not (0 <= layer_idx < len(model.blocks)):
        raise ValueError(f"Invalid layer_idx {layer_idx}, must be in [0, {len(model.blocks)})")

    if not neuron_indices:
        warnings.warn(f"No neuron indices provided for pruning MLP layer {layer_idx}")
        return

    # Get the MLP module for this layer
    try:
        mlp_module = model.blocks[layer_idx].mlp
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"Failed to access MLP module for layer {layer_idx}: {e}")

    # Get the neuron-axis width (``d_mlp``) for validation. For ``W_in`` the
    # neuron axis is axis 1 (columns); for ``W_out`` it is axis 0 (rows).
    mlp_width = None
    if hasattr(mlp_module, "W_in"):
        mlp_width = mlp_module.W_in.shape[1]
    elif hasattr(mlp_module, "W_out"):
        mlp_width = mlp_module.W_out.shape[0]
    else:
        warnings.warn(
            f"Layer {layer_idx} missing both W_in and W_out matrices, cannot determine MLP width"
        )
        return

    # Validate neuron indices against d_mlp.
    invalid_neurons = [idx for idx in neuron_indices if not (0 <= idx < mlp_width)]
    if invalid_neurons:
        raise ValueError(f"Invalid neuron indices {invalid_neurons}, must be in [0, {mlp_width})")

    # Zero out specific neurons in the MLP. Each indexing op is on the
    # neuron axis of the relevant matrix (see docstring above).
    if hasattr(mlp_module, "W_in"):
        try:
            for neuron_idx in neuron_indices:
                # W_in is [d_model, d_mlp] → neuron = column.
                mlp_module.W_in.data[:, neuron_idx] = 0.0
        except IndexError as e:
            raise RuntimeError(
                f"Failed to prune W_in weights for neurons {neuron_indices} in layer {layer_idx}: {e}"
            )
    else:
        warnings.warn(f"Layer {layer_idx} missing W_in matrix, skipping input weight pruning")

    if hasattr(mlp_module, "W_out"):
        try:
            for neuron_idx in neuron_indices:
                # W_out is [d_mlp, d_model] → neuron = row.
                mlp_module.W_out.data[neuron_idx, :] = 0.0
        except IndexError as e:
            raise RuntimeError(
                f"Failed to prune W_out weights for neurons {neuron_indices} in layer {layer_idx}: {e}"
            )
    else:
        warnings.warn(f"Layer {layer_idx} missing W_out matrix, skipping output weight pruning")

    # Zero gate weights for this neuron (SwiGLU MLP).
    if hasattr(mlp_module, "W_gate"):
        try:
            for neuron_idx in neuron_indices:
                # W_gate is [d_model, d_mlp] → neuron = column.
                mlp_module.W_gate.data[:, neuron_idx] = 0.0
        except IndexError as e:
            raise RuntimeError(
                f"Failed to prune W_gate weights for neurons {neuron_indices} in layer {layer_idx}: {e}"
            )
    else:
        warnings.warn(f"Layer {layer_idx} missing W_gate matrix, skipping gate weight pruning")

    # Zero biases for this neuron
    if hasattr(mlp_module, "b_in"):
        try:
            for neuron_idx in neuron_indices:
                mlp_module.b_in.data[neuron_idx] = 0.0
        except IndexError as e:
            raise RuntimeError(
                f"Failed to prune b_in bias for neurons {neuron_indices} in layer {layer_idx}: {e}"
            )
    else:
        warnings.warn(f"Layer {layer_idx} missing b_in bias, skipping input bias pruning")


def apply_node_pruning_to_weights(model: HookedTransformer, nodes_to_prune: List[str]):
    """
    Apply node-level pruning by directly modifying model weights.

    Args:
        model: The HookedTransformer model
        nodes_to_prune: List of node names to prune (e.g., ["A0.1", "MLP 2"])

    Raises:
        ValueError: If node names are in invalid format
        RuntimeError: If pruning operations fail
    """
    if not nodes_to_prune:
        warnings.warn("No nodes provided for pruning")
        return

    logger.info(f"Applying weight-based pruning to {len(nodes_to_prune)} nodes...")

    failed_nodes = []
    for node_name in nodes_to_prune:
        try:
            # Parse attention head nodes (e.g., "A0.1" -> layer 0, head 1)
            attn_match = re.match(r"A(\d+)\.(\d+)", node_name)
            if attn_match:
                layer_idx, head_idx = int(attn_match.group(1)), int(attn_match.group(2))
                logger.info(f"  Pruning attention head: layer {layer_idx}, head {head_idx}")
                zero_attention_head_weights(model, layer_idx, head_idx)
                continue

            # Parse MLP nodes (e.g., "MLP 2" -> layer 2)
            mlp_match = re.match(r"MLP (\d+)", node_name)
            if mlp_match:
                layer_idx = int(mlp_match.group(1))
                logger.info(f"  Pruning MLP layer: {layer_idx}")
                zero_mlp_weights(model, layer_idx)
                continue

            # Invalid node format
            error_msg = f"Unknown node format: {node_name}"
            logger.warning(error_msg)
            failed_nodes.append((node_name, error_msg))

        except (ValueError, RuntimeError) as e:
            error_msg = f"Failed to prune {node_name}: {e}"
            logger.error(error_msg)
            failed_nodes.append((node_name, error_msg))

    if failed_nodes:
        warnings.warn(
            f"Failed to prune {len(failed_nodes)} nodes: {[node for node, _ in failed_nodes]}"
        )
        logger.info(f"Failed nodes details: {failed_nodes}")


def apply_neuron_pruning_to_weights(
    model: HookedTransformer,
    pruned_mlp_neurons: Dict[int, List[int]],
    pruned_attn_neurons: Dict[Tuple[int, int], List[int]],
):
    """
    Apply neuron-level pruning by directly modifying model weights.

    Args:
        model: The HookedTransformer model
        pruned_mlp_neurons: Dict mapping layer_idx to list of neuron indices to prune
        pruned_attn_neurons: Dict mapping (layer_idx, head_idx) to list of neuron indices to prune

    Raises:
        ValueError: If pruning parameters are invalid
        RuntimeError: If pruning operations fail
    """
    logger.info("Applying weight-based neuron pruning...")

    # Debug information
    logger.info("  Model configuration:")
    logger.info(f"    - n_layers: {len(model.blocks)}")
    logger.info(f"    - n_heads: {model.cfg.n_heads}")
    logger.info(f"    - d_head: {model.cfg.d_head}")
    if hasattr(model.cfg, "d_mlp"):
        logger.info(f"    - d_mlp: {model.cfg.d_mlp}")

    failed_operations = []

    # Prune MLP neurons
    if pruned_mlp_neurons:
        logger.info(f"  Pruning MLP neurons in {len(pruned_mlp_neurons)} layers")
        for layer_idx, neuron_indices in pruned_mlp_neurons.items():
            try:
                logger.info(
                    f"  Pruning MLP neurons in layer {layer_idx}: {len(neuron_indices)} neurons (indices: {neuron_indices[:5]}{'...' if len(neuron_indices) > 5 else ''})"
                )
                zero_mlp_neuron_weights(model, layer_idx, neuron_indices)
            except (ValueError, RuntimeError) as e:
                error_msg = f"Failed to prune MLP neurons in layer {layer_idx}: {e}"
                logger.error(error_msg)
                failed_operations.append((f"MLP_{layer_idx}", error_msg))
    else:
        logger.info("  No MLP neurons to prune")

    # Prune attention neurons
    if pruned_attn_neurons:
        logger.info(f"  Pruning attention neurons in {len(pruned_attn_neurons)} heads")
        for (layer_idx, head_idx), neuron_indices in pruned_attn_neurons.items():
            try:
                logger.info(
                    f"  Pruning attention neurons in layer {layer_idx}, head {head_idx}: {len(neuron_indices)} neurons (indices: {neuron_indices[:5]}{'...' if len(neuron_indices) > 5 else ''})"
                )
                # Additional validation before calling the function (residual-space dims)
                if any(idx >= model.cfg.d_model for idx in neuron_indices):
                    logger.warning(f"Some neuron indices are >= d_model ({model.cfg.d_model})")
                    logger.warning("These indices must be valid residual dimensions [0, d_model)")
                zero_attention_neuron_weights(model, layer_idx, head_idx, neuron_indices)
            except (ValueError, RuntimeError) as e:
                error_msg = f"Failed to prune attention neurons in layer {layer_idx}, head {head_idx}: {e}"
                logger.error(error_msg)
                failed_operations.append((f"ATTN_{layer_idx}_{head_idx}", error_msg))
    else:
        logger.info("  No attention neurons to prune")

    if failed_operations:
        warnings.warn(
            f"Failed to prune {len(failed_operations)} neuron groups: {[op for op, _ in failed_operations]}"
        )
        logger.info(f"Failed operations details: {failed_operations}")


def create_weight_pruned_model(
    model: HookedTransformer, pruned_artifact: Union[List[str], Dict[str, Any]]
) -> HookedTransformer:
    """
    Create a new model with weights pruned based on the artifact.

    Args:
        model: The original HookedTransformer model
        pruned_artifact: Either a list of node names or a dict with MLP/attention neuron info

    Returns:
        A new model with pruned weights

    Raises:
        TypeError: If pruned_artifact is not a supported type
        RuntimeError: If model copying or pruning fails
    """
    if pruned_artifact is None:
        warnings.warn("No pruning artifact provided, returning original model")
        return model

    # Create a deep copy of the model to avoid modifying the original
    try:
        import copy

        pruned_model = copy.deepcopy(model)
    except Exception as e:
        raise RuntimeError(f"Failed to create deep copy of model: {e}")

    try:
        if isinstance(pruned_artifact, list):
            # Node-level pruning
            apply_node_pruning_to_weights(pruned_model, pruned_artifact)
        elif isinstance(pruned_artifact, dict):
            # Neuron-level pruning
            mlp_neurons = pruned_artifact.get("mlp", {})
            attn_neurons = pruned_artifact.get("attn", {})
            apply_neuron_pruning_to_weights(pruned_model, mlp_neurons, attn_neurons)
        else:
            raise TypeError(f"Unsupported pruning artifact type: {type(pruned_artifact)}")
    except Exception as e:
        raise RuntimeError(f"Failed to apply pruning to model: {e}")

    return pruned_model


def get_nodes_to_prune_from_weights(
    node_scores: Dict[str, float],
    target_sparsity: float,
    protected_nodes: List[str] = ["Resid Start"],
    pruning_scope: str = "both",
) -> List[str]:
    """
    Identifies which nodes to prune based on scores (same logic as hook-based pruning).
    This is a copy of the logic from node_pruner.py to maintain consistency.
    """
    if pruning_scope not in ["heads", "mlp", "both"]:
        raise ValueError(
            f"pruning_scope must be one of 'heads', 'mlp', or 'both', but got {pruning_scope}"
        )

    logger.info(f"\n--- Running Weight-Based Node Pruning (Scope: {pruning_scope.upper()}) ---")
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
        f"Found {len(attn_head_scores)} attention heads and {len(mlp_scores)} MLP layers to consider for pruning."
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

    logger.info("--- Weight-Based Pruning Finished ---")
    return nodes_to_prune
