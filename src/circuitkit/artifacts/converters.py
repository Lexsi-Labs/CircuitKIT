"""
Conversion utilities to create CircuitArtifact from different discovery methods.

Provides functions to convert outputs from ACDC, EAP, EAP-IG, and IBCircuit
discovery methods into the unified CircuitArtifact representation.
"""

import logging
from typing import Any, Dict, Optional

import torch

from .circuit_artifact import CircuitArtifact, Edge, Node, NodeType

logger = logging.getLogger(__name__)


def acdc_to_artifact(
    prune_scores: Dict[str, torch.Tensor],
    model_id: str,
    task: str,
    dataset: str,
    edges: Optional[list] = None,
    threshold: float = 0.0,
    granularity: str = "head",
) -> CircuitArtifact:
    """
    Convert ACDC prune scores to CircuitArtifact.

    Args:
        prune_scores: Dictionary mapping module names to edge importance tensors
        model_id: HuggingFace model identifier
        task: Task name
        dataset: Dataset name
        edges: Optional list of ACDC Edge objects for graph connectivity
        threshold: Importance threshold for filtering nodes
        granularity: Node granularity ("head", "neuron", "layer")

    Returns:
        CircuitArtifact instance

    Example:
        >>> artifact = acdc_to_artifact(
        ...     prune_scores=scores,
        ...     model_id="gpt2",
        ...     task="ioi",
        ...     dataset="ioi_dataset"
        ... )
    """
    artifact = CircuitArtifact(
        model_id=model_id,
        discovery_method="acdc",
        task=task,
        dataset=dataset,
        granularity=granularity,
        threshold=threshold,
    )

    artifact.algorithm_params = {
        "threshold": threshold,
        "granularity": granularity,
        "num_modules": len(prune_scores),
    }

    nodes_added = {}

    # Extract nodes from prune scores
    # Format: module_name -> scores tensor
    for module_name, scores_tensor in prune_scores.items():
        # Parse module name to get layer and component info
        # Expected formats: "blocks.0.attn.hook_v", "blocks.1.mlp.hook_result"
        parts = module_name.split(".")

        if len(parts) < 3:
            continue

        try:
            layer_idx = int(parts[1])
        except (ValueError, IndexError):
            logger.warning(f"Could not parse layer index from {module_name}")
            continue

        # Determine node type from module name
        if "attn" in module_name:
            node_type = NodeType.ATTENTION_HEAD
            component = "attn"
        elif "mlp" in module_name:
            node_type = NodeType.MLP_NEURON
            component = "mlp"
        else:
            logger.debug(f"Skipping module with unknown type: {module_name}")
            continue

        # Process scores (could be 1D for heads or 2D for neurons)
        scores_flat = scores_tensor.flatten().cpu().detach()

        for idx, importance in enumerate(scores_flat):
            importance_val = float(importance)

            if importance_val < threshold:
                continue

            node_id = f"L{layer_idx}_{component}_{idx}"
            node = Node(
                layer_idx=layer_idx,
                node_type=node_type,
                index=idx,
                importance=min(importance_val, 1.0),  # Normalize to [0, 1]
                name=node_id,
            )
            artifact.add_node(node_id, node)
            nodes_added[module_name] = node_id

    # Add edges if provided
    if edges:
        for edge_idx, edge_obj in enumerate(edges):
            try:
                src_node = edge_obj.src
                dst_node = edge_obj.dest

                src_id = f"L{src_node.layer}_{src_node.name.split('.')[-1]}"
                dst_id = f"L{dst_node.layer}_{dst_node.name.split('.')[-1]}"

                edge_weight = float(edge_obj.prune_score(prune_scores).mean())
                edge_weight = min(max(edge_weight, 0.0), 1.0)

                edge = Edge(
                    src_id=src_id,
                    dst_id=dst_id,
                    weight=edge_weight,
                    attribution="direct",
                )
                artifact.add_edge(f"E{edge_idx}", edge)
            except Exception as e:
                logger.debug(f"Could not add edge {edge_idx}: {e}")

    logger.info(
        f"Converted ACDC circuit: {len(artifact.nodes)} nodes, " f"{len(artifact.edges)} edges"
    )

    return artifact


def eap_to_artifact(
    node_scores: Dict[str, float],
    model_id: str,
    task: str,
    dataset: str,
    graph: Optional[Any] = None,
    threshold: float = 0.0,
    granularity: str = "head",
) -> CircuitArtifact:
    """
    Convert EAP node attribution scores to CircuitArtifact.

    Args:
        node_scores: Dictionary mapping node names to importance scores
        model_id: HuggingFace model identifier
        task: Task name
        dataset: Dataset name
        graph: Optional EAP Graph object for edge connectivity
        threshold: Importance threshold for filtering nodes
        granularity: Node granularity ("head", "neuron", "layer")

    Returns:
        CircuitArtifact instance

    Example:
        >>> artifact = eap_to_artifact(
        ...     node_scores=scores,
        ...     model_id="gpt2",
        ...     task="ioi",
        ...     dataset="ioi_dataset"
        ... )
    """
    artifact = CircuitArtifact(
        model_id=model_id,
        discovery_method="eap",
        task=task,
        dataset=dataset,
        granularity=granularity,
        threshold=threshold,
    )

    artifact.algorithm_params = {
        "threshold": threshold,
        "granularity": granularity,
        "num_nodes": len(node_scores),
    }

    # Parse node scores and create nodes
    # Expected format: "A0.0" (attention L0H0), "MLP 3" (MLP layer 3)
    for node_name, importance in node_scores.items():
        importance_val = float(importance)

        if importance_val < threshold:
            continue

        # Parse node name
        node_id = None
        node = None

        if node_name.startswith("A"):
            # Attention head: "A0.1" -> layer 0, head 1
            try:
                parts = node_name[1:].split(".")
                layer_idx = int(parts[0])
                head_idx = int(parts[1]) if len(parts) > 1 else 0
                node_id = f"L{layer_idx}H{head_idx}"
                node = Node(
                    layer_idx=layer_idx,
                    node_type=NodeType.ATTENTION_HEAD,
                    index=head_idx,
                    importance=min(importance_val, 1.0),
                    name=node_id,
                )
            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse attention node {node_name}: {e}")

        elif node_name.startswith("MLP"):
            # MLP neuron: "MLP 3" or "MLP3.5" -> layer 3, neuron 5
            try:
                parts = node_name[3:].strip().split(".")
                layer_idx = int(parts[0])
                neuron_idx = int(parts[1]) if len(parts) > 1 else 0
                node_id = f"L{layer_idx}M{neuron_idx}"
                node = Node(
                    layer_idx=layer_idx,
                    node_type=NodeType.MLP_NEURON,
                    index=neuron_idx,
                    importance=min(importance_val, 1.0),
                    name=node_id,
                )
            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse MLP node {node_name}: {e}")

        if node and node_id:
            artifact.add_node(node_id, node)

    # Add edges from graph if provided
    if graph:
        try:
            edge_idx = 0
            if hasattr(graph, "edges"):
                for edge_obj in graph.edges:
                    try:
                        src_name = str(edge_obj.src.name)
                        dst_name = str(edge_obj.dest.name)

                        # Convert names to node IDs
                        src_node_id = None
                        dst_node_id = None

                        if src_name.startswith("A"):
                            parts = src_name[1:].split(".")
                            src_node_id = f"L{parts[0]}H{parts[1] if len(parts) > 1 else 0}"
                        elif src_name.startswith("MLP"):
                            parts = src_name[3:].strip().split(".")
                            src_node_id = f"L{parts[0]}M{parts[1] if len(parts) > 1 else 0}"

                        if dst_name.startswith("A"):
                            parts = dst_name[1:].split(".")
                            dst_node_id = f"L{parts[0]}H{parts[1] if len(parts) > 1 else 0}"
                        elif dst_name.startswith("MLP"):
                            parts = dst_name[3:].strip().split(".")
                            dst_node_id = f"L{parts[0]}M{parts[1] if len(parts) > 1 else 0}"

                        if src_node_id and dst_node_id:
                            edge_weight = getattr(edge_obj, "weight", 0.5)
                            edge_weight = float(edge_weight)
                            edge_weight = min(max(edge_weight, 0.0), 1.0)

                            edge = Edge(
                                src_id=src_node_id,
                                dst_id=dst_node_id,
                                weight=edge_weight,
                                attribution="direct",
                            )
                            artifact.add_edge(f"E{edge_idx}", edge)
                            edge_idx += 1
                    except Exception as e:
                        logger.debug(f"Could not add edge from graph: {e}")
        except Exception as e:
            logger.warning(f"Could not extract edges from graph: {e}")

    logger.info(
        f"Converted EAP circuit: {len(artifact.nodes)} nodes, " f"{len(artifact.edges)} edges"
    )

    return artifact


def ibcircuit_to_artifact(
    node_scores: Dict[str, float],
    model_id: str,
    task: str,
    dataset: str,
    threshold: float = 0.0,
    granularity: str = "neuron",
) -> CircuitArtifact:
    """
    Convert IBCircuit node scores to CircuitArtifact.

    Args:
        node_scores: Dictionary mapping neuron identifiers to importance scores
        model_id: HuggingFace model identifier
        task: Task name
        dataset: Dataset name
        threshold: Importance threshold for filtering nodes
        granularity: Node granularity ("neuron" typically for IBCircuit)

    Returns:
        CircuitArtifact instance

    Example:
        >>> artifact = ibcircuit_to_artifact(
        ...     node_scores=scores,
        ...     model_id="gpt2",
        ...     task="ioi",
        ...     dataset="ioi_dataset"
        ... )
    """
    artifact = CircuitArtifact(
        model_id=model_id,
        discovery_method="ibcircuit",
        task=task,
        dataset=dataset,
        granularity=granularity,
        threshold=threshold,
    )

    artifact.algorithm_params = {
        "threshold": threshold,
        "granularity": granularity,
        "num_neurons": len(node_scores),
    }

    # Parse neuron identifiers
    # Expected format: "L2.N345" (layer 2, neuron 345)
    for neuron_id, importance in node_scores.items():
        importance_val = float(importance)

        if importance_val < threshold:
            continue

        try:
            parts = neuron_id.split(".")
            layer_idx = int(parts[0].replace("L", ""))
            neuron_idx = int(parts[1].replace("N", "")) if len(parts) > 1 else 0

            node = Node(
                layer_idx=layer_idx,
                node_type=NodeType.MLP_NEURON,
                index=neuron_idx,
                importance=min(importance_val, 1.0),
                name=neuron_id,
            )
            artifact.add_node(neuron_id, node)
        except (ValueError, IndexError) as e:
            logger.debug(f"Could not parse neuron identifier {neuron_id}: {e}")

    logger.info(f"Converted IBCircuit circuit: {len(artifact.nodes)} nodes")

    return artifact


def normalize_importance_scores(
    scores: Dict[str, float], method: str = "minmax"
) -> Dict[str, float]:
    """
    Normalize importance scores to [0, 1] range.

    Args:
        scores: Dictionary mapping node IDs to scores
        method: Normalization method ("minmax" or "zscore")

    Returns:
        Dictionary of normalized scores
    """
    if not scores:
        return {}

    score_values = list(scores.values())

    if method == "minmax":
        min_score = min(score_values)
        max_score = max(score_values)
        if max_score == min_score:
            return {node_id: 1.0 for node_id in scores}
        return {
            node_id: (score - min_score) / (max_score - min_score)
            for node_id, score in scores.items()
        }

    elif method == "zscore":
        import numpy as np

        mean = np.mean(score_values)
        std = np.std(score_values)
        if std == 0:
            return {node_id: 0.0 for node_id in scores}
        return {node_id: (score - mean) / std for node_id, score in scores.items()}

    else:
        raise ValueError(f"Unknown normalization method: {method}")
