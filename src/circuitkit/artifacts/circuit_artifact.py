"""
Circuit Artifact Schema: Unified representation across all discovery methods.

Provides standardized structure for representing circuits discovered by ACDC, EAP,
EAP-IG, and IBCircuit. Includes graph structure (nodes + edges), metadata,
serialization, validation, and conversion to intervention masks.

Schema Version: 1.0
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..utils.exceptions import DISCOVERY_ALGORITHMS

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Node types in a circuit graph."""

    ATTENTION_HEAD = "attn_head"
    MLP_NEURON = "mlp_neuron"
    ATTENTION_LAYER = "attn_layer"
    MLP_LAYER = "mlp_layer"
    EMBEDDING = "embedding"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class Node:
    """
    Represents a node (unit) in the circuit.

    Attributes:
        layer_idx: Layer index (0-indexed)
        node_type: Type of node (attention head, MLP neuron, etc.)
        index: Index within the layer (head index, neuron index, etc.)
        importance: Normalized importance score (0-1)
        name: Optional human-readable name (e.g., "L2H5" for layer 2, head 5)
    """

    layer_idx: int
    node_type: NodeType
    index: int
    importance: float
    name: Optional[str] = None

    def __post_init__(self):
        """Validate node constraints."""
        if not (0 <= self.importance <= 1):
            raise ValueError(f"importance must be in [0, 1], got {self.importance}")
        if self.layer_idx < 0:
            raise ValueError(f"layer_idx must be >= 0, got {self.layer_idx}")
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")

    def __hash__(self):
        """Allow hashing for set/dict use."""
        return hash((self.layer_idx, self.node_type, self.index))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "layer_idx": self.layer_idx,
            "node_type": self.node_type.value,
            "index": self.index,
            "importance": self.importance,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Reconstruct from dict."""
        _ALIASES = {"attention_head": "attn_head", "mlp": "mlp_neuron", "attn_layer": "attn_layer"}
        data_copy = dict(data)
        raw = data_copy["node_type"]
        data_copy["node_type"] = NodeType(_ALIASES.get(raw, raw))
        return cls(**data_copy)


@dataclass(frozen=True)
class Edge:
    """
    Represents an edge (connection) between two nodes.

    Attributes:
        src_id: Source node ID
        dst_id: Destination node ID
        weight: Edge importance (0-1)
        attribution: Type of attribution ("direct", "indirect", etc.)
    """

    src_id: str
    dst_id: str
    weight: float
    attribution: str = "direct"

    def __post_init__(self):
        """Validate edge constraints."""
        if not (0 <= self.weight <= 1):
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "weight": self.weight,
            "attribution": self.attribution,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        """Reconstruct from dict."""
        return cls(**data)


class CircuitArtifact:
    """
    Unified circuit representation across all discovery methods.

    Represents the results of circuit discovery as a directed graph of nodes
    (units) and edges (connections). Can be serialized, validated, converted
    to intervention masks, and analyzed for sparsity.

    Attributes:
        model_id: HuggingFace model identifier
        discovery_method: Method used for discovery ("acdc", "eap", "eap_ig", "ibcircuit")
        task: Task name for which circuit was discovered
        dataset: Dataset used for discovery
        granularity: Node granularity ("head", "neuron", "layer")
        threshold: Importance threshold for including nodes
        nodes: Dictionary mapping node_id -> Node
        edges: Dictionary mapping edge_id -> Edge
        metadata: Additional metadata (algorithm params, timestamps, etc.)
    """

    SCHEMA_VERSION = "1.0"

    def __init__(
        self,
        model_id: str,
        discovery_method: str,
        task: str,
        dataset: str,
        granularity: str = "head",
        threshold: float = 0.5,
    ):
        """
        Initialize a CircuitArtifact.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-2-7b")
            discovery_method: One of "acdc", "eap", "eap_ig", "ibcircuit"
            task: Task name (e.g., "ioi", "sva")
            dataset: Dataset name used for discovery
            granularity: Node granularity ("head", "neuron", "layer")
            threshold: Importance threshold for filtering nodes
        """
        # Validate inputs
        _normalised = discovery_method.replace("_", "-").lower()
        if _normalised not in DISCOVERY_ALGORITHMS:
            raise ValueError(
                f"discovery_method must be one of {sorted(DISCOVERY_ALGORITHMS)}, "
                f"got {discovery_method!r}"
            )
        if granularity not in ["head", "neuron", "layer"]:
            raise ValueError(
                f"granularity must be one of ['head', 'neuron', 'layer'], " f"got {granularity!r}"
            )
        if not (0 <= threshold <= 1):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        # Set metadata
        self.model_id = model_id
        self.discovery_method = discovery_method
        self.task = task
        self.dataset = dataset
        self.granularity = granularity
        self.threshold = threshold

        # Initialize graph structures
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}

        # Initialize metadata
        self.timestamp = datetime.now().isoformat()
        self.version = self.SCHEMA_VERSION
        self.algorithm_params: Dict[str, Any] = {}

    def add_node(self, node_id: str, node: Node) -> None:
        """
        Add a node to the circuit.

        Args:
            node_id: Unique node identifier
            node: Node object
        """
        if not isinstance(node, Node):
            raise TypeError(f"node must be Node instance, got {type(node)}")
        self.nodes[node_id] = node

    def add_edge(self, edge_id: str, edge: Edge) -> None:
        """
        Add an edge to the circuit.

        Args:
            edge_id: Unique edge identifier
            edge: Edge object
        """
        if not isinstance(edge, Edge):
            raise TypeError(f"edge must be Edge instance, got {type(edge)}")

        # Validate that source and destination nodes exist
        if edge.src_id not in self.nodes:
            logger.warning(f"Edge source {edge.src_id} not in nodes")
        if edge.dst_id not in self.nodes:
            logger.warning(f"Edge destination {edge.dst_id} not in nodes")

        self.edges[edge_id] = edge

    def add_node_batch(self, nodes: Dict[str, Node]) -> None:
        """
        Add multiple nodes at once.

        Args:
            nodes: Dictionary mapping node_id -> Node
        """
        for node_id, node in nodes.items():
            self.add_node(node_id, node)

    def add_edge_batch(self, edges: Dict[str, Edge]) -> None:
        """
        Add multiple edges at once.

        Args:
            edges: Dictionary mapping edge_id -> Edge
        """
        for edge_id, edge in edges.items():
            self.add_edge(edge_id, edge)

    def get_nodes_by_layer(self, layer_idx: int) -> Dict[str, Node]:
        """Get all nodes in a specific layer."""
        return {
            node_id: node for node_id, node in self.nodes.items() if node.layer_idx == layer_idx
        }

    def get_nodes_by_type(self, node_type: NodeType) -> Dict[str, Node]:
        """Get all nodes of a specific type."""
        return {
            node_id: node for node_id, node in self.nodes.items() if node.node_type == node_type
        }

    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Get all edges pointing to a node."""
        return [edge for edge in self.edges.values() if edge.dst_id == node_id]

    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get all edges originating from a node."""
        return [edge for edge in self.edges.values() if edge.src_id == node_id]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert artifact to JSON-serializable dictionary.

        Returns:
            Dictionary representation of the artifact
        """
        return {
            "version": self.version,
            "metadata": {
                "model_id": self.model_id,
                "discovery_method": self.discovery_method,
                "task": self.task,
                "dataset": self.dataset,
                "granularity": self.granularity,
                "threshold": self.threshold,
                "timestamp": self.timestamp,
                "algorithm_params": self.algorithm_params,
            },
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
        }

    def save_json(self, path: Path) -> None:
        """
        Serialize artifact to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved CircuitArtifact to {path}")

    @classmethod
    def load_json(cls, path: Path) -> "CircuitArtifact":
        """
        Deserialize artifact from JSON file.

        Args:
            path: Input file path

        Returns:
            CircuitArtifact instance
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitArtifact":
        """
        Reconstruct artifact from dictionary.

        Args:
            data: Dictionary with circuit data

        Returns:
            CircuitArtifact instance
        """
        metadata = data.get("metadata", {})

        # Create artifact instance
        artifact = cls(
            model_id=metadata.get("model_id", "unknown"),
            discovery_method=metadata.get("discovery_method", "unknown"),
            task=metadata.get("task", "unknown"),
            dataset=metadata.get("dataset", "unknown"),
            granularity=metadata.get("granularity", "head"),
            threshold=metadata.get("threshold", 0.5),
        )

        # Set additional metadata
        artifact.timestamp = metadata.get("timestamp", artifact.timestamp)
        artifact.version = data.get("version", cls.SCHEMA_VERSION)
        artifact.algorithm_params = metadata.get("algorithm_params", {})

        # Reconstruct nodes
        nodes_data = data.get("nodes", {})
        for node_id, node_data in nodes_data.items():
            node = Node.from_dict(node_data)
            artifact.add_node(node_id, node)

        # Reconstruct edges
        edges_data = data.get("edges", {})
        for edge_id, edge_data in edges_data.items():
            edge = Edge.from_dict(edge_data)
            artifact.add_edge(edge_id, edge)

        return artifact

    def to_mask(
        self,
        model: nn.Module,
        arch_cfg: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert circuit to binary intervention masks.

        Creates a dictionary of binary masks for each layer/module, where
        1 indicates "keep" and 0 indicates "prune".

        Args:
            model: HuggingFace model instance
            arch_cfg: Architecture configuration from arch_registry

        Returns:
            Dictionary mapping layer names to binary masks (torch.Tensor)

        Raises:
            ValueError: If circuit granularity is not supported
        """
        from circuitkit.applications import get_layers

        masks = {}

        try:
            layers = get_layers(model, arch_cfg)
            num_layers = len(layers)
        except Exception as e:
            logger.error(f"Failed to get layers from model: {e}")
            return masks

        if self.granularity == "head":
            # Create attention head masks
            for layer_idx in range(num_layers):
                num_heads = model.config.num_attention_heads
                mask = torch.ones(num_heads)

                # Mark nodes in this layer
                layer_nodes = self.get_nodes_by_layer(layer_idx)
                for node in layer_nodes.values():
                    if node.node_type == NodeType.ATTENTION_HEAD:
                        if node.importance >= self.threshold:
                            mask[node.index] = 1.0
                        else:
                            mask[node.index] = 0.0

                masks[f"layer_{layer_idx}_attn_heads"] = mask

        elif self.granularity == "neuron":
            # Create MLP neuron masks
            for layer_idx in range(num_layers):
                intermediate_size = model.config.intermediate_size
                mask = torch.ones(intermediate_size)

                # Mark nodes in this layer
                layer_nodes = self.get_nodes_by_layer(layer_idx)
                for node in layer_nodes.values():
                    if node.node_type == NodeType.MLP_NEURON:
                        if node.importance >= self.threshold:
                            mask[node.index] = 1.0
                        else:
                            mask[node.index] = 0.0

                masks[f"layer_{layer_idx}_mlp_neurons"] = mask

        elif self.granularity == "layer":
            # Create layer-level masks (all heads/neurons in layer or none)
            for layer_idx in range(num_layers):
                layer_nodes = self.get_nodes_by_layer(layer_idx)

                # Average importance across layer
                if layer_nodes:
                    avg_importance = sum(n.importance for n in layer_nodes.values()) / len(
                        layer_nodes
                    )
                    layer_mask = 1.0 if avg_importance >= self.threshold else 0.0
                else:
                    layer_mask = 1.0

                masks[f"layer_{layer_idx}"] = torch.tensor(layer_mask)

        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")

        return masks

    def get_sparsity(self) -> float:
        """
        Calculate circuit sparsity (fraction of nodes kept).

        Returns:
            Float in [0, 1] representing fraction of nodes above threshold
        """
        if not self.nodes:
            return 0.0

        kept = sum(1 for node in self.nodes.values() if node.importance >= self.threshold)
        return kept / len(self.nodes)

    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio (1 - sparsity).

        Returns:
            Float representing fraction of nodes below threshold (removed)
        """
        return 1.0 - self.get_sparsity()

    def validate(self) -> Dict[str, bool]:
        """
        Validate artifact consistency and correctness.

        Returns:
            Dictionary mapping check names to boolean results
        """
        checks = {}

        # Check metadata
        checks["has_model_id"] = bool(self.model_id)
        checks["valid_method"] = (
            self.discovery_method.replace("_", "-").lower() in DISCOVERY_ALGORITHMS
        )
        checks["valid_granularity"] = self.granularity in ["head", "neuron", "layer"]
        checks["valid_threshold"] = 0 <= self.threshold <= 1
        checks["has_task"] = bool(self.task)
        checks["has_dataset"] = bool(self.dataset)

        # Check graph structure
        checks["has_nodes"] = len(self.nodes) > 0
        checks["has_edges"] = len(self.edges) >= 0  # Edges are optional

        # Check node validity
        all_nodes_valid = all(
            isinstance(n, Node) and 0 <= n.importance <= 1 for n in self.nodes.values()
        )
        checks["all_nodes_valid"] = all_nodes_valid

        # Check edge validity
        all_edges_valid = True
        for edge in self.edges.values():
            if edge.src_id not in self.nodes or edge.dst_id not in self.nodes:
                all_edges_valid = False
                break
            if not (0 <= edge.weight <= 1):
                all_edges_valid = False
                break
        checks["all_edges_valid"] = all_edges_valid

        # Check timestamp format
        try:
            datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
            checks["valid_timestamp"] = True
        except (ValueError, AttributeError):
            checks["valid_timestamp"] = False

        return checks

    def summary(self) -> str:
        """
        Generate a human-readable summary of the circuit.

        Returns:
            String summary with key statistics
        """
        num_layers = len(set(n.layer_idx for n in self.nodes.values()))
        attn_nodes = sum(1 for n in self.nodes.values() if n.node_type == NodeType.ATTENTION_HEAD)
        mlp_nodes = sum(1 for n in self.nodes.values() if n.node_type == NodeType.MLP_NEURON)

        lines = [
            "CircuitArtifact Summary",
            f"  Model: {self.model_id}",
            f"  Task: {self.task}",
            f"  Discovery: {self.discovery_method}",
            f"  Dataset: {self.dataset}",
            f"  Nodes: {len(self.nodes)} ({attn_nodes} attention, {mlp_nodes} MLP)",
            f"  Edges: {len(self.edges)}",
            f"  Layers: {num_layers}",
            f"  Sparsity: {self.get_sparsity():.2%}",
            f"  Granularity: {self.granularity}",
            f"  Threshold: {self.threshold}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CircuitArtifact(model={self.model_id!r}, task={self.task!r}, "
            f"method={self.discovery_method!r}, nodes={len(self.nodes)}, "
            f"edges={len(self.edges)})"
        )
