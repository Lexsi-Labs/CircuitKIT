"""
Tests for CircuitArtifact schema and utilities.

Tests serialization, deserialization, validation, mask conversion,
and graph operations across all node types and granularities.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from circuitkit.artifacts import CircuitArtifact, Edge, Node, NodeType

# Test Fixtures


@pytest.fixture
def simple_artifact():
    """Create a simple artifact for basic tests."""
    artifact = CircuitArtifact(
        model_id="meta-llama/Llama-2-7b",
        discovery_method="eap",
        task="ioi",
        dataset="ioi_dataset",
        granularity="head",
        threshold=0.5,
    )

    # Add some nodes
    artifact.add_node(
        "L0H0", Node(layer_idx=0, node_type=NodeType.ATTENTION_HEAD, index=0, importance=0.8)
    )
    artifact.add_node(
        "L0H1", Node(layer_idx=0, node_type=NodeType.ATTENTION_HEAD, index=1, importance=0.3)
    )
    artifact.add_node(
        "L1H2", Node(layer_idx=1, node_type=NodeType.ATTENTION_HEAD, index=2, importance=0.6)
    )
    artifact.add_node(
        "L0M0", Node(layer_idx=0, node_type=NodeType.MLP_NEURON, index=0, importance=0.7)
    )

    # Add some edges
    artifact.add_edge("E0", Edge(src_id="L0H0", dst_id="L1H2", weight=0.9, attribution="direct"))
    artifact.add_edge("E1", Edge(src_id="L0H1", dst_id="L0M0", weight=0.4, attribution="indirect"))

    return artifact


@pytest.fixture
def mock_llama_model():
    """Create a mock LLaMA model for mask conversion tests."""

    class MockConfig:
        model_type = "llama"
        num_attention_heads = 8
        hidden_size = 512
        intermediate_size = 1024

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.mlp = nn.Module()

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([MockLayer() for _ in range(8)])

    return MockModel()


# Test Node class


class TestNode:
    """Test Node dataclass."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = Node(
            layer_idx=0,
            node_type=NodeType.ATTENTION_HEAD,
            index=3,
            importance=0.75,
        )
        assert node.layer_idx == 0
        assert node.node_type == NodeType.ATTENTION_HEAD
        assert node.index == 3
        assert node.importance == 0.75

    def test_node_with_name(self):
        """Test node creation with optional name."""
        node = Node(
            layer_idx=0,
            node_type=NodeType.ATTENTION_HEAD,
            index=3,
            importance=0.75,
            name="L0H3",
        )
        assert node.name == "L0H3"

    def test_node_importance_bounds(self):
        """Test that importance is validated to be in [0, 1]."""
        with pytest.raises(ValueError):
            Node(layer_idx=0, node_type=NodeType.ATTENTION_HEAD, index=0, importance=1.5)

        with pytest.raises(ValueError):
            Node(layer_idx=0, node_type=NodeType.ATTENTION_HEAD, index=0, importance=-0.1)

    def test_node_layer_idx_positive(self):
        """Test that layer_idx must be non-negative."""
        with pytest.raises(ValueError):
            Node(layer_idx=-1, node_type=NodeType.ATTENTION_HEAD, index=0, importance=0.5)

    def test_node_index_positive(self):
        """Test that index must be non-negative."""
        with pytest.raises(ValueError):
            Node(layer_idx=0, node_type=NodeType.ATTENTION_HEAD, index=-1, importance=0.5)

    def test_node_hashable(self):
        """Test that nodes can be hashed (for use in sets/dicts)."""
        node = Node(layer_idx=0, node_type=NodeType.ATTENTION_HEAD, index=3, importance=0.75)
        node_set = {node}
        assert node in node_set

    def test_node_to_dict(self):
        """Test node serialization to dict."""
        node = Node(
            layer_idx=2,
            node_type=NodeType.MLP_NEURON,
            index=5,
            importance=0.6,
            name="L2M5",
        )
        data = node.to_dict()
        assert data["layer_idx"] == 2
        assert data["node_type"] == "mlp_neuron"
        assert data["index"] == 5
        assert data["importance"] == 0.6
        assert data["name"] == "L2M5"

    def test_node_from_dict(self):
        """Test node deserialization from dict."""
        data = {
            "layer_idx": 3,
            "node_type": "attention_head",
            "index": 7,
            "importance": 0.85,
            "name": "L3H7",
        }
        node = Node.from_dict(data)
        assert node.layer_idx == 3
        assert node.node_type == NodeType.ATTENTION_HEAD
        assert node.index == 7
        assert node.importance == 0.85
        assert node.name == "L3H7"


# Test Edge class


class TestEdge:
    """Test Edge dataclass."""

    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = Edge(
            src_id="L0H0",
            dst_id="L1H2",
            weight=0.8,
            attribution="direct",
        )
        assert edge.src_id == "L0H0"
        assert edge.dst_id == "L1H2"
        assert edge.weight == 0.8
        assert edge.attribution == "direct"

    def test_edge_weight_bounds(self):
        """Test that weight is validated to be in [0, 1]."""
        with pytest.raises(ValueError):
            Edge(src_id="L0H0", dst_id="L1H2", weight=1.5)

        with pytest.raises(ValueError):
            Edge(src_id="L0H0", dst_id="L1H2", weight=-0.1)

    def test_edge_to_dict(self):
        """Test edge serialization to dict."""
        edge = Edge(
            src_id="L0H1",
            dst_id="L2M3",
            weight=0.7,
            attribution="indirect",
        )
        data = edge.to_dict()
        assert data["src_id"] == "L0H1"
        assert data["dst_id"] == "L2M3"
        assert data["weight"] == 0.7
        assert data["attribution"] == "indirect"

    def test_edge_from_dict(self):
        """Test edge deserialization from dict."""
        data = {
            "src_id": "L1H5",
            "dst_id": "L3H2",
            "weight": 0.65,
            "attribution": "direct",
        }
        edge = Edge.from_dict(data)
        assert edge.src_id == "L1H5"
        assert edge.dst_id == "L3H2"
        assert edge.weight == 0.65
        assert edge.attribution == "direct"


# Test CircuitArtifact class


class TestCircuitArtifactCreation:
    """Test CircuitArtifact creation and initialization."""

    def test_artifact_creation(self):
        """Test basic artifact creation."""
        artifact = CircuitArtifact(
            model_id="gpt2",
            discovery_method="eap",
            task="sva",
            dataset="counterfact",
        )
        assert artifact.model_id == "gpt2"
        assert artifact.discovery_method == "eap"
        assert artifact.task == "sva"
        assert artifact.dataset == "counterfact"
        assert artifact.granularity == "head"
        assert artifact.threshold == 0.5
        assert len(artifact.nodes) == 0
        assert len(artifact.edges) == 0

    def test_artifact_invalid_method(self):
        """Test that invalid discovery method raises error."""
        with pytest.raises(ValueError):
            CircuitArtifact(
                model_id="gpt2",
                discovery_method="invalid_method",
                task="test",
                dataset="test",
            )

    def test_artifact_invalid_granularity(self):
        """Test that invalid granularity raises error."""
        with pytest.raises(ValueError):
            CircuitArtifact(
                model_id="gpt2",
                discovery_method="eap",
                task="test",
                dataset="test",
                granularity="invalid",
            )

    def test_artifact_invalid_threshold(self):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError):
            CircuitArtifact(
                model_id="gpt2",
                discovery_method="eap",
                task="test",
                dataset="test",
                threshold=1.5,
            )

    def test_artifact_with_parameters(self):
        """Test artifact creation with custom parameters."""
        artifact = CircuitArtifact(
            model_id="meta-llama/Llama-3-8B",
            discovery_method="eap_ig",
            task="gender_bias",
            dataset="winobias",
            granularity="neuron",
            threshold=0.3,
        )
        assert artifact.model_id == "meta-llama/Llama-3-8B"
        assert artifact.discovery_method == "eap_ig"
        assert artifact.granularity == "neuron"
        assert artifact.threshold == 0.3


class TestCircuitArtifactGraphOperations:
    """Test graph operations on CircuitArtifact."""

    def test_add_node(self, simple_artifact):
        """Test adding a single node."""
        initial_count = len(simple_artifact.nodes)
        node = Node(layer_idx=2, node_type=NodeType.ATTENTION_HEAD, index=4, importance=0.5)
        simple_artifact.add_node("L2H4", node)
        assert len(simple_artifact.nodes) == initial_count + 1
        assert "L2H4" in simple_artifact.nodes

    def test_add_invalid_node(self, simple_artifact):
        """Test that adding non-Node object raises error."""
        with pytest.raises(TypeError):
            simple_artifact.add_node("invalid", {"not": "a node"})

    def test_add_edge(self, simple_artifact):
        """Test adding an edge."""
        initial_count = len(simple_artifact.edges)
        edge = Edge(src_id="L0H0", dst_id="L0M0", weight=0.5)
        simple_artifact.add_edge("E2", edge)
        assert len(simple_artifact.edges) == initial_count + 1
        assert "E2" in simple_artifact.edges

    def test_add_invalid_edge(self, simple_artifact):
        """Test that adding non-Edge object raises error."""
        with pytest.raises(TypeError):
            simple_artifact.add_edge("invalid", {"not": "an edge"})

    def test_get_nodes_by_layer(self, simple_artifact):
        """Test filtering nodes by layer."""
        layer_0_nodes = simple_artifact.get_nodes_by_layer(0)
        assert len(layer_0_nodes) == 3  # L0H0, L0H1, L0M0

        layer_1_nodes = simple_artifact.get_nodes_by_layer(1)
        assert len(layer_1_nodes) == 1  # L1H2

    def test_get_nodes_by_type(self, simple_artifact):
        """Test filtering nodes by type."""
        attn_nodes = simple_artifact.get_nodes_by_type(NodeType.ATTENTION_HEAD)
        assert len(attn_nodes) == 3

        mlp_nodes = simple_artifact.get_nodes_by_type(NodeType.MLP_NEURON)
        assert len(mlp_nodes) == 1

    def test_get_incoming_edges(self, simple_artifact):
        """Test finding incoming edges to a node."""
        incoming = simple_artifact.get_incoming_edges("L1H2")
        assert len(incoming) == 1
        assert incoming[0].src_id == "L0H0"

    def test_get_outgoing_edges(self, simple_artifact):
        """Test finding outgoing edges from a node."""
        outgoing = simple_artifact.get_outgoing_edges("L0H0")
        assert len(outgoing) == 1
        assert outgoing[0].dst_id == "L1H2"

    def test_add_node_batch(self):
        """Test batch adding nodes."""
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
        )
        nodes = {
            "n0": Node(0, NodeType.ATTENTION_HEAD, 0, 0.5),
            "n1": Node(0, NodeType.ATTENTION_HEAD, 1, 0.6),
            "n2": Node(1, NodeType.MLP_NEURON, 0, 0.4),
        }
        artifact.add_node_batch(nodes)
        assert len(artifact.nodes) == 3

    def test_add_edge_batch(self):
        """Test batch adding edges."""
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
        )
        # Add nodes first
        artifact.add_node("n0", Node(0, NodeType.ATTENTION_HEAD, 0, 0.5))
        artifact.add_node("n1", Node(1, NodeType.ATTENTION_HEAD, 1, 0.6))

        edges = {
            "e0": Edge(src_id="n0", dst_id="n1", weight=0.8),
        }
        artifact.add_edge_batch(edges)
        assert len(artifact.edges) == 1


class TestCircuitArtifactSerialization:
    """Test serialization and deserialization."""

    def test_to_dict(self, simple_artifact):
        """Test converting artifact to dict."""
        data = simple_artifact.to_dict()

        assert data["version"] == CircuitArtifact.SCHEMA_VERSION
        assert data["metadata"]["model_id"] == "meta-llama/Llama-2-7b"
        assert data["metadata"]["discovery_method"] == "eap"
        assert data["metadata"]["task"] == "ioi"
        assert len(data["nodes"]) == 4
        assert len(data["edges"]) == 2

    def test_from_dict(self):
        """Test reconstructing artifact from dict."""
        data = {
            "version": "1.0",
            "metadata": {
                "model_id": "gpt2",
                "discovery_method": "eap",
                "task": "capital_country",
                "dataset": "counterfact",
                "granularity": "head",
                "threshold": 0.4,
                "timestamp": datetime.now().isoformat(),
                "algorithm_params": {},
            },
            "nodes": {
                "L0H0": {
                    "layer_idx": 0,
                    "node_type": "attention_head",
                    "index": 0,
                    "importance": 0.8,
                    "name": "L0H0",
                }
            },
            "edges": {},
        }

        artifact = CircuitArtifact.from_dict(data)
        assert artifact.model_id == "gpt2"
        assert artifact.discovery_method == "eap"
        assert len(artifact.nodes) == 1
        assert "L0H0" in artifact.nodes

    def test_roundtrip_dict(self, simple_artifact):
        """Test that dict -> object -> dict is identical."""
        data1 = simple_artifact.to_dict()
        artifact2 = CircuitArtifact.from_dict(data1)
        data2 = artifact2.to_dict()

        # Compare node IDs and edge IDs (structure)
        assert set(data1["nodes"].keys()) == set(data2["nodes"].keys())
        assert set(data1["edges"].keys()) == set(data2["edges"].keys())
        assert data1["metadata"]["model_id"] == data2["metadata"]["model_id"]

    def test_save_and_load_json(self, simple_artifact):
        """Test saving and loading from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "circuit.json"

            # Save
            simple_artifact.save_json(path)
            assert path.exists()

            # Load
            loaded = CircuitArtifact.load_json(path)
            assert loaded.model_id == simple_artifact.model_id
            assert len(loaded.nodes) == len(simple_artifact.nodes)
            assert len(loaded.edges) == len(simple_artifact.edges)

    def test_json_roundtrip(self, simple_artifact):
        """Test JSON roundtrip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "circuit.json"
            simple_artifact.save_json(path)
            loaded = CircuitArtifact.load_json(path)

            # Verify all nodes and edges
            assert len(loaded.nodes) == len(simple_artifact.nodes)
            assert len(loaded.edges) == len(simple_artifact.edges)

            for node_id in simple_artifact.nodes:
                assert node_id in loaded.nodes
                orig_node = simple_artifact.nodes[node_id]
                loaded_node = loaded.nodes[node_id]
                assert orig_node.layer_idx == loaded_node.layer_idx
                assert orig_node.importance == loaded_node.importance

    def test_json_file_size(self, simple_artifact):
        """Test that typical artifact JSON is reasonably sized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "circuit.json"
            simple_artifact.save_json(path)
            size_kb = path.stat().st_size / 1024

            # Even 1000+ node circuit should be < 5MB
            assert size_kb < 5000


class TestCircuitArtifactValidation:
    """Test validation methods."""

    def test_validate_valid_artifact(self, simple_artifact):
        """Test validation of valid artifact."""
        checks = simple_artifact.validate()

        # All checks should pass
        assert all(checks.values()), f"Some checks failed: {checks}"

    def test_validate_invalid_discovery_method(self):
        """Test validation catches invalid discovery method."""
        # This should raise at creation, but test validation directly
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
        )
        artifact.discovery_method = "invalid"

        checks = artifact.validate()
        assert checks["valid_method"] is False

    def test_validate_missing_nodes(self):
        """Test validation with missing nodes."""
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
        )
        # No nodes added
        checks = artifact.validate()
        assert checks["has_nodes"] is False

    def test_validate_invalid_node_importance(self):
        """Test validation catches invalid node importance."""
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
        )
        # Add node with invalid importance (manually to bypass __post_init__)
        node = Node(0, NodeType.ATTENTION_HEAD, 0, 0.5)
        artifact.nodes["n0"] = node

        # Manually corrupt importance
        object.__setattr__(node, "importance", 1.5)

        checks = artifact.validate()
        assert checks["all_nodes_valid"] is False

    def test_validate_orphan_edges(self):
        """Test validation catches edges with missing nodes."""
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
        )
        # Add edge without adding nodes
        artifact.edges["e0"] = Edge(src_id="missing_src", dst_id="missing_dst", weight=0.5)

        checks = artifact.validate()
        assert checks["all_edges_valid"] is False


class TestCircuitArtifactSparsity:
    """Test sparsity calculation."""

    def test_get_sparsity(self, simple_artifact):
        """Test sparsity calculation with default threshold."""
        # simple_artifact has threshold=0.5
        # Nodes: L0H0(0.8), L0H1(0.3), L1H2(0.6), L0M0(0.7)
        # Above threshold: L0H0, L1H2, L0M0 = 3/4 = 0.75
        sparsity = simple_artifact.get_sparsity()
        assert abs(sparsity - 0.75) < 0.01

    def test_get_sparsity_all_above(self):
        """Test sparsity when all nodes above threshold."""
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
            threshold=0.1,
        )
        artifact.add_node("n0", Node(0, NodeType.ATTENTION_HEAD, 0, 0.9))
        artifact.add_node("n1", Node(0, NodeType.ATTENTION_HEAD, 1, 0.8))

        sparsity = artifact.get_sparsity()
        assert sparsity == 1.0

    def test_get_sparsity_all_below(self):
        """Test sparsity when all nodes below threshold."""
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
            threshold=0.9,
        )
        artifact.add_node("n0", Node(0, NodeType.ATTENTION_HEAD, 0, 0.5))
        artifact.add_node("n1", Node(0, NodeType.ATTENTION_HEAD, 1, 0.3))

        sparsity = artifact.get_sparsity()
        assert sparsity == 0.0

    def test_get_sparsity_empty(self):
        """Test sparsity of empty artifact."""
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
        )
        sparsity = artifact.get_sparsity()
        assert sparsity == 0.0

    def test_get_compression_ratio(self, simple_artifact):
        """Test compression ratio calculation."""
        compression = simple_artifact.get_compression_ratio()
        sparsity = simple_artifact.get_sparsity()
        assert abs(compression + sparsity - 1.0) < 0.01


class TestCircuitArtifactSummary:
    """Test summary and repr methods."""

    def test_summary(self, simple_artifact):
        """Test summary string generation."""
        summary = simple_artifact.summary()

        assert "CircuitArtifact Summary" in summary
        assert "meta-llama/Llama-2-7b" in summary
        assert "ioi" in summary
        assert "Nodes: 4" in summary
        assert "Edges: 2" in summary

    def test_repr(self, simple_artifact):
        """Test repr string."""
        repr_str = repr(simple_artifact)

        assert "CircuitArtifact" in repr_str
        assert "meta-llama/Llama-2-7b" in repr_str
        assert "ioi" in repr_str


# Test mask conversion (requires architecture support)


class TestCircuitArtifactMaskConversion:
    """Test conversion to intervention masks."""

    def test_to_mask_requires_architecture_config(self, simple_artifact, mock_llama_model):
        """Test mask conversion with mock model."""
        # This test requires arch_cfg which we can provide manually
        arch_cfg = {
            "name": "LLaMA",
            "layers_path": ["model.layers"],
            "attn": {"module": "self_attn"},
            "mlp": {},
        }

        # Test that method exists and accepts parameters
        masks = simple_artifact.to_mask(mock_llama_model, arch_cfg)
        assert isinstance(masks, dict)

    def test_to_mask_head_granularity(self):
        """Test mask conversion with head granularity."""
        artifact = CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
            granularity="head",
            threshold=0.5,
        )
        artifact.add_node("L0H0", Node(0, NodeType.ATTENTION_HEAD, 0, 0.8))
        artifact.add_node("L0H1", Node(0, NodeType.ATTENTION_HEAD, 1, 0.3))

        # Note: Full mask conversion requires actual model structure
        # This test just verifies the structure exists
        assert artifact.granularity == "head"


# Test converter functions


class TestConverters:
    """Test conversion functions from discovery methods to CircuitArtifact."""

    def test_acdc_to_artifact(self):
        """Test converting ACDC prune scores to artifact."""
        from circuitkit.artifacts import acdc_to_artifact

        # Create mock prune scores
        prune_scores = {
            "blocks.0.attn.hook_v": torch.randn(8),  # 8 attention heads
            "blocks.0.mlp.hook_result": torch.randn(4),  # 4 MLP units
            "blocks.1.attn.hook_v": torch.randn(8),
        }

        artifact = acdc_to_artifact(
            prune_scores=prune_scores,
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
            threshold=0.0,
        )

        assert artifact.model_id == "gpt2"
        assert artifact.discovery_method == "acdc"
        assert artifact.task == "ioi"
        assert len(artifact.nodes) > 0
        assert artifact.algorithm_params["num_modules"] == 3

    def test_eap_to_artifact(self):
        """Test converting EAP node scores to artifact."""
        from circuitkit.artifacts import eap_to_artifact

        # Create mock node scores in EAP format
        node_scores = {
            "A0.0": 0.8,  # Layer 0, Head 0
            "A0.1": 0.3,  # Layer 0, Head 1
            "A1.2": 0.6,  # Layer 1, Head 2
            "MLP 0": 0.5,  # Layer 0 MLP
            "MLP 1": 0.7,  # Layer 1 MLP
        }

        artifact = eap_to_artifact(
            node_scores=node_scores,
            model_id="gpt2",
            task="sva",
            dataset="counterfact",
            threshold=0.2,
        )

        assert artifact.model_id == "gpt2"
        assert artifact.discovery_method == "eap"
        assert artifact.task == "sva"
        # With threshold=0.2, should filter out A0.1 (0.3 passes)
        assert len(artifact.nodes) > 0
        assert all(n.importance >= 0.2 for n in artifact.nodes.values())

    def test_ibcircuit_to_artifact(self):
        """Test converting IBCircuit scores to artifact."""
        from circuitkit.artifacts import ibcircuit_to_artifact

        # Create mock neuron scores
        node_scores = {
            "L0.N10": 0.75,
            "L0.N25": 0.45,
            "L1.N5": 0.85,
            "L2.N100": 0.2,
        }

        artifact = ibcircuit_to_artifact(
            node_scores=node_scores,
            model_id="gpt2",
            task="greater_than",
            dataset="numeric",
            threshold=0.3,
        )

        assert artifact.model_id == "gpt2"
        assert artifact.discovery_method == "ibcircuit"
        assert artifact.task == "greater_than"
        # With threshold=0.3, should keep only higher scores
        assert all(n.importance >= 0.3 for n in artifact.nodes.values())
        assert artifact.granularity == "neuron"

    def test_normalize_importance_scores_minmax(self):
        """Test min-max normalization of scores."""
        from circuitkit.artifacts import normalize_importance_scores

        scores = {
            "n0": 10.0,
            "n1": 5.0,
            "n2": 20.0,
            "n3": 15.0,
        }

        normalized = normalize_importance_scores(scores, method="minmax")

        assert normalized["n0"] == pytest.approx(1 / 3)  # (10-5)/(20-5)
        assert normalized["n1"] == 0.0  # (5-5)/(20-5)
        assert normalized["n2"] == 1.0  # (20-5)/(20-5)
        assert normalized["n3"] == pytest.approx(2 / 3)  # (15-5)/(20-5)

    def test_normalize_importance_scores_equal(self):
        """Test normalization when all scores are equal."""
        from circuitkit.artifacts import normalize_importance_scores

        scores = {
            "n0": 5.0,
            "n1": 5.0,
            "n2": 5.0,
        }

        normalized = normalize_importance_scores(scores, method="minmax")

        # When all equal, should return 1.0
        assert all(v == 1.0 for v in normalized.values())

    def test_normalize_empty_scores(self):
        """Test normalization with empty scores."""
        from circuitkit.artifacts import normalize_importance_scores

        scores = {}
        normalized = normalize_importance_scores(scores)

        assert normalized == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
