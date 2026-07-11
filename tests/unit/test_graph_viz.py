"""
Tests for CircuitGraphVisualizer.

Tests cover:
- Initialization with various graph structures
- Node and edge extraction
- Hierarchical layout computation
- HTML export
- Interactive widget creation
- Filtering and thresholding
- Data export
"""

import json
import tempfile
from pathlib import Path

import pytest

from circuitkit.artifacts.scores import CircuitScores
from circuitkit.visualize.graph_viz import CircuitGraphVisualizer


def _expected_derived_edges(nodes):
    """Count the edges the visualizer derives from the node structure.

    The visualizer ignores any ``"edges"`` supplied in the input graph and
    instead connects every in-circuit node to every node in the immediately
    following layer. So the edge count is ``sum_L (|layer L| * |layer L+1|)``.
    """
    from collections import Counter

    per_layer = Counter(nd["layer"] for nd in nodes.values())
    return sum(per_layer[layer] * per_layer[layer + 1] for layer in per_layer)


class TestCircuitGraphVisualizer:
    """Test suite for CircuitGraphVisualizer."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample CircuitScores for testing."""
        return CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={
                "A0.0": 0.9,
                "A0.1": 0.8,
                "A1.0": 0.7,
                "A1.1": 0.6,
                "MLP 0": 0.5,
                "MLP 1": 0.4,
                "A2.0": 0.3,
                "A2.1": 0.2,
                "MLP 2": 0.1,
            },
            timestamp=CircuitScores.create_timestamp(),
        )

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph structure."""
        return {
            "nodes": {
                "A0.0": {"layer": 0, "type": "attn_head"},
                "A0.1": {"layer": 0, "type": "attn_head"},
                "A1.0": {"layer": 1, "type": "attn_head"},
                "A1.1": {"layer": 1, "type": "attn_head"},
                "MLP 0": {"layer": 0, "type": "mlp"},
                "MLP 1": {"layer": 1, "type": "mlp"},
                "A2.0": {"layer": 2, "type": "attn_head"},
                "A2.1": {"layer": 2, "type": "attn_head"},
                "MLP 2": {"layer": 2, "type": "mlp"},
            },
            "edges": [
                ("A0.0", "A1.0"),
                ("A0.1", "A1.1"),
                ("MLP 0", "A1.0"),
                ("A1.0", "A2.0"),
                ("A1.1", "A2.1"),
                ("MLP 1", "MLP 2"),
            ],
        }

    @pytest.fixture
    def edge_scores(self):
        """Create sample edge scores."""
        return {
            ("A0.0", "A1.0"): 0.95,
            ("A0.1", "A1.1"): 0.85,
            ("MLP 0", "A1.0"): 0.75,
            ("A1.0", "A2.0"): 0.65,
            ("A1.1", "A2.1"): 0.55,
            ("MLP 1", "MLP 2"): 0.45,
        }

    def test_initialization(self, sample_graph, sample_scores, edge_scores):
        """Test basic initialization."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores, edge_scores)

        assert viz.scores == sample_scores
        assert len(viz.nodes) == 9
        # Edges are derived from the node structure, not the input graph's
        # "edges" list (3 layers of 3 nodes -> 3*3 + 3*3 = 18).
        assert len(viz.edges) == _expected_derived_edges(viz.nodes)
        assert len(viz.node_scores) == 9

    def test_node_extraction(self, sample_scores):
        """Test node extraction from scores only."""
        graph = {"nodes": {}, "edges": []}
        viz = CircuitGraphVisualizer(graph, sample_scores)

        # Should extract all nodes from scores
        assert len(viz.nodes) == 9
        assert "A0.0" in viz.nodes
        assert "MLP 2" in viz.nodes

    def test_node_metadata_parsing(self, sample_scores):
        """Test parsing of node names."""
        graph = {"nodes": {}, "edges": []}
        viz = CircuitGraphVisualizer(graph, sample_scores)

        # Check attention head parsing
        assert viz.nodes["A0.0"]["type"] == "attn_head"
        assert viz.nodes["A0.0"]["layer"] == 0

        # Check MLP parsing
        assert viz.nodes["MLP 1"]["type"] == "mlp"
        assert viz.nodes["MLP 1"]["layer"] == 1

    def test_node_types(self, sample_graph, sample_scores):
        """Test node type inference."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores)

        assert viz.node_types["A0.0"] == "attn_head"
        assert viz.node_types["MLP 0"] == "mlp"

    def test_layout_hierarchy(self, sample_graph, sample_scores):
        """Test hierarchical layout computation."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores)
        layout = viz.layout_hierarchy()

        # Check all nodes have positions
        assert len(layout) == 9

        # Check x positions increase by layer
        a0_x = layout["A0.0"][0]
        a1_x = layout["A1.0"][0]
        a2_x = layout["A2.0"][0]

        assert a0_x < a1_x < a2_x

        # Check y positions are different within layer
        a0_0_y = layout["A0.0"][1]
        a0_1_y = layout["A0.1"][1]
        assert a0_0_y != a0_1_y

    def test_to_html(self, sample_graph, sample_scores, edge_scores):
        """Test HTML export."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores, edge_scores)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "circuit.html"
            html_string = viz.to_html(str(output_path))

            # Check file was created
            assert output_path.exists()

            # Check HTML content
            with open(output_path) as f:
                content = f.read()
                assert "d3" in content.lower()
                assert "A0.0" in content or "scatter" in content.lower()

            # Check return value is HTML string
            assert isinstance(html_string, str)
            assert "d3" in html_string.lower()

    def test_html_with_threshold(self, sample_graph, sample_scores):
        """Test HTML export with threshold filtering."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "circuit_filtered.html"
            viz.to_html(str(output_path), threshold=0.5)

            # File should exist
            assert output_path.exists()

    def test_html_custom_title(self, sample_graph, sample_scores):
        """Test HTML export with custom title."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "circuit.html"
            custom_title = "My Custom Circuit"
            viz.to_html(str(output_path), title=custom_title)

            with open(output_path) as f:
                content = f.read()
                # Title should appear in HTML
                assert custom_title in content

    def test_node_color_mapping(self, sample_graph, sample_scores):
        """Test node color mapping by type."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores)

        attn_color = viz._get_node_color("attn_head")
        mlp_color = viz._get_node_color("mlp")
        residual_color = viz._get_node_color("residual")

        assert attn_color != mlp_color
        assert mlp_color != residual_color

    def test_node_degree_stats(self, sample_graph, sample_scores):
        """Test node degree computation."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores)
        degrees = viz.get_node_degree_stats()

        # Check structure
        assert "A0.0" in degrees
        assert "in_degree" in degrees["A0.0"]
        assert "out_degree" in degrees["A0.0"]

        # Check values. Edges are derived, so A0.0 (layer 0) connects to every
        # layer-1 node (A1.0, A1.1, MLP 1) -> out_degree 3, and A1.0 receives
        # from every layer-0 node -> in_degree 3.
        assert degrees["A0.0"]["out_degree"] == 3
        assert degrees["A1.0"]["in_degree"] == 3

    def test_top_k_nodes(self, sample_graph, sample_scores):
        """Test top-k node extraction."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores)

        top_3 = viz.get_top_k_nodes(k=3)
        assert len(top_3) == 3

        # Check ordering (should be descending)
        scores = list(top_3.values())
        assert scores[0] >= scores[1] >= scores[2]

        # Check highest score is indeed A0.0
        assert list(top_3.keys())[0] == "A0.0"

    def test_filter_by_threshold(self, sample_graph, sample_scores):
        """Test graph filtering by threshold."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores)

        filtered_viz = viz.filter_by_threshold(threshold=0.5)

        # Should keep high-scoring nodes
        assert "A0.0" in filtered_viz.node_scores  # score 0.9
        assert "MLP 2" not in filtered_viz.node_scores  # score 0.1

    def test_filter_by_threshold_export(self, sample_graph, sample_scores):
        """Test filtering with immediate export."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "filtered.html"
            viz.filter_by_threshold(
                threshold=0.5, output_path=str(output_path), title="Filtered Circuit"
            )

            assert output_path.exists()

    def test_export_graph_data(self, sample_graph, sample_scores, edge_scores):
        """Test JSON export of graph data."""
        viz = CircuitGraphVisualizer(sample_graph, sample_scores, edge_scores)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "graph_data.json"
            viz.export_graph_data(str(output_path))

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            # Check structure
            assert "metadata" in data
            assert "nodes" in data
            assert "edges" in data

            # Check metadata
            assert data["metadata"]["task"] == "ioi"
            assert data["metadata"]["algorithm"] == "eap"

            # Check nodes
            assert len(data["nodes"]) == 9
            assert data["nodes"][0]["id"] in viz.nodes

            # Edges are derived from the node structure (see test_initialization).
            assert len(data["edges"]) == _expected_derived_edges(viz.nodes)

    def test_empty_graph(self, sample_scores):
        """Test with empty graph."""
        graph = {"nodes": {}, "edges": []}
        viz = CircuitGraphVisualizer(graph, sample_scores)

        # Should still work with nodes from scores
        assert len(viz.nodes) == len(sample_scores.node_scores)

    def test_input_edges_are_ignored(self, sample_scores):
        """Edges supplied in the input graph are ignored; edges are derived
        from the scored node structure instead (see the class docstring on
        CircuitGraphVisualizer)."""
        graph = {"nodes": {}, "edges": [("A0.0", "A1.0"), ("A1.0", "A2.0")]}
        viz = CircuitGraphVisualizer(graph, sample_scores)

        # Not the 2 input tuples — the full derived adjacent-layer edge set.
        assert len(viz.edges) == _expected_derived_edges(viz.nodes)
        assert len(viz.edges) > 2

    def test_normalize_scores(self, sample_scores):
        """Test score normalization."""
        viz = CircuitGraphVisualizer({"nodes": {}, "edges": []}, sample_scores)

        # Normalized scores should be in [0, 1]
        assert all(0 <= s <= 1 for s in viz.node_scores.values())

        # Max should be 1.0, min should be 0.0
        assert max(viz.node_scores.values()) == 1.0
        assert min(viz.node_scores.values()) == 0.0

    def test_layout_with_various_layer_counts(self):
        """Test layout with different numbers of layers."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={f"A{i}.0": 1.0 / (i + 1) for i in range(5)},
            timestamp=CircuitScores.create_timestamp(),
        )

        graph = {"nodes": {}, "edges": []}
        viz = CircuitGraphVisualizer(graph, scores)
        layout = viz.layout_hierarchy()

        assert len(layout) == 5

    def test_visualization_with_special_characters(self):
        """Test visualization with nodes having special names."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={
                "node_1": 0.5,
                "node_2": 0.3,
            },
            timestamp=CircuitScores.create_timestamp(),
        )

        graph = {
            "nodes": {
                "node_1": {"layer": 0, "type": "unknown"},
                "node_2": {"layer": 1, "type": "unknown"},
            },
            "edges": [("node_1", "node_2")],
        }

        viz = CircuitGraphVisualizer(graph, scores)
        assert len(viz.nodes) == 2
        assert "node_1" in viz.node_scores


class TestCircuitGraphVisualizerIntegration:
    """Integration tests with realistic circuits."""

    def test_realistic_transformer_circuit(self):
        """Test with realistic transformer circuit structure."""
        # Build a 3-layer transformer circuit
        nodes_dict = {}
        edges = []

        for layer in range(3):
            # Attention heads
            for head in range(8):
                node_name = f"A{layer}.{head}"
                nodes_dict[node_name] = {"layer": layer, "type": "attn_head"}

                # Connect to next layer
                if layer < 2:
                    next_head = head % 8
                    edges.append((node_name, f"A{layer + 1}.{next_head}"))

            # MLPs
            node_name = f"MLP {layer}"
            nodes_dict[node_name] = {"layer": layer, "type": "mlp"}

            if layer < 2:
                edges.append((node_name, f"MLP {layer + 1}"))

        # Create scores
        all_nodes = list(nodes_dict.keys())
        node_scores = {
            node: (len(all_nodes) - i) / len(all_nodes) for i, node in enumerate(all_nodes)
        }

        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores=node_scores,
            timestamp=CircuitScores.create_timestamp(),
        )

        graph = {"nodes": nodes_dict, "edges": edges}
        viz = CircuitGraphVisualizer(graph, scores)

        # Verify visualization can be created
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "realistic_circuit.html"
            viz.to_html(str(output_path))
            assert output_path.exists()
