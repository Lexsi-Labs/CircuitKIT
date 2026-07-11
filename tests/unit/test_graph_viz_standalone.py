"""
Standalone tests for CircuitGraphVisualizer.
"""

import json
import sys
import tempfile
from pathlib import Path

from circuitkit.artifacts.scores import CircuitScores
from circuitkit.visualize.graph_viz import CircuitGraphVisualizer


def test_basic_initialization():
    """Test basic initialization with sample graph."""
    print("TEST: Basic Initialization")

    scores = CircuitScores(
        task="ioi",
        model="gpt2",
        algorithm="eap",
        level="node",
        node_scores={"A0.0": 0.9, "A1.0": 0.7, "MLP 0": 0.5},
        timestamp=CircuitScores.create_timestamp(),
    )

    graph = {
        "nodes": {"A0.0": {"layer": 0}, "A1.0": {"layer": 1}, "MLP 0": {"layer": 0}},
        "edges": [("A0.0", "A1.0"), ("MLP 0", "A1.0")],
    }

    viz = CircuitGraphVisualizer(graph, scores)

    assert len(viz.nodes) == 3
    assert len(viz.edges) == 2
    assert len(viz.node_scores) == 3

    # Check normalization
    assert max(viz.node_scores.values()) == 1.0
    assert min(viz.node_scores.values()) == 0.0

    print("  PASSED: Created visualizer with 3 nodes and 2 edges")


def test_node_metadata_parsing():
    """Test parsing of node names."""
    print("TEST: Node Metadata Parsing")

    scores = CircuitScores(
        task="test",
        model="test",
        algorithm="eap",
        level="node",
        node_scores={
            "A0.1": 0.9,
            "A5.7": 0.8,
            "MLP 2": 0.7,
            "MLP 10": 0.6,
        },
        timestamp=CircuitScores.create_timestamp(),
    )

    graph = {"nodes": {}, "edges": []}
    viz = CircuitGraphVisualizer(graph, scores)

    # Check attention head parsing
    assert viz.nodes["A0.1"]["type"] == "attn_head"
    assert viz.nodes["A0.1"]["layer"] == 0
    assert viz.nodes["A0.1"]["head"] == 1

    assert viz.nodes["A5.7"]["layer"] == 5
    assert viz.nodes["A5.7"]["head"] == 7

    # Check MLP parsing
    assert viz.nodes["MLP 2"]["type"] == "mlp"
    assert viz.nodes["MLP 2"]["layer"] == 2

    assert viz.nodes["MLP 10"]["layer"] == 10

    print("  PASSED: Correctly parsed all node names")


def test_layout_hierarchy():
    """Test hierarchical layout computation."""
    print("TEST: Layout Hierarchy")

    scores = CircuitScores(
        task="test",
        model="test",
        algorithm="eap",
        level="node",
        node_scores={
            "A0.0": 0.9,
            "A0.1": 0.8,
            "A1.0": 0.7,
            "A1.1": 0.6,
            "A2.0": 0.5,
        },
        timestamp=CircuitScores.create_timestamp(),
    )

    graph = {"nodes": {}, "edges": []}
    viz = CircuitGraphVisualizer(graph, scores)
    layout = viz.layout_hierarchy()

    # Check all nodes have positions
    assert len(layout) == 5

    # Check x positions increase by layer
    a0_x = layout["A0.0"][0]
    a1_x = layout["A1.0"][0]
    a2_x = layout["A2.0"][0]

    assert a0_x < a1_x < a2_x
    print(f"  PASSED: Correct layer ordering (x: {a0_x:.1f} < {a1_x:.1f} < {a2_x:.1f})")

    # Check y positions differ within layer
    a0_0_y = layout["A0.0"][1]
    a0_1_y = layout["A0.1"][1]
    assert a0_0_y != a0_1_y
    print("  PASSED: Nodes within layer have different y positions")


def test_html_export():
    """Test HTML export."""
    print("TEST: HTML Export")

    scores = CircuitScores(
        task="ioi",
        model="gpt2",
        algorithm="eap",
        level="node",
        node_scores={"A0.0": 0.9, "A1.0": 0.7, "MLP 0": 0.5},
        timestamp=CircuitScores.create_timestamp(),
    )

    graph = {
        "nodes": {"A0.0": {"layer": 0}, "A1.0": {"layer": 1}, "MLP 0": {"layer": 0}},
        "edges": [("A0.0", "A1.0"), ("MLP 0", "A1.0")],
    }

    viz = CircuitGraphVisualizer(graph, scores)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "circuit.html"
        html_string = viz.to_html(str(output_path))

        # Check file exists
        assert output_path.exists()
        print(f"  PASSED: HTML file created at {output_path}")

        # Check content
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
            assert "d3" in content.lower()
            print(f"  PASSED: HTML contains plotly content ({len(content)} bytes)")

        # Check return value
        assert isinstance(html_string, str)
        assert "d3" in html_string.lower()
        print("  PASSED: Function returned HTML string")


def test_top_k_nodes():
    """Test top-k node selection."""
    print("TEST: Top-K Nodes")

    scores = CircuitScores(
        task="test",
        model="test",
        algorithm="eap",
        level="node",
        node_scores={
            "A0.0": 0.9,
            "A0.1": 0.8,
            "A1.0": 0.7,
            "A1.1": 0.6,
            "A2.0": 0.5,
        },
        timestamp=CircuitScores.create_timestamp(),
    )

    graph = {"nodes": {}, "edges": []}
    viz = CircuitGraphVisualizer(graph, scores)

    top_3 = viz.get_top_k_nodes(k=3)
    assert len(top_3) == 3

    # Check ordering
    scores_list = list(top_3.values())
    assert scores_list[0] >= scores_list[1] >= scores_list[2]

    # Check highest is A0.0
    assert list(top_3.keys())[0] == "A0.0"
    print(f"  PASSED: Top 3 nodes: {list(top_3.keys())}")


def test_threshold_filtering():
    """Test graph filtering by threshold."""
    print("TEST: Threshold Filtering")

    scores = CircuitScores(
        task="test",
        model="test",
        algorithm="eap",
        level="node",
        node_scores={
            "A0.0": 0.9,
            "A1.0": 0.7,
            "A2.0": 0.5,
            "A3.0": 0.3,
            "A4.0": 0.1,
        },
        timestamp=CircuitScores.create_timestamp(),
    )

    graph = {"nodes": {}, "edges": []}
    viz = CircuitGraphVisualizer(graph, scores)

    # Filter at 0.5 threshold
    filtered = viz.filter_by_threshold(threshold=0.5)

    # Should keep high-scoring nodes
    assert "A0.0" in filtered.node_scores  # 0.9 >= 0.5
    assert "A1.0" in filtered.node_scores  # 0.7 >= 0.5
    assert "A2.0" in filtered.node_scores  # 0.5 >= 0.5

    # Should drop low-scoring nodes
    assert "A3.0" not in filtered.node_scores  # 0.3 < 0.5
    assert "A4.0" not in filtered.node_scores  # 0.1 < 0.5

    print(f"  PASSED: Threshold 0.5 kept {len(filtered.node_scores)} nodes")


def test_graph_data_export():
    """Test JSON export of graph data."""
    print("TEST: Graph Data Export")

    scores = CircuitScores(
        task="ioi",
        model="gpt2",
        algorithm="eap",
        level="node",
        node_scores={"A0.0": 0.9, "A1.0": 0.7, "MLP 0": 0.5},
        timestamp=CircuitScores.create_timestamp(),
    )

    graph = {
        "nodes": {"A0.0": {"layer": 0}, "A1.0": {"layer": 1}, "MLP 0": {"layer": 0}},
        "edges": [("A0.0", "A1.0"), ("MLP 0", "A1.0")],
    }

    viz = CircuitGraphVisualizer(graph, scores)

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

        assert data["metadata"]["task"] == "ioi"
        assert data["metadata"]["algorithm"] == "eap"

        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2

        print(
            f"  PASSED: Exported graph with {len(data['nodes'])} nodes and {len(data['edges'])} edges"
        )


def test_node_degree_stats():
    """Test node degree computation."""
    print("TEST: Node Degree Stats")

    scores = CircuitScores(
        task="test",
        model="test",
        algorithm="eap",
        level="node",
        node_scores={"A0.0": 0.9, "A1.0": 0.7, "A2.0": 0.5},
        timestamp=CircuitScores.create_timestamp(),
    )

    graph = {
        "nodes": {
            "A0.0": {"layer": 0},
            "A1.0": {"layer": 1},
            "A2.0": {"layer": 2},
        },
        "edges": [
            ("A0.0", "A1.0"),
            ("A0.0", "A2.0"),
            ("A1.0", "A2.0"),
        ],
    }

    viz = CircuitGraphVisualizer(graph, scores)
    degrees = viz.get_node_degree_stats()

    # Edges are derived from the node layers (input "edges" is ignored): one
    # node per layer means exactly one adjacent-layer edge on each side.
    assert degrees["A0.0"]["out_degree"] == 1
    assert degrees["A1.0"]["in_degree"] == 1
    assert degrees["A1.0"]["out_degree"] == 1
    assert degrees["A2.0"]["in_degree"] == 1

    print(f"  PASSED: Computed degrees for {len(degrees)} nodes")


def test_realistic_circuit():
    """Test with realistic 3-layer transformer circuit."""
    print("TEST: Realistic 3-Layer Transformer Circuit")

    # Build 3-layer circuit
    nodes_dict = {}
    edges = []
    node_scores = {}

    score_idx = 0
    for layer in range(3):
        for head in range(8):
            node_name = f"A{layer}.{head}"
            nodes_dict[node_name] = {"layer": layer, "type": "attn_head"}
            node_scores[node_name] = 1.0 - (score_idx * 0.01)
            score_idx += 1

            if layer < 2:
                edges.append((node_name, f"A{layer + 1}.{head % 8}"))

        node_name = f"MLP {layer}"
        nodes_dict[node_name] = {"layer": layer, "type": "mlp"}
        node_scores[node_name] = 1.0 - (score_idx * 0.01)
        score_idx += 1

        if layer < 2:
            edges.append((node_name, f"MLP {layer + 1}"))

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

    # Verify structure
    assert len(viz.nodes) == 3 * 8 + 3  # 24 heads + 3 MLPs
    # Edges are derived, not from the input "edges": 9 nodes per layer, each
    # connecting to all 9 nodes in the next layer, across 2 layer transitions.
    assert len(viz.edges) == 9 * 9 * 2  # 162 edges

    # Test layout
    layout = viz.layout_hierarchy()
    assert len(layout) == 27

    # Test HTML export
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "realistic.html"
        viz.to_html(str(output_path))
        assert output_path.exists()

    print(f"  PASSED: Created realistic circuit with {len(viz.nodes)} nodes")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CircuitGraphVisualizer Standalone Tests")
    print("=" * 70 + "\n")

    tests = [
        test_basic_initialization,
        test_node_metadata_parsing,
        test_layout_hierarchy,
        test_html_export,
        test_top_k_nodes,
        test_threshold_filtering,
        test_graph_data_export,
        test_node_degree_stats,
        test_realistic_circuit,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
            import traceback

            traceback.print_exc()
        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
