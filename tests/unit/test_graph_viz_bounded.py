"""Tests for bounded circuit graph export (PR-L1): max_nodes/max_edges/
edge_threshold on ``_build_graph_data`` and ``CircuitGraphVisualizer.to_json``.
"""

import json
import tempfile
from pathlib import Path

from circuitkit.artifacts.scores import CircuitScores
from circuitkit.visualize.graph_viz import CircuitGraphVisualizer, _build_graph_data


def _make_gpt2_style_scores() -> CircuitScores:
    """156-node synthetic graph matching GPT-2 node-level discovery shape
    (12 layers x (12 heads + 1 MLP) = 156 nodes)."""
    node_scores = {}
    i = 0
    for layer in range(12):
        for head in range(12):
            i += 1
            node_scores[f"A{layer}.{head}"] = 1.0 / i
        i += 1
        node_scores[f"MLP {layer}"] = 1.0 / i
    return CircuitScores(
        task="ioi",
        model="gpt2",
        algorithm="eap",
        level="node",
        node_scores=node_scores,
        timestamp=CircuitScores.create_timestamp(),
    )


def _make_large_scores(n_layers: int = 100, heads_per_layer: int = 12) -> CircuitScores:
    """A 1,200+ node synthetic graph, well beyond GPT-2 scale."""
    node_scores = {}
    i = 0
    for layer in range(n_layers):
        for head in range(heads_per_layer):
            i += 1
            node_scores[f"A{layer}.{head}"] = 1.0 / i
    return CircuitScores(
        task="ioi",
        model="synthetic-large",
        algorithm="eap",
        level="node",
        node_scores=node_scores,
        timestamp=CircuitScores.create_timestamp(),
    )


class TestSmallGraphNoRegression:
    def test_small_graph_identical(self):
        """Default args (no bounding) reproduce the unbounded node/edge
        counts and metadata exactly as before this change."""
        scores = _make_gpt2_style_scores()
        assert len(scores.node_scores) == 156

        unbounded = _build_graph_data(scores)

        assert len(unbounded.nodes) == 156
        # 12 layers of 13 nodes each (12 heads + 1 MLP); edges connect every
        # node in layer L to every node in layer L+1, for the 11 adjacent pairs.
        expected_edges = 11 * (13 * 13)
        assert len(unbounded.edges) == expected_edges
        assert unbounded.metadata["n_nodes_hidden"] == 0
        assert unbounded.metadata["n_edges_hidden"] == 0
        assert unbounded.metadata["n_total_nodes"] == 156


class TestLargeGraphBounded:
    def test_large_graph_bounded(self):
        scores = _make_large_scores()
        assert len(scores.node_scores) == 1200

        bounded = _build_graph_data(scores, max_nodes=300, max_edges=3000)

        assert len(bounded.nodes) <= 300
        assert len(bounded.edges) <= 3000
        assert bounded.metadata["n_nodes_hidden"] > 0
        assert bounded.metadata["n_edges_hidden"] > 0
        assert bounded.metadata["n_nodes_hidden"] == 1200 - len(bounded.nodes)

    def test_max_nodes_keeps_highest_scored(self):
        scores = _make_large_scores(n_layers=5, heads_per_layer=10)
        bounded = _build_graph_data(scores, max_nodes=10)

        kept_names = {nd.name for nd in bounded.nodes}
        top_10_by_score = sorted(
            scores.node_scores.keys(),
            key=lambda n: (-abs(scores.node_scores[n]), n),
        )[:10]
        assert kept_names == set(top_10_by_score)

    def test_edge_threshold_filters_weak_edges(self):
        scores = _make_gpt2_style_scores()
        unfiltered = _build_graph_data(scores)
        filtered = _build_graph_data(scores, edge_threshold=0.9)

        assert len(filtered.edges) < len(unfiltered.edges)
        assert all(e.normalized_weight >= 0.9 for e in filtered.edges)
        assert filtered.metadata["n_edges_hidden"] == (len(unfiltered.edges) - len(filtered.edges))

    def test_max_edges_caps_and_keeps_strongest(self):
        scores = _make_gpt2_style_scores()
        bounded = _build_graph_data(scores, max_edges=50)

        assert len(bounded.edges) == 50
        weights = [e.normalized_weight for e in bounded.edges]
        assert weights == sorted(weights, reverse=True)


class TestToJsonSize:
    def test_to_json_size(self):
        """Bounded graph.json for a large graph stays well under 2 MB."""
        scores = _make_large_scores()
        viz = CircuitGraphVisualizer({"nodes": {}, "edges": []}, scores)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "graph.json"
            data = viz.to_json(str(out_path), max_nodes=300, max_edges=3000)

            assert out_path.exists()
            size_bytes = out_path.stat().st_size
            assert size_bytes < 2 * 1024 * 1024

            with open(out_path) as f:
                reloaded = json.load(f)
            assert reloaded == data
            assert len(reloaded["nodes"]) <= 300
            assert len(reloaded["edges"]) <= 3000
            assert reloaded["metadata"]["n_nodes_hidden"] > 0

    def test_to_json_unbounded_matches_export_graph_data(self):
        """With no bounds, to_json produces the same payload shape as the
        existing (unbounded) export_graph_data — no regression."""
        scores = _make_gpt2_style_scores()
        viz = CircuitGraphVisualizer({"nodes": {}, "edges": []}, scores)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "bounded.json"
            legacy_path = Path(tmpdir) / "legacy.json"

            bounded_data = viz.to_json(str(json_path))
            viz.export_graph_data(str(legacy_path))
            with open(legacy_path) as f:
                legacy_data = json.load(f)

            assert len(bounded_data["nodes"]) == len(legacy_data["nodes"])
            assert len(bounded_data["edges"]) == len(legacy_data["edges"])
