"""
Unit tests for graph.py — Graph, Node, AttentionNode, MLPNode, LogitNode, Edge.

These tests do NOT require a real model or CUDA.
"""

import json
import tempfile

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers — import the module under test using the project's package path.
# Adjust the import path to match your actual package layout.
# ---------------------------------------------------------------------------
from circuitkit.backends.eap.graph import (  # noqa: adjust path as needed
    AttentionNode,
    Graph,
    LogitNode,
    MLPNode,
    Node,
)

TINY_CFG = {"n_layers": 2, "n_heads": 2, "d_model": 64, "d_mlp": 128}


# ===========================================================================
# 1. Construction & node counts
# ===========================================================================
class TestGraphConstruction:
    def test_node_count(self):
        g = Graph.from_model(TINY_CFG)
        n_layers, n_heads = TINY_CFG["n_layers"], TINY_CFG["n_heads"]
        # input + (n_layers * n_heads) attn + n_layers mlp + logits
        expected = 1 + n_layers * n_heads + n_layers + 1
        assert len(g.nodes) == expected

    def test_has_input_and_logits_nodes(self):
        g = Graph.from_model(TINY_CFG)
        assert "input" in g.nodes
        assert "logits" in g.nodes

    def test_attention_node_names(self):
        g = Graph.from_model(TINY_CFG)
        for layer in range(TINY_CFG["n_layers"]):
            for head in range(TINY_CFG["n_heads"]):
                assert f"a{layer}.h{head}" in g.nodes

    def test_mlp_node_names(self):
        g = Graph.from_model(TINY_CFG)
        for layer in range(TINY_CFG["n_layers"]):
            assert f"m{layer}" in g.nodes

    def test_edge_names_are_unique(self):
        g = Graph.from_model(TINY_CFG)
        names = list(g.edges.keys())
        assert len(names) == len(set(names))

    def test_scores_tensor_shape(self):
        g = Graph.from_model(TINY_CFG)
        assert g.scores.shape == (g.n_forward, g.n_backward)

    def test_in_graph_tensor_shape(self):
        g = Graph.from_model(TINY_CFG)
        assert g.in_graph.shape == (g.n_forward, g.n_backward)

    def test_nodes_in_graph_shape(self):
        g = Graph.from_model(TINY_CFG)
        assert g.nodes_in_graph.shape[0] == g.n_forward


# ===========================================================================
# 2. Node type properties
# ===========================================================================
class TestNodeTypes:
    def test_attention_node_has_head_attr(self):
        g = Graph.from_model(TINY_CFG)
        node = g.nodes["a0.h1"]
        assert isinstance(node, AttentionNode)
        assert node.head == 1
        assert node.layer == 0

    def test_attention_node_has_three_qkv_inputs(self):
        g = Graph.from_model(TINY_CFG)
        node = g.nodes["a0.h0"]
        assert node.qkv_inputs is not None
        assert len(node.qkv_inputs) == 3

    def test_mlp_node_mlp_out_hooks(self):
        g = Graph.from_model(TINY_CFG)
        node = g.nodes["m0"]
        assert isinstance(node, MLPNode)
        assert "hook_mlp_in" in node.in_hook
        assert "hook_mlp_out" in node.out_hook

    def test_mlp_node_post_act_hooks(self):
        cfg = dict(TINY_CFG, mlp_hook="post_act")
        g = Graph.from_model(cfg)
        node = g.nodes["m0"]
        assert "hook_post" in node.in_hook
        assert "hook_post" in node.out_hook

    def test_mlp_d_neuron_mlp_out(self):
        g = Graph.from_model(TINY_CFG)
        node = g.nodes["m0"]
        assert node.d_neuron == TINY_CFG["d_model"]

    def test_mlp_d_neuron_post_act(self):
        cfg = dict(TINY_CFG, mlp_hook="post_act")
        g = Graph.from_model(cfg)
        node = g.nodes["m0"]
        assert node.d_neuron == TINY_CFG["d_mlp"]

    def test_logit_node_always_in_graph(self):
        g = Graph.from_model(TINY_CFG)
        assert g.nodes["logits"].in_graph is True

    def test_logit_node_in_graph_setter_raises(self):
        g = Graph.from_model(TINY_CFG)
        with pytest.raises(ValueError):
            g.nodes["logits"].in_graph = False


# ===========================================================================
# 3. Index consistency
# ===========================================================================
class TestIndexing:
    def test_forward_indices_are_unique(self):
        g = Graph.from_model(TINY_CFG)
        indices = [
            g.forward_index(n, attn_slice=False)
            for n in g.nodes.values()
            if not isinstance(n, LogitNode)
        ]
        assert len(indices) == len(set(indices))

    def test_prev_index_less_than_n_forward(self):
        g = Graph.from_model(TINY_CFG)
        for node in g.nodes.values():
            idx = g.prev_index(node)
            assert 0 <= idx <= g.n_forward

    def test_backward_index_within_range(self):
        g = Graph.from_model(TINY_CFG)
        for node in g.nodes.values():
            if isinstance(node, AttentionNode):
                for qkv in "qkv":
                    idx = g.backward_index(node, qkv=qkv, attn_slice=False)
                    assert 0 <= idx < g.n_backward
            elif not isinstance(node, (LogitNode,)):
                pass  # input has no backward index


# ===========================================================================
# 4. Score and membership state
# ===========================================================================
class TestNodeScoreState:
    def test_node_score_none_when_not_initialised(self):
        g = Graph.from_model(TINY_CFG)
        # node_scores not enabled by default
        if g.nodes_scores is None:
            assert g.nodes["a0.h0"].score is None

    def test_set_node_score_raises_when_no_scores_tensor(self):
        g = Graph.from_model(TINY_CFG)
        if g.nodes_scores is None:
            with pytest.raises(RuntimeError):
                g.nodes["a0.h0"].score = 1.0

    def test_node_in_graph_reflects_tensor(self):
        g = Graph.from_model(TINY_CFG)
        node = g.nodes["a0.h0"]
        fwd_idx = g.forward_index(node, attn_slice=False)
        g.nodes_in_graph[fwd_idx] = True
        assert node.in_graph is True
        g.nodes_in_graph[fwd_idx] = False
        assert node.in_graph is False

    def test_edge_in_graph_reflects_tensor(self):
        g = Graph.from_model(TINY_CFG)
        edge = next(iter(g.edges.values()))
        edge.in_graph = True
        assert edge.in_graph is True
        edge.in_graph = False
        assert edge.in_graph is False

    def test_node_score_roundtrip_with_node_scores_enabled(self):
        g = Graph.from_model(TINY_CFG, node_scores=True)
        node = g.nodes["m1"]
        node.score = 3.14
        assert abs(node.score.item() - 3.14) < 1e-4


# ===========================================================================
# 5. Pruning methods
# ===========================================================================
class TestPruning:
    def _graph_with_scores(self):
        g = Graph.from_model(TINY_CFG)
        # Assign distinct non-zero scores to all edges
        for i, edge in enumerate(g.edges.values()):
            edge.score = float(i + 1)
        return g

    def test_apply_topn_keeps_exactly_n(self):
        g = self._graph_with_scores()
        n = 5
        g.apply_topn(n, prune=False)
        in_graph_count = sum(1 for e in g.edges.values() if e.in_graph)
        assert in_graph_count == n

    def test_apply_topn_zero_clears_all(self):
        g = self._graph_with_scores()
        g.apply_topn(0)
        assert all(not e.in_graph for e in g.edges.values())

    def test_apply_topn_exceeds_total_keeps_all(self):
        g = self._graph_with_scores()
        total = len(g.edges)
        g.apply_topn(total + 999, prune=False)
        in_graph_count = sum(1 for e in g.edges.values() if e.in_graph)
        assert in_graph_count == total

    def test_apply_threshold_keeps_above(self):
        g = self._graph_with_scores()
        threshold = 5.0
        g.apply_threshold(threshold)
        for edge in g.edges.values():
            if edge.in_graph:
                assert abs(edge.score.item()) >= threshold

    def test_apply_threshold_zero_keeps_all_nonzero(self):
        g = self._graph_with_scores()
        g.apply_threshold(0.0)
        # All edges have score >= 1, so all should be in graph
        assert all(e.in_graph for e in g.edges.values())

    def test_apply_topn_uses_absolute_scores(self):
        """Negative scores should be treated by magnitude."""
        g = Graph.from_model(TINY_CFG)
        edges = list(g.edges.values())
        edges[0].score = -100.0
        for i, e in enumerate(edges[1:], 1):
            e.score = float(i)
        g.apply_topn(1, prune=False)  # <-- Add prune=False
        assert edges[0].in_graph

    def test_all_equal_scores_no_crash(self):
        g = Graph.from_model(TINY_CFG)
        for edge in g.edges.values():
            edge.score = 1.0
        g.apply_topn(3)  # Should not raise


# ===========================================================================
# 6. Serialisation round-trips
# ===========================================================================
class TestSerialisation:
    def _graph_with_data(self):
        g = Graph.from_model(TINY_CFG)
        for i, edge in enumerate(g.edges.values()):
            edge.score = float(i) * 0.1
        g.apply_topn(4)
        return g

    def test_pt_roundtrip_scores(self):
        g = self._graph_with_data()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            g.to_pt(f.name)
            g2 = Graph.from_pt(f.name)
        assert torch.allclose(g.scores, g2.scores)

    def test_pt_roundtrip_in_graph(self):
        g = self._graph_with_data()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            g.to_pt(f.name)
            g2 = Graph.from_pt(f.name)
        assert torch.equal(g.in_graph, g2.in_graph)

    def test_pt_roundtrip_nodes_in_graph(self):
        g = self._graph_with_data()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            g.to_pt(f.name)
            g2 = Graph.from_pt(f.name)
        assert torch.equal(g.nodes_in_graph, g2.nodes_in_graph)

    def test_pt_missing_optional_keys_loads_fine(self):
        """from_pt must succeed when nodes_scores / neurons_* keys are absent."""
        g = self._graph_with_data()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            g.to_pt(f.name)
            d = torch.load(f.name)
            # Strip optional keys
            for key in ("nodes_scores", "neurons_in_graph", "neurons_scores"):
                d.pop(key, None)
            torch.save(d, f.name)
            g2 = Graph.from_pt(f.name)
        assert g2.nodes_scores is None

    def test_pt_shape_mismatch_raises(self):
        g = self._graph_with_data()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            g.to_pt(f.name)
            d = torch.load(f.name)
            # Corrupt the shape
            d["edges_in_graph"] = torch.zeros(1, 1, dtype=torch.bool)
            torch.save(d, f.name)
            with pytest.raises(ValueError, match="they must match"):
                Graph.from_pt(f.name)

    def test_json_roundtrip(self):
        g = self._graph_with_data()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
            g.to_json(f.name)
            g2 = Graph.from_json(f.name)
        # Edge in_graph flags should match
        for name, edge in g.edges.items():
            assert edge.in_graph == g2.edges[name].in_graph

    def test_json_missing_key_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"cfg": TINY_CFG, "nodes": {}}, f)  # missing 'edges'
            fname = f.name
        with pytest.raises(ValueError, match="missing required top-level keys"):
            Graph.from_json(fname)

    def test_json_no_node_scores_sets_none(self):
        """When no node has a 'score' field, nodes_scores should be None."""
        g = self._graph_with_data()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
            g.to_json(f.name)
            # Read back, strip 'score' from every node
            with open(f.name) as jf:
                d = json.load(jf)
            for node_dict in d["nodes"].values():
                node_dict.pop("score", None)
            with open(f.name, "w") as jf:
                json.dump(d, jf)
            g2 = Graph.from_json(f.name)
        assert g2.nodes_scores is None
