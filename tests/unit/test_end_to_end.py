"""
End-to-end integration tests for the EAP/EAP-IG pipeline.

All tests require CUDA.  They exercise the full chain:
  graph construction → attribution → pruning → serialisation → adapter.
"""

import re
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

TINY_CFG = {"n_layers": 2, "n_heads": 2, "d_model": 64, "d_mlp": 128}

CLEAN_TEXTS = [
    "The cat sat on the mat",
    "A dog ran in the park",
    "She opened the red door",
    "He read the old book",
]
CORRUPTED_TEXTS = [
    "The cat sat on a hat",
    "A dog ran in the yard",
    "She closed the blue door",
    "He wrote the new book",
]
LABELS = [0, 1, 0, 1]

_CK_NAME_RE = re.compile(r"^(A\d+\.\d+|MLP \d+)$")


# ===========================================================================
# Helpers
# ===========================================================================


def _make_dataloader(batch_size=2):
    def collate(xs):
        c, r, lyr = zip(*xs)
        return list(c), list(r), torch.tensor(lyr)

    dataset = list(zip(CLEAN_TEXTS, CORRUPTED_TEXTS, LABELS))
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


def _simple_metric(logits, clean_logits, input_lengths, labels):
    """Differentiable scalar — last-token mean logit."""
    last = input_lengths - 1

    # .mean(-1) reduces the vocab dimension.
    # .sum() reduces the batch dimension to a single scalar for backprop.
    return logits[torch.arange(len(last)), last].mean(-1).sum()


# ===========================================================================
# 1. Full EAP edge pipeline
# ===========================================================================
class TestFullEAPPipeline:
    @requires_cuda
    def test_eap_apply_topn_gives_correct_count(self, tiny_model):
        from circuitkit.backends.eap.attribute import attribute
        from circuitkit.backends.eap.graph import Graph

        g = Graph.from_model(TINY_CFG)
        dl = _make_dataloader()
        attribute(tiny_model, g, dl, _simple_metric, method="EAP", quiet=True)

        n = 5
        # Disable pruning so disconnected edges aren't deleted
        g.apply_topn(n, prune=False)

        # Count how many edges were actually kept
        in_graph_count = sum(1 for e in g.edges.values() if e.in_graph)

        # Calculate how many edges actually had non-zero attribution scores
        non_zero_edges = (g.scores.abs() > 1e-7).sum().item()

        # The graph should contain n edges, unless there were fewer than n non-zero edges available
        expected_count = min(n, non_zero_edges)

        assert (
            in_graph_count == expected_count
        ), f"Expected {expected_count} edges, but got {in_graph_count}"

    @requires_cuda
    def test_eap_ig_pipeline_scores_finite(self, tiny_model):
        from circuitkit.backends.eap.attribute import attribute
        from circuitkit.backends.eap.graph import Graph

        g = Graph.from_model(TINY_CFG)
        dl = _make_dataloader()
        attribute(tiny_model, g, dl, _simple_metric, method="EAP-IG-inputs", ig_steps=2, quiet=True)
        assert torch.isfinite(g.scores).all()

    @requires_cuda
    def test_attribute_then_serialise_then_reload(self, tiny_model):
        from circuitkit.backends.eap.attribute import attribute
        from circuitkit.backends.eap.graph import Graph

        g = Graph.from_model(TINY_CFG)
        dl = _make_dataloader()
        attribute(tiny_model, g, dl, _simple_metric, method="EAP", quiet=True)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            g.to_pt(f.name)
            g2 = Graph.from_pt(f.name)

        assert torch.allclose(g.scores.cpu(), g2.scores.cpu())


# ===========================================================================
# 2. Full node pipeline → adapter
# ===========================================================================
class TestFullNodeAdapterPipeline:
    @requires_cuda
    def test_node_attribution_then_adapter_keys_valid(self, tiny_model):
        from circuitkit.backends.eap.attribute_node import attribute_node
        from circuitkit.backends.eap.circuit_kit_adapter import (
            convert_eap_graph_to_circuitkit_scores,
        )
        from circuitkit.backends.eap.graph import Graph

        g = Graph.from_model(TINY_CFG, node_scores=True)
        dl = _make_dataloader()
        attribute_node(tiny_model, g, dl, _simple_metric, method="EAP", quiet=True)

        result = convert_eap_graph_to_circuitkit_scores(g)
        assert len(result) > 0
        for key in result:
            assert _CK_NAME_RE.match(key), f"Bad key format: {key}"

    @requires_cuda
    def test_edge_scores_to_node_scores_adapter(self, tiny_model):
        from circuitkit.backends.eap.attribute import attribute
        from circuitkit.backends.eap.circuit_kit_adapter import (
            convert_eap_edge_scores_to_node_scores,
        )
        from circuitkit.backends.eap.graph import Graph

        g = Graph.from_model(TINY_CFG)
        dl = _make_dataloader()
        attribute(tiny_model, g, dl, _simple_metric, method="EAP", quiet=True)

        result = convert_eap_edge_scores_to_node_scores(g)
        assert len(result) > 0
        for key in result:
            assert _CK_NAME_RE.match(key)
        # Every non-input, non-logit node that has outgoing edges should appear
        for node in g.nodes.values():
            if node.child_edges and node.name not in ("input", "logits"):
                ck = _eap_to_ck(node.name)
                if ck:
                    assert ck in result, f"Missing node {node.name} → {ck}"

    @requires_cuda
    def test_node_scores_values_non_negative(self, tiny_model):
        from circuitkit.backends.eap.attribute_node import attribute_node
        from circuitkit.backends.eap.circuit_kit_adapter import (
            convert_eap_graph_to_circuitkit_scores,
        )
        from circuitkit.backends.eap.graph import Graph

        g = Graph.from_model(TINY_CFG, node_scores=True)
        dl = _make_dataloader()
        attribute_node(tiny_model, g, dl, _simple_metric, method="EAP", quiet=True)

        result = convert_eap_graph_to_circuitkit_scores(g)
        for key, val in result.items():
            assert val >= 0.0, f"Negative score for {key}: {val}"


# ===========================================================================
# 3. Cross-method consistency
# ===========================================================================
class TestCrossMethodConsistency:
    @requires_cuda
    def test_eap_and_clean_corrupted_rank_correlated(self, tiny_model):
        """
        EAP and clean-corrupted approximate the same quantity.
        Their per-node scores should have positive Spearman rank correlation.
        """
        from scipy.stats import spearmanr

        from circuitkit.backends.eap.attribute_node import attribute_node
        from circuitkit.backends.eap.graph import Graph

        dl = _make_dataloader()
        g_eap = Graph.from_model(TINY_CFG, node_scores=True)
        g_cc = Graph.from_model(TINY_CFG, node_scores=True)

        attribute_node(tiny_model, g_eap, dl, _simple_metric, method="EAP", quiet=True)
        attribute_node(
            tiny_model, g_cc, dl, _simple_metric, method="EAP-IG-inputs", ig_steps=2, quiet=True
        )

        s1 = g_eap.nodes_scores.cpu().numpy()
        s2 = g_cc.nodes_scores.cpu().numpy()
        corr, _ = spearmanr(s1, s2)
        # Loose threshold — just checking they're not anti-correlated
        assert corr > 0.0, f"Expected positive rank correlation, got {corr:.3f}"

    @requires_cuda
    def test_pt_roundtrip_preserves_edge_in_graph(self, tiny_model):
        from circuitkit.backends.eap.attribute import attribute
        from circuitkit.backends.eap.graph import Graph

        g = Graph.from_model(TINY_CFG)
        dl = _make_dataloader()
        attribute(tiny_model, g, dl, _simple_metric, method="EAP", quiet=True)
        g.apply_topn(6)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            g.to_pt(f.name)
            g2 = Graph.from_pt(f.name)

        for name, edge in g.edges.items():
            assert edge.in_graph == g2.edges[name].in_graph


# ===========================================================================
# Helper
# ===========================================================================
def _eap_to_ck(name: str) -> str:
    m = re.match(r"a(\d+)\.h(\d+)", name)
    if m:
        return f"A{m.group(1)}.{m.group(2)}"
    m = re.match(r"m(\d+)", name)
    if m:
        return f"MLP {m.group(1)}"
    return ""
