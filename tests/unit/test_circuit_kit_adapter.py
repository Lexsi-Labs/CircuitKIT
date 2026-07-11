"""
Unit tests for circuit_kit_adapter.py

Tests are pure Python / pure tensor — no model, no CUDA required.
"""

from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Import the functions under test.  Adjust path to match your package layout.
# ---------------------------------------------------------------------------
from circuitkit.backends.eap.circuit_kit_adapter import (
    calculate_manual_perplexity,
    convert_eap_edge_scores_to_node_scores,
    convert_eap_graph_to_circuitkit_scores,
)
from circuitkit.backends.eap.graph import AttentionNode, Graph, MLPNode

TINY_CFG = {"n_layers": 2, "n_heads": 2, "d_model": 64, "d_mlp": 128}


# ===========================================================================
# Helpers
# ===========================================================================


def _make_graph_with_node_scores(scores_dict: dict) -> Graph:
    """
    Build a Graph with node_scores=True and set each named node's score.
    `scores_dict` maps EAP node names (e.g. 'a0.h1', 'm1') to float values.
    """
    g = Graph.from_model(TINY_CFG, node_scores=True)
    for name, val in scores_dict.items():
        g.nodes[name].score = val
    return g


def _make_graph_with_edge_scores(scores_dict: dict) -> Graph:
    """
    Build a Graph and set each named edge's score.
    `scores_dict` maps EAP edge names to float values.
    """
    g = Graph.from_model(TINY_CFG)
    for name, val in scores_dict.items():
        if name in g.edges:
            g.edges[name].score = val
    return g


# ===========================================================================
# 1. convert_eap_graph_to_circuitkit_scores
# ===========================================================================
class TestConvertNodeScores:
    def test_raises_when_nodes_scores_none(self):
        g = Graph.from_model(TINY_CFG)  # node_scores not enabled
        assert g.nodes_scores is None
        with pytest.raises(ValueError):
            convert_eap_graph_to_circuitkit_scores(g)

    def test_attention_node_name_mapping(self):
        g = _make_graph_with_node_scores({"a1.h1": 0.5})
        result = convert_eap_graph_to_circuitkit_scores(g)
        assert "A1.1" in result

    def test_mlp_node_name_mapping(self):
        g = _make_graph_with_node_scores({"m1": 0.7})
        result = convert_eap_graph_to_circuitkit_scores(g)
        assert "MLP 1" in result

    def test_input_and_logit_nodes_excluded(self):
        g = Graph.from_model(TINY_CFG, node_scores=True)

        # Only assign scores to scoreable intermediate nodes (Attention and MLP)
        # This aligns with expected pipeline behavior.
        for node in g.nodes.values():
            if isinstance(node, (AttentionNode, MLPNode)):
                node.score = 1.0

        result = convert_eap_graph_to_circuitkit_scores(g)

        # Assert that despite existing in the graph, non-scoreable nodes
        # are successfully omitted from the output.
        assert "input" not in result
        assert "logits" not in result

    def test_scores_are_absolute_values(self):
        """Negative raw scores should become positive values in the output."""
        g = _make_graph_with_node_scores({"a0.h0": -3.0})
        result = convert_eap_graph_to_circuitkit_scores(g)
        assert result["A0.0"] == pytest.approx(3.0)

    def test_zero_score_maps_to_zero(self):
        g = _make_graph_with_node_scores({"m0": 0.0})
        result = convert_eap_graph_to_circuitkit_scores(g)
        assert result["MLP 0"] == pytest.approx(0.0)

    def test_all_attn_and_mlp_nodes_present(self):
        """Every attention head and MLP layer should appear in the output."""
        g = Graph.from_model(TINY_CFG, node_scores=True)
        result = convert_eap_graph_to_circuitkit_scores(g)
        for layer in range(TINY_CFG["n_layers"]):
            for head in range(TINY_CFG["n_heads"]):
                assert f"A{layer}.{head}" in result
            assert f"MLP {layer}" in result

    def test_output_count(self):
        g = Graph.from_model(TINY_CFG, node_scores=True)
        result = convert_eap_graph_to_circuitkit_scores(g)
        expected = TINY_CFG["n_layers"] * TINY_CFG["n_heads"] + TINY_CFG["n_layers"]
        assert len(result) == expected


# ===========================================================================
# 2. convert_eap_edge_scores_to_node_scores
# ===========================================================================
class TestConvertEdgeScores:
    def test_output_is_dict_of_floats(self):
        g = Graph.from_model(TINY_CFG)
        result = convert_eap_edge_scores_to_node_scores(g)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, float)

    def test_values_are_non_negative(self):
        """Mean of absolute values should always be >= 0."""
        g = Graph.from_model(TINY_CFG)
        for edge in g.edges.values():
            edge.score = -5.0
        result = convert_eap_edge_scores_to_node_scores(g)
        for v in result.values():
            assert v >= 0.0

    def test_single_edge_node_score_equals_abs_score(self):
        """
        Find a node that has exactly one outgoing edge, set its score, and
        verify the converted node score equals abs(score).
        """
        g = Graph.from_model(TINY_CFG)
        # Find a node with a single child edge
        target_node = None
        target_edge = None
        for node in g.nodes.values():
            if len(node.child_edges) == 1:
                target_node = node
                target_edge = next(iter(node.child_edges))
                break
        if target_node is None:
            pytest.skip("No node with exactly one outgoing edge in this config")

        raw_score = -7.5
        target_edge.score = raw_score

        result = convert_eap_edge_scores_to_node_scores(g)
        ck_name = _eap_to_ck_name(target_node.name)
        if ck_name:
            assert result[ck_name] == pytest.approx(abs(raw_score))

    def test_attention_name_mapping(self):
        g = Graph.from_model(TINY_CFG)
        result = convert_eap_edge_scores_to_node_scores(g)
        for key in result:
            # Must match A{layer}.{head} or MLP {layer}
            assert _is_valid_ck_name(key), f"Unexpected key format: {key}"

    def test_node_without_outgoing_edges_absent(self):
        """Logit node has no outgoing edges — it must not appear in output."""
        g = Graph.from_model(TINY_CFG)
        result = convert_eap_edge_scores_to_node_scores(g)
        assert "logits" not in result

    def test_empty_graph_no_crash(self):
        """Graph with no edge scores set should return a dict (possibly empty)."""
        g = Graph.from_model(TINY_CFG)
        result = convert_eap_edge_scores_to_node_scores(g)
        assert isinstance(result, dict)


# ===========================================================================
# 3. calculate_manual_perplexity
# ===========================================================================
class TestCalculateManualPerplexity:
    def _make_mock_model(self, vocab_size=100, d_model=64, device="cpu"):
        """
        Build a minimal mock that satisfies the function's interface:
          model(input_ids) -> logits [batch, seq, vocab]
          model.cfg.d_vocab -> int
        """
        import torch

        model = MagicMock()
        model.cfg.d_vocab = vocab_size

        def fake_forward(input_ids):
            batch, seq = input_ids.shape
            # Uniform logits so loss is deterministic
            return torch.zeros(batch, seq, vocab_size, device=device)

        model.side_effect = fake_forward
        model.__call__ = fake_forward
        return model

    def _make_mock_tokenizer(self, device="cpu"):
        tokenizer = MagicMock()

        def encode(text, return_tensors=None):
            # Encode as fixed 5-token sequence
            ids = torch.tensor([[1, 2, 3, 4, 5]])
            if return_tensors == "pt":
                return ids
            return [1, 2, 3, 4, 5]

        tokenizer.encode = encode
        return tokenizer

    def test_returns_inf_for_all_empty_texts(self):
        model = self._make_mock_model()
        tokenizer = MagicMock()
        tokenizer.encode = lambda text, **kw: torch.zeros(1, 1, dtype=torch.long)
        result = calculate_manual_perplexity(model, tokenizer, ["", ""], device="cpu")
        assert result == float("inf")

    def test_returns_finite_for_valid_texts(self):
        model = self._make_mock_model()
        tokenizer = self._make_mock_tokenizer()
        result = calculate_manual_perplexity(model, tokenizer, ["hello world"], device="cpu")
        assert isinstance(result, float)
        assert result > 0
        assert result != float("inf")

    def test_skips_single_token_sequences(self):
        """Sequences of length < 2 should be skipped; result still finite."""
        model = self._make_mock_model()
        tokenizer = MagicMock()
        call_count = 0

        def encode(text, return_tensors=None):
            nonlocal call_count
            call_count += 1
            # Alternate between length-1 (skip) and length-5 (valid)
            if call_count % 2 == 1:
                return torch.tensor([[1]])
            return torch.tensor([[1, 2, 3, 4, 5]])

        tokenizer.encode = encode
        result = calculate_manual_perplexity(model, tokenizer, ["skip", "valid"], device="cpu")
        assert result != float("inf")

    def test_empty_list_returns_inf(self):
        model = self._make_mock_model()
        tokenizer = self._make_mock_tokenizer()
        result = calculate_manual_perplexity(model, tokenizer, [], device="cpu")
        assert result == float("inf")

    def test_consistent_across_identical_inputs(self):
        model = self._make_mock_model()
        tokenizer = self._make_mock_tokenizer()
        texts = ["hello world"] * 3
        r1 = calculate_manual_perplexity(model, tokenizer, texts, device="cpu")
        r2 = calculate_manual_perplexity(model, tokenizer, texts, device="cpu")
        assert r1 == pytest.approx(r2)


# ===========================================================================
# Helpers
# ===========================================================================
import re  # noqa: E402 - import after intentional pre-import setup


def _eap_to_ck_name(eap_name: str) -> str:
    """Convert an EAP node name to expected CircuitKit key, or '' if not applicable."""
    m = re.match(r"a(\d+)\.h(\d+)", eap_name)
    if m:
        return f"A{m.group(1)}.{m.group(2)}"
    m = re.match(r"m(\d+)", eap_name)
    if m:
        return f"MLP {m.group(1)}"
    return ""


def _is_valid_ck_name(name: str) -> bool:
    return bool(re.match(r"^(A\d+\.\d+|MLP \d+)$", name))
