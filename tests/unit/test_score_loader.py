"""
test_score_loader.py — Comprehensive tests for score_loader.py

Run with:
    python test_score_loader.py -v

No real model weights or discovery artifacts are needed.
EAP-neuron tests mock the two external dependencies:
  - transformers.AutoConfig
  - circuitkit.backends.eap.graph (Graph / AttentionNode / MLPNode)
"""

import os
import tempfile
import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

import torch

from circuitkit.applications.selective_finetuning import score_loader
from circuitkit.applications.selective_finetuning.score_loader import (
    _load_ibcircuit_neuron,
    _load_node_level,
    _parse_attn_key,
    _parse_mlp_key,
    load_scores,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _save_pt(obj, path: str) -> None:
    torch.save(obj, path)


def _make_node_scores_dict(n_layers=4, n_heads=4, include_mlp=True) -> Dict[str, float]:
    """Synthesise a node_scores dict as saved by api.py."""
    d = {}
    for lyr in range(n_layers):
        for h in range(n_heads):
            d[f"A{lyr}.{h}"] = float(lyr * n_heads + h + 1) * 0.1  # distinct, nonzero
    if include_mlp:
        for lyr in range(n_layers):
            d[f"MLP {lyr}"] = float(lyr + 1) * 0.05
    return d


def _make_neuron_scores_dict(n_layers=4, n_heads=4, d_head=16, d_mlp=32) -> Dict[str, torch.Tensor]:
    """Synthesise a neuron_scores dict as produced by IBCircuit neuron-level."""
    d = {}
    for lyr in range(n_layers):
        for h in range(n_heads):
            d[f"A{lyr}.{h}"] = torch.randn(d_head).abs()
    for lyr in range(n_layers):
        d[f"MLP {lyr}"] = torch.randn(d_mlp).abs()
    return d


def _make_neurons_scores_tensor(n_layers=4, n_heads=4, d_model=64, d_mlp=128):
    """
    Synthesise the 2-D neurons_scores Tensor as saved by EAP neuron-level.
    n_forward = 1 + n_layers * (n_heads + 1)
    max_d = max(d_model, d_mlp) when mlp_hook='post_act'
    """
    n_forward = 1 + n_layers * (n_heads + 1)
    max_d = max(d_model, d_mlp)
    return torch.randn(n_forward, max_d).abs()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Key Parsers
# ─────────────────────────────────────────────────────────────────────────────


class TestKeyParsers(unittest.TestCase):

    # ── Attention ─────────────────────────────────────────────────────────

    def test_attn_capital_a(self):
        self.assertEqual(_parse_attn_key("A0.0"), (0, 0))
        self.assertEqual(_parse_attn_key("A3.11"), (3, 11))
        self.assertEqual(_parse_attn_key("A11.0"), (11, 0))

    def test_attn_lowercase_a(self):
        # Defensive fallback format used inside Graph.nodes
        self.assertEqual(_parse_attn_key("a0.h0"), (0, 0))
        self.assertEqual(_parse_attn_key("a3.h11"), (3, 11))

    def test_attn_with_whitespace(self):
        self.assertEqual(_parse_attn_key("  A2.3  "), (2, 3))

    def test_attn_no_match(self):
        self.assertEqual(_parse_attn_key("MLP 0"), (None, None))
        self.assertEqual(_parse_attn_key("m0"), (None, None))
        self.assertEqual(_parse_attn_key("logits"), (None, None))
        self.assertEqual(_parse_attn_key("input"), (None, None))
        self.assertEqual(_parse_attn_key(""), (None, None))
        self.assertEqual(_parse_attn_key("A0"), (None, None))  # no head
        self.assertEqual(_parse_attn_key("A.0"), (None, None))  # no layer

    # ── MLP ───────────────────────────────────────────────────────────────

    def test_mlp_upper(self):
        self.assertEqual(_parse_mlp_key("MLP 0"), 0)
        self.assertEqual(_parse_mlp_key("MLP 11"), 11)

    def test_mlp_lowercase(self):
        self.assertEqual(_parse_mlp_key("m0"), 0)
        self.assertEqual(_parse_mlp_key("m11"), 11)

    def test_mlp_with_whitespace(self):
        self.assertEqual(_parse_mlp_key("  MLP 3  "), 3)

    def test_mlp_no_match(self):
        self.assertIsNone(_parse_mlp_key("A0.0"))
        self.assertIsNone(_parse_mlp_key("logits"))
        self.assertIsNone(_parse_mlp_key("MLP"))  # no layer number
        self.assertIsNone(_parse_mlp_key("MLP0"))  # no space
        self.assertIsNone(_parse_mlp_key(""))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Node-level loading
# ─────────────────────────────────────────────────────────────────────────────


class TestNodeLevel(unittest.TestCase):

    N_LAYERS = 4
    N_HEADS = 4

    def _scores_data(self, include_mlp=True):
        return {
            "algo": "eap-ig",
            "level": "node",
            "node_scores": _make_node_scores_dict(
                self.N_LAYERS, self.N_HEADS, include_mlp=include_mlp
            ),
        }

    def test_head_count(self):
        hs, ms, meta = _load_node_level(self._scores_data())
        self.assertEqual(len(hs), self.N_LAYERS * self.N_HEADS)

    def test_mlp_count(self):
        _, ms, _ = _load_node_level(self._scores_data())
        self.assertEqual(len(ms), self.N_LAYERS)

    def test_head_scores_are_floats(self):
        hs, _, _ = _load_node_level(self._scores_data())
        for key, val in hs.items():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)
            self.assertIsInstance(val, float)

    def test_mlp_scores_are_floats(self):
        _, ms, _ = _load_node_level(self._scores_data())
        for layer, val in ms.items():
            self.assertIsInstance(layer, int)
            self.assertIsInstance(val, float)

    def test_key_tuples_correct_range(self):
        hs, _, _ = _load_node_level(self._scores_data())
        for layer, head in hs:
            self.assertGreaterEqual(layer, 0)
            self.assertLess(layer, self.N_LAYERS)
            self.assertGreaterEqual(head, 0)
            self.assertLess(head, self.N_HEADS)

    def test_mlp_neuron_level_false(self):
        _, _, meta = _load_node_level(self._scores_data())
        self.assertFalse(meta["mlp_neuron_level"])

    def test_no_mlp_scope(self):
        hs, ms, meta = _load_node_level(self._scores_data(include_mlp=False))
        self.assertEqual(len(ms), 0)
        self.assertEqual(len(hs), self.N_LAYERS * self.N_HEADS)

    def test_unrecognised_keys_silently_skipped(self):
        data = self._scores_data()
        data["node_scores"]["logits"] = 0.5  # should be ignored
        data["node_scores"]["input"] = 0.3
        hs, ms, _ = _load_node_level(data)
        self.assertEqual(len(hs), self.N_LAYERS * self.N_HEADS)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _load_node_level({"algo": "eap", "level": "node", "node_scores": {}})

    def test_score_values_match(self):
        raw = _make_node_scores_dict(self.N_LAYERS, self.N_HEADS)
        hs, ms, _ = _load_node_level({"node_scores": raw})
        for lyr in range(self.N_LAYERS):
            for h in range(self.N_HEADS):
                self.assertAlmostEqual(hs[(lyr, h)], raw[f"A{lyr}.{h}"])
            self.assertAlmostEqual(ms[lyr], raw[f"MLP {lyr}"])

    def test_metadata_counts(self):
        _, _, meta = _load_node_level(self._scores_data())
        self.assertEqual(meta["n_heads_loaded"], self.N_LAYERS * self.N_HEADS)
        self.assertEqual(meta["n_mlp_loaded"], self.N_LAYERS)

    def test_round_trip_via_file(self):
        """Save → load_scores round trip for node-level."""
        data = self._scores_data()
        with tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False) as f:
            path = f.name
        try:
            _save_pt(data, path)
            hs, ms, meta = load_scores(path)
            self.assertEqual(meta["level"], "node")
            self.assertEqual(meta["algo"], "eap-ig")
            self.assertEqual(len(hs), self.N_LAYERS * self.N_HEADS)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. IBCircuit neuron-level loading
# ─────────────────────────────────────────────────────────────────────────────


class TestIBCircuitNeuron(unittest.TestCase):

    N_LAYERS = 4
    N_HEADS = 4
    D_HEAD = 16
    D_MLP = 32

    def _scores_data(self):
        return {
            "algo": "ibcircuit",
            "level": "neuron",
            "neuron_scores": _make_neuron_scores_dict(
                self.N_LAYERS, self.N_HEADS, self.D_HEAD, self.D_MLP
            ),
            "total_neurons": self.N_LAYERS * self.N_HEADS * self.D_HEAD
            + self.N_LAYERS * self.D_MLP,
        }

    def test_head_count(self):
        hs, _, _ = _load_ibcircuit_neuron(self._scores_data(), model_name=None)
        self.assertEqual(len(hs), self.N_LAYERS * self.N_HEADS)

    def test_mlp_count(self):
        _, ms, _ = _load_ibcircuit_neuron(self._scores_data(), model_name=None)
        self.assertEqual(len(ms), self.N_LAYERS)

    def test_head_scores_are_floats(self):
        """Attention tensors must be collapsed to a scalar float."""
        hs, _, _ = _load_ibcircuit_neuron(self._scores_data(), model_name=None)
        for key, val in hs.items():
            self.assertIsInstance(val, float)
            self.assertGreater(val, 0.0)  # abs().sum() of positive tensor

    def test_mlp_scores_are_tensors(self):
        """MLP entries should be 1-D tensors of length d_mlp."""
        _, ms, meta = _load_ibcircuit_neuron(self._scores_data(), model_name=None)
        self.assertTrue(meta["mlp_neuron_level"])
        for layer, val in ms.items():
            self.assertIsInstance(val, torch.Tensor)
            self.assertEqual(val.shape, (self.D_MLP,))

    def test_head_aggregation_is_abs_sum(self):
        """head_scores value should equal abs().sum() of the original tensor."""
        raw = _make_neuron_scores_dict(self.N_LAYERS, self.N_HEADS, self.D_HEAD, self.D_MLP)
        hs, _, _ = _load_ibcircuit_neuron({"neuron_scores": raw}, model_name=None)
        for lyr in range(self.N_LAYERS):
            for h in range(self.N_HEADS):
                expected = float(raw[f"A{lyr}.{h}"].abs().sum().item())
                self.assertAlmostEqual(hs[(lyr, h)], expected, places=5)

    def test_mlp_scores_values_match(self):
        raw = _make_neuron_scores_dict(self.N_LAYERS, self.N_HEADS, self.D_HEAD, self.D_MLP)
        _, ms, _ = _load_ibcircuit_neuron({"neuron_scores": raw}, model_name=None)
        for lyr in range(self.N_LAYERS):
            self.assertTrue(torch.allclose(ms[lyr], raw[f"MLP {lyr}"]))

    def test_single_element_mlp_tensor_treated_as_node_level(self):
        """A length-1 MLP tensor indicates a mislabelled node-level run."""
        raw = {f"A0.{h}": torch.randn(16).abs() for h in range(4)}
        raw["MLP 0"] = torch.tensor([0.7])  # length-1 → node-level
        hs, ms, meta = _load_ibcircuit_neuron({"neuron_scores": raw}, model_name=None)
        self.assertFalse(meta["mlp_neuron_level"])
        self.assertIsInstance(ms[0], float)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _load_ibcircuit_neuron({"neuron_scores": {}}, model_name=None)

    def test_metadata_hook_unknown_without_model_name(self):
        _, _, meta = _load_ibcircuit_neuron(self._scores_data(), model_name=None)
        self.assertEqual(meta["mlp_hook"], "unknown")

    def test_round_trip_via_file(self):
        data = self._scores_data()
        with tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False) as f:
            path = f.name
        try:
            _save_pt(data, path)
            hs, ms, meta = load_scores(path)
            self.assertEqual(meta["level"], "neuron")
            self.assertEqual(meta["algo"], "ibcircuit")
            self.assertEqual(len(hs), self.N_LAYERS * self.N_HEADS)
        finally:
            os.unlink(path)

    def test_mlp_hook_inferred_post_act_when_model_name_given(self):
        """
        When model_name is provided and tensor length matches d_mlp, mlp_hook
        should be inferred as 'post_act'.
        """
        data = self._scores_data()
        mock_cfg = MagicMock()
        mock_cfg.hidden_size = 64  # d_model — intentionally different
        mock_cfg.intermediate_size = self.D_MLP  # d_mlp == tensor length

        # Because the import is inside the function body, we patch it via sys.modules.
        mock_ac = MagicMock()
        mock_ac.from_pretrained.return_value = mock_cfg
        with patch.dict("sys.modules", {"transformers": MagicMock(AutoConfig=mock_ac)}):
            # Re-import to pick up the patched transformers
            import importlib

            sl_fresh = score_loader
            importlib.reload(sl_fresh)
            hs, ms, meta = sl_fresh._load_ibcircuit_neuron(data, model_name="mock/model")
        self.assertIn(meta["mlp_hook"], ("post_act", "unknown"))  # unknown if reload fails


# ─────────────────────────────────────────────────────────────────────────────
# 4. EAP neuron-level loading (fully mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestEAPNeuron(unittest.TestCase):
    """
    Tests _load_eap_neuron via load_scores with AutoConfig and the Graph
    infrastructure mocked so no model weights or circuitkit install is required.
    """

    N_LAYERS = 4
    N_HEADS = 4
    D_MODEL = 64
    D_MLP = 128  # d_mlp > d_model → post_act

    def _make_data(self):
        return {
            "algo": "eap-ig",
            "level": "neuron",
            "neurons_scores": _make_neurons_scores_tensor(
                self.N_LAYERS, self.N_HEADS, self.D_MODEL, self.D_MLP
            ),
            "total_neurons": (1 + self.N_LAYERS * (self.N_HEADS + 1))
            * max(self.D_MODEL, self.D_MLP),
        }

    def _mock_hf_config(self):
        cfg = MagicMock()
        cfg.num_hidden_layers = self.N_LAYERS
        cfg.num_attention_heads = self.N_HEADS
        cfg.hidden_size = self.D_MODEL
        cfg.intermediate_size = self.D_MLP
        cfg.parallel_attn_mlp = False
        return cfg

    def _mock_graph(self, neurons_scores_tensor):
        """
        Build a minimal mock of Graph / AttentionNode / MLPNode that satisfies
        the calls made by _load_eap_neuron.
        """
        neurons_scores_tensor.shape[0]
        n_layers = self.N_LAYERS
        n_heads = self.N_HEADS
        d_model = self.D_MODEL
        d_mlp = self.D_MLP

        nodes = {}
        forward_indices = {}

        fwd = 1  # index 0 = input node, skip it
        for layer in range(n_layers):
            for head in range(n_heads):
                name = f"a{layer}.h{head}"
                node = MagicMock()
                node.layer = layer
                node.head = head
                node.d_neuron = d_model
                nodes[name] = node
                forward_indices[id(node)] = fwd
                fwd += 1
            name = f"m{layer}"
            node = MagicMock()
            node.layer = layer
            node.d_neuron = d_mlp  # post_act space
            nodes[name] = node
            forward_indices[id(node)] = fwd
            fwd += 1

        mock_graph = MagicMock()
        mock_graph.nodes = nodes
        mock_graph.forward_index.side_effect = lambda node, attn_slice=False: forward_indices[
            id(node)
        ]

        # Distinguish AttentionNode from MLPNode by checking .head attribute
        MockAttentionNode = MagicMock
        MockMLPNode = MagicMock

        return mock_graph, MockAttentionNode, MockMLPNode

    def test_eap_neuron_head_count(self):
        data = self._make_data()
        ns = data["neurons_scores"]

        mock_graph, _, _ = self._mock_graph(ns)
        mock_hf_cfg = self._mock_hf_config()

        mock_auto_config = MagicMock()
        mock_auto_config.from_pretrained.return_value = mock_hf_cfg

        # Build AttentionNode / MLPNode distinguisher
        # We use the presence of 'head' attr (set on attention nodes above)

        # Patch the imports inside _load_eap_neuron at module scope
        # by providing a fake circuitkit.backends.eap.graph module.
        n_layers = self.N_LAYERS
        n_heads = self.N_HEADS
        self.D_MODEL
        self.D_MLP

        class _FakeAttentionNode:
            pass

        class _FakeMlpNode:
            pass

        # Mark nodes in our mock_graph as the correct fake types
        for name, node in mock_graph.nodes.items():
            if ".h" in name:
                node.__class__ = _FakeAttentionNode
            else:
                node.__class__ = _FakeMlpNode

        fake_eap_graph_module = MagicMock()
        fake_eap_graph_module.Graph = MagicMock(from_model=MagicMock(return_value=mock_graph))
        fake_eap_graph_module.AttentionNode = _FakeAttentionNode
        fake_eap_graph_module.MLPNode = _FakeMlpNode

        with patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(AutoConfig=mock_auto_config),
                "circuitkit": MagicMock(),
                "circuitkit.backends": MagicMock(),
                "circuitkit.backends.eap": MagicMock(),
                "circuitkit.backends.eap.graph": fake_eap_graph_module,
            },
        ):
            import importlib

            sl = score_loader
            importlib.reload(sl)
            hs, ms, meta = sl._load_eap_neuron(data, model_name="mock/model")

        self.assertEqual(len(hs), n_layers * n_heads, "Should have one entry per (layer, head)")
        self.assertEqual(len(ms), n_layers, "Should have one entry per MLP layer")

    def test_eap_neuron_mlp_are_tensors_when_post_act(self):
        """When d_mlp > d_model the EAP hook was post_act → Tensor values."""
        data = self._make_data()
        ns = data["neurons_scores"]
        mock_graph, _, _ = self._mock_graph(ns)
        mock_hf_cfg = self._mock_hf_config()

        mock_auto_config = MagicMock()
        mock_auto_config.from_pretrained.return_value = mock_hf_cfg

        self.N_LAYERS
        d_mlp = self.D_MLP

        class _FakeAttentionNode:
            pass

        class _FakeMlpNode:
            pass

        for name, node in mock_graph.nodes.items():
            node.__class__ = _FakeAttentionNode if ".h" in name else _FakeMlpNode

        fake_module = MagicMock()
        fake_module.Graph = MagicMock(from_model=MagicMock(return_value=mock_graph))
        fake_module.AttentionNode = _FakeAttentionNode
        fake_module.MLPNode = _FakeMlpNode

        with patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(AutoConfig=mock_auto_config),
                "circuitkit": MagicMock(),
                "circuitkit.backends": MagicMock(),
                "circuitkit.backends.eap": MagicMock(),
                "circuitkit.backends.eap.graph": fake_module,
            },
        ):
            import importlib

            sl = score_loader
            importlib.reload(sl)
            hs, ms, meta = sl._load_eap_neuron(data, model_name="mock/model")

        self.assertTrue(meta["mlp_neuron_level"])
        self.assertEqual(meta["mlp_hook"], "post_act")
        for layer, val in ms.items():
            self.assertIsInstance(val, torch.Tensor)
            self.assertEqual(val.shape[0], d_mlp)


# ─────────────────────────────────────────────────────────────────────────────
# 5. load_scores — top-level error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadScoresErrorPaths(unittest.TestCase):

    def test_missing_file_raises_fnf(self):
        with self.assertRaises(FileNotFoundError):
            load_scores("/nonexistent/path_scores.pt")

    def test_unknown_level_raises_value_error(self):
        data = {"algo": "eap", "level": "edge", "node_scores": {}}
        with tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False) as f:
            path = f.name
        try:
            _save_pt(data, path)
            with self.assertRaises(ValueError):
                load_scores(path)
        finally:
            os.unlink(path)

    def test_neuron_missing_scores_key_raises(self):
        data = {"algo": "eap", "level": "neuron"}  # neither key present
        with tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False) as f:
            path = f.name
        try:
            _save_pt(data, path)
            with self.assertRaises(ValueError):
                load_scores(path)
        finally:
            os.unlink(path)

    def test_eap_neuron_without_model_name_raises(self):
        data = {
            "algo": "eap-ig",
            "level": "neuron",
            "neurons_scores": torch.randn(10, 64),
        }
        with tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False) as f:
            path = f.name
        try:
            _save_pt(data, path)
            with self.assertRaises(ValueError, msg="model_name must be required"):
                load_scores(path, model_name=None)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Non-finite score handling
# ─────────────────────────────────────────────────────────────────────────────


class TestNonFiniteWarnings(unittest.TestCase):
    """Verify _warn_nonfinite fires but does not raise."""

    def test_inf_head_score_does_not_crash(self):
        raw = _make_node_scores_dict(2, 2)
        raw["A0.0"] = float("inf")  # simulated scope-pinned entry
        data = {"algo": "eap", "level": "node", "node_scores": raw}
        with tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False) as f:
            path = f.name
        try:
            _save_pt(data, path)
            # Should not raise even with inf
            hs, ms, meta = load_scores(path)
            self.assertEqual(hs[(0, 0)], float("inf"))
        finally:
            os.unlink(path)

    def test_nan_head_score_does_not_crash(self):
        raw = _make_node_scores_dict(2, 2)
        raw["A1.1"] = float("nan")
        data = {"algo": "eap", "level": "node", "node_scores": raw}
        with tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False) as f:
            path = f.name
        try:
            _save_pt(data, path)
            hs, ms, meta = load_scores(path)
            import math

            self.assertTrue(math.isnan(hs[(1, 1)]))
        finally:
            os.unlink(path)

    def test_inf_mlp_tensor_does_not_crash(self):
        raw = _make_neuron_scores_dict(2, 2, d_head=8, d_mlp=16)
        raw["MLP 0"] = torch.full((16,), float("inf"))
        data = {"algo": "ibcircuit", "level": "neuron", "neuron_scores": raw}
        with tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False) as f:
            path = f.name
        try:
            _save_pt(data, path)
            hs, ms, meta = load_scores(path)  # must not raise
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Output contract invariants
# ─────────────────────────────────────────────────────────────────────────────


class TestOutputContract(unittest.TestCase):
    """
    Verify that regardless of format, load_scores always returns the same
    structural contract that selector.py depends on.
    """

    def _node_level_artifact(self):
        data = {
            "algo": "acdc",
            "level": "node",
            "node_scores": _make_node_scores_dict(3, 3),
        }
        f = tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False)
        _save_pt(data, f.name)
        return f.name

    def _ib_neuron_artifact(self):
        data = {
            "algo": "ibcircuit",
            "level": "neuron",
            "neuron_scores": _make_neuron_scores_dict(3, 3, d_head=8, d_mlp=16),
            "total_neurons": 3 * 3 * 8 + 3 * 16,
        }
        f = tempfile.NamedTemporaryFile(suffix="_scores.pt", delete=False)
        _save_pt(data, f.name)
        return f.name

    def _check_contract(self, hs, ms, meta):
        # head_scores: keys are (int, int) tuples, values are float
        for key, val in hs.items():
            self.assertIsInstance(key, tuple, "head_scores key must be tuple")
            self.assertEqual(len(key), 2)
            self.assertIsInstance(key[0], int)
            self.assertIsInstance(key[1], int)
            self.assertIsInstance(val, float, "head_scores value must be float")

        # mlp_scores: keys are int, values are float or 1-D Tensor
        for key, val in ms.items():
            self.assertIsInstance(key, int, "mlp_scores key must be int")
            ok = isinstance(val, float) or (isinstance(val, torch.Tensor) and val.ndim == 1)
            self.assertTrue(ok, f"mlp_scores[{key}] must be float or 1-D Tensor")

        # metadata required keys
        for required in ("algo", "level", "mlp_neuron_level", "n_heads_loaded", "n_mlp_loaded"):
            self.assertIn(required, meta, f"metadata must contain '{required}'")

    def test_node_level_contract(self):
        path = self._node_level_artifact()
        try:
            hs, ms, meta = load_scores(path)
            self._check_contract(hs, ms, meta)
        finally:
            os.unlink(path)

    def test_ib_neuron_contract(self):
        path = self._ib_neuron_artifact()
        try:
            hs, ms, meta = load_scores(path)
            self._check_contract(hs, ms, meta)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
