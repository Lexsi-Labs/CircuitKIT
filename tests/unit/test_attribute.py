"""
Unit tests for attribute.py — edge-level attribution dispatcher.

All tests that run a real model require CUDA and are guarded with
pytest.mark.skipif. Tests that only check dispatcher logic use mocks and
run on CPU.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Import the module under test.  Adjust path to match your package layout.
# ---------------------------------------------------------------------------
from circuitkit.backends.eap.attribute import attribute
from circuitkit.backends.eap.graph import Graph

TINY_CFG = {"n_layers": 2, "n_heads": 2, "d_model": 64, "d_mlp": 128}

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ===========================================================================
# Helpers
# ===========================================================================


def _make_mock_model(cfg_overrides: dict = None):
    """
    A mock HookedTransformer whose cfg passes all assertions in attribute().
    """
    model = MagicMock()
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.n_key_value_heads = None
    model.cfg.d_model = 64
    model.cfg.dtype = torch.float32
    model.cfg.device = "cpu"
    model.cfg.n_layers = 2
    model.cfg.n_heads = 2
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(model.cfg, k, v)
    return model


def _make_tiny_dataloader(batch_size=2):
    clean = ["The cat sat on the mat", "A dog ran in the park"]
    corrupted = ["The cat sat on a hat", "A dog ran in the yard"]
    labels = torch.tensor([0, 1])

    def collate(xs):
        c, r, lyr = zip(*xs)
        return list(c), list(r), torch.tensor(lyr)

    dataset = list(zip(clean, corrupted, labels.tolist()))
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


# ===========================================================================
# 1. Dispatcher validation (no CUDA needed — patched scoring functions)
# ===========================================================================
class TestAttributeDispatcherValidation:
    def _patched_scores(self, graph):
        """Return a zero scores tensor of the right shape."""
        return torch.zeros(graph.n_forward, graph.n_backward)

    def test_raises_if_use_attn_result_false(self):
        model = _make_mock_model({"use_attn_result": False})
        g = Graph.from_model(TINY_CFG)
        with pytest.raises(ValueError, match="use_attn_result"):
            attribute(model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP")

    def test_raises_if_use_split_qkv_input_false(self):
        model = _make_mock_model({"use_split_qkv_input": False})
        g = Graph.from_model(TINY_CFG)
        with pytest.raises(ValueError, match="use_split_qkv_input"):
            attribute(model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP")

    def test_raises_if_use_hook_mlp_in_false(self):
        model = _make_mock_model({"use_hook_mlp_in": False})
        g = Graph.from_model(TINY_CFG)
        with pytest.raises(ValueError, match="use_hook_mlp_in"):
            attribute(model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP")

    def test_raises_for_unknown_method(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG)
        with pytest.raises(ValueError, match="method"):
            attribute(model, g, _make_tiny_dataloader(), lambda *a: None, method="BOGUS")

    def test_raises_for_invalid_aggregation(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG)
        with pytest.raises(ValueError, match="aggregation"):
            attribute(
                model,
                g,
                _make_tiny_dataloader(),
                lambda *a: None,
                method="EAP",
                aggregation="invalid",
            )

    def test_eap_ig_inputs_raises_for_non_patching_intervention(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG)
        with patch("circuitkit.backends.eap.attribute.get_scores_eap_ig") as mock_fn:
            mock_fn.return_value = torch.zeros(g.n_forward, g.n_backward)
            with pytest.raises(ValueError, match="patching"):
                attribute(
                    model,
                    g,
                    _make_tiny_dataloader(),
                    lambda *a: None,
                    method="EAP-IG-inputs",
                    intervention="zero",
                )

    def test_clean_corrupted_raises_for_non_patching_intervention(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG)
        with pytest.raises(ValueError, match="patching"):
            attribute(
                model,
                g,
                _make_tiny_dataloader(),
                lambda *a: None,
                method="clean-corrupted",
                intervention="mean",
            )

    def test_aggregation_mean_divides_by_d_model(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG)
        dummy_scores = torch.full((g.n_forward, g.n_backward), 2.0)

        with patch(
            "circuitkit.backends.eap.attribute.get_scores_eap", return_value=dummy_scores.clone()
        ):
            attribute(
                model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP", aggregation="mean"
            )
        expected = 2.0 / model.cfg.d_model
        assert torch.allclose(g.scores, torch.full_like(g.scores, expected))

    def test_scores_written_into_graph_in_place(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG)
        sentinel = torch.full((g.n_forward, g.n_backward), 9.9)

        with patch(
            "circuitkit.backends.eap.attribute.get_scores_eap", return_value=sentinel.clone()
        ):
            attribute(
                model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP", aggregation="sum"
            )
        assert torch.allclose(g.scores, sentinel)


# ===========================================================================
# 2. Real-model smoke tests (require CUDA)
# ===========================================================================
class TestAttributeWithRealModel:
    @requires_cuda
    def test_eap_returns_correct_shape(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG)
        attribute(tiny_model, g, tiny_dataloader, logit_diff_metric, method="EAP", quiet=True)
        assert g.scores.shape == (g.n_forward, g.n_backward)

    @requires_cuda
    def test_eap_scores_are_finite(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG)
        attribute(tiny_model, g, tiny_dataloader, logit_diff_metric, method="EAP", quiet=True)
        assert torch.isfinite(g.scores).all()

    @requires_cuda
    def test_eap_zero_intervention_differs_from_patching(
        self, tiny_model, tiny_dataloader, logit_diff_metric
    ):
        """Zero ablation and patching should produce different (non-identical) scores."""
        g_patch = Graph.from_model(TINY_CFG)
        g_zero = Graph.from_model(TINY_CFG)
        attribute(
            tiny_model,
            g_patch,
            tiny_dataloader,
            logit_diff_metric,
            method="EAP",
            intervention="patching",
            quiet=True,
        )
        attribute(
            tiny_model,
            g_zero,
            tiny_dataloader,
            logit_diff_metric,
            method="EAP",
            intervention="zero",
            quiet=True,
        )
        assert not torch.allclose(g_patch.scores, g_zero.scores)

    @requires_cuda
    def test_eap_identical_clean_corrupted_near_zero(self, tiny_model, logit_diff_metric):
        """When clean == corrupted the activation difference is zero → scores ≈ 0."""
        texts = ["The cat sat on the mat"] * 4
        labels = torch.zeros(4, dtype=torch.long)

        def collate(xs):
            c, r, lyr = zip(*xs)
            return list(c), list(r), torch.tensor(lyr)

        dataset = list(zip(texts, texts, labels.tolist()))
        dl = DataLoader(dataset, batch_size=4, collate_fn=collate)
        g = Graph.from_model(TINY_CFG)
        attribute(tiny_model, g, dl, logit_diff_metric, method="EAP", quiet=True)
        assert g.scores.abs().max().item() < 1e-3

    @requires_cuda
    def test_eap_ig_inputs_scores_finite(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG)
        attribute(
            tiny_model,
            g,
            tiny_dataloader,
            logit_diff_metric,
            method="EAP-IG-inputs",
            ig_steps=2,
            quiet=True,
        )
        assert torch.isfinite(g.scores).all()

    @requires_cuda
    def test_single_batch_vs_multi_batch_consistent(self, tiny_model, logit_diff_metric):
        """Averaging over 1 batch-of-4 should match averaging over 2 batches-of-2."""
        clean = [
            "The cat sat on the mat",
            "A dog ran in the park",
            "She opened the red door",
            "He read the old book",
        ]
        corrupted = [
            "The cat sat on a hat",
            "A dog ran in the yard",
            "She closed the blue door",
            "He wrote the new book",
        ]
        labels = [0, 1, 0, 1]

        def collate(xs):
            c, r, lyr = zip(*xs)
            return list(c), list(r), torch.tensor(lyr)

        dataset = list(zip(clean, corrupted, labels))
        dl_1 = DataLoader(dataset, batch_size=4, collate_fn=collate)
        dl_2 = DataLoader(dataset, batch_size=2, collate_fn=collate)

        g1 = Graph.from_model(TINY_CFG)
        g2 = Graph.from_model(TINY_CFG)
        attribute(tiny_model, g1, dl_1, logit_diff_metric, method="EAP", quiet=True)
        attribute(tiny_model, g2, dl_2, logit_diff_metric, method="EAP", quiet=True)
        assert torch.allclose(g1.scores, g2.scores, atol=1e-5)
