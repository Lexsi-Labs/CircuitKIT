"""
Unit tests for attribute_node.py — node-level attribution dispatcher.

Dispatcher-validation tests run on CPU with mocks.
Real-model smoke tests require CUDA.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from circuitkit.backends.eap.attribute_node import attribute_node
from circuitkit.backends.eap.graph import Graph

TINY_CFG = {"n_layers": 2, "n_heads": 2, "d_model": 64, "d_mlp": 128}

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ===========================================================================
# Helpers
# ===========================================================================


def _make_mock_model(cfg_overrides: dict = None):
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
# 1. Dispatcher validation (no CUDA)
# ===========================================================================
class TestAttributeNodeDispatcherValidation:
    def test_raises_if_use_attn_result_false(self):
        model = _make_mock_model({"use_attn_result": False})
        g = Graph.from_model(TINY_CFG, node_scores=True)
        with pytest.raises(ValueError, match="use_attn_result"):
            attribute_node(model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP")

    def test_raises_if_use_split_qkv_input_false(self):
        model = _make_mock_model({"use_split_qkv_input": False})
        g = Graph.from_model(TINY_CFG, node_scores=True)
        with pytest.raises(ValueError, match="use_split_qkv_input"):
            attribute_node(model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP")

    def test_raises_if_use_hook_mlp_in_false(self):
        model = _make_mock_model({"use_hook_mlp_in": False})
        g = Graph.from_model(TINY_CFG, node_scores=True)
        with pytest.raises(ValueError, match="use_hook_mlp_in"):
            attribute_node(model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP")

    def test_raises_for_unknown_method(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG, node_scores=True)
        with pytest.raises(ValueError, match="method"):
            attribute_node(model, g, _make_tiny_dataloader(), lambda *a: None, method="NOTAMETHOD")

    def test_raises_for_invalid_aggregation(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG, node_scores=True)
        with pytest.raises(ValueError, match="aggregation"):
            attribute_node(
                model,
                g,
                _make_tiny_dataloader(),
                lambda *a: None,
                method="EAP",
                aggregation="wrong",
            )

    def test_eap_ig_inputs_raises_for_non_patching(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG, node_scores=True)
        with pytest.raises(ValueError, match="patching"):
            attribute_node(
                model,
                g,
                _make_tiny_dataloader(),
                lambda *a: None,
                method="EAP-IG-inputs",
                intervention="zero",
            )

    def test_aggregation_mean_divides_by_d_model(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG, node_scores=True)
        dummy = torch.full((g.n_forward,), 4.0)

        with patch(
            "circuitkit.backends.eap.attribute_node.get_scores_eap", return_value=dummy.clone()
        ):
            attribute_node(
                model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP", aggregation="mean"
            )
        expected = 4.0 / model.cfg.d_model
        assert torch.allclose(g.nodes_scores, torch.full_like(g.nodes_scores, expected))

    def test_neuron_false_writes_to_nodes_scores(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG, node_scores=True)
        dummy_node = torch.full((g.n_forward,), 1.0)

        with patch(
            "circuitkit.backends.eap.attribute_node.get_scores_eap", return_value=dummy_node.clone()
        ):
            attribute_node(
                model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP", neuron=False
            )
        assert g.nodes_scores is not None
        assert g.nodes_scores.shape == (g.n_forward,)

    def test_neuron_true_writes_to_neurons_scores(self):
        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG, node_scores=True, neuron_level=True)
        max_d = model.cfg.d_model
        dummy_neuron = torch.ones(g.n_forward, max_d)

        with patch(
            "circuitkit.backends.eap.attribute_node.get_scores_eap",
            return_value=dummy_neuron.clone(),
        ):
            attribute_node(
                model, g, _make_tiny_dataloader(), lambda *a: None, method="EAP", neuron=True
            )
        assert g.neurons_scores is not None


# ===========================================================================
# 2. make_hooks_and_matrices (node-level) — shape checks without a real model
# ===========================================================================
class TestMakeHooksAndMatrices:
    def test_activation_difference_shape(self):
        from circuitkit.backends.eap.attribute_node import make_hooks_and_matrices  # noqa

        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG, node_scores=True)
        scores = torch.zeros(g.n_forward)
        batch_size, n_pos = 2, 8

        (fwd_corr, fwd_clean, bwd), act_diff = make_hooks_and_matrices(
            model, g, batch_size, n_pos, scores, neuron=False
        )
        # [batch, pos, n_forward, max_d]
        assert act_diff.shape[0] == batch_size
        assert act_diff.shape[1] == n_pos
        assert act_diff.shape[2] == g.n_forward

    def test_hook_list_lengths(self):
        from circuitkit.backends.eap.attribute_node import make_hooks_and_matrices  # noqa

        model = _make_mock_model()
        g = Graph.from_model(TINY_CFG, node_scores=True)
        scores = torch.zeros(g.n_forward)

        (fwd_corr, fwd_clean, bwd), _ = make_hooks_and_matrices(
            model, g, 2, 8, scores, neuron=False
        )
        # input + n_layers attn + n_layers mlp = 1 + 2 + 2 = 5 forward hooks each
        n_layers = TINY_CFG["n_layers"]
        expected_fwd = 1 + n_layers + n_layers
        assert len(fwd_corr) == expected_fwd
        assert len(fwd_clean) == expected_fwd

    def test_post_act_max_d_equals_d_mlp(self):
        from circuitkit.backends.eap.attribute_node import make_hooks_and_matrices  # noqa

        cfg = dict(TINY_CFG, mlp_hook="post_act")
        model = _make_mock_model()
        model.cfg.d_model = 64
        # Monkey-patch d_mlp into cfg; make_hooks_and_matrices reads graph.cfg
        g = Graph.from_model(cfg, node_scores=True)
        scores = torch.zeros(g.n_forward, max(64, 128))  # neuron shape

        (_, _, _), act_diff = make_hooks_and_matrices(model, g, 2, 8, scores, neuron=True)
        # max_d should be max(d_model, d_mlp) = 128
        assert act_diff.shape[-1] == 128


# ===========================================================================
# 3. Real-model smoke tests (require CUDA)
# ===========================================================================
class TestAttributeNodeWithRealModel:
    @requires_cuda
    def test_eap_node_scores_shape(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG, node_scores=True)
        attribute_node(tiny_model, g, tiny_dataloader, logit_diff_metric, method="EAP", quiet=True)
        assert g.nodes_scores.shape == (g.n_forward,)

    @requires_cuda
    def test_eap_node_scores_finite(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG, node_scores=True)
        attribute_node(tiny_model, g, tiny_dataloader, logit_diff_metric, method="EAP", quiet=True)
        assert torch.isfinite(g.nodes_scores).all()

    @requires_cuda
    def test_eap_neuron_scores_shape(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG, node_scores=True, neuron_level=True)
        attribute_node(
            tiny_model, g, tiny_dataloader, logit_diff_metric, method="EAP", neuron=True, quiet=True
        )
        assert g.neurons_scores is not None
        assert g.neurons_scores.shape[0] == g.n_forward

    @requires_cuda
    def test_eap_ig_inputs_node_scores_finite(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG, node_scores=True)
        attribute_node(
            tiny_model,
            g,
            tiny_dataloader,
            logit_diff_metric,
            method="EAP-IG-inputs",
            ig_steps=2,
            quiet=True,
        )
        assert torch.isfinite(g.nodes_scores).all()

    @requires_cuda
    def test_clean_corrupted_node_scores_finite(
        self, tiny_model, tiny_dataloader, logit_diff_metric
    ):
        g = Graph.from_model(TINY_CFG, node_scores=True)
        attribute_node(
            tiny_model, g, tiny_dataloader, logit_diff_metric, method="EAP-IG-inputs", quiet=True
        )
        assert torch.isfinite(g.nodes_scores).all()


# ===========================================================================
# 4. Phase-2 algorithm ports (atp-gd, eap-gp, relp) — added in v0.4
# ===========================================================================
class TestPhase2Algorithms:
    """Dispatcher-level tests for the new Phase-2 algorithm keys."""

    def test_atp_gd_method_recognised(self):
        from circuitkit.backends.eap.attribute_node import (
            get_scores_atp_grad_drop,
            get_scores_eap_gp,
            get_scores_relp,
        )

        # Importing without error is the smoke test; symbols are wired.
        assert callable(get_scores_atp_grad_drop)
        assert callable(get_scores_relp)
        assert callable(get_scores_eap_gp)

    def test_dispatcher_rejects_unknown_method(self):
        # Sanity: the new keys are NOT silently swallowed by some
        # default branch.
        model = _make_mock_model()
        with pytest.raises(ValueError, match="method must be in"):
            attribute_node(
                model,
                MagicMock(),
                _make_tiny_dataloader(),
                lambda *_: torch.tensor(0.0),
                method="totally-fake-method",
            )

    def test_relp_hook_builder_picks_identity_for_non_gated(self):
        from circuitkit.backends.eap.attribute_node import _build_relp_hooks

        # Mock a non-gated GPT-2-style model.
        m = _make_mock_model()
        m.cfg.gated_mlp = False
        m.hook_dict = {
            f"blocks.{lyr}.{p}": None
            for lyr in range(2)
            for p in (
                "attn.hook_pattern",
                "ln1.hook_scale",
                "ln2.hook_scale",
                "mlp.hook_pre",
                "mlp.hook_post",
            )
        }
        hooks = _build_relp_hooks(m)
        names = [h[0] for h in hooks]
        # Non-gated path should hook BOTH hook_pre and hook_post (Identity-rule).
        assert any("mlp.hook_pre" in n for n in names)
        assert any("mlp.hook_post" in n for n in names)

    def test_relp_hook_builder_picks_half_rule_for_gated(self):
        from circuitkit.backends.eap.attribute_node import _build_relp_hooks

        m = _make_mock_model()
        m.cfg.gated_mlp = True
        m.hook_dict = {
            f"blocks.{lyr}.{p}": None
            for lyr in range(2)
            for p in (
                "attn.hook_pattern",
                "ln1.hook_scale",
                "ln2.hook_scale",
                "mlp.hook_pre",
                "mlp.hook_post",
            )
        }
        hooks = _build_relp_hooks(m)
        # Gated path should hook hook_post (Half-rule) but NOT hook_pre.
        names = [h[0] for h in hooks]
        post_count = sum(1 for n in names if "mlp.hook_post" in n)
        pre_count = sum(1 for n in names if "mlp.hook_pre" in n)
        assert post_count == 2  # one per layer
        assert pre_count == 0  # Identity-rule's pre-hook is skipped on gated

    @requires_cuda
    def test_atp_gd_node_scores_finite(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG, node_scores=True)
        attribute_node(
            tiny_model, g, tiny_dataloader, logit_diff_metric, method="atp-gd", quiet=True
        )
        assert torch.isfinite(g.nodes_scores).all()
        # AtP+GD aggregates abs scores; result should be non-negative.
        assert (g.nodes_scores >= -1e-6).all()

    @requires_cuda
    def test_eap_gp_node_scores_finite(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG, node_scores=True)
        attribute_node(
            tiny_model,
            g,
            tiny_dataloader,
            logit_diff_metric,
            method="eap-gp",
            ig_steps=3,
            quiet=True,
        )
        assert torch.isfinite(g.nodes_scores).all()

    @requires_cuda
    def test_relp_node_scores_finite(self, tiny_model, tiny_dataloader, logit_diff_metric):
        g = Graph.from_model(TINY_CFG, node_scores=True)
        attribute_node(tiny_model, g, tiny_dataloader, logit_diff_metric, method="relp", quiet=True)
        assert torch.isfinite(g.nodes_scores).all()

    @requires_cuda
    def test_exact_post_fix_no_uniform_scores(self, tiny_model, tiny_dataloader, logit_diff_metric):
        """Regression test for the get_scores_exact in_graph corruption bug:
        before the fix every node ended up with the same score (~baseline +
        corrupted-metric). After the fix scores should vary across nodes."""
        g = Graph.from_model(TINY_CFG, node_scores=True)
        attribute_node(
            tiny_model, g, tiny_dataloader, logit_diff_metric, method="exact", quiet=True
        )
        scores = g.nodes_scores.flatten()
        # Score variance must be greater than ~ULP (would be zero before fix).
        assert scores.std().item() > 1e-4, (
            "Scores collapsed to ~uniform — get_scores_exact may have "
            "regressed the in_graph snapshot/restore fix"
        )
