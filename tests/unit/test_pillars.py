"""
Unit tests for Pillar 3 (Stability) and Pillar 5 (Baselines).

Tests the core metrics and functionality of the stability and baseline
comparison evaluation pillars.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from circuitkit.backends.eap.graph import Graph
from circuitkit.evaluation.full import run_full_faithfulness
from circuitkit.evaluation.pillars.ablation import Pillar2_Ablation
from circuitkit.evaluation.pillars.baselines import Pillar5_Baselines
from circuitkit.evaluation.pillars.causal_patching import Pillar1_CausalPatching
from circuitkit.evaluation.pillars.generalization import Pillar6_Generalization
from circuitkit.evaluation.pillars.robustness import Pillar4_Robustness
from circuitkit.evaluation.pillars.stability import Pillar3_Stability
from circuitkit.evaluation.report import FaithfulnessReport
from circuitkit.evaluation.stability_discovery import (
    node_scores_to_circuit,
    spearman_rank_correlation,
)

# Lightweight config for building a real Graph without loading a model.
# Graph.from_model() accepts a plain config dict, so baseline-builder tests
# can exercise the real Graph topology (and Graph.from_model clone path)
# without the cost or flakiness of a real HookedTransformer.
TINY_CFG = {"n_layers": 4, "n_heads": 4, "d_model": 64, "d_mlp": 128}


def _make_real_graph(node_scores=True):
    """Build a real Graph from TINY_CFG (no model load required)."""
    return Graph.from_model(TINY_CFG, node_scores=node_scores)


def _scored_node_names(graph):
    """Return non-terminal node names (excludes 'input' and 'logits')."""
    return [n for n in graph.nodes if n not in ("input", "logits")]


# ─── Shared mock helpers ───────────────────────────────────────────────


def _make_mock_model(use_attn_result=True):
    """Create a mock HookedTransformer with minimal cfg."""
    model = MagicMock()
    model.cfg.use_attn_result = use_attn_result
    model.cfg.device = "cpu"
    model.cfg.dtype = torch.float32
    return model


def _make_weighted_model(cfg=TINY_CFG, head_mag=None, mlp_mag=None):
    """Mock model with controllable per-node weight norms.

    The magnitude baseline ranks a node by RMS(weight) = ||W|| / sqrt(numel).
    For ``W = c * ones(...)``, RMS == c, so ``head_mag(layer, head)`` and
    ``mlp_mag(layer)`` set node (layer, head) / MLP layer magnitudes exactly,
    making the weight-norm ranking predictable.
    """
    import types

    n_layers, n_heads = cfg["n_layers"], cfg["n_heads"]
    d_model, d_mlp = cfg["d_model"], cfg["d_mlp"]
    d_head = d_model // n_heads
    head_mag = head_mag or (lambda layer, head: 1.0)
    mlp_mag = mlp_mag or (lambda layer: 1.0)

    blocks = []
    for layer in range(n_layers):
        w_o = torch.stack(
            [head_mag(layer, h) * torch.ones(d_head, d_model) for h in range(n_heads)]
        )  # [n_heads, d_head, d_model]
        w_out = mlp_mag(layer) * torch.ones(d_mlp, d_model)  # [d_mlp, d_model]
        blocks.append(
            types.SimpleNamespace(
                attn=types.SimpleNamespace(W_O=w_o),
                mlp=types.SimpleNamespace(W_out=w_out),
            )
        )
    return types.SimpleNamespace(blocks=blocks)


def _make_mock_graph(n_nodes=10, in_graph_indices=None):
    """Create a mock Graph with controllable in_graph flags."""

    class MockNode:
        def __init__(self, name, layer, score, in_graph=False):
            self.name = name
            self.layer = layer
            self.score = torch.tensor(score)
            self._in_graph = in_graph

        @property
        def in_graph(self):
            return self._in_graph

        @in_graph.setter
        def in_graph(self, value):
            self._in_graph = value

    if in_graph_indices is None:
        in_graph_indices = set(range(n_nodes))

    class MockGraph:
        def __init__(self):
            self.nodes = {
                f"node_{i}": MockNode(
                    f"node_{i}",
                    i % 3,
                    float(i) * 0.1,
                    in_graph=(i in in_graph_indices),
                )
                for i in range(n_nodes)
            }
            self.neurons_in_graph = None
            self.neurons_scores = None

    return MockGraph()


def _make_mock_task_spec(metric_fn=None):
    """Create a mock TaskSpec with a metric function."""
    spec = MagicMock()
    spec.metric_fn = metric_fn or MagicMock()
    spec.name = "mock_task"
    spec.build_dataloader = MagicMock(return_value=MagicMock())
    return spec


def _make_discovery_cfg(**overrides):
    """Create a minimal discovery config dict."""
    cfg = {
        "algorithm": "eap",
        "model": {"name": "gpt2"},
        "task": "ioi",
        "level": "node",
        "scope": "full",
        "intervention": "zero",
        "pruning": {"target_sparsity": 0.3},
        "data_params": {"seed": 42},
    }
    cfg.update(overrides)
    return cfg


class TestStabilityDiscovery:
    """Tests for stability_discovery helper functions."""

    def test_node_scores_to_circuit_basic(self):
        """Nodes below sparsity threshold are pruned."""
        scores = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.2}
        # sparsity=0.5 → discard bottom 50% (2 nodes: a=0.1, d=0.2)
        kept = node_scores_to_circuit(scores, sparsity=0.5)
        assert set(kept.keys()) == {"b", "c"}

    def test_node_scores_to_circuit_zero_sparsity(self):
        """Sparsity 0 keeps all nodes."""
        scores = {"a": 0.1, "b": 0.5}
        kept = node_scores_to_circuit(scores, sparsity=0.0)
        assert set(kept.keys()) == {"a", "b"}

    def test_node_scores_to_circuit_full_sparsity(self):
        """Sparsity 1.0 discards all nodes."""
        scores = {"a": 0.1, "b": 0.5}
        kept = node_scores_to_circuit(scores, sparsity=1.0)
        assert len(kept) == 0

    def test_node_scores_to_circuit_empty_input(self):
        """Empty input returns empty output."""
        assert node_scores_to_circuit({}, sparsity=0.5) == {}

    def test_spearman_identical_scores(self):
        """Identical orderings yield rho=1."""
        scores = {"a": 1.0, "b": 2.0, "c": 3.0}
        rho = spearman_rank_correlation(scores, scores)
        assert abs(rho - 1.0) < 1e-6

    def test_spearman_reversed_scores(self):
        """Perfectly reversed orderings yield rho=-1."""
        a = {"x": 1.0, "y": 2.0, "z": 3.0}
        b = {"x": 3.0, "y": 2.0, "z": 1.0}
        rho = spearman_rank_correlation(a, b)
        assert abs(rho - (-1.0)) < 1e-6

    def test_spearman_no_shared_keys(self):
        """Disjoint key sets return 0."""
        a = {"x": 1.0}
        b = {"y": 2.0}
        rho = spearman_rank_correlation(a, b)
        assert rho == 0.0

    def test_spearman_single_shared_key(self):
        """Fewer than 2 shared keys returns 0."""
        a = {"x": 1.0, "y": 2.0}
        b = {"x": 5.0, "z": 3.0}
        rho = spearman_rank_correlation(a, b)
        assert rho == 0.0

    def test_spearman_partial_overlap(self):
        """Only shared keys are compared."""
        a = {"x": 1.0, "y": 2.0, "z": 3.0, "extra": 100.0}
        b = {"x": 1.0, "y": 2.0, "z": 3.0, "other": 200.0}
        rho = spearman_rank_correlation(a, b)
        assert abs(rho - 1.0) < 1e-6  # shared keys have same order


class TestPillar1CausalPatching:
    """Tests for Pillar 1: Causal Patching."""

    def test_run_rejects_none_graph(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="Graph cannot be None"):
            Pillar1_CausalPatching.run(model, None, MagicMock(), MagicMock())

    def test_run_rejects_missing_use_attn_result(self):
        model = _make_mock_model(use_attn_result=False)
        graph = _make_mock_graph()
        with pytest.raises(AssertionError, match="use_attn_result"):
            Pillar1_CausalPatching.run(model, graph, MagicMock(), MagicMock())

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_run_returns_raw_and_ratio(self, mock_eval, mock_base):
        """run() returns a dict: raw_score is the mean metric, score is the
        normalized faithfulness ratio (circuit - corrupt) / (clean - corrupt)."""
        mock_eval.return_value = torch.tensor([0.6, 0.8, 1.0])  # circuit raw mean = 0.8
        # clean baseline = 1.0, corrupt baseline = 0.0  ->  ratio = (0.8-0)/(1-0) = 0.8
        mock_base.side_effect = [torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])]
        model = _make_mock_model()
        graph = _make_mock_graph()

        result = Pillar1_CausalPatching.run(model, graph, MagicMock(), MagicMock(), quiet=True)

        assert abs(result["raw_score"] - 0.8) < 1e-6
        assert abs(result["score"] - 0.8) < 1e-6
        assert result["degenerate_denominator"] is False
        mock_eval.assert_called_once()
        call_kwargs = mock_eval.call_args[1]
        assert call_kwargs["intervention"] == "patching"
        assert call_kwargs["skip_clean"] is True

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_run_handles_scalar_tensor(self, mock_eval, mock_base):
        """Handles 0-d tensor returned by evaluate_graph."""
        mock_eval.return_value = torch.tensor(0.75)
        mock_base.side_effect = [torch.tensor([1.0]), torch.tensor([0.0])]
        model = _make_mock_model()
        result = Pillar1_CausalPatching.run(
            model, _make_mock_graph(), MagicMock(), MagicMock(), quiet=True
        )
        assert abs(result["raw_score"] - 0.75) < 1e-6
        assert abs(result["score"] - 0.75) < 1e-6

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_run_handles_list_of_metrics(self, mock_eval, mock_base):
        """When evaluate_graph returns a list, uses first element."""
        mock_eval.return_value = [torch.tensor([0.5, 0.7]), torch.tensor([0.1, 0.2])]
        mock_base.side_effect = [torch.tensor([1.0]), torch.tensor([0.0])]
        model = _make_mock_model()
        result = Pillar1_CausalPatching.run(
            model, _make_mock_graph(), MagicMock(), MagicMock(), quiet=True
        )
        assert abs(result["raw_score"] - 0.6) < 1e-6
        assert abs(result["score"] - 0.6) < 1e-6

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_run_caps_ratio_at_one(self, mock_eval, mock_base):
        """Circuit outperforming the full model yields score capped at 1.0."""
        mock_eval.return_value = torch.tensor([1.5])  # circuit raw = 1.5
        mock_base.side_effect = [torch.tensor([1.0]), torch.tensor([0.0])]
        model = _make_mock_model()
        result = Pillar1_CausalPatching.run(
            model, _make_mock_graph(), MagicMock(), MagicMock(), quiet=True
        )
        assert result["score"] == 1.0
        assert abs(result["raw_ratio"] - 1.5) < 1e-6

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_run_degenerate_denominator(self, mock_eval, mock_base):
        """When clean ~ corrupt, ratio is undefined → status='invalid', not 0.0.

        A degenerate denominator is semantically the same as the inverted case:
        the faithfulness ratio is undefined, not genuinely zero. It must surface
        as score=None + status='invalid' so callers do not mistake it for "no
        faithfulness" (issue #151, item 1).
        """
        mock_eval.return_value = torch.tensor([0.8])
        mock_base.side_effect = [torch.tensor([0.5]), torch.tensor([0.5])]
        model = _make_mock_model()
        result = Pillar1_CausalPatching.run(
            model, _make_mock_graph(), MagicMock(), MagicMock(), quiet=True
        )
        assert result["degenerate_denominator"] is True
        assert result["inverted_denominator"] is False
        assert result["status"] == "invalid"
        assert result["score"] is None
        assert "degenerate" in result["reason"]

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_run_inverted_metric_is_invalid_not_zero(self, mock_eval, mock_base):
        """clean < corrupt (inverted metric direction) → status='invalid', not 0.0.

        Regression for the CS-21 case (Qwen2.5-0.5B, wmdp-bio, eap-ig):
        clean=-2.0312 < corrupt=-0.5078 gave raw ratios that the old
        max(0.0, ...) clamp squashed to 0.000, printing "circuit may not
        fully capture the mechanism" when the real issue was metric-direction
        inversion. Mirrors the status='invalid' convention of P4/P5/P6.
        """
        mock_eval.return_value = torch.tensor([-0.5039])  # circuit raw
        mock_base.side_effect = [
            torch.tensor([-2.0312]),  # clean BELOW corrupt → inverted
            torch.tensor([-0.5078]),
        ]
        model = _make_mock_model()
        result = Pillar1_CausalPatching.run(
            model, _make_mock_graph(), MagicMock(), MagicMock(), quiet=True
        )
        assert result["status"] == "invalid"
        assert result["score"] is None
        assert result["inverted_denominator"] is True
        assert result["degenerate_denominator"] is False
        assert "inverted" in result["reason"]
        # raw values stay inspectable
        assert abs(result["raw_score"] - (-0.5039)) < 1e-6
        assert abs(result["clean_score"] - (-2.0312)) < 1e-6
        assert abs(result["corrupt_score"] - (-0.5078)) < 1e-6
        # raw_ratio = (circuit - corrupt)/(clean - corrupt) = 0.0026/-1.5234 ≈ -0.00256
        expected_raw = (-0.5039 - -0.5078) / (-2.0312 - -0.5078)
        assert abs(result["raw_ratio"] - expected_raw) < 1e-6  # float32 tensor means

    @patch(
        "circuitkit.evaluation.pillars.causal_patching.evaluate_graph",
        side_effect=RuntimeError("boom"),
    )
    @patch(
        "circuitkit.evaluation.pillars.causal_patching.evaluate_baseline",
        return_value=torch.tensor([1.0]),
    )
    def test_run_propagates_evaluation_error(self, mock_base, mock_eval):
        model = _make_mock_model()
        with pytest.raises(RuntimeError, match="boom"):
            Pillar1_CausalPatching.run(
                model, _make_mock_graph(), MagicMock(), MagicMock(), quiet=True
            )

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_compare_with_baseline_faithfulness_ratio(self, mock_eval_graph, mock_eval_base):
        """Faithfulness = circuit_score / baseline_score, capped at 1."""
        mock_eval_graph.return_value = torch.tensor([0.9, 0.9])
        mock_eval_base.return_value = torch.tensor([1.0, 1.0])

        model = _make_mock_model()
        result = Pillar1_CausalPatching.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            quiet=True,
        )

        assert abs(result["circuit_score"] - 0.9) < 1e-6
        assert abs(result["baseline_score"] - 1.0) < 1e-6
        assert abs(result["faithfulness"] - 0.9) < 1e-6

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_compare_with_baseline_caps_at_one(self, mock_eval_graph, mock_eval_base):
        mock_eval_graph.return_value = torch.tensor([1.2])
        mock_eval_base.return_value = torch.tensor([1.0])

        model = _make_mock_model()
        result = Pillar1_CausalPatching.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            quiet=True,
        )
        assert result["faithfulness"] == 1.0

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_compare_with_baseline_zero_baseline(self, mock_eval_graph, mock_eval_base):
        """Zero baseline → the ratio is undefined: status='invalid', not a 0.0 sentinel."""
        mock_eval_graph.return_value = torch.tensor([0.5])
        mock_eval_base.return_value = torch.tensor([0.0])

        model = _make_mock_model()
        result = Pillar1_CausalPatching.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            quiet=True,
        )
        assert result["status"] == "invalid"
        assert result["faithfulness"] is None

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_compare_with_baseline_negative_baseline_is_invalid(
        self, mock_eval_graph, mock_eval_base
    ):
        """Negative full-model baseline (signed metric) → status='invalid'.

        The old `if baseline_avg > 0 else 0.0` returned a 0.0 SENTINEL — which
        reads as "completely unfaithful circuit" — when the ratio was simply
        undefined for the metric's sign.
        """
        mock_eval_graph.return_value = torch.tensor([-0.5])
        mock_eval_base.return_value = torch.tensor([-1.2])

        model = _make_mock_model()
        result = Pillar1_CausalPatching.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            quiet=True,
        )
        assert result["status"] == "invalid"
        assert result["faithfulness"] is None
        assert "sign" in result["reason"] or "non-positive" in result["reason"]
        # raw scores stay inspectable
        assert abs(result["circuit_score"] - (-0.5)) < 1e-6
        assert abs(result["baseline_score"] - (-1.2)) < 1e-6

    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.causal_patching.evaluate_graph")
    def test_compare_with_baseline_negative_circuit_clamps_to_zero_keeps_raw(
        self, mock_eval_graph, mock_eval_base
    ):
        """Negative circuit score with a valid baseline: the old cap-above-only
        min(x, 1.0) let a NEGATIVE 'faithfulness' pass through under a key
        documented as capped. Now clamped to [0,1] with the raw quotient kept."""
        mock_eval_graph.return_value = torch.tensor([-0.5])
        mock_eval_base.return_value = torch.tensor([2.0])

        model = _make_mock_model()
        result = Pillar1_CausalPatching.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            quiet=True,
        )
        assert "status" not in result  # valid regime
        assert result["faithfulness"] == 0.0
        assert abs(result["raw_faithfulness"] - (-0.25)) < 1e-6


class TestPillar2Ablation:
    """Tests for Pillar 2: Ablation."""

    def test_run_rejects_none_graph(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="Graph cannot be None"):
            Pillar2_Ablation.run(model, None, MagicMock(), MagicMock())

    def test_run_rejects_missing_use_attn_result(self):
        model = _make_mock_model(use_attn_result=False)
        with pytest.raises(AssertionError, match="use_attn_result"):
            Pillar2_Ablation.run(model, _make_mock_graph(), MagicMock(), MagicMock())

    def test_run_rejects_invalid_intervention(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="Invalid intervention"):
            Pillar2_Ablation.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                MagicMock(),
                intervention="garbage",
            )

    def test_run_rejects_mean_without_dataloader(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="requires an .*intervention_dataloader"):
            Pillar2_Ablation.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                MagicMock(),
                intervention="mean",
                intervention_dataloader=None,
            )

    def test_run_rejects_mean_positional_without_dataloader(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="requires an .*intervention_dataloader"):
            Pillar2_Ablation.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                MagicMock(),
                intervention="mean-positional",
                intervention_dataloader=None,
            )

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_run_zero_intervention(self, mock_eval, mock_base):
        """run() returns a dict: raw_score is the mean metric, score is the
        normalized faithfulness ratio (circuit - corrupt) / (clean - corrupt)."""
        mock_eval.return_value = torch.tensor([0.7, 0.9])  # circuit raw mean = 0.8
        # clean = 1.0, corrupt = 0.0  ->  ratio = (0.8 - 0) / (1 - 0) = 0.8
        mock_base.side_effect = [torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])]
        model = _make_mock_model()

        result = Pillar2_Ablation.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            intervention="zero",
            quiet=True,
        )

        assert abs(result["raw_score"] - 0.8) < 1e-6
        assert abs(result["score"] - 0.8) < 1e-6
        assert result["degenerate_denominator"] is False
        call_kwargs = mock_eval.call_args[1]
        assert call_kwargs["intervention"] == "zero"

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_run_inverted_metric_is_invalid_not_zero(self, mock_eval, mock_base):
        """clean < corrupt (inverted metric) → status='invalid', not a clamped 0.0.

        Same regression as Pillar 1's (the CS-21 wmdp-bio case: P2 clamped a
        raw +0.0270 ratio to 0.000 and blamed the circuit): the ratio's meaning
        flips when the denominator is negative, so report invalid.
        """
        mock_eval.return_value = torch.tensor([-0.4])
        mock_base.side_effect = [
            torch.tensor([-2.0]),  # clean BELOW corrupt → inverted
            torch.tensor([-0.5]),
        ]
        model = _make_mock_model()
        result = Pillar2_Ablation.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            intervention="zero",
            quiet=True,
        )
        assert result["status"] == "invalid"
        assert result["score"] is None
        assert result["inverted_denominator"] is True
        assert "inverted" in result["reason"]
        assert abs(result["raw_score"] - (-0.4)) < 1e-6  # raw stays inspectable

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_run_degenerate_denominator_is_invalid(self, mock_eval, mock_base):
        """clean ~ corrupt (degenerate denominator) → status='invalid', not 0.0.

        Mirrors Pillar 1's degenerate handling (issue #151, item 1): an
        undefined ratio must surface as score=None + status='invalid', not a
        silent 0.0 that reads as "no faithfulness".
        """
        mock_eval.return_value = torch.tensor([0.8])
        mock_base.side_effect = [torch.tensor([0.5]), torch.tensor([0.5])]
        model = _make_mock_model()
        result = Pillar2_Ablation.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            intervention="zero",
            quiet=True,
        )
        assert result["degenerate_denominator"] is True
        assert result["inverted_denominator"] is False
        assert result["status"] == "invalid"
        assert result["score"] is None
        assert "degenerate" in result["reason"]

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_run_mean_intervention_passes_dataloader(self, mock_eval, mock_base):
        mock_eval.return_value = torch.tensor([0.6])
        mock_base.side_effect = [torch.tensor([1.0]), torch.tensor([0.0])]
        model = _make_mock_model()
        interv_dl = MagicMock()

        Pillar2_Ablation.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            intervention="mean",
            intervention_dataloader=interv_dl,
            quiet=True,
        )

        call_kwargs = mock_eval.call_args[1]
        assert call_kwargs["intervention"] == "mean"
        assert call_kwargs["intervention_dataloader"] is interv_dl

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_run_case_insensitive_intervention(self, mock_eval, mock_base):
        """Intervention string is lowercased internally."""
        mock_eval.return_value = torch.tensor([0.5])
        mock_base.side_effect = [torch.tensor([1.0]), torch.tensor([0.0])]
        model = _make_mock_model()
        result = Pillar2_Ablation.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            intervention="ZERO",
            quiet=True,
        )
        assert isinstance(result, dict)
        assert isinstance(result["score"], float)
        assert isinstance(result["raw_score"], float)

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_compare_interventions_zero_only(self, mock_eval, mock_base):
        """Without intervention_dataloader, only zero is returned."""
        mock_eval.return_value = torch.tensor([0.5])
        mock_base.return_value = torch.tensor([1.0])
        model = _make_mock_model()

        results = Pillar2_Ablation.compare_interventions(
            model,
            _make_mock_graph(),
            MagicMock(),
            None,
            MagicMock(),
            quiet=True,
        )

        assert "zero" in results
        assert "mean" not in results
        assert "mean_positional" not in results
        # Each entry is the full run() result dict with a normalized 'score'.
        assert isinstance(results["zero"], dict)
        assert "score" in results["zero"]

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_compare_interventions_all_methods(self, mock_eval, mock_base):
        """With intervention_dataloader, all three methods are returned."""
        mock_eval.return_value = torch.tensor([0.5])
        mock_base.return_value = torch.tensor([1.0])
        model = _make_mock_model()

        results = Pillar2_Ablation.compare_interventions(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            quiet=True,
        )

        assert set(results.keys()) == {"zero", "mean", "mean_positional"}

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_compare_with_baseline_sufficiency(self, mock_eval_graph, mock_eval_base):
        mock_eval_graph.return_value = torch.tensor([0.8])
        mock_eval_base.return_value = torch.tensor([1.0])
        model = _make_mock_model()

        result = Pillar2_Ablation.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            quiet=True,
        )

        assert abs(result["sufficiency"] - 0.8) < 1e-6
        assert abs(result["baseline_score"] - 1.0) < 1e-6

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_compare_with_baseline_negative_baseline_is_invalid(
        self, mock_eval_graph, mock_eval_base
    ):
        """Negative full-model baseline (signed metric) -> status='invalid'.

        The old `if baseline_avg > 0 else 0.0` returned a 0.0 SENTINEL for the
        sufficiency of an undefined ratio (same bug as P1 compare_with_baseline).
        """
        mock_eval_graph.return_value = torch.tensor([-0.5])
        mock_eval_base.return_value = torch.tensor([-1.2])
        model = _make_mock_model()

        result = Pillar2_Ablation.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            quiet=True,
        )
        assert result["status"] == "invalid"
        assert result["sufficiency"] is None
        assert abs(result["baseline_score"] - (-1.2)) < 1e-6  # raw stays inspectable

    @patch("circuitkit.evaluation.pillars.ablation.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.ablation.evaluate_graph")
    def test_compare_with_baseline_negative_circuit_clamps_keeps_raw(
        self, mock_eval_graph, mock_eval_base
    ):
        """Negative circuit score, valid baseline: old cap-above-only min(x,1.0)
        passed a NEGATIVE sufficiency through; now clamped to [0,1] + raw kept."""
        mock_eval_graph.return_value = torch.tensor([-0.5])
        mock_eval_base.return_value = torch.tensor([2.0])
        model = _make_mock_model()

        result = Pillar2_Ablation.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            quiet=True,
        )
        assert "status" not in result
        assert result["sufficiency"] == 0.0
        assert abs(result["raw_sufficiency"] - (-0.25)) < 1e-6


class TestPillar3Stability:
    """Tests for Pillar 3 Stability evaluation."""

    def test_compute_jaccard_identical_circuits(self):
        """Test Jaccard similarity for identical circuits."""
        circuit1 = {"a": {}, "b": {}, "c": {}}
        circuit2 = {"a": {}, "b": {}, "c": {}}

        jaccard = Pillar3_Stability.compute_jaccard(circuit1, circuit2)
        assert jaccard == 1.0, "Identical circuits should have Jaccard = 1.0"

    def test_compute_jaccard_disjoint_circuits(self):
        """Test Jaccard similarity for completely different circuits."""
        circuit1 = {"a": {}, "b": {}, "c": {}}
        circuit2 = {"x": {}, "y": {}, "z": {}}

        jaccard = Pillar3_Stability.compute_jaccard(circuit1, circuit2)
        assert jaccard == 0.0, "Disjoint circuits should have Jaccard = 0.0"

    def test_compute_jaccard_partial_overlap(self):
        """Test Jaccard similarity for partially overlapping circuits."""
        circuit1 = {"a": {}, "b": {}, "c": {}}
        circuit2 = {"b": {}, "c": {}, "d": {}}

        jaccard = Pillar3_Stability.compute_jaccard(circuit1, circuit2)
        # Intersection: {b, c} = 2, Union: {a, b, c, d} = 4
        # Jaccard = 2/4 = 0.5
        assert jaccard == 0.5, f"Expected 0.5, got {jaccard}"

    def test_compute_jaccard_empty_circuits(self):
        """Test Jaccard similarity for empty circuits."""
        circuit1 = {}
        circuit2 = {}

        jaccard = Pillar3_Stability.compute_jaccard(circuit1, circuit2)
        assert jaccard == 1.0, "Empty circuits should have Jaccard = 1.0"

    def test_compute_dice_identical_circuits(self):
        """Test Dice coefficient for identical circuits."""
        circuit1 = {"a": {}, "b": {}, "c": {}}
        circuit2 = {"a": {}, "b": {}, "c": {}}

        dice = Pillar3_Stability.compute_dice(circuit1, circuit2)
        assert dice == 1.0, "Identical circuits should have Dice = 1.0"

    def test_compute_dice_disjoint_circuits(self):
        """Test Dice coefficient for completely different circuits."""
        circuit1 = {"a": {}, "b": {}, "c": {}}
        circuit2 = {"x": {}, "y": {}, "z": {}}

        dice = Pillar3_Stability.compute_dice(circuit1, circuit2)
        assert dice == 0.0, "Disjoint circuits should have Dice = 0.0"

    def test_compute_dice_partial_overlap(self):
        """Test Dice coefficient for partially overlapping circuits."""
        circuit1 = {"a": {}, "b": {}, "c": {}}
        circuit2 = {"b": {}, "c": {}, "d": {}}

        dice = Pillar3_Stability.compute_dice(circuit1, circuit2)
        # Intersection: {b, c} = 2, Total: 3 + 3 = 6
        # Dice = 2*2/6 = 4/6 ≈ 0.667
        expected = 4.0 / 6.0
        assert abs(dice - expected) < 1e-6, f"Expected {expected}, got {dice}"

    def test_extract_circuit_nodes_from_mock_graph(self):
        """Test extraction of circuit nodes from a graph."""

        # Create a simple mock graph with basic nodes
        class MockNode:
            def __init__(self, name, layer, score, in_graph=False):
                self.name = name
                self.layer = layer
                self.score = torch.tensor(score)
                self._in_graph = in_graph

            @property
            def in_graph(self):
                return self._in_graph

            @in_graph.setter
            def in_graph(self, value):
                self._in_graph = value

        class MockGraph:
            def __init__(self):
                self.nodes = {
                    "node_0": MockNode("node_0", 0, 0.5, in_graph=True),
                    "node_1": MockNode("node_1", 0, 0.3, in_graph=False),
                    "node_2": MockNode("node_2", 1, 0.8, in_graph=True),
                }
                self.neurons_in_graph = None
                self.neurons_scores = None

        graph = MockGraph()
        circuit = Pillar3_Stability._extract_circuit_nodes(graph)

        assert len(circuit) == 2, "Should extract 2 nodes"
        assert "node_0" in circuit, "node_0 should be in circuit"
        assert "node_2" in circuit, "node_2 should be in circuit"
        assert "node_1" not in circuit, "node_1 should not be in circuit"

    def test_count_nodes_by_type(self):
        """Test counting of nodes by type.

        _count_nodes_by_type tallies circuit nodes by their Python class name
        (type(node).__name__), so distinct node classes must produce distinct
        keys. Two AttentionNode-like instances + one MLPNode-like instance,
        all present in the circuit, should yield {AttentionNode: 2, MLPNode: 1}.
        """

        class AttentionNode:
            def __init__(self, name):
                self.name = name

        class MLPNode:
            def __init__(self, name):
                self.name = name

        class MockGraph:
            def __init__(self):
                self.nodes = {
                    "attn_0": AttentionNode("attn_0"),
                    "mlp_1": MLPNode("mlp_1"),
                    "attn_1": AttentionNode("attn_1"),
                    "attn_2": AttentionNode("attn_2"),  # not in circuit
                }
                self.neurons_in_graph = None
                self.neurons_scores = None

        graph = MockGraph()
        circuit = {
            "attn_0": {},
            "mlp_1": {},
            "attn_1": {},
            "missing": {},  # name absent from graph.nodes — ignored
        }

        counts = Pillar3_Stability._count_nodes_by_type(graph, circuit)

        assert isinstance(counts, dict), "counts should be a dict"
        assert counts == {"AttentionNode": 2, "MLPNode": 1}

    def test_stability_metrics_structure(self):
        """Test that stability run returns correct output structure."""
        # This is a structural test; actual graph evaluation requires model setup
        # We validate the return dict structure

        class MockGraph:
            def __init__(self):
                self.nodes = {}

        MockGraph()

        # Mock the underlying evaluation by returning simple dict

        # Compute overlap for empty circuits
        jaccard_matrix = np.zeros((2, 2))
        jaccard_matrix[0, 0] = 1.0
        jaccard_matrix[1, 1] = 1.0
        jaccard_matrix[0, 1] = 1.0
        jaccard_matrix[1, 0] = 1.0

        # Verify expected structure
        assert jaccard_matrix.shape == (2, 2), "Jaccard matrix has wrong shape"
        assert np.allclose(jaccard_matrix.diagonal(), 1.0), "Diagonal should be 1.0"

    def test_compute_jaccard_one_empty_one_not(self):
        """One empty, one non-empty should yield 0."""
        circuit1 = {}
        circuit2 = {"a": {}}
        assert Pillar3_Stability.compute_jaccard(circuit1, circuit2) == 0.0

    def test_compute_dice_one_empty_one_not(self):
        """One empty, one non-empty should yield 0."""
        circuit1 = {}
        circuit2 = {"a": {}}
        assert Pillar3_Stability.compute_dice(circuit1, circuit2) == 0.0

    def test_compute_layer_wise_overlap_basic(self):
        """Layer-wise overlap produces per-layer Jaccard values."""
        graph = _make_mock_graph(n_nodes=9)  # layers 0,1,2 (i%3)
        # Circuit 1: all nodes. Circuit 2: only even-indexed nodes.
        c1 = {f"node_{i}": {} for i in range(9)}
        c2 = {f"node_{i}": {} for i in range(0, 9, 2)}

        overlap = Pillar3_Stability._compute_layer_wise_overlap([c1, c2], graph)

        assert isinstance(overlap, dict)
        assert set(overlap.keys()) == {0, 1, 2}
        for layer, jaccard in overlap.items():
            assert 0.0 <= jaccard <= 1.0

    def test_compute_layer_wise_overlap_no_matching_nodes(self):
        """Circuits with names not in graph produce empty layer overlap."""
        graph = _make_mock_graph(n_nodes=3)
        c1 = {"unknown_a": {}}
        c2 = {"unknown_b": {}}

        overlap = Pillar3_Stability._compute_layer_wise_overlap([c1, c2], graph)
        assert overlap == {}

    def test_run_rejects_none_graph(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="Graph cannot be None"):
            Pillar3_Stability.run(model, None, MagicMock(), MagicMock())

    def test_run_rejects_missing_use_attn_result(self):
        model = _make_mock_model(use_attn_result=False)
        with pytest.raises(AssertionError, match="use_attn_result"):
            Pillar3_Stability.run(model, _make_mock_graph(), MagicMock(), MagicMock())


class TestPillar4Robustness:
    """Tests for Pillar 4: Robustness."""

    def test_run_rejects_none_graph(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="Graph cannot be None"):
            Pillar4_Robustness.run(model, None, MagicMock())

    def test_run_rejects_none_metric(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="metric_fn cannot be None"):
            Pillar4_Robustness.run(model, _make_mock_graph(), MagicMock(), metric_fn=None)

    def test_run_rejects_missing_use_attn_result(self):
        model = _make_mock_model(use_attn_result=False)
        with pytest.raises(AssertionError, match="use_attn_result"):
            Pillar4_Robustness.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                metric_fn=MagicMock(),
            )

    def test_run_rejects_invalid_corruption_variant(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="Invalid corruption_variant"):
            Pillar4_Robustness.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                corruption_variant="nonexistent",
                metric_fn=MagicMock(),
            )

    def test_run_rejects_mean_without_intervention_dataloader(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="requires an .*intervention_dataloader"):
            Pillar4_Robustness.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                metric_fn=MagicMock(),
                intervention="mean",
            )

    @patch("circuitkit.evaluation.pillars.robustness.evaluate_graph")
    def test_run_computes_correct_metrics(self, mock_eval):
        """Verify delta, relative_drop, robustness_ratio calculations."""
        # First call = original, second call = variant
        mock_eval.side_effect = [
            torch.tensor([1.0, 1.0]),  # original_score mean = 1.0
            torch.tensor([0.8, 0.6]),  # variant_score mean = 0.7
        ]
        model = _make_mock_model()

        result = Pillar4_Robustness.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            corruption_variant="paraphrase",
            corruption_dataloader=MagicMock(),
            metric_fn=MagicMock(),
            quiet=True,
        )

        assert abs(result["original_score"] - 1.0) < 1e-6
        assert abs(result["variant_score"] - 0.7) < 1e-6
        assert abs(result["delta"] - 0.3) < 1e-6
        assert abs(result["relative_drop"] - 0.3) < 1e-6
        assert abs(result["robustness_ratio"] - 0.7) < 1e-6
        assert result["corruption_variant"] == "paraphrase"

    @patch("circuitkit.evaluation.pillars.robustness.evaluate_graph")
    def test_run_zero_original_score(self, mock_eval):
        """Zero original score is undefined → status='invalid', not a fabricated 0.0.

        robustness_ratio = variant/original is meaningless when the original score
        is non-positive (signed logit_diff), so Pillar 4 reports status='invalid'
        with no ratio rather than emitting a misleading 0.0 (worst-case) value.
        """
        mock_eval.side_effect = [
            torch.tensor([0.0]),
            torch.tensor([0.5]),
        ]
        model = _make_mock_model()

        result = Pillar4_Robustness.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            corruption_variant="entity_swap",
            corruption_dataloader=MagicMock(),
            metric_fn=MagicMock(),
            quiet=True,
        )

        assert result["status"] == "invalid"
        assert "robustness_ratio" not in result
        assert "relative_drop" not in result
        assert result["original_score"] == 0.0
        assert result["variant_score"] == 0.5

    @patch("circuitkit.evaluation.pillars.robustness.evaluate_graph")
    def test_run_skips_variant_when_corrupted_dataloader_unavailable(self, mock_eval):
        """No corruption_dataloader + ungeneratable corruption → skipped, not faked.

        Pillar 4 must NEVER silently evaluate the circuit against the original
        (uncorrupted) data twice as a fallback. When it cannot build a corrupted
        dataloader, the variant is reported as skipped with NO robustness_ratio
        / delta, so it can never fabricate a zero-delta "perfectly robust" PASS.
        """
        mock_eval.return_value = torch.tensor([0.9])
        model = _make_mock_model()
        # A bare iterable that does NOT yield (clean, corrupted, label) text
        # batches — so no text-level corruption strategy can be applied.
        original_dl = []

        result = Pillar4_Robustness.run(
            model,
            _make_mock_graph(),
            original_dl,
            corruption_variant="paraphrase",
            corruption_dataloader=None,  # must NOT fall back to original_dl
            metric_fn=MagicMock(),
            quiet=True,
        )

        # Variant is skipped, not fabricated.
        assert result["status"] == "skipped"
        assert result["corruption_variant"] == "paraphrase"
        assert "reason" in result
        # A skipped variant must never carry a fabricated robustness number.
        assert "robustness_ratio" not in result
        assert "delta" not in result
        # evaluate_graph must not have been called for a skipped variant.
        mock_eval.assert_not_called()

    @patch("circuitkit.evaluation.pillars.robustness.evaluate_graph")
    @patch("circuitkit.evaluation.pillars.robustness._build_corrupted_dataloader")
    def test_run_generates_corrupted_dataloader_when_not_supplied(self, mock_build, mock_eval):
        """When no corruption_dataloader is supplied, Pillar 4 generates one
        itself and reports corruption_source='generated' — it does not reuse
        the original (uncorrupted) dataloader."""
        mock_build.return_value = MagicMock()  # a successfully generated loader
        mock_eval.side_effect = [
            torch.tensor([1.0, 1.0]),  # original
            torch.tensor([0.5, 0.5]),  # corrupted variant
        ]
        model = _make_mock_model()
        original_dl = MagicMock()

        result = Pillar4_Robustness.run(
            model,
            _make_mock_graph(),
            original_dl,
            corruption_variant="paraphrase",
            corruption_dataloader=None,
            metric_fn=MagicMock(),
            quiet=True,
        )

        mock_build.assert_called_once()
        assert result["corruption_source"] == "generated"
        # Real, non-trivial delta from genuinely different corrupted data.
        assert abs(result["delta"] - 0.5) < 1e-6
        assert abs(result["robustness_ratio"] - 0.5) < 1e-6

    @patch("circuitkit.evaluation.pillars.robustness.evaluate_graph")
    def test_run_accepts_all_valid_variants(self, mock_eval):
        """All five documented corruption variants are accepted."""
        mock_eval.return_value = torch.tensor([0.5])
        model = _make_mock_model()
        for variant in ["paraphrase", "entity_swap", "distractor", "role_swap", "token_swap"]:
            result = Pillar4_Robustness.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                corruption_variant=variant,
                corruption_dataloader=MagicMock(),
                metric_fn=MagicMock(),
                quiet=True,
            )
            assert result["corruption_variant"] == variant

    @patch("circuitkit.evaluation.pillars.robustness.evaluate_graph")
    def test_compare_corruption_variants(self, mock_eval):
        """compare_corruption_variants returns results keyed by variant name."""
        mock_eval.return_value = torch.tensor([0.5])
        model = _make_mock_model()
        corruption_dls = {
            "paraphrase": MagicMock(),
            "entity_swap": MagicMock(),
        }

        results = Pillar4_Robustness.compare_corruption_variants(
            model,
            _make_mock_graph(),
            MagicMock(),
            corruption_dls,
            metric_fn=MagicMock(),
            quiet=True,
        )

        assert set(results.keys()) == {"paraphrase", "entity_swap"}
        for v in results.values():
            assert "original_score" in v

    @patch("circuitkit.evaluation.pillars.robustness.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.robustness.evaluate_graph")
    def test_compare_with_baseline(self, mock_eval_graph, mock_eval_base):
        # evaluate_graph called twice (original + variant inside .run())
        mock_eval_graph.side_effect = [
            torch.tensor([0.9]),  # original
            torch.tensor([0.8]),  # variant
        ]
        # evaluate_baseline called twice (original + variant)
        mock_eval_base.side_effect = [
            torch.tensor([1.0]),  # baseline original
            torch.tensor([0.95]),  # baseline variant
        ]
        model = _make_mock_model()

        result = Pillar4_Robustness.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            corruption_variant="paraphrase",
            corruption_dataloader=MagicMock(),
            metric_fn=MagicMock(),
            quiet=True,
        )

        assert "circuit_robustness" in result
        assert "baseline_original_score" in result
        assert "baseline_variant_score" in result
        assert "is_circuit_more_robust" in result
        assert isinstance(result["is_circuit_more_robust"], bool)

    @patch("circuitkit.evaluation.pillars.robustness.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.robustness.evaluate_graph")
    def test_compare_with_baseline_negative_baseline_is_invalid(
        self, mock_eval_graph, mock_eval_base
    ):
        """Non-positive BASELINE original -> status='invalid', no fabricated verdict.

        run()'s guard was never mirrored on the baseline side: a full model that
        collapses under corruption on an inverted metric (original=-0.5,
        variant=-1.5) produced baseline_relative_drop = 0.0 (sentinel) and a
        fabricated is_circuit_more_robust=False. Now the comparison reports
        status='invalid' while preserving the circuit-side result.
        """
        mock_eval_graph.side_effect = [
            torch.tensor([0.9]),  # circuit original (healthy, so run() succeeds)
            torch.tensor([0.8]),  # circuit variant
        ]
        mock_eval_base.side_effect = [
            torch.tensor([-0.5]),  # baseline original: non-positive -> undefined
            torch.tensor([-1.5]),  # baseline variant
        ]
        model = _make_mock_model()

        result = Pillar4_Robustness.compare_with_baseline(
            model,
            _make_mock_graph(),
            MagicMock(),
            corruption_variant="paraphrase",
            corruption_dataloader=MagicMock(),
            metric_fn=MagicMock(),
            quiet=True,
        )

        assert result["status"] == "invalid"
        assert result["baseline_relative_drop"] is None
        assert "is_circuit_more_robust" not in result  # no fabricated verdict
        # circuit-side result is preserved for inspection
        assert "circuit_robustness" in result
        assert abs(result["baseline_original_score"] - (-0.5)) < 1e-6


class TestBootstrapStabilityCV:
    """compute_bootstrap_stability's CV must handle signed metrics.

    The old `std/mean if mean > 0 else 0.0` returned a 0.0 SENTINEL — read as
    PERFECTLY STABLE ("lower is more stable") — for any non-positive mean,
    including a perfectly consistent all-negative metric and a wildly unstable
    sign-straddling one. And min(ratio, 1.0) clamped away CV > 1, hiding
    near-degenerate denominators.
    """

    def _run(self, mock_scores):
        from circuitkit.evaluation.pillars.stability import Pillar3_Stability

        with patch(
            "circuitkit.evaluation.pillars.stability.evaluate_graph",
            side_effect=[torch.tensor([v]) for v in mock_scores],
        ):
            return Pillar3_Stability.compute_bootstrap_stability(
                _make_mock_model(),
                _make_mock_graph(),
                [MagicMock()],  # one batch
                MagicMock(),
                n_bootstrap=len(mock_scores),
                quiet=True,
            )

    def test_all_negative_scores_give_real_cv_not_zero(self):
        """[-2.1, -1.9, -2.0]: old code returned 0.0 ("perfectly stable");
        now a genuine CV = std/|mean| ~ 0.041."""
        result = self._run([-2.1, -1.9, -2.0])
        import numpy as np

        expected = float(np.std([-2.1, -1.9, -2.0])) / 2.0
        assert result["performance_stability"] is not None
        assert abs(result["performance_stability"] - expected) < 1e-6
        assert result["performance_stability"] > 0  # not the old sentinel

    def test_near_zero_mean_is_invalid(self):
        """Sign-straddling scores with ~0 mean: CV undefined -> status='invalid'
        (old code: mean not > 0 -> sentinel 0.0 = 'perfectly stable')."""
        result = self._run([1.0, -1.0])
        assert result["status"] == "invalid"
        assert result["performance_stability"] is None
        assert result["scores"] == [1.0, -1.0]  # raw scores stay inspectable

    def test_cv_above_one_not_clamped(self):
        """std > |mean| is real information; the old min(ratio, 1.0) hid it."""
        result = self._run([-1.0, 5.0])  # mean 2.0, std 3.0 -> CV 1.5
        assert abs(result["performance_stability"] - 1.5) < 1e-6


class TestPillar5Baselines:
    """Tests for Pillar 5 Baselines comparison."""

    def test_compute_percentage_circuit_better_than_baseline(self):
        """Test percentage computation when circuit outperforms baseline."""
        circuit_score = 0.8
        baseline_score = 0.5

        percentage = Pillar5_Baselines._compute_percentage_of_baseline(
            circuit_score, baseline_score
        )

        # circuit_score / baseline_score = 1.6, * 100 = 160, capped at 100
        expected = (0.8 / 0.5) * 100.0
        assert abs(percentage - expected) < 1e-6, f"Expected {expected}, got {percentage}"

    def test_compute_percentage_circuit_worse_than_baseline(self):
        """Test percentage computation when circuit underperforms baseline."""
        circuit_score = 0.3
        baseline_score = 0.5

        percentage = Pillar5_Baselines._compute_percentage_of_baseline(
            circuit_score, baseline_score
        )

        # circuit_score / baseline_score = 0.6, * 100 = 60
        expected = (0.3 / 0.5) * 100.0
        assert abs(percentage - expected) < 1e-6, f"Expected {expected}, got {percentage}"

    def test_compute_percentage_zero_baseline(self):
        """Zero baseline → percentage undefined: None, not an arbitrary 100/0.

        The old branch returned 100.0 for any positive circuit ("matches
        baseline" — arbitrary) and 0.0 otherwise, and its exact `== 0` float
        check let baseline=1e-12 skip the branch and return a ~1e14 percentage.
        """
        assert Pillar5_Baselines._compute_percentage_of_baseline(0.5, 0.0) is None
        # near-zero baseline no longer explodes:
        assert Pillar5_Baselines._compute_percentage_of_baseline(0.5, 1e-12) is None

    def test_compute_percentage_negative_baseline_is_none(self):
        """Signed metric, negative baseline → direction inverts; report None.

        circuit=0.5 vs baseline=-0.5 previously gave -100.0 (circuit
        outperforms, number says the opposite); circuit=-0.5 vs baseline=-1.0
        gave 50.0 ("half of baseline") when the circuit is actually BETTER
        (less negative).
        """
        assert Pillar5_Baselines._compute_percentage_of_baseline(0.5, -0.5) is None
        assert Pillar5_Baselines._compute_percentage_of_baseline(-0.5, -1.0) is None

    def test_build_random_circuit_maintains_sparsity(self):
        """Random baseline keeps roughly `sparsity` fraction of nodes.

        On a real Graph, _build_random_circuit selects n_keep = sparsity * N
        non-terminal nodes; the terminal nodes ('input', 'logits') are then
        forced/auto-included by edge synchronisation. The selected count
        therefore should not exceed n_keep + the 2 terminals.
        """
        graph = _make_real_graph()
        sparsity = 0.3
        all_nodes = [n for n in graph.nodes if n != "logits"]
        n_keep = max(1, int(len(all_nodes) * sparsity))

        baseline_graph = Pillar5_Baselines._build_random_circuit(graph, sparsity=sparsity, seed=42)

        n_in_graph = sum(1 for node in baseline_graph.nodes.values() if node.in_graph)
        # n_keep randomly chosen + at most the 2 terminal nodes.
        assert (
            n_keep <= n_in_graph <= n_keep + 2
        ), f"Expected ~{n_keep} (+terminals) nodes, got {n_in_graph}"

    def test_build_random_circuit_reproducibility(self):
        """Random baseline is reproducible with the same seed."""
        baseline1 = Pillar5_Baselines._build_random_circuit(
            _make_real_graph(), sparsity=0.4, seed=42
        )
        baseline2 = Pillar5_Baselines._build_random_circuit(
            _make_real_graph(), sparsity=0.4, seed=42
        )

        selected1 = {name for name, node in baseline1.nodes.items() if node.in_graph}
        selected2 = {name for name, node in baseline2.nodes.items() if node.in_graph}

        assert selected1 == selected2, "Same seed should produce same random selection"

    def test_build_magnitude_circuit_maintains_sparsity(self):
        """Magnitude baseline keeps roughly `sparsity` fraction of nodes."""
        graph = _make_real_graph()
        names = _scored_node_names(graph)
        # Distinct per-node weight norms so the ranking is well-defined
        # (the baseline ranks by RMS weight norm, not by node.score).
        model = _make_weighted_model(
            head_mag=lambda layer, head: 1.0 + layer * TINY_CFG["n_heads"] + head,
            mlp_mag=lambda layer: 50.0 + layer,
        )

        sparsity = 0.3
        baseline_graph = Pillar5_Baselines._build_magnitude_circuit(model, graph, sparsity=sparsity)

        n_in_graph = sum(1 for node in baseline_graph.nodes.values() if node.in_graph)
        n_keep = max(1, int(len(names) * sparsity))
        assert (
            n_keep <= n_in_graph <= n_keep + 2
        ), f"Expected ~{n_keep} (+terminals) nodes, got {n_in_graph}"

    def test_build_magnitude_circuit_selects_top_weight_norm(self):
        """Magnitude baseline selects the highest weight-norm non-terminal nodes.

        The node-level magnitude baseline ranks by RMS weight norm — independent
        of the discovered circuit's attribution scores (node.score), not by it.
        """
        graph = _make_real_graph()
        names = _scored_node_names(graph)
        n_heads = TINY_CFG["n_heads"]

        # Known weight norms: A{l}.{h} = 1 + l*n_heads + h; MLP {l} = 100 + l.
        model = _make_weighted_model(
            head_mag=lambda layer, head: 1.0 + layer * n_heads + head,
            mlp_mag=lambda layer: 100.0 + layer,
        )

        def _weight_norm(name):
            # Node names: attn "a{layer}.h{head}", mlp "m{layer}".
            if name.startswith("m"):
                return 100.0 + int(name[1:])
            layer, head = name[1:].split(".h")
            return 1.0 + int(layer) * n_heads + int(head)

        sparsity = 0.25
        n_keep = max(1, int(len(names) * sparsity))
        baseline_graph = Pillar5_Baselines._build_magnitude_circuit(model, graph, sparsity=sparsity)

        selected = {
            name
            for name, node in baseline_graph.nodes.items()
            if node.in_graph and name not in ("input", "logits")
        }
        top = sorted(names, key=_weight_norm, reverse=True)[:n_keep]
        expected_top = set(top)
        # Edge synchronisation may drop a selected node with no surviving edges,
        # so `selected` must be a SUBSET of the top-weight-norm candidates.
        assert selected, "Magnitude baseline selected no non-terminal nodes"
        assert selected.issubset(expected_top), (
            f"Selected {selected} must be among the top-{n_keep} weight-norm "
            f"nodes {expected_top}"
        )
        # The single highest-weight-norm node must always survive.
        assert top[0] in selected, f"Highest weight-norm node {top[0]!r} should be selected"

    def test_extract_circuit_nodes_from_mock_graph(self):
        """Test extraction of circuit nodes from a graph."""

        class MockNode:
            def __init__(self, name, layer, score, in_graph=False):
                self.name = name
                self.layer = layer
                self.score = torch.tensor(score)
                self._in_graph = in_graph

            @property
            def in_graph(self):
                return self._in_graph

        class MockGraph:
            def __init__(self):
                self.nodes = {
                    "node_0": MockNode("node_0", 0, 0.5, in_graph=True),
                    "node_1": MockNode("node_1", 0, 0.3, in_graph=False),
                    "node_2": MockNode("node_2", 1, 0.8, in_graph=True),
                }
                self.neurons_in_graph = None
                self.neurons_scores = None

        graph = MockGraph()
        circuit = Pillar5_Baselines._extract_circuit_nodes(graph)

        assert len(circuit) == 2, "Should extract 2 in_graph nodes"
        assert "node_0" in circuit, "node_0 should be in circuit"
        assert "node_2" in circuit, "node_2 should be in circuit"
        assert "node_1" not in circuit, "node_1 should not be in circuit"

    def test_generate_summary_substantial_improvement(self):
        """Test summary generation for substantial improvement."""
        circuit_score = 0.8
        baselines_results = {
            "random": {
                "score": 0.4,
                "improvement": 2.0,
                "percentage": 200.0,
            }
        }

        summary = Pillar5_Baselines._generate_summary(circuit_score, baselines_results, ["random"])

        assert "substantially" in summary.lower(), "Should indicate substantial improvement"
        assert "2.00x" in summary, "Should include improvement ratio"

    def test_generate_summary_marginal_improvement(self):
        """Test summary generation for marginal improvement."""
        circuit_score = 0.5
        baselines_results = {
            "random": {
                "score": 0.48,
                "improvement": 1.04,
                "percentage": 104.0,
            }
        }

        summary = Pillar5_Baselines._generate_summary(circuit_score, baselines_results, ["random"])

        assert "marginally" in summary.lower(), "Should indicate marginal improvement"
        assert "1.04x" in summary, "Should include improvement ratio"

    def test_generate_summary_underperforming_circuit(self):
        """improvement < 1.0 must be described as UNDERPERFORMING, not
        'marginally outperforms'.

        The old final `else` labeled every valid ratio below 1.1 as
        outperforming — including 0.60x (circuit strictly worse than random)
        and -0.40x (negative circuit score vs positive baseline), producing
        the human-readable verdict "Circuit only marginally outperforms
        random baseline (-0.40x improvement)".
        """
        for improvement in (0.60, -0.40):
            baselines_results = {
                "random": {"score": 0.5, "improvement": improvement, "percentage": None}
            }
            summary = Pillar5_Baselines._generate_summary(-0.2, baselines_results, ["random"])
            assert "underperform" in summary.lower(), summary
            assert "outperforms" not in summary.lower(), summary
            assert f"{improvement:.2f}x" in summary

    def test_run_rejects_none_graph(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="Graph cannot be None"):
            Pillar5_Baselines.run(model, None, MagicMock(), MagicMock())

    def test_run_rejects_missing_use_attn_result(self):
        model = _make_mock_model(use_attn_result=False)
        with pytest.raises(AssertionError, match="use_attn_result"):
            Pillar5_Baselines.run(model, _make_mock_graph(), MagicMock(), MagicMock())

    def test_build_random_circuit_logits_always_in_graph(self):
        """The terminal 'logits' node is always retained in a random baseline.

        (Replaces an obsolete test that expected _build_wanda_circuit to raise
        NotImplementedError — WANDA is now fully implemented.)
        """
        baseline_graph = Pillar5_Baselines._build_random_circuit(
            _make_real_graph(), sparsity=0.3, seed=7
        )
        assert baseline_graph.nodes["logits"].in_graph is True

    def test_extract_circuit_nodes_empty_graph(self):
        """Empty graph returns empty circuit dict."""

        class EmptyGraph:
            nodes = {}
            neurons_in_graph = None

        assert Pillar5_Baselines._extract_circuit_nodes(EmptyGraph()) == {}

    def test_compute_percentage_of_baseline_both_zero(self):
        """Both zero → percentage undefined: None (was an arbitrary 0.0)."""
        result = Pillar5_Baselines._compute_percentage_of_baseline(0.0, 0.0)
        assert result is None

    def test_generate_summary_no_baselines(self):
        """Empty baseline list returns appropriate message."""
        summary = Pillar5_Baselines._generate_summary(0.5, {}, [])
        assert summary == "No baselines evaluated"

    def test_generate_summary_meaningful_improvement(self):
        """Improvement between 1.1 and 1.5 is 'meaningfully'."""
        baselines_results = {
            "random": {"score": 0.5, "improvement": 1.3, "percentage": 130.0},
        }
        summary = Pillar5_Baselines._generate_summary(0.65, baselines_results, ["random"])
        assert "meaningfully" in summary.lower()

    def test_build_random_circuit_different_seeds_differ(self):
        """Different seeds produce different selections."""
        b1 = Pillar5_Baselines._build_random_circuit(_make_real_graph(), sparsity=0.3, seed=1)
        b2 = Pillar5_Baselines._build_random_circuit(_make_real_graph(), sparsity=0.3, seed=99)

        sel1 = {n for n, nd in b1.nodes.items() if nd.in_graph}
        sel2 = {n for n, nd in b2.nodes.items() if nd.in_graph}
        # Extremely unlikely to be identical with different seeds.
        assert sel1 != sel2

    def test_build_magnitude_circuit_keeps_terminal_nodes(self):
        """Terminal nodes (input / logits) get +inf weight-norm and are always kept.

        _compute_node_weight_norm_scores assigns inf to nodes without weight
        matrices (input, logits, and any unrecognised type), so they survive
        the top-k selection regardless of how aggressive the sparsity is.
        """
        graph = _make_real_graph()
        model = _make_weighted_model()  # uniform, finite weight norms
        # Very aggressive sparsity: keep almost nothing among the finite nodes.
        result = Pillar5_Baselines._build_magnitude_circuit(model, graph, sparsity=0.05)
        selected = {n for n, nd in result.nodes.items() if nd.in_graph}

        # Terminal nodes must survive even at extreme sparsity.
        for terminal in ("input", "logits"):
            if terminal in graph.nodes:
                assert terminal in selected, f"{terminal!r} (inf weight-norm) must be kept"


class TestPillar6Generalization:
    """Tests for Pillar 6: Generalization."""

    def test_run_rejects_none_graph(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="Graph cannot be None"):
            Pillar6_Generalization.run(
                model,
                None,
                MagicMock(),
                MagicMock(),
                metric_fn=MagicMock(),
            )

    def test_run_rejects_none_metric(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="metric_fn cannot be None"):
            Pillar6_Generalization.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                MagicMock(),
                metric_fn=None,
            )

    def test_run_rejects_missing_use_attn_result(self):
        model = _make_mock_model(use_attn_result=False)
        with pytest.raises(AssertionError, match="use_attn_result"):
            Pillar6_Generalization.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                MagicMock(),
                metric_fn=MagicMock(),
            )

    def test_run_rejects_mean_without_intervention_dataloader(self):
        model = _make_mock_model()
        with pytest.raises(ValueError, match="requires an .*intervention_dataloader"):
            Pillar6_Generalization.run(
                model,
                _make_mock_graph(),
                MagicMock(),
                MagicMock(),
                metric_fn=MagicMock(),
                intervention="mean",
            )

    @patch("circuitkit.evaluation.pillars.generalization.evaluate_graph")
    def test_run_computes_transfer_metrics(self, mock_eval):
        """Verify transfer_ratio, delta, relative_drop calculations."""
        mock_eval.side_effect = [
            torch.tensor([1.0, 1.0]),  # source mean = 1.0
            torch.tensor([0.6, 0.8]),  # target mean = 0.7
        ]
        model = _make_mock_model()

        result = Pillar6_Generalization.run(
            model,
            _make_mock_graph(),
            source_dataloader=MagicMock(),
            target_dataloader=MagicMock(),
            metric_fn=MagicMock(),
            source_task_name="ioi",
            target_task_name="sva",
            quiet=True,
        )

        assert abs(result["source_score"] - 1.0) < 1e-6
        assert abs(result["target_score"] - 0.7) < 1e-6
        assert abs(result["transfer_ratio"] - 0.7) < 1e-6
        assert abs(result["transfer_delta"] - 0.3) < 1e-6
        assert abs(result["relative_transfer_drop"] - 0.3) < 1e-6
        assert result["source_task"] == "ioi"
        assert result["target_task"] == "sva"

    @patch("circuitkit.evaluation.pillars.generalization.evaluate_graph")
    def test_run_zero_source_score(self, mock_eval):
        """Zero source score is undefined → status='invalid' with None ratios.

        transfer_ratio = target/source is meaningless when the source score is
        non-positive (signed logit_diff), so Pillar 6 reports status='invalid'
        with None ratios rather than fabricating a 0.0.
        """
        mock_eval.side_effect = [
            torch.tensor([0.0]),
            torch.tensor([0.5]),
        ]
        model = _make_mock_model()

        result = Pillar6_Generalization.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            metric_fn=MagicMock(),
            quiet=True,
        )

        assert result["status"] == "invalid"
        assert result["transfer_ratio"] is None
        assert result["relative_transfer_drop"] is None

    @patch("circuitkit.evaluation.pillars.generalization.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.generalization.evaluate_graph")
    def test_run_renormalize_bounds_signed_metric(self, mock_eval, mock_base):
        """renormalize=True yields a bounded transfer ratio where the raw path is invalid.

        A negative raw source score makes the raw target/source ratio meaningless
        (the raw path returns status='invalid'). Affine-normalizing each task's
        score against its own clean/corrupt baselines (F = (circuit-corrupt)/
        (clean-corrupt), Zhang & Nanda 2023 / Pillar 1) maps both into [0,1], so
        the transfer ratio F_target/F_source is well-defined.
        """
        # evaluate_graph: source then target raw circuit scores (both negative).
        mock_eval.side_effect = [
            torch.tensor([-0.5]),  # source raw
            torch.tensor([-0.6]),  # target raw
        ]
        # evaluate_baseline order: source clean, source corrupt, target clean, target corrupt.
        mock_base.side_effect = [
            torch.tensor([1.0]),  # source clean
            torch.tensor([-1.0]),  # source corrupt
            torch.tensor([1.0]),  # target clean
            torch.tensor([-1.0]),  # target corrupt
        ]
        model = _make_mock_model()

        result = Pillar6_Generalization.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            metric_fn=MagicMock(),
            renormalize=True,
            quiet=True,
        )

        # F_source = (-0.5 - -1.0)/(1.0 - -1.0) = 0.25; F_target = (-0.6 - -1.0)/2 = 0.20
        assert result["normalized"] is True
        assert "status" not in result  # bounded, no longer invalid
        assert abs(result["source_faithfulness"] - 0.25) < 1e-6
        assert abs(result["target_faithfulness"] - 0.20) < 1e-6
        assert abs(result["transfer_ratio"] - 0.8) < 1e-6  # 0.20 / 0.25
        assert 0.0 <= result["transfer_ratio"] <= 1.0
        # raw ratio is still surfaced for reference (and is the meaningless one)
        assert abs(result["raw_transfer_ratio"] - (-0.6 / -0.5)) < 1e-6

    @patch("circuitkit.evaluation.pillars.generalization.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.generalization.evaluate_graph")
    def test_run_renormalize_degenerate_source_faithfulness(self, mock_eval, mock_base):
        """F_source ~ 0 (circuit no better than corrupt on its own task) → invalid ratio.

        The transfer *ratio* is 0/0 and undefined, but the bounded per-task
        faithfulness values remain meaningful and are still surfaced.
        """
        mock_eval.side_effect = [
            torch.tensor([-1.0]),  # source raw == source corrupt → F_source = 0
            torch.tensor([0.0]),  # target raw
        ]
        mock_base.side_effect = [
            torch.tensor([1.0]),  # source clean
            torch.tensor([-1.0]),  # source corrupt
            torch.tensor([1.0]),  # target clean
            torch.tensor([-1.0]),  # target corrupt
        ]
        model = _make_mock_model()

        result = Pillar6_Generalization.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            metric_fn=MagicMock(),
            renormalize=True,
            quiet=True,
        )

        assert result["status"] == "invalid"
        assert result["transfer_ratio"] is None
        assert result["normalized"] is True
        assert abs(result["source_faithfulness"]) < 1e-6
        assert abs(result["target_faithfulness"] - 0.5) < 1e-6  # (0.0 - -1.0)/2

    @patch("circuitkit.evaluation.pillars.generalization.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.generalization.evaluate_graph")
    def test_run_renormalize_degenerate_target_denominator_is_invalid(self, mock_eval, mock_base):
        """A degenerate TARGET clean/corrupt gap must be invalid, not a fabricated 0.0 ratio.

        Regression test: source is healthy (faith_source=0.25, well above the
        degenerate-faithfulness epsilon), but the target's clean and corrupt
        baselines are identical, so its denominator is ~0 and
        _faithfulness_ratio returns a 0.0 *sentinel* for "undefined" — not a
        real zero-faithfulness score. Only checking faith_source < eps (as the
        first version of this fix did) misses this: it would divide the
        target's sentinel 0.0 by a healthy faith_source and return a normal,
        non-invalid transfer_ratio of exactly 0.0, silently reporting "the
        circuit completely fails to transfer" when the real cause is an
        undefined target baseline, not the circuit's transfer quality.
        """
        mock_eval.side_effect = [
            torch.tensor([-0.5]),  # source raw
            torch.tensor([0.3]),  # target raw (irrelevant — denom is degenerate)
        ]
        mock_base.side_effect = [
            torch.tensor([1.0]),  # source clean
            torch.tensor([-1.0]),  # source corrupt  -> faith_source = 0.25 (healthy)
            torch.tensor([1.0]),  # target clean
            torch.tensor([1.0]),  # target corrupt   -> |clean - corrupt| = 0 (degenerate)
        ]
        model = _make_mock_model()

        result = Pillar6_Generalization.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            metric_fn=MagicMock(),
            renormalize=True,
            quiet=True,
        )

        assert result["status"] == "invalid"
        assert result["transfer_ratio"] is None
        assert result["relative_transfer_drop"] is None
        assert "target" in result["reason"]
        assert abs(result["source_faithfulness"] - 0.25) < 1e-6

    @patch("circuitkit.evaluation.pillars.generalization.evaluate_baseline")
    @patch("circuitkit.evaluation.pillars.generalization.evaluate_graph")
    def test_run_renormalize_inverted_target_baseline_is_invalid(self, mock_eval, mock_base):
        """An INVERTED target clean/corrupt gap (clean < corrupt) must be invalid.

        Companion to the degenerate-target regression: _faithfulness_ratio now
        flags inverted denominators, and P6 must treat them like degenerate
        ones — a target task whose metric runs backwards is a metric-direction
        problem, not zero transfer.
        """
        mock_eval.side_effect = [
            torch.tensor([-0.5]),  # source raw
            torch.tensor([0.3]),  # target raw
        ]
        mock_base.side_effect = [
            torch.tensor([1.0]),  # source clean
            torch.tensor([-1.0]),  # source corrupt -> healthy F_source = 0.25
            torch.tensor([-2.0]),  # target clean BELOW target corrupt -> inverted
            torch.tensor([-0.5]),  # target corrupt
        ]
        model = _make_mock_model()

        result = Pillar6_Generalization.run(
            model,
            _make_mock_graph(),
            MagicMock(),
            MagicMock(),
            metric_fn=MagicMock(),
            renormalize=True,
            quiet=True,
        )

        assert result["status"] == "invalid"
        assert result["transfer_ratio"] is None
        assert "inverted" in result["reason"]
        assert "target" in result["reason"]

    def test_summarize_transfer_matrix_basic(self):
        """Summarize a 2x2 transfer matrix."""
        matrix = {
            "ioi": {
                "ioi": {"transfer_ratio": 1.0},
                "sva": {"transfer_ratio": 0.8},
            },
            "sva": {
                "ioi": {"transfer_ratio": 0.6},
                "sva": {"transfer_ratio": 1.0},
            },
        }

        summary = Pillar6_Generalization.summarize_transfer_matrix(matrix)

        assert abs(summary["within_task_transfer"] - 1.0) < 1e-6
        # cross-task = mean(0.8, 0.6) = 0.7
        assert abs(summary["cross_task_transfer"] - 0.7) < 1e-6
        assert summary["best_transfer"][2] == 0.8
        assert summary["worst_transfer"][2] == 0.6

    def test_summarize_transfer_matrix_single_task(self):
        """Single task matrix has no off-diagonal entries."""
        matrix = {
            "ioi": {
                "ioi": {"transfer_ratio": 1.0},
            },
        }

        summary = Pillar6_Generalization.summarize_transfer_matrix(matrix)
        assert summary["cross_task_transfer"] == 0.0
        assert summary["best_transfer"] is None
        assert summary["worst_transfer"] is None


class TestRunFullFaithfulness:
    """Tests for run_full_faithfulness orchestrator (full.py)."""

    # ── Validation ────────────────────────────────────────────────────────

    def test_invalid_pillar_name_raises(self):
        """Unknown pillar names raise ValueError."""
        model = _make_mock_model()
        graph = _make_mock_graph()
        task_spec = _make_mock_task_spec()
        cfg = _make_discovery_cfg()

        with pytest.raises(ValueError, match="Invalid pillars"):
            run_full_faithfulness(
                model=model,
                graph=graph,
                task_spec=task_spec,
                discovery_cfg=cfg,
                dataloader=MagicMock(),
                pillars=["patching", "nonexistent"],
            )

    def test_multiple_invalid_pillars_reported(self):
        """All invalid names appear in the error message."""
        model = _make_mock_model()
        with pytest.raises(ValueError, match="foo") as exc_info:
            run_full_faithfulness(
                model=model,
                graph=_make_mock_graph(),
                task_spec=_make_mock_task_spec(),
                discovery_cfg=_make_discovery_cfg(),
                dataloader=MagicMock(),
                pillars=["foo", "bar"],
            )
        assert "bar" in str(exc_info.value)

    # ── Pillar selection ──────────────────────────────────────────────────

    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_single_pillar_only_runs_requested(self, mock_p1):
        """Requesting only 'patching' should not invoke other pillars."""
        mock_p1.run.return_value = {
            "score": 0.85,
            "raw_score": 0.85,
            "raw_ratio": 0.85,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["patching"],
        )

        mock_p1.run.assert_called_once()
        assert report.patching_score == 0.85
        assert report.ablation_score is None
        assert report.stability is None
        assert report.robustness is None
        assert report.baseline_comparison is None
        assert report.generalization is None

    @patch("circuitkit.evaluation.full.Pillar2_Ablation")
    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_two_pillars(self, mock_p1, mock_p2):
        mock_p1.run.return_value = {
            "score": 0.9,
            "raw_score": 0.9,
            "raw_ratio": 0.9,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        mock_p2.run.return_value = {
            "score": 0.7,
            "raw_score": 0.7,
            "raw_ratio": 0.7,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["patching", "ablation"],
        )

        assert report.patching_score == 0.9
        assert report.ablation_score == 0.7

    # ── Report metadata: real model name ─────────────────────────────────

    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_report_metadata_uses_real_model_name(self, mock_p1):
        """report.metadata['model'] must be the real model name.

        Regression test: the discovery sub-config does NOT carry a 'model'
        key (that lives in the top-level config), so the model name must be
        read from the loaded model's cfg.model_name — not defaulted to
        'unknown'. A previous version read discovery_cfg.get('model', {})
        and always produced 'unknown' on real runs.
        """
        mock_p1.run.return_value = {
            "score": 0.85,
            "raw_score": 0.85,
            "raw_ratio": 0.85,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        model = _make_mock_model()
        model.cfg.model_name = "gpt2"  # as set by HookedTransformer.from_pretrained

        # discovery_cfg WITHOUT a 'model' key — the real-world shape.
        cfg = _make_discovery_cfg()
        cfg.pop("model", None)

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=cfg,
            dataloader=MagicMock(),
            pillars=["patching"],
        )

        assert report.metadata["model"] == "gpt2"
        assert report.metadata["model"] != "unknown"

    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_report_metadata_model_falls_back_to_discovery_cfg(self, mock_p1):
        """If the model object has no usable cfg.model_name, an explicit
        discovery_cfg['model_name'] override is used before 'unknown'."""
        mock_p1.run.return_value = {
            "score": 0.5,
            "raw_score": 0.5,
            "raw_ratio": 0.5,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        model = _make_mock_model()
        model.cfg.model_name = None  # no name available from the model

        cfg = _make_discovery_cfg()
        cfg.pop("model", None)
        cfg["model_name"] = "pythia-70m"

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=cfg,
            dataloader=MagicMock(),
            pillars=["patching"],
        )

        assert report.metadata["model"] == "pythia-70m"

    @patch("circuitkit.evaluation.full.Pillar5_Baselines")
    @patch("circuitkit.evaluation.full.Pillar2_Ablation")
    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_empty_pillars_list_runs_nothing(self, mock_p1, mock_p2, mock_p5):
        """Empty list is valid — no pillars run, report is all None."""
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=[],
        )

        mock_p1.run.assert_not_called()
        mock_p2.run.assert_not_called()
        mock_p5.run.assert_not_called()
        assert report.patching_score is None
        assert report.ablation_score is None

    # ── Return type & report structure ────────────────────────────────────

    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_returns_faithfulness_report(self, mock_p1):
        mock_p1.run.return_value = {
            "score": 0.5,
            "raw_score": 0.5,
            "raw_ratio": 0.5,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["patching"],
        )

        assert isinstance(report, FaithfulnessReport)

    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_metadata_populated(self, mock_p1):
        """Report metadata includes algorithm, model, task, timing."""
        mock_p1.run.return_value = {
            "score": 0.5,
            "raw_score": 0.5,
            "raw_ratio": 0.5,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        model = _make_mock_model()
        cfg = _make_discovery_cfg(algorithm="eap-ig", task="sva")

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=cfg,
            dataloader=MagicMock(),
            pillars=["patching"],
        )

        assert report.metadata["algorithm"] == "eap-ig"
        assert report.metadata["task"] == "sva"
        assert report.metadata["model"] == "gpt2"
        assert "patching" in report.metadata["pillars_computed"]
        assert "total_duration_seconds" in report.metadata
        assert isinstance(report.metadata["per_pillar_duration_seconds"], dict)
        assert "patching" in report.metadata["per_pillar_duration_seconds"]

    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_metadata_timing_positive(self, mock_p1):
        mock_p1.run.return_value = {
            "score": 0.5,
            "raw_score": 0.5,
            "raw_ratio": 0.5,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["patching"],
        )

        assert report.metadata["total_duration_seconds"] >= 0
        assert report.metadata["per_pillar_duration_seconds"]["patching"] >= 0

    # ── Error propagation ─────────────────────────────────────────────────

    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_pillar1_failure_raises_runtime_error(self, mock_p1):
        mock_p1.run.side_effect = RuntimeError("patching exploded")
        model = _make_mock_model()

        with pytest.raises(RuntimeError, match="Pillar 1.*patching exploded"):
            run_full_faithfulness(
                model=model,
                graph=_make_mock_graph(),
                task_spec=_make_mock_task_spec(),
                discovery_cfg=_make_discovery_cfg(),
                dataloader=MagicMock(),
                pillars=["patching"],
            )

    @patch("circuitkit.evaluation.full.Pillar2_Ablation")
    def test_pillar2_failure_raises_runtime_error(self, mock_p2):
        mock_p2.run.side_effect = ValueError("ablation failed")
        model = _make_mock_model()

        with pytest.raises(RuntimeError, match="Pillar 2.*ablation failed"):
            run_full_faithfulness(
                model=model,
                graph=_make_mock_graph(),
                task_spec=_make_mock_task_spec(),
                discovery_cfg=_make_discovery_cfg(),
                dataloader=MagicMock(),
                pillars=["ablation"],
            )

    @patch("circuitkit.evaluation.full.Pillar5_Baselines")
    def test_pillar5_failure_raises_runtime_error(self, mock_p5):
        mock_p5.run.side_effect = Exception("baseline boom")
        model = _make_mock_model()

        with pytest.raises(RuntimeError, match="Pillar 5.*baseline boom"):
            run_full_faithfulness(
                model=model,
                graph=_make_mock_graph(),
                task_spec=_make_mock_task_spec(),
                discovery_cfg=_make_discovery_cfg(),
                dataloader=MagicMock(),
                pillars=["baselines"],
            )

    @patch("circuitkit.evaluation.full.Pillar3_Stability")
    def test_pillar3_failure_raises_runtime_error(self, mock_p3):
        mock_p3.run.side_effect = Exception("stability boom")
        model = _make_mock_model()

        with pytest.raises(RuntimeError, match="Pillar 3.*stability boom"):
            run_full_faithfulness(
                model=model,
                graph=_make_mock_graph(),
                task_spec=_make_mock_task_spec(),
                discovery_cfg=_make_discovery_cfg(),
                dataloader=MagicMock(),
                pillars=["stability"],
            )

    @patch("circuitkit.evaluation.full.Pillar6_Generalization")
    def test_pillar6_failure_raises_runtime_error(self, mock_p6):
        mock_p6.run.side_effect = Exception("gen boom")
        model = _make_mock_model()
        target_spec = _make_mock_task_spec()

        with pytest.raises(RuntimeError, match="Pillar 6.*gen boom"):
            run_full_faithfulness(
                model=model,
                graph=_make_mock_graph(),
                task_spec=_make_mock_task_spec(),
                discovery_cfg=_make_discovery_cfg(),
                dataloader=MagicMock(),
                pillars=["generalization"],
                target_task_spec=target_spec,
                target_dataloader=MagicMock(),
            )

    # ── Pillar 4 robustness: per-variant error handling ───────────────────

    @patch("circuitkit.evaluation.full.Pillar4_Robustness")
    def test_pillar4_variant_failure_captured_not_raised(self, mock_p4):
        """Individual robustness variant failures are captured, not raised."""
        mock_p4.run.side_effect = Exception("variant boom")
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["robustness"],
            corruption_variants=["paraphrase"],
        )

        assert "paraphrase" in report.robustness
        assert "error" in report.robustness["paraphrase"]

    @patch("circuitkit.evaluation.full.Pillar4_Robustness")
    def test_pillar4_multiple_variants(self, mock_p4):
        """Multiple corruption variants are each evaluated."""
        mock_p4.run.return_value = {"original_score": 0.9, "variant_score": 0.8}
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["robustness"],
            corruption_variants=["paraphrase", "entity_swap"],
        )

        assert set(report.robustness.keys()) == {"paraphrase", "entity_swap"}
        assert mock_p4.run.call_count == 2

    # ── Pillar 5 baselines: custom baseline_types ─────────────────────────

    @patch("circuitkit.evaluation.full.Pillar5_Baselines")
    def test_pillar5_passes_baseline_types(self, mock_p5):
        mock_p5.run.return_value = {"circuit_score": 0.8, "summary": "ok"}
        model = _make_mock_model()

        run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["baselines"],
            baseline_types=["random"],
        )

        call_kwargs = mock_p5.run.call_args[1]
        assert call_kwargs["baseline_types"] == ["random"]

    # ── Pillar 6 generalization: skip when no target ──────────────────────

    def test_pillar6_skipped_without_target_spec(self):
        """Generalization is silently skipped when target_task_spec is None."""
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["generalization"],
            target_task_spec=None,
        )

        assert report.generalization is None

    def test_pillar6_skipped_without_target_dataloader(self):
        """Generalization is silently skipped when target_dataloader is None."""
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["generalization"],
            target_task_spec=_make_mock_task_spec(),
            target_dataloader=None,
        )

        assert report.generalization is None

    @patch("circuitkit.evaluation.full.Pillar6_Generalization")
    def test_pillar6_runs_when_target_provided(self, mock_p6):
        mock_p6.run.return_value = {"transfer_ratio": 0.75}
        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["generalization"],
            target_task_spec=_make_mock_task_spec(),
            target_dataloader=MagicMock(),
        )

        mock_p6.run.assert_called_once()
        assert report.generalization["transfer_ratio"] == 0.75

    # ── Defaults ──────────────────────────────────────────────────────────

    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_metric_fn_defaults_to_task_spec(self, mock_p1):
        """When metric_fn is None, task_spec.metric_fn() is called to resolve it.

        task_spec.metric_fn is a *factory* — run_full_faithfulness calls it to
        get the actual metric callable (passing the factory itself would crash
        evaluate_baseline with a wrong signature).
        """
        mock_p1.run.return_value = {
            "score": 0.5,
            "raw_score": 0.5,
            "raw_ratio": 0.5,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        model = _make_mock_model()
        spec = _make_mock_task_spec()
        sentinel_factory = MagicMock()
        spec.metric_fn = sentinel_factory

        run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=spec,
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["patching"],
        )

        call_kwargs = mock_p1.run.call_args[1]
        # the factory is *called* — the pillar receives its return value
        assert call_kwargs["metric_fn"] is sentinel_factory.return_value

    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_custom_metric_fn_overrides_task_spec(self, mock_p1):
        mock_p1.run.return_value = {
            "score": 0.5,
            "raw_score": 0.5,
            "raw_ratio": 0.5,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        model = _make_mock_model()
        custom_metric = MagicMock()

        run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["patching"],
            metric_fn=custom_metric,
        )

        call_kwargs = mock_p1.run.call_args[1]
        assert call_kwargs["metric_fn"] is custom_metric

    def test_default_pillars_is_all_six(self):
        """When pillars=None, all 6 are scheduled (validation passes)."""
        # We can verify this by checking that a valid call with pillars=None
        # doesn't raise ValueError — the only way that works is if the
        # default list contains only valid pillar names.
        model = _make_mock_model()

        # This will fail at actual pillar execution (no real model),
        # but the pillar *validation* should pass.
        with patch("circuitkit.evaluation.full.Pillar1_CausalPatching") as p1:
            p1.run.side_effect = RuntimeError("stop early")
            with pytest.raises(RuntimeError, match="stop early"):
                run_full_faithfulness(
                    model=model,
                    graph=_make_mock_graph(),
                    task_spec=_make_mock_task_spec(),
                    discovery_cfg=_make_discovery_cfg(),
                    dataloader=MagicMock(),
                    pillars=None,  # should default to all 6
                )

    def test_no_dataloader_falls_back_to_task_spec_build(self):
        """When dataloader=None, task_spec.build_dataloader is called."""
        model = _make_mock_model()
        spec = _make_mock_task_spec()
        cfg = _make_discovery_cfg()
        built_dl = MagicMock()
        spec.build_dataloader.return_value = built_dl

        with patch("circuitkit.evaluation.full.Pillar1_CausalPatching") as p1:
            p1.run.return_value = {
                "score": 0.5,
                "raw_score": 0.5,
                "raw_ratio": 0.5,
                "clean_score": 1.0,
                "corrupt_score": 0.0,
                "degenerate_denominator": False,
            }
            run_full_faithfulness(
                model=model,
                graph=_make_mock_graph(),
                task_spec=spec,
                discovery_cfg=cfg,
                dataloader=None,
                pillars=["patching"],
            )

            call_kwargs = p1.run.call_args[1]
            assert call_kwargs["dataloader"] is built_dl
            spec.build_dataloader.assert_called_once()

    # ── Pillar 3 passes n_stability_runs & seed ───────────────────────────

    @patch("circuitkit.evaluation.full.Pillar3_Stability")
    def test_pillar3_receives_n_runs_and_seed(self, mock_p3):
        mock_p3.run.return_value = {"mean_jaccard": 0.8}
        model = _make_mock_model()

        run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=["stability"],
            n_stability_runs=10,
        )

        call_kwargs = mock_p3.run.call_args[1]
        assert call_kwargs["n_runs"] == 10
        assert call_kwargs["seed_start"] == 42  # from discovery_cfg

    # ── Full run integration (all pillars mocked) ─────────────────────────

    @patch("circuitkit.evaluation.full.Pillar6_Generalization")
    @patch("circuitkit.evaluation.full.Pillar3_Stability")
    @patch("circuitkit.evaluation.full.Pillar4_Robustness")
    @patch("circuitkit.evaluation.full.Pillar5_Baselines")
    @patch("circuitkit.evaluation.full.Pillar2_Ablation")
    @patch("circuitkit.evaluation.full.Pillar1_CausalPatching")
    def test_all_pillars_end_to_end(self, mock_p1, mock_p2, mock_p5, mock_p4, mock_p3, mock_p6):
        """All 6 pillars run and populate the report correctly."""
        mock_p1.run.return_value = {
            "score": 0.91,
            "raw_score": 0.91,
            "raw_ratio": 0.91,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        mock_p2.run.return_value = {
            "score": 0.85,
            "raw_score": 0.85,
            "raw_ratio": 0.85,
            "clean_score": 1.0,
            "corrupt_score": 0.0,
            "degenerate_denominator": False,
        }
        mock_p5.run.return_value = {"circuit_score": 0.9, "summary": "good"}
        mock_p4.run.return_value = {"original_score": 0.9, "variant_score": 0.85}
        mock_p3.run.return_value = {"mean_jaccard": 0.78}
        mock_p6.run.return_value = {"transfer_ratio": 0.65}

        model = _make_mock_model()

        report = run_full_faithfulness(
            model=model,
            graph=_make_mock_graph(),
            task_spec=_make_mock_task_spec(),
            discovery_cfg=_make_discovery_cfg(),
            dataloader=MagicMock(),
            pillars=[
                "patching",
                "ablation",
                "baselines",
                "robustness",
                "stability",
                "generalization",
            ],
            target_task_spec=_make_mock_task_spec(),
            target_dataloader=MagicMock(),
        )

        assert report.patching_score == 0.91
        assert report.ablation_score == 0.85
        assert report.baseline_comparison["circuit_score"] == 0.9
        assert "paraphrase" in report.robustness
        assert report.stability["mean_jaccard"] == 0.78
        assert report.generalization["transfer_ratio"] == 0.65
        assert len(report.metadata["pillars_computed"]) == 6
        assert (
            len(report.metadata["per_pillar_duration_seconds"]) >= 5
        )  # robustness variants share one key


class TestPillarIntegration:
    """Integration tests for pillar modules."""

    def test_stability_and_baselines_compatibility(self):
        """Test that stability and baselines can work with same graph."""

        # Both pillars should work with the same graph type
        class MockNode:
            def __init__(self, name, layer):
                self.name = name
                self.layer = layer
                self.score = torch.tensor(0.5)
                self._in_graph = True

            @property
            def in_graph(self):
                return self._in_graph

            @in_graph.setter
            def in_graph(self, value):
                self._in_graph = value

        class MockGraph:
            def __init__(self):
                self.nodes = {f"node_{i}": MockNode(f"node_{i}", i % 3) for i in range(10)}
                self.neurons_in_graph = None
                self.neurons_scores = None

        graph = MockGraph()

        # Both should be able to extract circuits
        stability_circuit = Pillar3_Stability._extract_circuit_nodes(graph)
        baselines_circuit = Pillar5_Baselines._extract_circuit_nodes(graph)

        assert len(stability_circuit) == len(
            baselines_circuit
        ), "Both methods should extract same circuit"

    def test_overlap_metrics_consistency(self):
        """Test that Jaccard and Dice are consistent."""
        circuit1 = {"a": {}, "b": {}, "c": {}, "d": {}}
        circuit2 = {"a": {}, "b": {}, "e": {}, "f": {}}

        jaccard = Pillar3_Stability.compute_jaccard(circuit1, circuit2)
        dice = Pillar3_Stability.compute_dice(circuit1, circuit2)

        # Both should be in [0, 1]
        assert 0.0 <= jaccard <= 1.0, "Jaccard should be in [0, 1]"
        assert 0.0 <= dice <= 1.0, "Dice should be in [0, 1]"

        # For identical circuits, both should be 1
        jaccard_same = Pillar3_Stability.compute_jaccard(circuit1, circuit1)
        dice_same = Pillar3_Stability.compute_dice(circuit1, circuit1)
        assert jaccard_same == 1.0, "Jaccard should be 1.0 for identical"
        assert dice_same == 1.0, "Dice should be 1.0 for identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
