"""
Tests for baseline implementations.
"""

import torch
import torch.nn as nn

from circuitkit.benchmarks.baselines import (
    GptqBaseline,
    MagnitudeBaseline,
    RandomBaseline,
    SparseGPTBaseline,
    WandaBaseline,
)


class _TinyModel(nn.Module):
    """Minimal real nn.Module used in place of a Mock.

    Using a real module keeps it ``copy.deepcopy``-able (Mock objects
    are not) and gives a stable, re-iterable ``named_parameters()``.
    """

    def __init__(self, param_specs):
        super().__init__()
        for name, tensor in param_specs.items():
            self.register_parameter(name, nn.Parameter(tensor.clone()))

    @property
    def device(self):
        return torch.device("cpu")


def _make_model(num_params=3):
    """Create a real model with deterministic parameters.

    ``register_parameter`` does not allow dotted names, so parameters
    are registered with sanitized names; the baselines only iterate
    ``named_parameters()`` so the exact names do not matter.
    """
    specs = {
        "layer1_weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "layer1_bias": torch.tensor([0.1, 0.2]),
        "layer2_weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        "layer2_bias": torch.tensor([0.3, 0.4]),
    }
    chosen = dict(list(specs.items())[:num_params])
    return _TinyModel(chosen)


class TestMagnitudeBaseline:
    """Test MagnitudeBaseline."""

    def setup_method(self):
        """Setup test fixtures."""
        self.baseline = MagnitudeBaseline(verbose=False)
        self.model = _make_model(num_params=4)

    def test_score_parameters(self):
        """Test parameter scoring."""
        scores = self.baseline.score_parameters(self.model)

        assert len(scores) == 4
        assert all(isinstance(v, float) for v in scores.values())
        # Higher magnitude should have higher score
        assert scores["layer2_weight"] > scores["layer1_weight"]

    def test_select_parameters(self):
        """Test parameter selection."""
        selection = self.baseline.select_parameters(self.model, target_sparsity=0.5)

        assert len(selection) == 4
        assert sum(selection.values()) == 2  # Keep 50% of 4 parameters

    def test_prune_model(self):
        """Test model pruning."""
        pruned = self.baseline.prune_model(self.model, target_sparsity=0.5, inplace=False)

        assert pruned is not None

    def test_sparsity_mask(self):
        """Test sparsity mask generation."""
        masks = self.baseline.get_sparsity_mask(self.model, target_sparsity=0.5)

        assert len(masks) == 4
        assert all(
            m.shape == p.shape for m, (_, p) in zip(masks.values(), self.model.named_parameters())
        )

    def test_layer_importance(self):
        """Test per-layer importance computation."""
        layer_imp = self.baseline.compute_layer_importance(self.model)

        assert len(layer_imp) > 0
        assert all(isinstance(v, float) for v in layer_imp.values())


class TestWandaBaseline:
    """Test WandaBaseline."""

    def setup_method(self):
        """Setup test fixtures."""
        self.baseline = WandaBaseline(verbose=False)
        self.model = _make_model(num_params=2)

    def test_score_parameters(self):
        """Test WANDA scoring."""
        scores = self.baseline.score_parameters(self.model)

        assert len(scores) == 2
        assert all(isinstance(v, float) for v in scores.values())

    def test_select_parameters(self):
        """Test parameter selection."""
        selection = self.baseline.select_parameters(self.model, target_sparsity=0.5)

        assert len(selection) == 2

    def test_prune_model(self):
        """Test model pruning with WANDA."""
        pruned = self.baseline.prune_model(self.model, target_sparsity=0.5, inplace=False)

        assert pruned is not None


class TestRandomBaseline:
    """Test RandomBaseline."""

    def setup_method(self):
        """Setup test fixtures."""
        self.baseline = RandomBaseline(seed=42, verbose=False)
        self.model = _make_model(num_params=4)

    def test_select_parameters_reproducible(self):
        """Test that random selection is reproducible with seed."""
        baseline1 = RandomBaseline(seed=42, verbose=False)
        baseline2 = RandomBaseline(seed=42, verbose=False)

        selection1 = baseline1.select_parameters(self.model, 0.5)
        selection2 = baseline2.select_parameters(self.model, 0.5)

        assert selection1 == selection2

    def test_different_sparsity_levels(self):
        """Test different sparsity levels.

        ``target_sparsity`` is the fraction of parameters *removed*, so
        a higher sparsity must keep fewer parameters.
        """
        sel_10 = self.baseline.select_parameters(self.model, 0.1)
        sel_50 = self.baseline.select_parameters(self.model, 0.5)

        count_10 = sum(sel_10.values())
        count_50 = sum(sel_50.values())

        assert count_50 < count_10

    def test_prune_model(self):
        """Test random pruning."""
        pruned = self.baseline.prune_model(self.model, target_sparsity=0.5, inplace=False)

        assert pruned is not None


class TestGptqBaseline:
    """Test GptqBaseline."""

    def setup_method(self):
        """Setup test fixtures."""
        self.baseline = GptqBaseline(bits=4, verbose=False)
        self.model = _make_model(num_params=2)

    def test_quantize_parameter(self):
        """Test parameter quantization."""
        param = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        quantized = self.baseline.quantize_parameter(param, bits=4)

        assert quantized.shape == param.shape

    def test_quantize_model(self):
        """Test model quantization."""
        quantized = self.baseline.quantize_model(self.model, inplace=False)

        assert quantized is not None

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        ratio = self.baseline.compute_compression_ratio(self.model)

        assert ratio == 8.0  # 32 bits / 4 bits

    def test_size_estimate(self):
        """Test size estimation."""
        size_est = self.baseline.get_size_estimate(self.model)

        assert "original_size_mb" in size_est
        assert "quantized_size_mb" in size_est
        assert "compression_ratio" in size_est
        assert size_est["compression_ratio"] == 8.0


class TestSparseGPTBaseline:
    """Test SparseGPTBaseline."""

    def setup_method(self):
        """Setup test fixtures."""
        self.baseline = SparseGPTBaseline(verbose=False)
        self.model = _make_model(num_params=3)

    def test_compute_importance_scores(self):
        """Test importance score computation."""
        scores = self.baseline.compute_importance_scores(self.model)

        assert len(scores) == 3
        assert all(isinstance(v, torch.Tensor) for v in scores.values())

    def test_select_parameters(self):
        """Test parameter selection."""
        masks = self.baseline.select_parameters(self.model, target_sparsity=0.5)

        assert len(masks) == 3

    def test_prune_model(self):
        """Test SparseGPT pruning."""
        pruned = self.baseline.prune_model(self.model, target_sparsity=0.5, inplace=False)

        assert pruned is not None


class TestBaselineComparison:
    """Test baseline comparisons."""

    def test_baseline_consistency(self):
        """Test that pruning baselines follow the same interface.

        ``GptqBaseline`` is a quantization baseline and intentionally
        exposes a different (quantize-oriented) interface, so it is not
        expected to provide ``prune_model``/``get_sparsity_mask``.
        """
        pruning_baselines = [
            MagnitudeBaseline(),
            WandaBaseline(),
            RandomBaseline(),
            SparseGPTBaseline(),
        ]

        for baseline in pruning_baselines:
            assert hasattr(baseline, "prune_model")
            assert hasattr(baseline, "select_parameters")

        # Quantization baseline exposes a quantization interface.
        quant = GptqBaseline()
        assert hasattr(quant, "quantize_model")
        assert hasattr(quant, "compute_compression_ratio")
