"""
Tests for activation saliency visualization.
"""

import json

import numpy as np
import pytest

from circuitkit.visualize.saliency import ActivationSaliencyVisualizer


class TestActivationSaliencyVisualizer:
    """Test activation saliency visualizer."""

    @pytest.fixture
    def sample_activations(self):
        """Create sample activations for testing."""
        return {
            "layer_0": np.random.rand(10),
            "layer_1": np.random.rand(10),
            "layer_2": np.random.rand(10),
        }

    @pytest.fixture
    def sample_tokens(self):
        """Create sample tokens."""
        return ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "!"]

    def test_initialization(self, sample_activations):
        """Test visualizer initialization."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        assert viz.activations == sample_activations
        assert len(viz.normalized_activations) == 3

    def test_initialization_with_tokens(self, sample_activations, sample_tokens):
        """Test initialization with tokens."""
        viz = ActivationSaliencyVisualizer(
            sample_activations,
            tokens=sample_tokens,
        )
        assert viz.tokens == sample_tokens

    def test_normalization(self, sample_activations):
        """Test that activations are normalized to [0, 1]."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        for layer, normalized in viz.normalized_activations.items():
            assert normalized.min() >= 0
            assert normalized.max() <= 1

    def test_invalid_activation_type(self):
        """Test that invalid activation types raise error."""
        with pytest.raises(TypeError):
            ActivationSaliencyVisualizer({"layer_0": "invalid"})

    def test_plot_layer_heatmaps(self, sample_activations):
        """Test layer heatmap generation."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        fig = viz.plot_layer_heatmaps()
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_aggregate_heatmap(self, sample_activations):
        """Test aggregate heatmap generation."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        fig = viz.plot_aggregate_heatmap(aggregation="mean")
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_aggregate_max(self, sample_activations):
        """Test aggregate with max aggregation."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        fig = viz.plot_aggregate_heatmap(aggregation="max")
        assert fig is not None

    def test_plot_aggregate_sum(self, sample_activations):
        """Test aggregate with sum aggregation."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        fig = viz.plot_aggregate_heatmap(aggregation="sum")
        assert fig is not None

    def test_plot_layer_comparison(self, sample_activations):
        """Test layer comparison visualization."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        fig = viz.plot_layer_comparison()
        assert fig is not None
        assert len(fig.data) > 0

    def test_export_html(self, sample_activations, tmp_path):
        """Test HTML export."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        output_file = tmp_path / "test.html"
        viz.export_to_html(str(output_file))
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_export_json(self, sample_activations, tmp_path):
        """Test JSON export."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        output_file = tmp_path / "test.json"
        viz.export_to_json(str(output_file))
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert "layers" in data
            assert "metadata" in data

    def test_get_top_tokens(self, sample_activations, sample_tokens):
        """Test getting top tokens."""
        viz = ActivationSaliencyVisualizer(sample_activations, tokens=sample_tokens)
        top_tokens = viz.get_top_tokens("layer_0", k=3)
        assert len(top_tokens) == 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top_tokens)

    def test_get_saliency_summary(self, sample_activations):
        """Test getting saliency summary."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        summary = viz.get_saliency_summary()
        assert "layer_0" in summary
        assert "mean" in summary["layer_0"]
        assert "max" in summary["layer_0"]
        assert "std" in summary["layer_0"]

    def test_multi_dimensional_activations(self):
        """Test handling of multi-dimensional activations."""
        activations = {
            "layer_0": np.random.rand(10, 64),  # (seq_len, hidden_dim)
            "layer_1": np.random.rand(10, 64),
        }
        viz = ActivationSaliencyVisualizer(activations)
        # Should aggregate to (10,) per layer
        assert all(len(v) == 10 for v in viz.normalized_activations.values())

    def test_zero_activation(self):
        """Test handling of zero activations."""
        activations = {
            "layer_0": np.zeros(10),
            "layer_1": np.zeros(10),
        }
        viz = ActivationSaliencyVisualizer(activations)
        assert all(np.allclose(v, 0) for v in viz.normalized_activations.values())

    def test_export_all_plot_types(self, sample_activations, tmp_path):
        """Test exporting all plot types."""
        viz = ActivationSaliencyVisualizer(sample_activations)

        for plot_type in ["layer_heatmaps", "aggregate", "comparison"]:
            output_file = tmp_path / f"test_{plot_type}.html"
            viz.export_to_html(str(output_file), plot_type=plot_type)
            assert output_file.exists()

    def test_invalid_aggregation(self, sample_activations):
        """Test that invalid aggregation raises error."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        with pytest.raises(ValueError):
            viz.plot_aggregate_heatmap(aggregation="invalid")

    def test_get_top_tokens_invalid_layer(self, sample_activations, sample_tokens):
        """Test error on invalid layer."""
        viz = ActivationSaliencyVisualizer(sample_activations, tokens=sample_tokens)
        with pytest.raises(ValueError):
            viz.get_top_tokens("invalid_layer")

    def test_with_list_activations(self, sample_tokens):
        """Test initialization with list activations."""
        activations = {
            "layer_0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "layer_1": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        }
        viz = ActivationSaliencyVisualizer(activations, tokens=sample_tokens)
        assert len(viz.normalized_activations) == 2

    def test_normalization_correctness(self, sample_activations):
        """Verify normalization produces correct min=0, max=1 boundary values."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        for layer, normalized in viz.normalized_activations.items():
            if normalized.max() > 0:  # skip zero arrays
                assert np.isclose(normalized.min(), 0.0), f"{layer} min should be 0"
                assert np.isclose(normalized.max(), 1.0), f"{layer} max should be 1"

    def test_get_top_tokens_ordering(self, sample_tokens):
        """Top tokens must be sorted by activation descending."""
        activations = {"layer_0": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 1.0])}
        viz = ActivationSaliencyVisualizer(activations, tokens=sample_tokens)
        top = viz.get_top_tokens("layer_0", k=3)
        scores = [score for _, score in top]
        assert scores == sorted(scores, reverse=True), "Top tokens must be sorted descending"
        assert top[0][0] == "!"  # index 9 has value 1.0

    def test_get_top_tokens_k_exceeds_length(self, sample_activations, sample_tokens):
        """k larger than token count should not crash — returns all tokens."""
        viz = ActivationSaliencyVisualizer(sample_activations, tokens=sample_tokens)
        top = viz.get_top_tokens("layer_0", k=100)
        assert len(top) <= len(sample_tokens)

    def test_export_json_content_correctness(self, sample_activations, sample_tokens, tmp_path):
        """JSON export must contain correct layer data and metadata values."""
        viz = ActivationSaliencyVisualizer(sample_activations, tokens=sample_tokens)
        output_file = tmp_path / "test.json"
        viz.export_to_json(str(output_file))
        with open(output_file) as f:
            data = json.load(f)
        assert data["tokens"] == sample_tokens
        assert data["metadata"]["num_layers"] == 3
        assert data["metadata"]["num_tokens"] == 10
        assert set(data["layers"].keys()) == set(sample_activations.keys())
        for layer, values in data["layers"].items():
            assert len(values) == 10
            assert all(0.0 <= v <= 1.0 for v in values)

    def test_multidim_activation_reduction_values(self):
        """2D activations should reduce via max(abs) across hidden dim."""
        raw = np.zeros((5, 8))
        raw[2, 3] = 5.0  # position 2 should dominate after reduction
        viz = ActivationSaliencyVisualizer({"layer_0": raw})
        reduced = viz.normalized_activations["layer_0"]
        assert np.isclose(reduced[2], 1.0), "Position with max abs should normalize to 1.0"
        assert np.isclose(reduced[0], 0.0)

    def test_export_html_invalid_plot_type(self, sample_activations, tmp_path):
        viz = ActivationSaliencyVisualizer(sample_activations)
        with pytest.raises(ValueError):
            viz.export_to_html(str(tmp_path / "out.html"), plot_type="nonexistent")

    def test_token_length_mismatch(self):
        """Mismatched token and activation lengths should raise an error."""
        activations = {"layer_0": np.random.rand(10)}
        with pytest.raises(ValueError, match="tokens"):
            ActivationSaliencyVisualizer(activations, tokens=["only", "three", "tokens"])

    def test_plot_layer_heatmaps_subset(self, sample_activations):
        """Should render correctly when only a subset of layers is requested."""
        viz = ActivationSaliencyVisualizer(sample_activations)
        fig = viz.plot_layer_heatmaps(layers=["layer_0", "layer_1"])
        assert len(fig.data) == 2
