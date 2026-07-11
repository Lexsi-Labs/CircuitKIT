"""
Tests for feature saliency visualization.
"""

import json

import numpy as np
import pytest

from circuitkit.visualize.feature_saliency import FeatureSaliencyVisualizer


class TestFeatureSaliencyVisualizer:
    """Test feature saliency visualizer."""

    @pytest.fixture
    def sample_attributions(self):
        """Create sample node attributions."""
        return {
            "A0.0": 0.8,
            "A0.1": 0.6,
            "A1.0": 0.9,
            "MLP0": 0.5,
            "MLP1": 0.7,
        }

    @pytest.fixture
    def sample_activations(self):
        """Create sample activation data."""
        return {
            "original": {
                "layer_0": np.random.rand(10, 64),
                "layer_1": np.random.rand(10, 64),
            },
            "corrupted": {
                "layer_0": np.random.rand(10, 64),
                "layer_1": np.random.rand(10, 64),
            },
        }

    def test_initialization(self, sample_attributions):
        """Test visualizer initialization."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        assert len(viz.normalized_attributions) == 5

    def test_normalization(self, sample_attributions):
        """Test attribution normalization to [0, 1]."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        for score in viz.normalized_attributions.values():
            assert 0 <= score <= 1

    def test_plot_importance_bar(self, sample_attributions):
        """Test importance bar chart."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        fig = viz.plot_importance_bar()
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_importance_bar_top_k(self, sample_attributions):
        """Test importance bar with top-k filtering outputs exactly k bars."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        fig = viz.plot_importance_bar(top_k=3)
        assert fig is not None
        assert len(fig.data) == 1
        # In horizontal bar charts, y contains the categories (nodes)
        assert len(fig.data[0].y) == 3
        assert len(fig.data[0].x) == 3

    def test_plot_network_saliency(self, sample_attributions):
        """Test network saliency visualization renders correct node count."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        fig = viz.plot_network_saliency()
        assert fig is not None
        assert len(fig.data) == 1
        # Ensure all 5 nodes are plotted as scatter points
        assert len(fig.data[0].x) == 5
        assert len(fig.data[0].marker.size) == 5

    def test_plot_comparison_multidimensional(self, sample_attributions, sample_activations):
        """Test comparison visualization correctly flattens multi-dimensional activations."""
        viz = FeatureSaliencyVisualizer(
            sample_attributions,
            original_activations=sample_activations["original"],
            corrupted_activations=sample_activations["corrupted"],
        )
        fig = viz.plot_comparison()

        assert fig is not None
        assert len(fig.data) == 3  # Original, Corrupted, Difference heatmaps

        # The fixture has 2 layers, and sequences of length 10 (shape: 10, 64)
        # The code should max-pool across the last dimension, resulting in a (2, 10) grid
        z_data = fig.data[0].z
        assert len(z_data) == 2  # 2 layers (y-axis)
        assert len(z_data[0]) == 10  # 10 sequence positions (x-axis)

    def test_comparison_missing_data(self, sample_attributions):
        """Test error when comparison data is incomplete."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        with pytest.raises(ValueError):
            viz.plot_comparison()

    def test_export_html(self, sample_attributions, tmp_path):
        """Test HTML export."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        output_file = tmp_path / "test.html"
        viz.export_to_html(str(output_file))
        assert output_file.exists()

    def test_export_json(self, sample_attributions, tmp_path):
        """Test JSON export."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        output_file = tmp_path / "test.json"
        viz.export_to_json(str(output_file))
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert "node_attributions" in data
            assert "method" in data

    def test_get_top_nodes(self, sample_attributions):
        """Test getting top nodes."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        top_nodes = viz.get_top_nodes(k=3)
        assert len(top_nodes) == 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top_nodes)
        # Should be sorted descending
        assert top_nodes[0][1] >= top_nodes[1][1]

    def test_get_attribution_summary(self, sample_attributions):
        """Test getting attribution summary includes all expected statistics."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        summary = viz.get_attribution_summary()

        expected_keys = ["mean", "median", "max", "min", "std", "num_nodes", "method"]
        for key in expected_keys:
            assert key in summary, f"Missing expected key '{key}' in summary"

        # Verify basic math sanity (max >= min)
        assert summary["max"] >= summary["min"]

    def test_negative_attributions(self):
        """Test handling of negative attributions (uses absolute value)."""
        attributions = {
            "A0.0": -0.8,
            "A0.1": 0.6,
            "A1.0": -0.9,
        }
        viz = FeatureSaliencyVisualizer(attributions)
        # All normalized should be positive
        assert all(v >= 0 for v in viz.normalized_attributions.values())

    def test_zero_attributions(self):
        """Test handling of zero attributions."""
        attributions = {"A0.0": 0, "A0.1": 0}
        viz = FeatureSaliencyVisualizer(attributions)
        assert all(v == 0 for v in viz.normalized_attributions.values())

    def test_method_tracking(self):
        """Test that attribution method is tracked."""
        attributions = {"A0.0": 0.5}
        viz = FeatureSaliencyVisualizer(attributions, method="patching")
        assert viz.method == "patching"
        summary = viz.get_attribution_summary()
        assert summary["method"] == "patching"

    def test_invalid_plot_type(self, sample_attributions):
        """Test error on invalid plot type."""
        viz = FeatureSaliencyVisualizer(sample_attributions)
        with pytest.raises(ValueError):
            viz.export_to_html("test.html", plot_type="invalid")

    def test_export_all_plot_types(self, sample_attributions, tmp_path):
        """Test exporting all plot types."""
        original = {"layer_0": np.random.rand(10)}
        corrupted = {"layer_0": np.random.rand(10)}

        viz = FeatureSaliencyVisualizer(
            sample_attributions,
            original_activations=original,
            corrupted_activations=corrupted,
        )

        for plot_type in ["importance_bar", "network", "comparison"]:
            output_file = tmp_path / f"test_{plot_type}.html"
            viz.export_to_html(str(output_file), plot_type=plot_type)
            assert output_file.exists()

    def test_node_names_parameter(self, sample_attributions):
        """Test specifying custom node names."""
        custom_names = ["Node1", "Node2", "Node3", "Node4", "Node5"]
        viz = FeatureSaliencyVisualizer(
            sample_attributions,
            node_names=custom_names,
        )
        assert len(viz.node_names) == 5

    def test_comparison_mismatched_keys(self, sample_attributions):
        """Test error is raised when original and corrupted activations have different layers."""
        original = {"layer_0": np.random.rand(10)}
        corrupted = {"layer_1": np.random.rand(10)}  # Different key

        with pytest.raises(ValueError, match="same keys"):
            FeatureSaliencyVisualizer(
                sample_attributions,
                original_activations=original,
                corrupted_activations=corrupted,
            )

    def test_node_names_length_mismatch(self, sample_attributions):
        """Test error is raised if custom node names don't match attributions length."""
        custom_names = ["JustOneNode"]  # Missing 4 nodes

        with pytest.raises(ValueError, match="Length of node_names must match"):
            FeatureSaliencyVisualizer(
                sample_attributions,
                node_names=custom_names,
            )
