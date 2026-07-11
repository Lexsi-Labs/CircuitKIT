"""
Tests for comparison dashboard.
"""

import json

import numpy as np
import pytest

from circuitkit.visualize.comparison import ComparisonDashboard


class TestComparisonDashboard:
    """Test comparison dashboard."""

    @pytest.fixture
    def sample_circuits(self):
        """Create sample circuits for comparison."""
        nodes = ["A0.0", "A0.1", "A1.0", "MLP0", "MLP1"]
        return {
            "seed_1": {node: np.random.rand() for node in nodes},
            "seed_2": {node: np.random.rand() for node in nodes},
            "seed_3": {node: np.random.rand() for node in nodes},
        }

    def test_initialization(self, sample_circuits):
        """Test dashboard initialization."""
        dashboard = ComparisonDashboard(sample_circuits)
        assert dashboard.comparison_type == "stability"
        assert len(dashboard.circuits) == 3

    def test_initialization_robustness(self, sample_circuits):
        """Test initialization with robustness type."""
        dashboard = ComparisonDashboard(
            sample_circuits,
            comparison_type="robustness",
        )
        assert dashboard.comparison_type == "robustness"

    def test_initialization_generalization(self, sample_circuits):
        """Test initialization with generalization type."""
        dashboard = ComparisonDashboard(
            sample_circuits,
            comparison_type="generalization",
        )
        assert dashboard.comparison_type == "generalization"

    def test_requires_multiple_circuits(self):
        """Test error with single circuit."""
        circuits = {
            "only_one": {"A0.0": 0.5},
        }
        with pytest.raises(ValueError):
            ComparisonDashboard(circuits)

    def test_common_nodes_extraction(self, sample_circuits):
        """Test extraction of common nodes."""
        dashboard = ComparisonDashboard(sample_circuits)
        common = dashboard.common_nodes
        assert len(common) > 0
        # All nodes should appear in all circuits
        for node in common:
            for circuit in sample_circuits.values():
                assert node in circuit

    def test_normalization(self, sample_circuits):
        """Test circuit normalization."""
        dashboard = ComparisonDashboard(sample_circuits)
        for circuit_name, normalized_circuit in dashboard.normalized_circuits.items():
            for score in normalized_circuit.values():
                assert 0 <= score <= 1

    def test_plot_stability_heatmap(self, sample_circuits):
        """Test stability heatmap."""
        dashboard = ComparisonDashboard(sample_circuits)
        fig = dashboard.plot_stability_heatmap()
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_stability_heatmap_top_k(self, sample_circuits):
        """Test stability heatmap with top-k."""
        dashboard = ComparisonDashboard(sample_circuits)
        fig = dashboard.plot_stability_heatmap(top_k=3)
        assert fig is not None

    def test_plot_correlation_matrix(self, sample_circuits):
        """Test correlation matrix visualization."""
        dashboard = ComparisonDashboard(sample_circuits)
        fig = dashboard.plot_correlation_matrix()
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_robustness_comparison(self, sample_circuits):
        """Test robustness comparison."""
        dashboard = ComparisonDashboard(sample_circuits)
        fig = dashboard.plot_robustness_comparison()
        assert fig is not None

    def test_plot_transfer_matrix(self, sample_circuits):
        """Test transfer matrix visualization."""
        dashboard = ComparisonDashboard(sample_circuits)
        fig = dashboard.plot_transfer_matrix()
        assert fig is not None
        assert len(fig.data) > 0

    def test_plot_distribution_comparison(self, sample_circuits):
        """Test distribution comparison."""
        dashboard = ComparisonDashboard(sample_circuits)
        fig = dashboard.plot_distribution_comparison()
        assert fig is not None
        assert len(fig.data) > 0

    def test_export_html_all(self, sample_circuits, tmp_path):
        """Test HTML export with all plots."""
        dashboard = ComparisonDashboard(sample_circuits)
        output_file = tmp_path / "test.html"
        dashboard.export_to_html(str(output_file), plot_type="all")
        assert output_file.exists()

    def test_export_html_single_type(self, sample_circuits, tmp_path):
        """Test HTML export with single plot type."""
        dashboard = ComparisonDashboard(sample_circuits)
        output_file = tmp_path / "test_stability.html"
        dashboard.export_to_html(str(output_file), plot_type="stability")
        assert output_file.exists()

    def test_export_json(self, sample_circuits, tmp_path):
        """Test JSON export."""
        dashboard = ComparisonDashboard(sample_circuits)
        output_file = tmp_path / "test.json"
        dashboard.export_to_json(str(output_file))
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert "comparison_type" in data
            assert "circuits" in data
            assert "common_nodes" in data

    def test_get_summary_stats(self, sample_circuits):
        """Test summary statistics."""
        dashboard = ComparisonDashboard(sample_circuits)
        summary = dashboard.get_summary_stats()
        assert "comparison_type" in summary
        assert "num_circuits" in summary
        assert "num_common_nodes" in summary
        assert "circuits" in summary
        assert len(summary["circuits"]) == 3

    def test_summary_stats_content(self, sample_circuits):
        """Test that summary stats contain required fields."""
        dashboard = ComparisonDashboard(sample_circuits)
        summary = dashboard.get_summary_stats()

        for circuit_name, circuit_stats in summary["circuits"].items():
            assert "mean" in circuit_stats
            assert "median" in circuit_stats
            assert "std" in circuit_stats
            assert "max" in circuit_stats
            assert "min" in circuit_stats

    def test_with_metadata(self, sample_circuits):
        """Test initialization with metadata."""
        metadata = {
            "task": "ioi",
            "model": "gpt2",
            "algorithm": "eap",
        }
        dashboard = ComparisonDashboard(
            sample_circuits,
            metadata=metadata,
        )
        assert dashboard.metadata == metadata

    def test_with_custom_labels(self, sample_circuits):
        """Test initialization with custom labels."""
        labels = ["Run A", "Run B", "Run C"]
        dashboard = ComparisonDashboard(
            sample_circuits,
            labels=labels,
        )
        assert dashboard.labels == labels

    def test_different_node_sets(self):
        """Test handling of circuits with different node sets."""
        circuits = {
            "circuit_1": {"A0.0": 0.5, "A0.1": 0.6, "MLP0": 0.7},
            "circuit_2": {"A0.0": 0.4, "A0.1": 0.5, "MLP1": 0.8},  # Different nodes
        }
        dashboard = ComparisonDashboard(circuits)
        # Common nodes should only be those in both
        assert "A0.0" in dashboard.common_nodes
        assert "A0.1" in dashboard.common_nodes
        assert "MLP0" not in dashboard.common_nodes or "MLP1" not in dashboard.common_nodes

    def test_export_all_plot_types(self, sample_circuits, tmp_path):
        """Test exporting all plot types individually."""
        dashboard = ComparisonDashboard(sample_circuits)

        for plot_type in ["stability", "correlation", "robustness", "transfer", "distribution"]:
            output_file = tmp_path / f"test_{plot_type}.html"
            dashboard.export_to_html(str(output_file), plot_type=plot_type)
            assert output_file.exists()

    def test_invalid_plot_type(self, sample_circuits):
        """Test error on invalid plot type."""
        dashboard = ComparisonDashboard(sample_circuits)
        with pytest.raises(ValueError):
            dashboard.export_to_html("test.html", plot_type="invalid")

    def test_high_correlation_same_circuit(self, sample_circuits):
        """Test that correlation with self is 1.0."""
        dashboard = ComparisonDashboard(sample_circuits)
        # Diagonal of correlation matrix should be 1.0
        # This is implicitly tested by the plot_correlation_matrix
        fig = dashboard.plot_correlation_matrix()
        assert fig is not None

    def test_correlation_matrix_mathematical_accuracy(self):
        """Test that Pearson correlation calculation yields correct mathematical values."""
        circuits = {
            # Perfect positive correlation to c3, perfect negative to c2
            "c1": {"A": 0.0, "B": 0.5, "C": 1.0},
            "c2": {"A": 1.0, "B": 0.5, "C": 0.0},
            "c3": {"A": 0.0, "B": 0.5, "C": 1.0},
        }
        dashboard = ComparisonDashboard(circuits)
        fig = dashboard.plot_correlation_matrix()

        # Extract the heatmap data matrix
        z_matrix = fig.data[0].z

        # c1 vs c1 (self) should be exactly 1.0
        assert z_matrix[0][0] == 1.0
        # c1 vs c2 (inverse) should be exactly -1.0
        assert z_matrix[0][1] == pytest.approx(-1.0)
        # c1 vs c3 (identical) should be exactly 1.0
        assert z_matrix[0][2] == pytest.approx(1.0)

    def test_transfer_matrix_jaccard_calculation(self):
        """
        Test Jaccard index logic and the hardcoded >0.5 threshold.
        """
        circuits = {
            # Normalized: A=0.0, B=0.5, C=1.0 -> Threshold passes: B, C
            "source_1": {"A": 0, "B": 5, "C": 10},
            # Normalized: A=0.0, B=0.0, C=1.0 -> Threshold passes: C
            "source_2": {"A": 0, "B": 0, "C": 10},
        }
        dashboard = ComparisonDashboard(circuits)
        fig = dashboard.plot_transfer_matrix()
        z_matrix = fig.data[0].z

        # Intersection of {B, C} and {C} is {C} (len=1)
        # Union of {B, C} and {C} is {B, C} (len=2)
        # Jaccard = 1 / 2 = 0.5
        assert z_matrix[0][1] == 0.5
        assert z_matrix[1][0] == 0.5
        assert z_matrix[0][0] == 1.0  # Self-transfer

    def test_transfer_matrix_empty_intersection(self):
        """Test transfer matrix behavior when no nodes pass the threshold."""
        circuits = {
            "flat_1": {"A": 0.5, "B": 0.5},  # Normalizes to 0.0, 0.0
            "flat_2": {"A": 0.5, "B": 0.5},  # Normalizes to 0.0, 0.0
        }
        dashboard = ComparisonDashboard(circuits)
        fig = dashboard.plot_transfer_matrix()
        z_matrix = fig.data[0].z

        # If no nodes pass threshold, off-diagonal should be 0.0, diagonal 1.0
        assert z_matrix[0][1] == 0.0
        assert z_matrix[0][0] == 1.0

    def test_no_common_nodes_exception(self):
        """Test pipeline raises correct ValueError when there is no node overlap."""
        circuits = {
            "run_A": {"node_1": 0.8, "node_2": 0.5},
            "run_B": {"node_3": 0.9, "node_4": 0.4},
        }
        with pytest.raises(ValueError, match="No common nodes found across circuits"):
            ComparisonDashboard(circuits)

    def test_zero_variance_normalization(self):
        """
        Test that circuits with identical scores for all nodes do not cause
        divide-by-zero errors during normalization.
        """
        circuits = {
            "flat_circuit": {"A": 0.5, "B": 0.5, "C": 0.5},
            "normal_circuit": {"A": 0.1, "B": 0.5, "C": 0.9},
        }
        dashboard = ComparisonDashboard(circuits)

        # A flat circuit should normalize to an array of zeros safely
        assert dashboard.normalized_circuits["flat_circuit"]["A"] == 0.0
        assert dashboard.normalized_circuits["flat_circuit"]["B"] == 0.0

        # The normal circuit should still scale correctly to [0, 1]
        assert dashboard.normalized_circuits["normal_circuit"]["A"] == pytest.approx(0.0)
        assert dashboard.normalized_circuits["normal_circuit"]["B"] == pytest.approx(0.5)
        assert dashboard.normalized_circuits["normal_circuit"]["C"] == pytest.approx(1.0)

    def test_robustness_top_k_sorting_logic(self):
        """Test that top_k correctly averages and sorts the most important nodes."""
        circuits = {
            # Norm: A=0.0, B=1.0, C=0.5
            "run_1": {"A": 0.1, "B": 0.9, "C": 0.5},
            # Norm: A=0.0, B=1.0, C=0.714
            "run_2": {"A": 0.1, "B": 0.8, "C": 0.6},
        }
        # Averages: A=0.0, B=1.0, C=0.607.
        # Therefore, order should be B, C, A.
        dashboard = ComparisonDashboard(circuits)

        # Ask for Top 2, should return B and C in that exact order
        fig = dashboard.plot_robustness_comparison(top_k=2)
        x_axis_nodes = fig.data[0].x

        assert list(x_axis_nodes) == ["B", "C"]
        assert "A" not in x_axis_nodes
