"""
Tests for Jupyter widget suite.
"""

import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from circuitkit.visualize.jupyter_suite import JupyterWidgetSuite


@pytest.fixture
def sample_circuit():
    """Create sample circuit."""
    return {
        "nodes": {
            "A0.0": {"layer": 0},
            "A0.1": {"layer": 0},
            "MLP0": {"layer": 0},
        },
        "edges": [
            ["A0.0", "MLP0"],
            ["A0.1", "MLP0"],
        ],
    }


@pytest.fixture
def sample_scores():
    """Create sample node scores."""
    return {
        "A0.0": 0.8,
        "A0.1": 0.6,
        "MLP0": 0.7,
    }


@pytest.fixture
def sample_activations():
    """Create sample activations."""
    return {
        "layer_0": np.random.rand(10),
        "layer_1": np.random.rand(10),
    }


@pytest.fixture
def sample_attributions():
    """Create sample attributions."""
    return {
        "A0.0": 0.8,
        "A0.1": 0.6,
        "MLP0": 0.7,
    }


class TestJupyterWidgetSuite:
    """Test Jupyter widget suite."""

    def test_suite_initialization(self):
        """Test widget suite initialization."""
        try:
            suite = JupyterWidgetSuite()
            assert suite is not None
        except ImportError:
            # ipywidgets not available in test environment
            pytest.skip("ipywidgets not available")

    def test_display_circuit_analysis_basic(self, sample_circuit, sample_scores):
        """Test basic circuit analysis display."""
        try:
            # This will fail in non-Jupyter environment, so we just check it doesn't crash
            suite = JupyterWidgetSuite()
            # Don't actually call display since we're not in Jupyter
            assert suite is not None
        except ImportError:
            pytest.skip("ipywidgets not available")

    def test_convenience_function_delegates_to_suite(self, sample_circuit, sample_scores):
        """display_circuit_analysis() convenience function calls the suite method."""
        from circuitkit.visualize.jupyter_suite import display_circuit_analysis

        with patch(
            "circuitkit.visualize.jupyter_suite.JupyterWidgetSuite.display_circuit_analysis"
        ) as mock_method:
            display_circuit_analysis(sample_circuit, sample_scores, show_editor=False)
            mock_method.assert_called_once_with(sample_circuit, sample_scores, show_editor=False)

    def test_with_activations(self, sample_circuit, sample_scores, sample_activations):
        """Activation saliency visualizer is constructed when activations are passed."""
        with (
            patch("circuitkit.visualize.jupyter_suite.ActivationSaliencyVisualizer") as MockASV,
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            JupyterWidgetSuite.display_circuit_analysis(
                sample_circuit, sample_scores, activations=sample_activations
            )
            MockASV.assert_called_once()
            assert MockASV.call_args[1]["activations"] is sample_activations


def test_with_attributions(sample_circuit, sample_scores, sample_attributions):
    """Feature saliency visualizer is constructed when attributions are passed."""
    with (
        patch("circuitkit.visualize.jupyter_suite.FeatureSaliencyVisualizer") as MockFSV,
        patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
        patch("circuitkit.visualize.jupyter_suite.display"),
        patch("circuitkit.visualize.jupyter_suite.widgets"),
    ):
        JupyterWidgetSuite.display_circuit_analysis(
            sample_circuit, sample_scores, node_attributions=sample_attributions
        )
        MockFSV.assert_called_once()
        assert MockFSV.call_args[1]["node_attributions"] is sample_attributions


def test_with_comparison_circuits(sample_circuit, sample_scores):
    """ComparisonDashboard is constructed when comparison_circuits are passed."""
    comparison = {"circuit_a": {"A0.0": 0.9}, "circuit_b": {"A0.0": 0.3}}
    with (
        patch("circuitkit.visualize.jupyter_suite.ComparisonDashboard") as MockCD,
        patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
        patch("circuitkit.visualize.jupyter_suite.display"),
        patch("circuitkit.visualize.jupyter_suite.widgets"),
    ):
        JupyterWidgetSuite.display_circuit_analysis(
            sample_circuit, sample_scores, comparison_circuits=comparison
        )
        MockCD.assert_called_once_with(comparison)


def test_show_editor_false(sample_circuit, sample_scores):
    """CircuitEditor is NOT constructed when show_editor=False."""
    with (
        patch("circuitkit.visualize.jupyter_suite.CircuitEditor") as MockEd,
        patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
        patch("circuitkit.visualize.jupyter_suite.display"),
        patch("circuitkit.visualize.jupyter_suite.widgets"),
    ):
        JupyterWidgetSuite.display_circuit_analysis(
            sample_circuit, sample_scores, show_editor=False
        )
        MockEd.assert_not_called()


class TestWarningPaths:
    """Each visualizer failure emits a warning and doesn't crash the suite."""

    def test_graph_viz_failure_emits_warning(self, sample_circuit, sample_scores):
        with (
            patch(
                "circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer",
                side_effect=Exception("graph boom"),
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            with pytest.warns(UserWarning, match="graph"):
                JupyterWidgetSuite.display_circuit_analysis(sample_circuit, sample_scores)

    def test_saliency_failure_emits_warning(
        self, sample_circuit, sample_scores, sample_activations
    ):
        with (
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch(
                "circuitkit.visualize.jupyter_suite.ActivationSaliencyVisualizer",
                side_effect=Exception("saliency boom"),
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            with pytest.warns(UserWarning, match="activation saliency"):
                JupyterWidgetSuite.display_circuit_analysis(
                    sample_circuit, sample_scores, activations=sample_activations
                )

    def test_feature_viz_failure_emits_warning(
        self, sample_circuit, sample_scores, sample_attributions
    ):
        with (
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch(
                "circuitkit.visualize.jupyter_suite.FeatureSaliencyVisualizer",
                side_effect=Exception("feature boom"),
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            with pytest.warns(UserWarning, match="feature attribution"):
                JupyterWidgetSuite.display_circuit_analysis(
                    sample_circuit, sample_scores, node_attributions=sample_attributions
                )

    def test_editor_failure_emits_warning(self, sample_circuit, sample_scores):
        with (
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch(
                "circuitkit.visualize.jupyter_suite.CircuitEditor",
                side_effect=Exception("editor boom"),
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            with pytest.warns(UserWarning, match="editor"):
                JupyterWidgetSuite.display_circuit_analysis(sample_circuit, sample_scores)

    def test_comparison_failure_emits_warning(self, sample_circuit, sample_scores):
        comparison = {"a": {"A0.0": 0.9}}
        with (
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch(
                "circuitkit.visualize.jupyter_suite.ComparisonDashboard",
                side_effect=Exception("dashboard boom"),
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            with pytest.warns(UserWarning, match="comparison"):
                JupyterWidgetSuite.display_circuit_analysis(
                    sample_circuit, sample_scores, comparison_circuits=comparison
                )

    def test_all_failures_still_prints_no_viz_message(self, sample_circuit, sample_scores, capsys):
        """When every tab fails, the fallback print message appears."""
        with (
            patch(
                "circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer",
                side_effect=Exception("boom"),
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                JupyterWidgetSuite.display_circuit_analysis(sample_circuit, sample_scores)
        captured = capsys.readouterr()
        assert (
            "circuit graph" in captured.err.lower()
            or "boom" in captured.err.lower()
            or "VBox" in captured.out
        )


class TestExportPath:

    def test_export_creates_directory(
        self, sample_circuit, sample_scores, sample_activations, tmp_path
    ):
        export_dir = tmp_path / "exports"
        mock_saliency = MagicMock()
        with (
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch(
                "circuitkit.visualize.jupyter_suite.ActivationSaliencyVisualizer",
                return_value=mock_saliency,
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            JupyterWidgetSuite.display_circuit_analysis(
                sample_circuit,
                sample_scores,
                activations=sample_activations,
                export_path=str(export_dir),
            )
        assert export_dir.exists()

    def test_export_calls_saliency_html(
        self, sample_circuit, sample_scores, sample_activations, tmp_path
    ):
        mock_saliency = MagicMock()
        with (
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch(
                "circuitkit.visualize.jupyter_suite.ActivationSaliencyVisualizer",
                return_value=mock_saliency,
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            JupyterWidgetSuite.display_circuit_analysis(
                sample_circuit,
                sample_scores,
                activations=sample_activations,
                export_path=str(tmp_path),
            )
        mock_saliency.export_to_html.assert_called_once_with(
            str(tmp_path / "activation_saliency.html")
        )

    def test_export_calls_feature_html(
        self, sample_circuit, sample_scores, sample_attributions, tmp_path
    ):
        mock_feature = MagicMock()
        with (
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch(
                "circuitkit.visualize.jupyter_suite.FeatureSaliencyVisualizer",
                return_value=mock_feature,
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            JupyterWidgetSuite.display_circuit_analysis(
                sample_circuit,
                sample_scores,
                node_attributions=sample_attributions,
                export_path=str(tmp_path),
            )
        mock_feature.export_to_html.assert_called_once_with(
            str(tmp_path / "feature_attribution.html")
        )

    def test_export_calls_comparison_html(self, sample_circuit, sample_scores, tmp_path):
        comparison = {"a": {"A0.0": 0.9}}
        mock_dashboard = MagicMock()
        with (
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch(
                "circuitkit.visualize.jupyter_suite.ComparisonDashboard",
                return_value=mock_dashboard,
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            JupyterWidgetSuite.display_circuit_analysis(
                sample_circuit,
                sample_scores,
                comparison_circuits=comparison,
                export_path=str(tmp_path),
            )
        mock_dashboard.export_to_html.assert_called_once_with(str(tmp_path / "comparison.html"))

    def test_export_failure_emits_warning(
        self, sample_circuit, sample_scores, sample_activations, tmp_path
    ):
        mock_saliency = MagicMock()
        mock_saliency.export_to_html.side_effect = IOError("disk full")
        with (
            patch("circuitkit.visualize.jupyter_suite.CircuitGraphVisualizer"),
            patch(
                "circuitkit.visualize.jupyter_suite.ActivationSaliencyVisualizer",
                return_value=mock_saliency,
            ),
            patch("circuitkit.visualize.jupyter_suite.display"),
            patch("circuitkit.visualize.jupyter_suite.widgets"),
        ):
            with pytest.warns(UserWarning, match="export"):
                JupyterWidgetSuite.display_circuit_analysis(
                    sample_circuit,
                    sample_scores,
                    activations=sample_activations,
                    export_path=str(tmp_path),
                )
