"""
Jupyter widget suite that packages all circuit visualization components.

Provides a unified entry point for displaying comprehensive circuit analysis
within Jupyter notebooks.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional


import logging

logger = logging.getLogger(__name__)

try:
    import ipywidgets as widgets
    from IPython.display import display

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

from .comparison import ComparisonDashboard
from .editor import CircuitEditor
from .feature_saliency import FeatureSaliencyVisualizer
from .graph_viz import CircuitGraphVisualizer
from .saliency import ActivationSaliencyVisualizer


class JupyterWidgetSuite:
    """
    Unified Jupyter widget suite for comprehensive circuit analysis.

    Wraps all visualization components:
    - CircuitGraphVisualizer
    - ActivationSaliencyVisualizer
    - FeatureSaliencyVisualizer
    - CircuitEditor
    - ComparisonDashboard

    Provides single entry point: display_circuit_analysis()
    """

    def __init__(self):
        """Initialize the widget suite."""
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets is required for JupyterWidgetSuite. "
                "Install with: pip install ipywidgets"
            )

    @staticmethod
    def display_circuit_analysis(
        circuit: Dict[str, Any],
        node_scores: Dict[str, float],
        tokens: Optional[List[str]] = None,
        activations: Optional[Dict[str, Any]] = None,
        node_attributions: Optional[Dict[str, float]] = None,
        edge_scores: Optional[Dict[tuple, float]] = None,
        comparison_circuits: Optional[Dict[str, Dict[str, float]]] = None,
        show_editor: bool = True,
        export_path: Optional[str] = None,
    ) -> None:
        """
        Display comprehensive circuit analysis dashboard.

        Args:
            circuit: Circuit structure with 'nodes' and 'edges'.
            node_scores: Dict of node importance scores.
            tokens: Optional list of token strings.
            activations: Optional dict of layer activations for saliency.
            node_attributions: Optional dict of node attribution scores.
            edge_scores: Optional dict of edge importance scores.
            comparison_circuits: Optional dict of circuits for comparison.
            show_editor: Whether to show the interactive editor.
            export_path: Optional path to export analysis to HTML.
        """
        from ..artifacts.scores import CircuitScores


        # Create tabs for different visualizations
        tab_children = []
        tab_titles = []

        # Tab 1: Circuit Graph
        try:
            # Create dummy scores if needed
            scores = CircuitScores(
                task="analysis",
                model="unknown",
                algorithm="eap-ig",
                level="node",
                node_scores=node_scores,
                timestamp="",
            )

            graph_viz = CircuitGraphVisualizer(circuit, scores, edge_scores)
            graph_fig = graph_viz.plot_graph()

            graph_widget = widgets.Output()
            with graph_widget:
                display(graph_fig)

            tab_children.append(graph_widget)
            tab_titles.append("Circuit Graph")
        except Exception as e:
            warnings.warn(f"Failed to create circuit graph: {e}")

        # Tab 2: Activation Saliency
        if activations:
            try:
                saliency_viz = ActivationSaliencyVisualizer(
                    activations=activations,
                    tokens=tokens,
                )

                saliency_widget = widgets.Output()
                with saliency_widget:
                    fig = saliency_viz.plot_layer_comparison()
                    display(fig)

                tab_children.append(saliency_widget)
                tab_titles.append("Activation Saliency")
            except Exception as e:
                warnings.warn(f"Failed to create activation saliency: {e}")

        # Tab 3: Feature Attribution
        if node_attributions:
            try:
                feature_viz = FeatureSaliencyVisualizer(
                    node_attributions=node_attributions,
                )

                feature_widget = widgets.Output()
                with feature_widget:
                    fig = feature_viz.plot_importance_bar(top_k=20)
                    display(fig)

                tab_children.append(feature_widget)
                tab_titles.append("Feature Attribution")
            except Exception as e:
                warnings.warn(f"Failed to create feature attribution: {e}")

        # Tab 4: Interactive Editor
        if show_editor:
            try:
                editor = CircuitEditor(circuit)

                editor_widget = widgets.Output()
                with editor_widget:
                    editor.display()

                tab_children.append(editor_widget)
                tab_titles.append("Interactive Editor")
            except Exception as e:
                warnings.warn(f"Failed to create circuit editor: {e}")

        # Tab 5: Comparison Dashboard
        if comparison_circuits:
            try:
                dashboard = ComparisonDashboard(comparison_circuits)

                comparison_widget = widgets.Output()
                with comparison_widget:
                    fig = dashboard.plot_stability_heatmap()
                    display(fig)

                tab_children.append(comparison_widget)
                tab_titles.append("Comparison")
            except Exception as e:
                warnings.warn(f"Failed to create comparison dashboard: {e}")

        # Create tab container
        if tab_children:
            tabs = widgets.Tab(children=tab_children)
            for i, title in enumerate(tab_titles):
                tabs.set_title(i, title)

            # Display main UI
            title_widget = widgets.HTML("<h2>Circuit Analysis Dashboard</h2>")

            display(title_widget)
            display(tabs)

            # Export option
            if export_path:
                try:
                    export_dir = Path(export_path)
                    export_dir.mkdir(parents=True, exist_ok=True)

                    if activations:
                        saliency_viz.export_to_html(str(export_dir / "activation_saliency.html"))

                    if node_attributions:
                        feature_viz.export_to_html(str(export_dir / "feature_attribution.html"))

                    if comparison_circuits:
                        dashboard.export_to_html(str(export_dir / "comparison.html"))

                    logger.info(f"Analysis exported to {export_path}")
                except Exception as e:
                    warnings.warn(f"Failed to export analysis: {e}")
        else:
            logger.info("No visualizations could be created with provided data.")


def display_circuit_analysis(
    circuit: Dict[str, Any],
    node_scores: Dict[str, float],
    **kwargs,
) -> None:
    """
    Convenience function to display circuit analysis.

    Args:
        circuit: Circuit structure with 'nodes' and 'edges'.
        node_scores: Dict of node importance scores.
        **kwargs: Additional arguments passed to JupyterWidgetSuite.display_circuit_analysis()
    """
    suite = JupyterWidgetSuite()
    suite.display_circuit_analysis(circuit, node_scores, **kwargs)
