"""
CircuitKit visualization module.

Provides tools for visualizing circuit discovery results, including:
- Graph visualization (nodes and edges)
- Activation saliency heatmaps
- Feature attribution maps
- Interactive circuit editor
- Comparison dashboards
- Jupyter widget suite
- Streamlit dashboard
- Visualization gallery
- Static HTML exports
"""

from .comparison import ComparisonDashboard
from .editor import CircuitEditor
from .feature_saliency import FeatureSaliencyVisualizer
from .gallery import GalleryGenerator
from .graph_viz import CircuitGraphVisualizer
from .jupyter_suite import JupyterWidgetSuite, display_circuit_analysis
from .saliency import ActivationSaliencyVisualizer
from .theme import PALETTE, get_node_color, get_plotly_layout, get_d3_theme
from .d3_template import render_d3_circuit_html

__all__ = [
    "CircuitGraphVisualizer",
    "ActivationSaliencyVisualizer",
    "FeatureSaliencyVisualizer",
    "CircuitEditor",
    "ComparisonDashboard",
    "JupyterWidgetSuite",
    "display_circuit_analysis",
    "GalleryGenerator",
    "render_d3_circuit_html",
    "PALETTE",
    "get_d3_theme",
    "get_node_color",
    "get_plotly_layout",
]


# ---------------------------------------------------------------------------
# Deprecated pre-1.0 alias (H1 in the 1.0.0 audit).
#
# ``circuitkit.visualize`` used to be the top-level entry point; it was renamed
# to ``circuitkit.visualize_circuit`` in 1.0.0. The name ``visualize`` is now
# this subpackage, so the old *call* site ``ck.visualize(circuit, ...)`` would
# otherwise raise "module is not callable". Making the package object callable
# keeps that call working with a DeprecationWarning, delegating to the new
# name. A plain ``module.__getattr__`` shim on the parent package cannot do
# this: once the subpackage is imported (e.g. ``circuitkit.visualize.graph_viz``
# or any ``mock.patch("circuitkit.visualize....")``) the submodule shadows the
# shim, so the warning must live on the package object itself. Removal planned
# for 2.0.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types


class _CallableVisualizeModule(_types.ModuleType):
    """Module subclass so ``circuitkit.visualize(...)`` stays callable."""

    def __call__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "circuitkit.visualize(...) was renamed to "
            "circuitkit.visualize_circuit(...) in 1.0.0; this deprecated alias "
            "will be removed in 2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..quick import visualize_circuit

        return visualize_circuit(*args, **kwargs)


_sys.modules[__name__].__class__ = _CallableVisualizeModule
