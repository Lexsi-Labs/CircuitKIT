# visualize

Tools for visualizing circuit discovery results: interactive graphs, saliency and feature-attribution maps, comparison dashboards, editors, galleries, and HTML/Jupyter/Streamlit exports.

## Key modules

- `graph_viz.py` — `CircuitGraphVisualizer`: shared data prep with Plotly notebook widget and D3.js HTML export paths for circuit graphs.
- `d3_template.py` — `render_d3_circuit_html`: renders a self-contained, zoom/pan/threshold-slider D3.js HTML page from graph payloads.
- `saliency.py` — `ActivationSaliencyVisualizer`: token/position activation heatmaps across layers.
- `feature_saliency.py` — `FeatureSaliencyVisualizer`: node-level feature attribution maps (gradient or patching based).
- `comparison.py` — `ComparisonDashboard`: side-by-side circuit comparison across seeds, corruptions, or tasks.
- `editor.py` — `CircuitEditor`: interactive ipywidgets Jupyter editor for adding/removing/toggling nodes and edges.
- `jupyter_suite.py` — `JupyterWidgetSuite` and `display_circuit_analysis`: unified Jupyter entry point bundling all visualizers.
- `gallery.py` — `GalleryGenerator`: auto-generated interactive HTML gallery of circuit analyses.
- `streamlit_app.py` — self-contained multi-page Streamlit dashboard (`streamlit run streamlit_app.py`).
- `plotter.py` — networkx/Plotly plotting helpers over ACDC `PruneScores`.
- `theme.py` — design-system single source of truth: `PALETTE`, `get_node_color`, `get_plotly_layout`, `get_d3_theme`.

## Public API

`CircuitGraphVisualizer`, `ActivationSaliencyVisualizer`, `FeatureSaliencyVisualizer`, `CircuitEditor`, `ComparisonDashboard`, `JupyterWidgetSuite`, `display_circuit_analysis`, `GalleryGenerator`, `render_d3_circuit_html`, plus `theme` helpers (`PALETTE`, `get_node_color`, `get_plotly_layout`, `get_d3_theme`).

## How it fits

Reads `artifacts` (notably `CircuitScores`) and backend prune-scores, turning discovery output into interactive or exportable visualizations for notebooks, browsers, and dashboards.
