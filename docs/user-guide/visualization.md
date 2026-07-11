# Visualization

`ck.visualize_circuit()` has three modes: an interactive Plotly graph, a
comparison dashboard, and a Streamlit dashboard. The graph and comparison modes
export to standalone HTML.

## Quick start

```python
import circuitkit as ck

circuit = ck.load_scores("./circuit.pt")
model = ck.load_model("gpt2")

# Interactive graph view
fig = ck.visualize_circuit(circuit, mode="graph")
fig.show()

# Export to HTML
ck.visualize_circuit(circuit, mode="graph", output="./circuit_graph.html")
```

Via Pipeline:
```python
pipe.visualize(mode="graph", output="./circuit.html")
```

## Mode 1: Graph view

Renders the circuit as a directed graph: nodes are attention heads and MLP layers, edges represent information flow weighted by attribution score.

```python
fig = ck.visualize_circuit(circuit, mode="graph")
```

- Nodes coloured by type (blue = attention head, amber = MLP)
- Node size proportional to importance score
- Edges only for components at or above sparsity threshold
- Hover tooltips with layer, head index, and importance score

Graph mode delegates to `Circuit.plot`, which takes only an optional output path.
Pass `output` to write an HTML export, or omit it for the inline widget:

```python
ck.visualize_circuit(circuit, mode="graph", output="./circuit_graph.html")
```

## Mode 2: Comparison dashboard

Compare two circuits side-by-side: algorithm comparisons, cross-task transfer, or before/after pruning.

```python
circuit_a = ck.load_scores("./circuit_eap_ig.pt")
circuit_b = ck.load_scores("./circuit_acdc.pt")

dashboard = ck.visualize_circuit(
    circuit_a,
    mode="comparison",
    second_circuit=circuit_b,
    comparison_type="stability",   # "stability", "robustness", or "generalization"
    labels=["EAP-IG", "ACDC"],
)
```

Both circuits must carry node scores. Passing `output="./compare.html"` writes the
dashboard to HTML and returns `None`; without `output` you get the
`ComparisonDashboard` instance back.

`comparison_type` accepts `"stability"` (default), `"robustness"`, or
`"generalization"`. It labels the comparison in the exported metadata; the HTML
export itself renders all available views (stability heatmap, correlation matrix,
robustness, transfer matrix, and score distribution).

## Mode 3: Streamlit dashboard

Launch the interactive Streamlit dashboard as a background subprocess.

```python
ck.visualize_circuit(circuit, mode="dashboard")
```

This runs `streamlit run` on the bundled `circuitkit/visualize/streamlit_app.py`
and returns `None`. Streamlit must be installed, and the app opens in your
browser rather than rendering inline.

## CLI

There is no `circuitkit visualize` CLI command. Visualization is currently available only through the Python API shown above (`ck.visualize_circuit()` / `Pipeline.visualize()`).

## Jupyter widgets

In a Jupyter notebook, the visualizations render inline automatically. For a
bundled multi-tab analysis view (graph, saliency, feature attribution, editor,
comparison), use `display_circuit_analysis` from `circuitkit.visualize`:

```python
from circuitkit.visualize import display_circuit_analysis

display_circuit_analysis(
    circuit={"nodes": [...], "edges": [...]},
    node_scores=circuit.scores,
)
```

This requires `ipywidgets`. See the `JupyterWidgetSuite` class in
`circuitkit.visualize` for the full set of tabs and options.

## Next steps

- [:octicons-arrow-right-24: Applications](applications.md)
- [:octicons-arrow-right-24: Selectors](selectors.md)
