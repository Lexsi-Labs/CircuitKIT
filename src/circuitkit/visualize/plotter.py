import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch as t
from sklearn.preprocessing import minmax_scale

from circuitkit.backends.acdc.types import PruneScores
from circuitkit.backends.acdc.utils.tensor_ops import flat_prune_scores
from .theme import PALETTE


import logging

logger = logging.getLogger(__name__)

def _build_graph(p_model, prune_scores: PruneScores, threshold: float = 0.0):
    """Helper function to build a networkx graph from model and scores."""
    G = nx.DiGraph()

    # Normalize scores for coloring and thickness
    # Convert to float32 before moving to CPU and numpy, as bfloat16 is not
    # directly supported by some numpy/sklearn operations.
    flat_scores = flat_prune_scores(prune_scores).abs().to(t.float32).cpu().numpy()

    if len(flat_scores) > 0 and flat_scores.max() > 0:
        normalized_scores = minmax_scale(flat_scores)
    else:
        normalized_scores = np.zeros_like(flat_scores)

    score_idx = 0

    # Layer mapping for positioning
    layers = {}
    for node in p_model.nodes:
        if node.layer not in layers:
            layers[node.layer] = []
        layers[node.layer].append(node.name)
        G.add_node(node.name, layer=node.layer)

    for edge in p_model.edges:
        score = edge.prune_score(prune_scores).item()
        # Handle cases where scores might not be perfectly aligned (e.g., if graph changes)
        if score_idx < len(normalized_scores):
            norm_score = normalized_scores[score_idx]
        else:
            norm_score = 0
        score_idx += 1

        if abs(norm_score) >= threshold:
            G.add_edge(
                edge.src.name,
                edge.dest.name,
                score=score,
                weight=abs(norm_score) * 5 + 0.5,  # For line thickness
                color_val=norm_score,  # For color mapping
            )

    # Calculate positions
    pos = nx.multipartite_layout(G, subset_key="layer")
    return G, pos


def _create_plotly_figure(G, pos, title):
    """Helper function to create a Plotly figure from a networkx graph."""
    edge_traces = []
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ck_edge_cmap",
        [PALETTE.edge_low, "#E85D04", PALETTE.edge_high],
    )

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        color_val = edge[2]["color_val"]
        weight = edge[2]["weight"]

        color = "rgb" + str(tuple(int(c * 255) for c in cmap(color_val)[:3]))

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                line=dict(width=weight, color=color),
                hoverinfo="none",
                mode="lines",
            )
        )

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=False,  # Hiding color scale for simplicity
            color=PALETTE.attn_head,
            size=10,
            line_width=2,
        ),
    )

    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Circuit Graph",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


def visualize_static_circuit(p_model, prune_scores, threshold, file_path, title):
    """Visualizes a circuit with a fixed threshold. Good for ACDC."""
    flat_scores = flat_prune_scores(prune_scores).abs().to(t.float32).cpu().numpy()
    if flat_scores.max() > 0:
        # Heuristic for ACDC: find a threshold that shows *some* but not all edges.
        # This assumes scores are not binary but have some variance.
        non_zero_scores = flat_scores[flat_scores > 1e-6]
        if len(non_zero_scores) > 0:
            norm_threshold = np.quantile(non_zero_scores, 0.1) / flat_scores.max()
        else:
            norm_threshold = 1.0  # Show nothing if all scores are zero
    else:
        norm_threshold = 1.0

    G, pos = _build_graph(p_model, prune_scores, threshold=norm_threshold)
    fig = _create_plotly_figure(G, pos, title)
    fig.write_html(file_path)


def visualize_eap_circuit(p_model, prune_scores, default_threshold, file_path, title):
    """
    Creates a static visualization for EAP scores.
    """
    G, pos = _build_graph(p_model, prune_scores, threshold=default_threshold)
    fig = _create_plotly_figure(G, pos, f"{title} (Threshold: {default_threshold:.2f})")
    fig.write_html(file_path)


def visualize_eap_interactive(p_model, prune_scores, title="Interactive EAP Circuit"):
    """
    Creates an interactive EAP circuit visualization for use in Jupyter Notebooks.
    """
    try:
        import plotly.graph_objects as go
        from ipywidgets import FloatSlider, interact

    except ImportError:
        logger.info("Please install ipywidgets to use interactive visualization: pip install ipywidgets")
        return

    # Build the full graph once to get positions for all nodes
    full_G, pos = _build_graph(p_model, prune_scores, threshold=0.0)

    # Use FigureWidget for efficient updates
    fig = go.FigureWidget(
        layout=go.Layout(title=title, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
    )

    def update_plot(threshold):
        # <--- START OF CHANGE --->
        # Use batch_update to correctly modify the FigureWidget
        with fig.batch_update():
            # Filter edges based on the threshold
            G_filtered, _ = _build_graph(p_model, prune_scores, threshold=threshold)

            # Clear all existing traces from the figure
            fig.data = []

            # --- Add Edge Traces ---
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "ck_edge_cmap",
                [PALETTE.edge_low, "#E85D04", PALETTE.edge_high],
            )
            for edge in G_filtered.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                color_val = edge[2]["color_val"]
                weight = edge[2]["weight"]
                color = "rgb" + str(tuple(int(c * 255) for c in cmap(color_val)[:3]))
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(width=weight, color=color),
                        hoverinfo="none",
                    )
                )

            # --- Add Node Trace ---
            node_x, node_y, node_text = [], [], []
            for node in full_G.nodes():  # Use full_G to keep all nodes in place
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)

            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text",
                    text=node_text,
                    textposition="top center",
                    marker=dict(size=10, color=PALETTE.attn_head),
                    hoverinfo="text",
                )
            )

            fig.layout.title = f"{title} (Threshold: {threshold:.2f})"
        # <--- END OF CHANGE --->

    # Initial plot
    update_plot(threshold=0.5)

    # Create the slider and link it to the update function
    slider = FloatSlider(min=0.0, max=1.0, step=0.01, value=0.5, description="Threshold")
    interact(update_plot, threshold=slider)

    return fig
