"""
Circuit graph visualization: Plotly notebook widget + D3.js HTML export.

Two rendering paths that share a single data-preparation layer:

* **Plotly** (``interactive_widget``) — returns a ``go.FigureWidget`` for
  inline Jupyter display with a threshold slider.
* **D3.js** (``to_html``) — writes a self-contained HTML file with full
  zoom/pan, hover tooltips, and a real-time threshold slider.

Both paths consume the same ``_GraphData`` structure built from a
``CircuitScores`` artifact and an optional list of pruned node names.

Key design decisions
--------------------
* **All 156 scored nodes are shown** (for GPT-2 node-level discovery) so the
  viewer can see the full model and identify where the circuit sits.
* **Circuit nodes** (kept = ``scores.keys() - pruned_nodes``) are rendered
  with full color, larger size, and solid borders.
* **Pruned nodes** are rendered as small, low-opacity gray markers so the
  background model structure is legible without competing with the circuit.
* **Edges** connect adjacent-layer circuit nodes, weighted by the product of
  the two endpoint raw scores (normalized within edge set). The threshold
  slider filters edges by this normalized weight.
* **Log-scale normalization** of raw scores drives node size/opacity and
  edge color/width. Scores span orders of magnitude, so values are spread
  on a true log scale — this prevents the MLP 0 outlier (score ≈ 1.33 vs.
  next ≈ 0.075) from making all other nodes and edges appear equally
  unimportant, as happens with linear (or log1p) normalization.
* **Layout**: X = layer index; Y = evenly spaced within layer, MLP centered.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import plotly.graph_objects as go

from ..artifacts.scores import CircuitScores
from .theme import PALETTE, FONT_FAMILY, FONT_MONO, get_node_color, get_plotly_layout, get_d3_theme


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

@dataclass
class _NodeData:
    """All per-node values needed for rendering."""
    name: str
    node_type: str          # "attn_head" | "mlp" | "unknown"
    layer: int
    head: Optional[int]     # None for MLP nodes
    raw_score: float
    log_norm_score: float   # log1p(raw) normalized to [0, 1] across all nodes
    in_circuit: bool        # True = kept (not pruned)
    rank: int               # rank by raw score descending (1 = most important)
    x: float                # layout x coordinate
    y: float                # layout y coordinate


@dataclass
class _EdgeData:
    """All per-edge values needed for rendering."""
    src: str
    dst: str
    # weight = product of raw scores, log1p-normalized across edge set to [0,1]
    normalized_weight: float
    raw_weight: float       # raw score product, for tooltip display


@dataclass
class _GraphData:
    """Complete graph ready for rendering. Built once, consumed by both renderers."""
    nodes: List[_NodeData]
    edges: List[_EdgeData]
    n_layers: int
    circuit_node_ids: Set[str]
    metadata: Dict[str, Any]   # task, model, algorithm, timestamp


# ---------------------------------------------------------------------------
# Node name parser
# ---------------------------------------------------------------------------

def _parse_node_name(name: str) -> Tuple[str, int, Optional[int]]:
    """
    Parse a node name into (node_type, layer, head).

    Supported formats:
        ``"A0.1"``  → ``("attn_head", 0, 1)``
        ``"MLP 3"`` → ``("mlp", 3, None)``

    Returns:
        Tuple of (node_type, layer, head_index_or_None).
    """
    if name.startswith("A") and "." in name:
        parts = name[1:].split(".")
        if len(parts) == 2:
            try:
                return "attn_head", int(parts[0]), int(parts[1])
            except ValueError:
                pass
    elif name.startswith("MLP"):
        parts = name.split()
        if len(parts) == 2:
            try:
                return "mlp", int(parts[1]), None
            except ValueError:
                pass
    return "unknown", 0, None


# ---------------------------------------------------------------------------
# Layout engine
# ---------------------------------------------------------------------------

def _compute_layout(
    nodes: List[_NodeData],
    x_spacing: float = 1.8,
    y_spacing: float = 0.72,
) -> List[_NodeData]:
    """
    Assign (x, y) coordinates to every node.

    Layout strategy:
    - X axis = layer index * x_spacing.
    - Within each layer, attention heads are sorted by head index and
      distributed symmetrically around y=0.  The MLP node for that layer
      is placed at the vertical center (y=0) between the head clusters
      using a small y offset so it doesn't collide with head A*.0.

    The result keeps all nodes of a layer at the same x so the viewer
    can read the circuit depth left-to-right.

    Args:
        nodes:      Flat list of _NodeData (x, y will be mutated in place).
        x_spacing:  Horizontal distance between adjacent layers.
        y_spacing:  Vertical gap between nodes within a layer.

    Returns:
        The same list with x, y filled in.
    """
    # Group by layer
    from collections import defaultdict
    by_layer: Dict[int, List[_NodeData]] = defaultdict(list)
    for nd in nodes:
        by_layer[nd.layer].append(nd)

    for layer, layer_nodes in by_layer.items():
        x = layer * x_spacing

        attn_nodes = sorted(
            [n for n in layer_nodes if n.node_type == "attn_head"],
            key=lambda n: n.head if n.head is not None else 0,
        )
        mlp_nodes = [n for n in layer_nodes if n.node_type == "mlp"]
        other_nodes = [n for n in layer_nodes if n.node_type == "unknown"]

        # Distribute attention heads symmetrically around y=0
        n_attn = len(attn_nodes)
        for i, nd in enumerate(attn_nodes):
            nd.x = x
            nd.y = (i - (n_attn - 1) / 2.0) * y_spacing

        # Place MLP just below the attn cluster (predictable, never overlapping)
        mlp_y_offset = ((n_attn - 1) / 2.0 + 1.2) * y_spacing
        for nd in mlp_nodes:
            nd.x = x
            nd.y = mlp_y_offset

        # Anything else goes below MLP
        for j, nd in enumerate(other_nodes):
            nd.x = x
            nd.y = mlp_y_offset + (j + 1) * y_spacing

    return nodes


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """Linearly interpolate between two ``#RRGGBB`` colors at t ∈ [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    r1, g1, b1 = (int(c1[i:i + 2], 16) for i in (1, 3, 5))
    r2, g2, b2 = (int(c2[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02X}{:02X}{:02X}".format(
        round(r1 + (r2 - r1) * t),
        round(g1 + (g2 - g1) * t),
        round(b1 + (b2 - b1) * t),
    )


# ---------------------------------------------------------------------------
# Score normalization helpers
# ---------------------------------------------------------------------------

def _log_normalize(values: List[float]) -> List[float]:
    """
    Min-max normalize on a true logarithmic scale to [0, 1].

    Importance scores routinely span several orders of magnitude (e.g.
    MLP 0 ≈ 1.33 while typical heads sit at 1e-2…1e-4, and edge weights —
    score *products* — reach 1e-6). A plain log(v) spreads those orders of
    magnitude evenly across [0, 1], so a single outlier no longer compresses
    every other value to ≈ 0.

    Note: an earlier version used log1p, which is ~identity for v ≪ 1 and
    therefore degenerated to linear min-max exactly in the regime where the
    outlier problem occurs.

    Zeros (and anything below the smallest positive value) are clamped to
    the smallest positive value before taking the log.

    Args:
        values: Raw importance scores (non-negative).

    Returns:
        Normalized scores in the same order.
    """
    positive = [v for v in values if v > 0]
    if not positive:
        return [1.0] * len(values)
    floor = min(positive)
    log_vals = [math.log(max(v, floor)) for v in values]
    mn, mx = min(log_vals), max(log_vals)
    if mx == mn:
        return [1.0] * len(log_vals)
    return [(v - mn) / (mx - mn) for v in log_vals]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph_data(
    scores: CircuitScores,
    pruned_node_names: Optional[List[str]] = None,
) -> _GraphData:
    """
    Build a ``_GraphData`` from a ``CircuitScores`` artifact.

    The pruned_node_names list tells us which nodes were *removed* by the
    discovery algorithm. Everything in scores.node_scores that is NOT in
    pruned_node_names is considered "in circuit".

    Edge construction:
        For every pair (src, dst) where both are in-circuit nodes and
        dst is in the immediately adjacent layer (layer(dst) == layer(src)+1),
        create an edge with weight = raw_score(src) * raw_score(dst).
        Weights are then log1p-normalized across all edges to [0, 1].

    Args:
        scores:             CircuitScores with all node_scores.
        pruned_node_names:  List of node names that were pruned (removed).
                            Defaults to empty (all nodes considered in circuit).

    Returns:
        A fully populated _GraphData instance.
    """
    pruned_set: Set[str] = set(pruned_node_names or [])
    raw_scores: Dict[str, float] = dict(scores.node_scores)

    # --- Rank by raw score descending ---
    ranked = sorted(raw_scores.keys(), key=lambda k: raw_scores[k], reverse=True)
    rank_map = {name: i + 1 for i, name in enumerate(ranked)}

    # --- Log-normalize raw scores for visual sizing ---
    all_names = list(raw_scores.keys())
    log_norms = _log_normalize([raw_scores[n] for n in all_names])
    log_norm_map = {n: v for n, v in zip(all_names, log_norms)}

    # --- Build NodeData list ---
    node_list: List[_NodeData] = []
    for name in all_names:
        ntype, layer, head = _parse_node_name(name)
        node_list.append(_NodeData(
            name=name,
            node_type=ntype,
            layer=layer,
            head=head,
            raw_score=raw_scores[name],
            log_norm_score=log_norm_map[name],
            in_circuit=(name not in pruned_set),
            rank=rank_map[name],
            x=0.0,  # filled by layout
            y=0.0,
        ))

    # --- Assign layout positions ---
    node_list = _compute_layout(node_list)

    # --- Build EdgeData list (circuit nodes only, adjacent layers) ---
    circuit_nodes = {nd.name: nd for nd in node_list if nd.in_circuit}
    raw_edge_weights: List[float] = []
    proto_edges: List[Tuple[str, str, float]] = []

    for src_name, src_nd in circuit_nodes.items():
        for dst_name, dst_nd in circuit_nodes.items():
            if dst_nd.layer == src_nd.layer + 1:
                raw_w = raw_scores[src_name] * raw_scores[dst_name]
                proto_edges.append((src_name, dst_name, raw_w))
                raw_edge_weights.append(raw_w)

    # Log-normalize edge weights independently of node scores
    if raw_edge_weights:
        log_edge_norms = _log_normalize(raw_edge_weights)
    else:
        log_edge_norms = []

    edge_list: List[_EdgeData] = [
        _EdgeData(
            src=src,
            dst=dst,
            normalized_weight=log_edge_norms[i],
            raw_weight=raw_w,
        )
        for i, (src, dst, raw_w) in enumerate(proto_edges)
    ]

    n_layers = max((nd.layer for nd in node_list), default=0) + 1

    circuit_ids = {nd.name for nd in node_list if nd.in_circuit}

    return _GraphData(
        nodes=node_list,
        edges=edge_list,
        n_layers=n_layers,
        circuit_node_ids=circuit_ids,
        metadata={
            "task": scores.task,
            "model": scores.model,
            "algorithm": scores.algorithm,
            "timestamp": scores.timestamp,
            "n_total_nodes": len(node_list),
            "n_circuit_nodes": len(circuit_ids),
            "n_pruned_nodes": len(pruned_set),
        },
    )


# ---------------------------------------------------------------------------
# Plotly renderer (for Jupyter notebooks)
# ---------------------------------------------------------------------------

def _make_plotly_figure(
    graph: _GraphData,
    threshold: float = 0.0,
    title: Optional[str] = None,
    width: int = 1200,
    height: int = 700,
    node_size_scale: float = 18.0,
) -> go.Figure:
    """
    Build a Plotly ``go.Figure`` from _GraphData at the given threshold.

    This is called on initial render and on every slider update.

    Args:
        graph:           Prepared _GraphData.
        threshold:       Minimum normalized edge weight to display [0, 1].
        title:           Figure title.
        width:           Canvas width in pixels.
        height:          Canvas height in pixels.
        node_size_scale: Maximum node marker size (log_norm=1.0 → this size).

    Returns:
        A ``go.Figure`` with edge + node traces.
    """
    # --- Edge traces (one per visible edge) ---
    #
    # Plotly cannot vary color/width per segment within a single Scatter trace,
    # so we emit individual traces. Edges below threshold are simply skipped.
    # Alpha is also encoded in the RGBA color based on normalized weight.
    traces: List[go.BaseTraceType] = []

    visible_edges = [e for e in graph.edges if e.normalized_weight >= threshold]

    for edge in visible_edges:
        src_nd = next(n for n in graph.nodes if n.name == edge.src)
        dst_nd = next(n for n in graph.nodes if n.name == edge.dst)

        # Interpolate hue between edge_low and edge_high based on weight,
        # with alpha keeping every edge at least faintly visible.
        alpha = 0.35 + edge.normalized_weight * 0.65  # 0.35 … 1.0
        hex_color = _lerp_hex(PALETTE.edge_low, PALETTE.edge_high, edge.normalized_weight)
        r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
        color = f"rgba({r}, {g}, {b}, {alpha:.2f})"

        traces.append(go.Scatter(
            x=[src_nd.x, dst_nd.x],
            y=[src_nd.y, dst_nd.y],
            mode="lines",
            line=dict(width=0.8 + edge.normalized_weight * 3.0, color=color),
            hovertemplate=(
                f"<b>{edge.src} → {edge.dst}</b><br>"
                f"Edge strength: {edge.normalized_weight:.3f}<br>"
                f"Raw weight: {edge.raw_weight:.6f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # --- Build node index for O(1) lookup ---
    node_map = {nd.name: nd for nd in graph.nodes}

    # --- Separate in-circuit vs background nodes ---
    circuit_nodes = [nd for nd in graph.nodes if nd.in_circuit]
    background_nodes = [nd for nd in graph.nodes if not nd.in_circuit]

    def _node_hover(nd: _NodeData) -> str:
        head_str = f", head {nd.head}" if nd.head is not None else ""
        circuit_str = "✓ In circuit" if nd.in_circuit else "✗ Pruned"
        return (
            f"<b>{nd.name}</b><br>"
            f"Type: {nd.node_type}{head_str}<br>"
            f"Layer: {nd.layer}<br>"
            f"Score: {nd.raw_score:.6f}<br>"
            f"Rank: #{nd.rank} of {len(graph.nodes)}<br>"
            f"{circuit_str}"
            "<extra></extra>"
        )

    # --- Background (pruned) nodes ---
    if background_nodes:
        traces.append(go.Scatter(
            x=[nd.x for nd in background_nodes],
            y=[nd.y for nd in background_nodes],
            mode="markers",
            marker=dict(
                size=6,
                color=PALETTE.background_node,
                line=dict(color=PALETTE.background_node_stroke, width=1),
                opacity=0.45,
            ),
            hovertemplate=[_node_hover(nd) for nd in background_nodes],
            showlegend=False,
            name="pruned",
        ))

    # --- In-circuit nodes (colored by type) ---
    if circuit_nodes:
        sizes = [
            max(9.0, node_size_scale * nd.log_norm_score + 6.0)
            for nd in circuit_nodes
        ]
        colors = [get_node_color(nd.node_type) for nd in circuit_nodes]
        opacity = [0.55 + nd.log_norm_score * 0.45 for nd in circuit_nodes]

        traces.append(go.Scatter(
            x=[nd.x for nd in circuit_nodes],
            y=[nd.y for nd in circuit_nodes],
            mode="markers+text",
            text=[nd.name for nd in circuit_nodes],
            textposition="top center",
            textfont=dict(
                family=FONT_MONO,
                size=9,
                color=PALETTE.text_primary,
            ),
            marker=dict(
                size=sizes,
                color=colors,
                opacity=opacity,
                line=dict(color=PALETTE.circuit_stroke, width=1.5),
            ),
            hovertemplate=[_node_hover(nd) for nd in circuit_nodes],
            showlegend=False,
            name="circuit",
        ))

    # --- Layer annotation ticks along the bottom ---
    annotations = []
    n_layers = graph.n_layers
    # Place labels below the lowest node regardless of head count per layer
    # (Plotly's y-axis points up, so "below" is the minimum y).
    label_y = min((nd.y for nd in graph.nodes), default=0.0) - 1.5 * 0.72
    for layer_idx in range(n_layers):
        x_coord = layer_idx * 1.8  # must match x_spacing in _compute_layout
        annotations.append(dict(
            x=x_coord,
            y=label_y,
            text=f"L{layer_idx}",
            showarrow=False,
            font=dict(family=FONT_FAMILY, size=10, color=PALETTE.text_secondary),
            xanchor="center",
        ))

    if title is None:
        m = graph.metadata
        title = (
            f"{m.get('task','').upper()} Circuit — {m.get('model','')} "
            f"/ {m.get('algorithm','')}"
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        **get_plotly_layout(
            title=title,
            width=width,
            height=height,
        ),
        annotations=annotations,
    )
    return fig


# ---------------------------------------------------------------------------
# Public visualizer class
# ---------------------------------------------------------------------------

class CircuitGraphVisualizer:
    """
    Interactive circuit graph visualizer.

    Renders the full model graph with the discovered subcircuit highlighted.
    Supports two output modes:

    * ``interactive_widget()`` — Plotly ``FigureWidget`` + ipywidgets threshold
      slider for use in Jupyter notebooks.
    * ``to_html()`` — self-contained D3.js HTML file with zoom/pan, hover
      tooltips, and a real-time threshold slider.

    Example::

        viz = CircuitGraphVisualizer(scores, pruned_nodes=circuit.nodes)
        viz.to_html("circuit.html", title="IOI Circuit")   # D3 export
        fig = viz.interactive_widget()                     # Jupyter widget
    """

    def __init__(
        self,
        graph: Dict[str, Any],
        scores: CircuitScores,
        edge_scores: Optional[Dict[Tuple[str, str], float]] = None,
        pruned_nodes: Optional[List[str]] = None,
    ) -> None:
        """
        Initialise the visualizer.

        Args:
            graph:        Circuit graph dict (``"nodes"``, ``"edges"`` keys).
                          The ``"edges"`` value is intentionally ignored —
                          edges are derived from the scored node structure so
                          that the visualizer works correctly even when
                          ``circuit.plot()`` passes an empty edge list.
            scores:       ``CircuitScores`` artifact (all 156 scored nodes).
            edge_scores:  Optional pre-computed per-edge attribution values.
                          Currently reserved for future per-edge EAP scores.
            pruned_nodes: List of node names that were *removed* by the
                          discovery algorithm. Everything else is "in circuit".
                          Defaults to an empty list (all nodes shown as
                          in-circuit).
        """
        self.scores = scores
        self.node_scores = scores.normalize_scores(method="minmax")
        self.edge_scores = edge_scores or {}
        self.pruned_nodes: List[str] = list(pruned_nodes or [])

        # The graph dict is still accepted for API compatibility, but we
        # ignore its edges and derive our own from the score data.
        self.graph = graph
        self.node_types = {
            name: _parse_node_name(name)[0]
            for name in scores.node_scores.keys()
        }

        # Build the internal graph representation once.
        self._graph_data: _GraphData = _build_graph_data(scores, self.pruned_nodes)

    # ------------------------------------------------------------------
    # Parsed-graph accessors
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Dict[str, Dict[str, Any]]:
        """Parsed nodes keyed by name: {name: {type, layer, head, score, in_circuit}}."""
        return {
            nd.name: {
                "type": nd.node_type,
                "layer": nd.layer,
                "head": nd.head,
                "score": nd.raw_score,
                "in_circuit": nd.in_circuit,
            }
            for nd in self._graph_data.nodes
        }

    @property
    def edges(self) -> List[Dict[str, Any]]:
        """Derived edges as {src, dst, weight} dicts."""
        return [
            {"src": e.src, "dst": e.dst, "weight": e.normalized_weight}
            for e in self._graph_data.edges
        ]

    # ------------------------------------------------------------------
    # Public output methods
    # ------------------------------------------------------------------

    def to_html(
        self,
        output_path: str,
        title: Optional[str] = None,
        default_threshold: float = 0.0,
        **_kw: Any,
    ) -> str:
        """
        Export the circuit graph as a self-contained D3.js HTML file.

        The exported file requires no external dependencies (D3.js is
        bundled via CDN) and works in any modern browser.

        Features:
        - Scroll-to-zoom, drag-to-pan (mouse + trackpad).
        - Hover tooltips on nodes and edges.
        - Real-time threshold slider for edge filtering.
        - Toggle to show/hide pruned (background) nodes.
        - Color legend with node-type key.

        Args:
            output_path:        File path for the HTML file.
            title:              Figure title. Defaults to
                                ``"<TASK> Circuit — <model> / <algo>"``.
            default_threshold:  Initial edge threshold [0, 1].

        Returns:
            The full HTML string (also written to ``output_path``).
        """
        from .d3_template import render_d3_circuit_html

        if title is None:
            m = self._graph_data.metadata
            title = (
                f"{m.get('task','').upper()} Circuit — "
                f"{m.get('model','')} / {m.get('algorithm','')}"
            )

        export_data = self._build_export_data()
        html = render_d3_circuit_html(
            graph_data=export_data,
            theme=get_d3_theme(),
            title=title,
            default_threshold=default_threshold,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return html

    def interactive_widget(
        self,
        width: int = 1200,
        height: int = 700,
        node_size_scale: float = 18.0,
    ) -> go.FigureWidget:
        """
        Return a Plotly ``FigureWidget`` with a threshold slider for Jupyter.

        The slider filters edges by their log-normalized weight so the user
        can progressively reveal or hide lower-importance connections.

        Args:
            width:           Canvas width in pixels.
            height:          Canvas height in pixels.
            node_size_scale: Maximum marker size for the highest-scored node.

        Returns:
            A ``go.FigureWidget`` (also linked to an ipywidgets slider that
            is displayed when you call this inside a notebook).
        """
        try:
            from ipywidgets import FloatSlider, HBox, VBox, HTML as HTML_widget, interact
            from IPython.display import display
        except ImportError:
            raise ImportError(
                "ipywidgets and IPython are required for interactive_widget(). "
                "Install with: pip install ipywidgets"
            )

        m = self._graph_data.metadata
        default_title = (
            f"{m.get('task','').upper()} Circuit — "
            f"{m.get('model','')} / {m.get('algorithm','')}"
        )

        initial_fig = _make_plotly_figure(
            self._graph_data,
            threshold=0.0,
            title=default_title,
            width=width,
            height=height,
            node_size_scale=node_size_scale,
        )
        fig_widget = go.FigureWidget(initial_fig)

        def _update(threshold: float) -> None:
            new_fig = _make_plotly_figure(
                self._graph_data,
                threshold=threshold,
                title=f"{default_title} (edge threshold ≥ {threshold:.2f})",
                width=width,
                height=height,
                node_size_scale=node_size_scale,
            )
            with fig_widget.batch_update():
                fig_widget.data = []
                for trace in new_fig.data:
                    fig_widget.add_trace(trace)
                fig_widget.layout.title.text = new_fig.layout.title.text

        slider = FloatSlider(
            min=0.0,
            max=1.0,
            step=0.01,
            value=0.0,
            description="Edge threshold:",
            style={"description_width": "initial"},
            layout={"width": "500px"},
        )

        meta = self._graph_data.metadata
        info_html = HTML_widget(
            f"<span style='font-family: {FONT_FAMILY}; font-size: 12px; "
            f"color: {PALETTE.text_secondary};'>"
            f"Nodes: {meta['n_circuit_nodes']} in circuit, "
            f"{meta['n_pruned_nodes']} pruned | "
            f"Edges: {len(self._graph_data.edges)} total | "
            f"Drag slider to filter low-importance edges"
            f"</span>"
        )

        interact(_update, threshold=slider)
        return fig_widget

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _build_export_data(self) -> Dict[str, Any]:
        """
        Build the JSON-serializable graph payload for the D3.js renderer.

        Returns:
            Dict with keys ``metadata``, ``nodes``, ``edges``,
            ``n_layers``, ``circuit_node_ids``.
        """
        gd = self._graph_data
        nodes_payload = [
            {
                "id": nd.name,
                "type": nd.node_type,
                "layer": nd.layer,
                "head": nd.head,
                "raw_score": nd.raw_score,
                "log_norm_score": nd.log_norm_score,
                "in_circuit": nd.in_circuit,
                "rank": nd.rank,
                "x": nd.x,
                "y": nd.y,
            }
            for nd in gd.nodes
        ]
        edges_payload = [
            {
                "source": e.src,
                "target": e.dst,
                "normalized_weight": e.normalized_weight,
                "raw_weight": e.raw_weight,
            }
            for e in gd.edges
        ]
        return {
            "metadata": gd.metadata,
            "nodes": nodes_payload,
            "edges": edges_payload,
            "n_layers": gd.n_layers,
            "circuit_node_ids": sorted(gd.circuit_node_ids),
        }

    def export_graph_data(self, output_path: str) -> None:
        """
        Export the graph payload as JSON (for external tooling).

        Args:
            output_path: Path to write JSON file.
        """
        data = self._build_export_data()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # Legacy compatibility helpers (retained from original API)
    # ------------------------------------------------------------------

    def layout_hierarchy(self) -> Dict[str, Tuple[float, float]]:
        """
        Return a ``{node_name: (x, y)}`` dict for the current layout.

        Kept for backward compatibility with code that reads positions
        directly from the visualizer.
        """
        return {nd.name: (nd.x, nd.y) for nd in self._graph_data.nodes}

    def get_top_k_nodes(self, k: int = 10) -> Dict[str, float]:
        """
        Return the top-k highest-scoring nodes as ``{name: normalized_score}``.
        """
        ranked = sorted(
            self.node_scores.items(), key=lambda x: x[1], reverse=True
        )
        return dict(ranked[:k])

    def get_node_degree_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Return per-node ``{in_degree, out_degree}`` from the circuit edge set.
        """
        degrees: Dict[str, Dict[str, int]] = {
            nd.name: {"in_degree": 0, "out_degree": 0}
            for nd in self._graph_data.nodes
        }
        for edge in self._graph_data.edges:
            if edge.src in degrees:
                degrees[edge.src]["out_degree"] += 1
            if edge.dst in degrees:
                degrees[edge.dst]["in_degree"] += 1
        return degrees

    def filter_by_threshold(
        self,
        threshold: float,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> "CircuitGraphVisualizer":
        """
        Return a new visualizer with nodes below the score threshold removed.

        Args:
            threshold:    Minimum normalized node score [0, 1].
            output_path:  If given, also write the filtered graph to HTML.
            title:        Optional title override.

        Returns:
            New ``CircuitGraphVisualizer`` instance.
        """
        filtered_scores_dict = {
            name: score
            for name, score in self.node_scores.items()
            if score >= threshold
        }
        filtered_cs = CircuitScores(
            task=self.scores.task,
            model=self.scores.model,
            algorithm=self.scores.algorithm,
            level=self.scores.level,
            node_scores=filtered_scores_dict,
            timestamp=self.scores.timestamp,
            version=self.scores.version,
            discovery_cfg=self.scores.discovery_cfg,
        )
        filtered_graph = {"nodes": {k: {} for k in filtered_scores_dict}, "edges": []}
        filtered_pruned = [
            n for n in self.pruned_nodes if n in filtered_scores_dict
        ]
        viz = CircuitGraphVisualizer(
            filtered_graph, filtered_cs, self.edge_scores, filtered_pruned
        )
        if output_path:
            viz.to_html(output_path, title=title)
        return viz

    @staticmethod
    def _get_node_color(node_type: str) -> str:
        """Thin wrapper kept for backward compatibility."""
        return get_node_color(node_type)
