from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import plotly.graph_objects as go
import torch as t
from ordered_set import OrderedSet

from .types import COLOR_PALETTE, Edge, Node, PruneScores
from .utils.misc import repo_path_to_abs_path
from .utils.patchable_model import PatchableModel


def node_name(n: str, unembed: bool = False) -> str:
    if n == "Resid End":
        return "Unembed" if unembed else ""
    elif n == "Resid Start":
        return "Embed"
    else:
        return n[:-2] if n[-1] in ["Q", "K", "V"] else n


def t_fmt(x: Any, seq_dim: int, seq_idx: Optional[int], line_break: str = "\n") -> str:
    if type(x) is not t.Tensor:
        return str(x)
    seq_sl = (-1 if x.size(-1) < 5 else slice(None)) if seq_idx is None else seq_idx
    idx = tuple([0] + [slice(None)] * (seq_dim - 1) + [seq_sl])
    if (arr := x[idx].squeeze()).ndim == 1:
        return f"[{', '.join([f'{v:.2f}'.rstrip('0') for v in arr.tolist()[:2]])} ...]"
    return str(x[idx]).lstrip("tensor(").rstrip(")").replace("\n", line_break)


def net_viz(
    model: PatchableModel,
    seq_edges: Set[Edge],
    prune_scores: Optional[PruneScores],
    vert_interval: Tuple[float, float],
    seq_idx: Optional[int] = None,
    score_threshold: float = 1e-2,
    layer_spacing: bool = False,
    orientation: Literal["h", "v"] = "h",
) -> Tuple[go.Sankey, int]:
    """
    Draw the sankey diagram for a single token position.

    Args:
        model: The model to visualize.
        seq_edges: The edges to visualize for a single token position.
        prune_scores: The edge scores to use for the visualization.
        vert_interval: The vertical interval to place the diagram in the figure.
        seq_idx: The token position being visualized.
        score_threshold: Edges with scores below this are not shown.
        layer_spacing: If True, nodes are spaced according to their layer.
        orientation: The orientation of the sankey diagram ("h" or "v").

    Returns:
        The sankey diagram for the given token position.
    """
    nodes: OrderedSet[Node] = OrderedSet(model.nodes)
    un = False if orientation == "h" else True

    # Define the sankey nodes
    viz_nodes: Dict[str, Node] = dict([(node_name(n.name, un), n) for n in nodes])
    node_idxs: Dict[str, int] = dict([(n, i) for i, n in enumerate(viz_nodes.keys())])
    lyr_nodes: Dict[int, List[str]] = defaultdict(list)
    for n in viz_nodes.values():
        lyr_nodes[n.layer].append(n.name)
    graph_nodes = {
        "label": ["" for _, _ in viz_nodes.items()],
        "color": ["rgba(0,0,0,0.0)" for _, _ in viz_nodes.items()],
        "line": dict(width=0.0),
    }

    # Define the sankey edges
    sources, targets, values, colors = [], [], [], []
    included_layer_nodes: Dict[int, List[str]] = defaultdict(list)
    for e in seq_edges:
        if prune_scores is None:
            no_edge_score_error = "Visualization requires patch mode or PruneScores."
            assert e.dest.module(model).curr_src_outs is not None, no_edge_score_error
            edge_score = e.patch_mask(model).data[e.patch_idx].item()
        else:
            edge_score = prune_scores[e.dest.module_name][e.patch_idx].item()

        if abs(edge_score) < score_threshold:
            continue

        color_idx = len(sources) % len(COLOR_PALETTE)
        sources.append(node_idxs[node_name(e.src.name, un)])
        graph_nodes["label"][node_idxs[node_name(e.src.name, un)]] = node_name(e.src.name, un)
        graph_nodes["color"][node_idxs[node_name(e.src.name, un)]] = COLOR_PALETTE[color_idx]
        targets.append(node_idxs[node_name(e.dest.name, un)])
        graph_nodes["label"][node_idxs[node_name(e.dest.name, un)]] = node_name(e.dest.name, un)
        graph_nodes["color"][node_idxs[node_name(e.dest.name, un)]] = COLOR_PALETTE[color_idx]
        values.append(0.8 if prune_scores is None else abs(edge_score))
        if edge_score > 0:
            edge_color = "rgba(0,0,255,0.3)"
        else:
            edge_color = "rgba(255,0,0,0.3)"
        colors.append(edge_color)
        included_layer_nodes[e.src.layer].append(e.src.name)
        included_layer_nodes[e.dest.layer].append(e.dest.name)

    included_layer_count = len(included_layer_nodes)
    if layer_spacing:
        # Add ghost edges to horizontally align nodes to the correct layer
        for i in lyr_nodes.keys():
            if i not in included_layer_nodes:
                included_layer_nodes[i] = [lyr_nodes[i][0]]

        ordered_lyr_nodes = [n for _, n in sorted(included_layer_nodes.items())]

        ghost_edge_val = 1e-6
        for l1_n, l2_n in zip(ordered_lyr_nodes[:-1], ordered_lyr_nodes[1:]):
            for l1 in l1_n:
                sources.append(node_idxs[node_name(l1, un)])
                targets.append(node_idxs[node_name(l2_n[0], un)])
                values.append(ghost_edge_val)
                colors.append("rgba(0,255,0,0.0)")
            for l2 in l2_n:
                sources.append(node_idxs[node_name(l1_n[0], un)])
                targets.append(node_idxs[node_name(l2, un)])
                values.append(ghost_edge_val)
                colors.append("rgba(0,255,0,0.0)")

    return (
        go.Sankey(
            arrangement="perpendicular",
            node=graph_nodes,
            link={
                "arrowlen": 25,
                "source": sources,
                "target": targets,
                "value": values,
                "color": colors,
            },
            orientation=orientation,
            domain={"y": vert_interval},
        ),
        included_layer_count,
    )


def draw_seq_graph(
    model: PatchableModel,
    prune_scores: Optional[PruneScores] = None,
    score_threshold: float = 1e-2,
    show_all_seq_pos: bool = False,
    seq_labels: Optional[List[str]] = None,
    layer_spacing: bool = False,
    orientation: Literal["h", "v"] = "h",
    display_ipython: bool = True,
    file_path: Optional[str] = None,
) -> go.Figure:
    """
    Draw the sankey for all token positions in a `PatchableModel`.

    Args:
        model: The model to visualize.
        prune_scores: Edge scores to visualize. If `None`, current masks are used.
        score_threshold: Minimum absolute edge score to show.
        show_all_seq_pos: If `True`, show all token positions.
        seq_labels: Labels for each token position.
        layer_spacing: If `True`, nodes are spaced by layer.
        orientation: Orientation of the sankey diagram ("h" or "v").
        display_ipython: If `True`, display in the current ipython environment.
        file_path: If provided, save the diagram to this file path.
    """
    seq_len = model.seq_len or 1

    # Calculate the vertical interval for each sub-diagram
    edge_scores = (
        prune_scores.values()
        if prune_scores is not None
        else model.current_patch_masks_as_prune_scores().values()
    )
    total_ps = max(
        sum([t.clamp(v.abs() - score_threshold, min=0).sum() for v in edge_scores]), 1e-2
    )

    intervals = {}
    if seq_len > 1:
        sankey_heights: Dict[Optional[int], float] = defaultdict(float)
        for patch_mask in edge_scores:
            ps_seq_tots = t.clamp(patch_mask.abs() - score_threshold, min=0.0)
            ps_seq_tots = ps_seq_tots.sum(dim=list(range(1, patch_mask.ndim)))
            for seq_idx, ps_seq_tot in enumerate(ps_seq_tots):
                if ps_seq_tot > 0 or show_all_seq_pos:
                    sankey_heights[seq_idx] += ps_seq_tot.item()

        for seq_idx in model.edge_dict.keys():
            if show_all_seq_pos:
                min_h = total_ps / (len(model.edge_dict) * 2)
                sankey_heights[seq_idx] = max(sankey_heights[seq_idx], min_h)

        n_figs = len(sankey_heights)
        margin_h: float = total_ps / (n_figs * 2) if n_figs > 1 else 0
        total_h = sum(sankey_heights.values()) + margin_h * (n_figs - 1)

        interval_start = total_h
        for seq_idx, height in sorted(sankey_heights.items(), key=lambda x: x[0] or -1):
            interval_end = interval_start - (margin_h if len(intervals) > 0 else 0)
            interval_start = interval_end - height
            intervals[seq_idx] = (
                max(interval_start / total_h, 1e-6),
                min(interval_end / total_h, 1 - 1e-6),
            )
    else:
        intervals = {list(model.edge_dict.keys())[0]: (0, 1)}

    # Draw the sankey for each token position
    sankeys, n_layers = [], 0
    for seq_idx, vert_interval in intervals.items():
        edge_set = set(model.edge_dict[seq_idx])
        viz, n_layers = net_viz(
            model=model,
            seq_edges=edge_set,
            prune_scores=prune_scores,
            vert_interval=vert_interval,
            seq_idx=seq_idx,
            score_threshold=score_threshold,
            layer_spacing=layer_spacing,
            orientation=orientation,
        )
        sankeys.append(viz)

    if orientation == "h":
        h = max(250 * len(sankeys), 400)
        w = max(50 * n_layers, 600)
    else:
        h = max(50 * n_layers, 600)
        w = max(700 * len(sankeys), 800)
    layout = go.Layout(height=h, width=w, plot_bgcolor="white")
    fig = go.Figure(data=sankeys, layout=layout)

    if seq_labels:
        for fig_idx, seq_idx in enumerate(intervals.keys()):
            seq_label = "All tokens" if seq_idx is None else seq_labels[seq_idx]
            y_range: Tuple[float, float] = fig.data[fig_idx].domain["y"]  # type: ignore
            fig.add_annotation(
                x=-0.17,
                y=(y_range[0] + y_range[1]) / 2,
                text=f"<b>{seq_label}</b>",
                showarrow=False,
                xref="paper",
                yref="paper",
            )
    if display_ipython:
        fig.show()
    if file_path:
        absolute_path: Path = repo_path_to_abs_path(file_path)
        fig.write_image(str(absolute_path))
    return fig
