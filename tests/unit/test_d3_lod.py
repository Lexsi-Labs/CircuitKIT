"""Tests for the D3 template's level-of-detail (LOD) render-mode selection
(PR-L1): canvas above NODE_SVG_LIMIT/EDGE_SVG_LIMIT, SVG below."""

from circuitkit.visualize.d3_template import (
    EDGE_SVG_LIMIT,
    NODE_SVG_LIMIT,
    render_d3_circuit_html,
)
from circuitkit.visualize.theme import get_d3_theme


def _graph_data(n_nodes: int, n_edges: int) -> dict:
    nodes = [
        {
            "id": f"A0.{i}",
            "type": "attn_head",
            "layer": 0,
            "head": i,
            "raw_score": 1.0,
            "log_norm_score": 1.0,
            "in_circuit": True,
            "rank": i + 1,
            "x": float(i),
            "y": 0.0,
        }
        for i in range(n_nodes)
    ]
    edges = [
        {
            "source": f"A0.{i % max(n_nodes, 1)}",
            "target": f"A0.{(i + 1) % max(n_nodes, 1)}",
            "normalized_weight": 0.5,
            "raw_weight": 0.5,
        }
        for i in range(n_edges)
    ]
    return {
        "metadata": {
            "task": "ioi",
            "model": "gpt2",
            "algorithm": "eap",
            "n_circuit_nodes": n_nodes,
            "n_pruned_nodes": 0,
        },
        "nodes": nodes,
        "edges": edges,
        "n_layers": 1,
        "circuit_node_ids": [n["id"] for n in nodes],
    }


def test_lod_switch_small_graph_uses_svg():
    small = _graph_data(n_nodes=9, n_edges=6)
    html = render_d3_circuit_html(small, get_d3_theme())
    assert 'data-ck-render-mode="svg"' in html
    assert 'data-ck-render-mode="canvas"' not in html


def test_lod_switch_many_nodes_uses_canvas():
    large = _graph_data(n_nodes=NODE_SVG_LIMIT + 50, n_edges=10)
    html = render_d3_circuit_html(large, get_d3_theme())
    assert 'data-ck-render-mode="canvas"' in html


def test_lod_switch_many_edges_uses_canvas():
    large = _graph_data(n_nodes=10, n_edges=EDGE_SVG_LIMIT + 500)
    html = render_d3_circuit_html(large, get_d3_theme())
    assert 'data-ck-render-mode="canvas"' in html


def test_lod_switch_at_exact_limit_still_svg():
    at_limit = _graph_data(n_nodes=NODE_SVG_LIMIT, n_edges=EDGE_SVG_LIMIT)
    html = render_d3_circuit_html(at_limit, get_d3_theme())
    assert 'data-ck-render-mode="svg"' in html
