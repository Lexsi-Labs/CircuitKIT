"""Tests for the D3 template's level-of-detail (LOD) render-mode selection
(PR-L1): canvas above NODE_SVG_LIMIT/EDGE_SVG_LIMIT, SVG below. Also the
opening threshold for very large graphs (CK-25): no size refusal, only the
strongest INITIAL_VISIBLE_EDGES edges drawn at first."""

import json
import re

import pytest

from circuitkit.visualize.d3_template import (
    EDGE_SVG_LIMIT,
    INITIAL_VISIBLE_EDGES,
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


def _embedded(html: str) -> tuple:
    """Return the embedded graph payload and the page's opening threshold."""
    graph = json.loads(re.search(r"const GRAPH\s*= (.*);$", html, re.M).group(1))
    threshold = float(re.search(r"const INITIAL_THRESHOLD = ([0-9.]+);", html).group(1))
    return graph, threshold


def test_huge_graph_opens_with_the_strongest_edges_and_no_size_refusal():
    n_edges = INITIAL_VISIBLE_EDGES + 5000
    data = _graph_data(n_nodes=50, n_edges=n_edges)
    for i, edge in enumerate(data["edges"]):
        edge["normalized_weight"] = (i % 997) / 996  # spread over [0, 1], many ties
    html = render_d3_circuit_html(data, get_d3_theme())

    assert "too large" not in html
    assert "MAX_RENDERABLE" not in html
    graph, t = _embedded(html)
    assert len(graph["edges"]) == n_edges  # nothing is dropped from the page

    weights = [e["normalized_weight"] for e in graph["edges"]]
    assert round(t * 100) == pytest.approx(t * 100)  # a multiple of the 0.01 step
    assert sum(w >= t for w in weights) <= INITIAL_VISIBLE_EDGES
    # Minimal: one step lower would draw more than the budget.
    assert sum(w >= t - 0.01 for w in weights) > INITIAL_VISIBLE_EDGES


@pytest.mark.parametrize(
    "n_edges, requested, expected",
    [
        (6, None, 0.0),  # small graph: everything is drawn
        (6, 0.37, 0.37),  # an explicit value is used as given
        (INITIAL_VISIBLE_EDGES + 1, 0.0, 0.0),  # ... even on a huge graph
    ],
)
def test_opening_threshold_is_automatic_only_when_not_given(n_edges, requested, expected):
    html = render_d3_circuit_html(
        _graph_data(n_nodes=9, n_edges=n_edges), get_d3_theme(), default_threshold=requested
    )
    assert _embedded(html)[1] == expected
    assert f'value="{expected}"' in html  # the slider starts at the same value
    assert "const AUTO_FILTERED = false;" in html  # so no "edges hidden" note
