"""Tests for the circuit graph export hardening (CK-25): edge labelling, pinned and
integrity-checked scripts, and safe embedding of the payload in the HTML.

Everything here is offline: the script hashes are compared with the module
constants, never fetched.
"""

import base64
import json
import re

from circuitkit.artifacts.scores import CircuitScores
from circuitkit.visualize import d3_template
from circuitkit.visualize.d3_template import (
    D3_SCRIPT_SRI,
    D3_SCRIPT_URL,
    ELKJS_SCRIPT_SRI,
    ELKJS_SCRIPT_URL,
    _json_for_script,
)
from circuitkit.visualize.graph_viz import EDGE_KIND_LAYER_ADJACENT, CircuitGraphVisualizer

_LS = chr(0x2028)  # U+2028 LINE SEPARATOR
_PS = chr(0x2029)  # U+2029 PARAGRAPH SEPARATOR
_HOSTILE = f"</script><script>alert(1)</script>{_LS}{_PS}&<!--"


def _scores(task: str = "ioi", model: str = "gpt2", extra: str = "") -> CircuitScores:
    node_scores = {"A0.0": 0.9, "A0.1": 0.5, "MLP 1": 0.7, "A2.0": 0.3}
    if extra:
        node_scores[extra] = 0.1
    return CircuitScores(
        task=task,
        model=model,
        algorithm="eap",
        level="node",
        node_scores=node_scores,
        timestamp=CircuitScores.create_timestamp(),
    )


def _viz(scores: CircuitScores, pruned=None) -> CircuitGraphVisualizer:
    return CircuitGraphVisualizer({"nodes": {}, "edges": []}, scores, pruned_nodes=pruned)


def _embedded_graph(html: str) -> dict:
    """Parse the ``const GRAPH = ...;`` payload back out of a rendered page."""
    match = re.search(r"const GRAPH\s*=\s*(.*);\r?\nconst CK_THEME", html)
    assert match, "GRAPH constant not found in the page"
    return json.loads(match.group(1))


# ---------------------------------------------------------------------------
# Edge kind
# ---------------------------------------------------------------------------


class TestEdgeKind:
    def test_value_is_layer_adjacent(self):
        assert EDGE_KIND_LAYER_ADJACENT == "layer_adjacent"

    def test_to_json_carries_edge_kind(self, tmp_path):
        out = tmp_path / "graph.json"
        returned = _viz(_scores()).to_json(str(out))
        assert returned["metadata"]["edge_kind"] == "layer_adjacent"
        assert json.loads(out.read_text(encoding="utf-8"))["metadata"]["edge_kind"] == (
            "layer_adjacent"
        )

    def test_bounded_to_json_keeps_edge_kind(self, tmp_path):
        returned = _viz(_scores()).to_json(str(tmp_path / "g.json"), max_nodes=2, max_edges=1)
        assert returned["metadata"]["edge_kind"] == "layer_adjacent"

    def test_export_graph_data_carries_edge_kind(self, tmp_path):
        out = tmp_path / "export.json"
        _viz(_scores()).export_graph_data(str(out))
        assert json.loads(out.read_text(encoding="utf-8"))["metadata"]["edge_kind"] == (
            "layer_adjacent"
        )

    def test_html_embeds_edge_kind_and_explains_it(self, tmp_path):
        html = _viz(_scores()).to_html(str(tmp_path / "g.html"))
        assert _embedded_graph(html)["metadata"]["edge_kind"] == "layer_adjacent"
        assert "not measured connections" in html


# ---------------------------------------------------------------------------
# Pinned, integrity-checked scripts
# ---------------------------------------------------------------------------


def _script_tags(html: str) -> list:
    return re.findall(r"<script\b[^>]*\bsrc=[^>]*>", html)


class TestPinnedScripts:
    def test_both_script_tags_are_pinned_and_checked(self):
        html = d3_template.render_d3_circuit_html(_viz(_scores())._build_export_data(), {})
        tags = _script_tags(html)
        assert len(tags) == 2

        expected = {
            D3_SCRIPT_URL: D3_SCRIPT_SRI,
            ELKJS_SCRIPT_URL: ELKJS_SCRIPT_SRI,
        }
        seen = {}
        for tag in tags:
            src = re.search(r'src="([^"]+)"', tag).group(1)
            integrity = re.search(r'integrity="([^"]+)"', tag)
            assert integrity, f"no integrity attribute on {tag}"
            assert 'crossorigin="anonymous"' in tag
            seen[src] = integrity.group(1)
        assert seen == expected

    def test_urls_name_an_exact_version(self):
        assert re.search(r"/elkjs@\d+\.\d+\.\d+/", ELKJS_SCRIPT_URL)
        assert re.search(r"/d3/\d+\.\d+\.\d+/", D3_SCRIPT_URL)

    def test_elkjs_is_the_package_file_not_a_cdn_minified_one(self):
        # elkjs ships no .min.js; jsDelivr minifies on request, and those bytes
        # can change without a release, which would break the integrity check.
        assert ELKJS_SCRIPT_URL.endswith("/lib/elk.bundled.js")

    def test_integrity_values_are_sha384_digests(self):
        for sri in (D3_SCRIPT_SRI, ELKJS_SCRIPT_SRI):
            algorithm, _, digest = sri.partition("-")
            assert algorithm == "sha384"
            assert len(base64.b64decode(digest, validate=True)) == 48

    def test_docstring_no_longer_claims_no_external_dependencies(self):
        from circuitkit.visualize.graph_viz import CircuitGraphVisualizer as Viz

        assert "no external dependencies" not in (Viz.to_html.__doc__ or "").lower()
        assert "no additional dependencies" not in (d3_template.__doc__ or "").lower()


# ---------------------------------------------------------------------------
# Safe embedding
# ---------------------------------------------------------------------------


class TestSafeEmbedding:
    def test_json_for_script_escapes_and_round_trips(self):
        value = {"k": _HOSTILE, "nested": [{"a&b": "<>"}], "n": 1.5, "none": None}
        text = _json_for_script(value)
        for ch in ("<", ">", "&", _LS, _PS):
            assert ch not in text
        assert json.loads(text) == value

    def test_json_for_script_keeps_other_non_ascii(self):
        value = {"label": "Circuit " + chr(0x2014) + " " + chr(0xE9) + chr(0x4E2D)}
        text = _json_for_script(value)
        assert chr(0x2014) in text and chr(0x4E2D) in text
        assert json.loads(text) == value

    def test_hostile_names_cannot_break_out_of_the_data_script(self, tmp_path):
        scores = _scores(task=_HOSTILE, model=_HOSTILE, extra=_HOSTILE)
        viz = _viz(scores, pruned=["A2.0"])
        html = viz.to_html(str(tmp_path / "g.html"))

        # No name can form a script tag in the page.
        assert "<script>alert(1)" not in html
        assert "alert(1)</script>" not in html

        # The first </script> after the data constant is the real end of the
        # page's own script, so the data cannot have closed it early. The line
        # separators (which end a line in JavaScript source) are escaped there.
        data_start = html.index("const GRAPH")
        script_end = html.index("</script>", data_start)
        script_body = html[data_start:script_end]
        assert "end initGraph" in script_body
        assert _LS not in script_body and _PS not in script_body
        assert html.count("</script>") == 3  # two <script src> tags + the page script

        # The embedded data is the exported payload, unchanged.
        expected = json.loads(json.dumps(viz._build_export_data()))
        assert _embedded_graph(html) == expected
        assert _embedded_graph(html)["metadata"]["task"] == _HOSTILE

    def test_title_is_html_escaped(self, tmp_path):
        # The default title upper-cases the task, hence the capitals.
        viz = _viz(_scores(task="</title><h1>x</h1>"))
        html = viz.to_html(str(tmp_path / "g.html"))
        assert "</TITLE><H1>" not in html
        assert "&lt;/TITLE&gt;&lt;H1&gt;" in html


class TestTooltipEscaping:
    def test_tooltips_escape_every_value_from_the_scores_file(self):
        html = d3_template.render_d3_circuit_html(_viz(_scores())._build_export_data(), {})
        assert "function esc(v)" in html
        # No node or edge field reaches a tooltip's innerHTML unescaped.
        raw = re.findall(r"\$\{(?:nearest|node|edge)\.(?:id|source|target|layer|head|rank)\}", html)
        assert raw == []
        for field in ("nearest.id", "node.id", "edge.source", "edge.target"):
            assert f"esc({field})" in html
