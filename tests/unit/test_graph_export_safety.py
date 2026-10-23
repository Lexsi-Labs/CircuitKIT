"""Tests for the circuit graph export hardening (CK-25): edge labelling, pinned and
integrity-checked scripts, and safe embedding of the payload and tooltips in the HTML.

One test per contract, to keep the fast tier small.

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
    def test_json_exports_carry_edge_kind(self, tmp_path):
        # Bounded to_json and unbounded export_graph_data take different paths
        # through _build_graph_data; both must label the edges.
        viz = _viz(_scores())
        bounded = viz.to_json(str(tmp_path / "g.json"), max_nodes=2, max_edges=1)
        viz.export_graph_data(str(tmp_path / "export.json"))
        full = json.loads((tmp_path / "export.json").read_text(encoding="utf-8"))
        assert bounded["metadata"]["edge_kind"] == EDGE_KIND_LAYER_ADJACENT == "layer_adjacent"
        assert full["metadata"]["edge_kind"] == EDGE_KIND_LAYER_ADJACENT

    def test_html_embeds_edge_kind_and_explains_it(self, tmp_path):
        html = _viz(_scores()).to_html(str(tmp_path / "g.html"))
        assert _embedded_graph(html)["metadata"]["edge_kind"] == EDGE_KIND_LAYER_ADJACENT
        assert "not measured connections" in html


class TestPinnedScripts:
    def test_both_scripts_are_pinned_and_integrity_checked(self):
        html = d3_template.render_d3_circuit_html(_viz(_scores())._build_export_data(), {})
        tags = re.findall(r"<script\b[^>]*\bsrc=[^>]*>", html)
        seen = {}
        for tag in tags:
            assert 'crossorigin="anonymous"' in tag
            seen[re.search(r'src="([^"]+)"', tag).group(1)] = re.search(
                r'integrity="([^"]+)"', tag
            ).group(1)
        assert seen == {D3_SCRIPT_URL: D3_SCRIPT_SRI, ELKJS_SCRIPT_URL: ELKJS_SCRIPT_SRI}

        assert re.search(r"/d3/\d+\.\d+\.\d+/", D3_SCRIPT_URL)
        # elkjs ships no .min.js; jsDelivr minifies on request, and those bytes
        # can change without a release, which would break the integrity check.
        assert re.search(r"/elkjs@\d+\.\d+\.\d+/lib/elk\.bundled\.js$", ELKJS_SCRIPT_URL)
        for sri in seen.values():
            algorithm, _, digest = sri.partition("-")
            assert algorithm == "sha384"
            assert len(base64.b64decode(digest, validate=True)) == 48


class TestSafeEmbedding:
    def test_json_for_script_escapes_and_round_trips(self):
        value = {"k": _HOSTILE, "nested": [{"a&b": "<>"}], "label": "—é中"}
        text = _json_for_script(value)
        for ch in ("<", ">", "&", _LS, _PS):
            assert ch not in text
        assert "中" in text  # other non-ASCII stays readable
        assert json.loads(text) == value

    def test_hostile_names_cannot_break_the_page(self, tmp_path):
        viz = _viz(_scores(task=_HOSTILE, model=_HOSTILE, extra=_HOSTILE), pruned=["A2.0"])
        html = viz.to_html(str(tmp_path / "g.html"))

        assert "<script>alert(1)" not in html
        # The default title upper-cases the task; it is HTML-escaped in <title> and <h1>.
        assert "&lt;/SCRIPT&gt;&lt;SCRIPT&gt;" in html
        # The first </script> after the data is the real end of the page's own
        # script, so the data cannot have closed it early.
        data_start = html.index("const GRAPH")
        script_body = html[data_start : html.index("</script>", data_start)]
        assert "end initGraph" in script_body
        assert _LS not in script_body and _PS not in script_body
        assert html.count("</script>") == 3  # two <script src> tags + the page script
        # The embedded data is the exported payload, unchanged.
        assert _embedded_graph(html) == json.loads(json.dumps(viz._build_export_data()))


class TestTooltipEscaping:
    def test_tooltips_escape_every_value_from_the_scores_file(self):
        html = d3_template.render_d3_circuit_html(_viz(_scores())._build_export_data(), {})
        assert "function esc(v)" in html
        # No node or edge field reaches a tooltip's innerHTML unescaped.
        raw = re.findall(r"\$\{(?:nearest|node|edge)\.(?:id|source|target|layer|head|rank)\}", html)
        assert raw == []
        for field in ("nearest.id", "node.id", "edge.source", "edge.target"):
            assert f"esc({field})" in html
