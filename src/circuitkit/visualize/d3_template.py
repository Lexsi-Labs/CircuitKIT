"""
D3.js HTML template renderer for CircuitKit circuit graphs.

``render_d3_circuit_html`` takes the graph payload produced by
``CircuitGraphVisualizer._build_export_data()`` and returns a fully
self-contained HTML string that can be written to disk and opened in any
modern browser with no additional dependencies.

The D3.js library (v7) is loaded from the official CDN. Everything else
(layout coordinates, colors, theme values) is embedded directly as a JSON
constant so the file works offline once the CDN script has been cached.

Visual features
---------------
* Scroll-to-zoom, drag-to-pan (mouse + trackpad) with smooth transitions.
* Hover tooltip on nodes: name, type, layer, score, rank, circuit status.
* Hover tooltip on edges: source → target, normalized weight.
* On node hover: connected edges and neighbor nodes highlight; all others dim.
* Real-time threshold slider that fades out edges below the threshold using
  D3 transitions (no full redraw needed).
* Toggle checkbox to show/hide pruned (background) nodes.
* Color legend showing node types.
* Layer labels along the x-axis.
* Responsive: SVG fills the viewport; layout coordinates are in data space
  and D3 zoom maps them to screen space.
"""

from __future__ import annotations

import json
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_d3_circuit_html(
    graph_data: Dict[str, Any],
    theme: Dict[str, Any],
    title: str = "Circuit Graph",
    default_threshold: float = 0.0,
) -> str:
    """
    Build a self-contained D3.js HTML page from the circuit graph payload.

    Args:
        graph_data:         Output of ``CircuitGraphVisualizer._build_export_data()``.
        theme:              Output of ``get_d3_theme()`` — design system dict.
        title:              Page and figure title.
        default_threshold:  Initial edge threshold slider value [0, 1].

    Returns:
        Complete HTML string.
    """
    # Embed data as JSON constants; json.dumps handles escaping.
    graph_json = json.dumps(graph_data, ensure_ascii=False)
    theme_json = json.dumps(theme, ensure_ascii=False)

    return _HTML_TEMPLATE.format(
        title=title,
        graph_json=graph_json,
        theme_json=theme_json,
        default_threshold=default_threshold,
    )


# ---------------------------------------------------------------------------
# HTML / JS template
# ---------------------------------------------------------------------------

# Kept as a module-level constant so it is compiled once and can be inspected
# or overridden easily. Curly braces that are NOT f-string placeholders are
# escaped as {{ }}.

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/elkjs/lib/elk.bundled.min.js"></script>
<style>
/* ---- Reset & base ---- */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

/* ---- CSS variables (set by JS from theme) ---- */
:root {{
  --font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --mono: "JetBrains Mono", "Fira Code", monospace;
  --bg: #FFFFFF;
  --bg-paper: #F8F9FA;
  --text: #212529;
  --text-muted: #6C757D;
  --border: #E9ECEF;
  --accent: #1D4ED8;
}}

/* ---- Top bar ---- */
#topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 52px;
  background: var(--bg-paper);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
  gap: 16px;
  flex-wrap: wrap;
}}

#topbar h1 {{
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 380px;
}}

/* ---- Controls ---- */
.controls {{
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}}

.control-group {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
}}

#threshold-slider {{
  width: 160px;
  accent-color: var(--accent);
  cursor: pointer;
}}

#threshold-value {{
  min-width: 32px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text);
}}

#toggle-background {{
  accent-color: var(--accent);
  cursor: pointer;
}}

.stats-pill {{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 3px 10px;
  font-size: 11px;
  color: var(--text-muted);
  white-space: nowrap;
}}

/* ---- SVG canvas ---- */
#graph-container {{
  flex: 1;
  position: relative;
  overflow: hidden;
}}

#graph-svg {{
  width: 100%;
  height: 100%;
  cursor: grab;
}}

#graph-svg:active {{ cursor: grabbing; }}

/* Edge and node styles */
.edge {{
  stroke-linecap: round;
  transition: opacity 0.25s ease;
  /* Hover is handled by the invisible wide .edge-hit lines drawn on top. */
  pointer-events: none;
}}

.edge-hit {{
  stroke: transparent;
  stroke-width: 9;
  pointer-events: stroke;
}}

.node-bg {{
  transition: opacity 0.25s ease;
}}

.node-circuit {{
  transition: opacity 0.25s ease;
  cursor: pointer;
}}

.node-label {{
  font-family: var(--mono);
  font-size: 8.5px;
  fill: var(--text);
  pointer-events: none;
  transition: opacity 0.25s ease;
}}

.layer-label {{
  font-family: var(--font);
  font-size: 10px;
  fill: var(--text-muted);
  text-anchor: middle;
  pointer-events: none;
}}

/* ---- Tooltip ---- */
#tooltip {{
  position: fixed;
  background: var(--bg-paper);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 9px 12px;
  font-size: 12px;
  line-height: 1.6;
  color: var(--text);
  pointer-events: none;
  opacity: 0;
  box-shadow: 0 4px 16px rgba(0,0,0,0.1);
  max-width: 220px;
  transition: opacity 0.15s ease;
  z-index: 9999;
}}

#tooltip.visible {{ opacity: 1; }}

#tooltip .tt-name {{
  font-family: var(--mono);
  font-weight: 700;
  font-size: 13px;
  margin-bottom: 4px;
}}

#tooltip .tt-badge {{
  display: inline-block;
  padding: 1px 6px;
  border-radius: 10px;
  font-size: 10px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 4px;
}}

#tooltip .tt-row {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
}}

#tooltip .tt-key {{ color: var(--text-muted); }}

#tooltip .tt-status-in  {{ color: #2DC653; font-weight: 600; }}
#tooltip .tt-status-out {{ color: #6C757D; }}

/* ---- Legend ---- */
#legend {{
  position: absolute;
  bottom: 20px;
  right: 20px;
  background: var(--bg-paper);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 11px;
  color: var(--text-muted);
  line-height: 1.8;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  min-width: 140px;
}}

#legend h3 {{
  font-size: 11px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}}

.legend-row {{
  display: flex;
  align-items: center;
  gap: 8px;
}}

.legend-dot {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}}

/* ---- Reset button ---- */
#reset-zoom {{
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px 10px;
  font-size: 12px;
  cursor: pointer;
  color: var(--text);
  font-family: var(--font);
}}

#reset-zoom:hover {{
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}}

/* ---- Too-large fallback ---- */
#graph-toolarge {{
  display: none;
  flex: 1;
  align-items: center;
  justify-content: center;
  padding: 40px;
}}

.toolarge-box {{
  max-width: 480px;
  text-align: center;
  color: var(--text-muted);
  font-size: 13px;
  line-height: 1.6;
}}

.toolarge-box h2 {{
  font-size: 16px;
  color: var(--text);
  margin-bottom: 10px;
}}

.toolarge-box p {{
  margin-bottom: 8px;
}}
</style>
</head>
<body>

<!-- Top bar: title + controls -->
<div id="topbar">
  <h1 id="title-text">{title}</h1>

  <div class="controls">
    <div class="control-group">
      <label for="threshold-slider">Edge threshold</label>
      <input type="range" id="threshold-slider"
             min="0" max="1" step="0.01" value="{default_threshold}"/>
      <span id="threshold-value">{default_threshold:.2f}</span>
    </div>

    <div class="control-group">
      <input type="checkbox" id="toggle-background" checked/>
      <label for="toggle-background">Show pruned nodes</label>
    </div>

    <button id="reset-zoom">⟳ Reset view</button>

    <div class="stats-pill" id="stats-pill">—</div>
  </div>
</div>

<!-- Main SVG canvas -->
<div id="graph-container">
  <svg id="graph-svg">
    <defs><!-- arrowhead markers are generated by JS, binned by edge weight --></defs>
    <g id="root-g">
      <g id="edges-g"></g>
      <g id="edges-hit-g"></g>
      <g id="nodes-bg-g"></g>
      <g id="nodes-circuit-g"></g>
      <g id="labels-g"></g>
      <g id="layer-labels-g"></g>
    </g>
  </svg>

  <!-- Tooltip (absolutely positioned) -->
  <div id="tooltip"></div>

  <!-- Legend -->
  <div id="legend">
    <h3>Node types</h3>
    <div id="legend-items"></div>
    <div style="margin-top:8px; border-top: 1px solid var(--border); padding-top:8px;">
      <div class="legend-row">
        <div class="legend-dot"
             style="background:#CED4DA; border: 1px solid #ADB5BD;"></div>
        <span>Pruned node</span>
      </div>
    </div>
  </div>
</div>

<!-- Too-large fallback (hidden unless the size guard trips) -->
<div id="graph-toolarge">
  <div class="toolarge-box">
    <h2>This circuit is too large to render interactively</h2>
    <p id="toolarge-detail"></p>
    <p>Try raising the edge threshold, increasing sparsity to prune more
       nodes, or exporting a static image instead.</p>
  </div>
</div>

<script>
// ============================================================
// Embedded data
// ============================================================
const GRAPH   = {graph_json};
const CK_THEME = {theme_json};

// ============================================================
// Apply CSS variables from theme
// ============================================================
(function applyTheme() {{
  const p = CK_THEME.palette;
  const root = document.documentElement.style;
  root.setProperty("--bg",        p.bg);
  root.setProperty("--bg-paper",  p.bg_paper);
  root.setProperty("--text",      p.text_primary);
  root.setProperty("--text-muted",p.text_secondary);
  root.setProperty("--border",    p.grid);
  root.setProperty("--accent",    p.accent);
}})();

// ============================================================
// Helpers
// ============================================================
function nodeColor(type) {{
  return CK_THEME.node_colors[type] || CK_THEME.palette.unknown;
}}

function fmt(v, decimals=4) {{
  return Number(v).toFixed(decimals);
}}

// ============================================================
// Legend
// ============================================================
(function buildLegend() {{
  const items = document.getElementById("legend-items");
  const types = [
    ["attn_head", "Attention head"],
    ["mlp",       "MLP sublayer"],
    ["residual",  "Residual stream"],
    ["embed",     "Embed / Unembed"],
    ["unknown",   "Other"],
  ];
  types.forEach(([type, label]) => {{
    const row = document.createElement("div");
    row.className = "legend-row";
    row.innerHTML = `
      <div class="legend-dot" style="background:${{nodeColor(type)}}"></div>
      <span>${{label}}</span>`;
    items.appendChild(row);
  }});
}})();

// ============================================================
// Stats pill (kept in sync with the threshold slider)
// ============================================================
function updateStats(visibleEdges) {{
  const m = GRAPH.metadata;
  const edgeText = visibleEdges === GRAPH.edges.length
    ? `${{GRAPH.edges.length}} edges`
    : `${{visibleEdges}} / ${{GRAPH.edges.length}} edges`;
  document.getElementById("stats-pill").textContent =
    `${{m.n_circuit_nodes}} circuit · ${{m.n_pruned_nodes}} pruned · ${{edgeText}}`;
}}
updateStats(GRAPH.edges.length);

// ============================================================
// D3 setup
// ============================================================
const svg   = d3.select("#graph-svg");
const rootG = d3.select("#root-g");
const edgesG  = d3.select("#edges-g");
const edgesHitG = d3.select("#edges-hit-g");
const nodesBgG = d3.select("#nodes-bg-g");
const nodesCircuitG = d3.select("#nodes-circuit-g");
const labelsG = d3.select("#labels-g");
const layerLabelsG = d3.select("#layer-labels-g");

// ============================================================
// Size guard — bail out before attempting an interactive layout
// ============================================================
const MAX_RENDERABLE_NODES = 600;
const MAX_RENDERABLE_EDGES = 1500;

function markReady() {{
  // Signal that the page has reached a stable render state, so headless
  // capture tools can wait on it instead of a blind timeout.
  window.__ckGraphReady = true;
  document.body.setAttribute("data-ck-ready", "1");
}}

function showTooLarge() {{
  document.getElementById("graph-container").style.display = "none";
  document.getElementById("legend").style.display = "none";
  document.getElementById("toolarge-detail").textContent =
    `This circuit has ${{GRAPH.nodes.length}} nodes and ${{GRAPH.edges.length}} edges ` +
    `(limits: ${{MAX_RENDERABLE_NODES}} nodes / ${{MAX_RENDERABLE_EDGES}} edges).`;
  document.getElementById("graph-toolarge").style.display = "flex";
  markReady();
}}

if (GRAPH.nodes.length > MAX_RENDERABLE_NODES || GRAPH.edges.length > MAX_RENDERABLE_EDGES) {{
  showTooLarge();
}} else {{
  initGraph().catch(err => {{
    console.error("Circuit graph failed to initialise:", err);
    markReady();  // don't leave capture tools waiting forever
  }});
}}

async function initGraph() {{

// ---- Build lookup maps ----
const nodeById = new Map(GRAPH.nodes.map(n => [n.id, n]));

// ---- Node size: log_norm_score → radius, FIXED pixel range ----
// Bounded via d3.scaleSqrt (area, not radius, scales with value) so no
// single high-score node can ever render larger than this range,
// regardless of how the underlying scores are distributed.
const scoreExtent = d3.extent(GRAPH.nodes, d => d.log_norm_score);
const radiusScale = d3.scaleSqrt()
  .domain([scoreExtent[0] ?? 0, scoreExtent[1] ?? 1])
  .range([5, 16])
  .clamp(true);

function nodeRadius(node) {{
  return node.in_circuit ? radiusScale(node.log_norm_score) : 4;
}}

// ---- Edge stroke-width: normalized_weight → width, FIXED pixel range ----
const edgeWidthScale = d3.scaleLinear()
  .domain([0, 1])
  .range([0.75, 4])
  .clamp(true);

// ---- Edge colour interpolator ----
const edgeColorScale = d3.scaleSequential()
  .domain([0, 1])
  .interpolator(d3.interpolateRgb(
    CK_THEME.palette.edge_low,
    CK_THEME.palette.edge_high,
  ));

// ---- Arrowhead markers, binned by edge weight ----
// SVG markers cannot inherit per-edge stroke color in all browsers, so we
// pre-generate a small bank of markers along the edge color scale and pick
// the nearest bin per edge. This keeps weak-edge arrowheads as faint as
// their lines instead of a fixed solid red.
const N_ARROW_BINS = 8;
const svgDefs = svg.select("defs");
for (let i = 0; i < N_ARROW_BINS; i++) {{
  const t = (i + 0.5) / N_ARROW_BINS;
  svgDefs.append("marker")
    .attr("id", `arrowhead-${{i}}`)
    .attr("markerUnits", "userSpaceOnUse")
    .attr("markerWidth", 8)
    .attr("markerHeight", 6)
    .attr("refX", 7)
    .attr("refY", 3)
    .attr("orient", "auto")
    .append("polygon")
    .attr("points", "0 0, 8 3, 0 6")
    .attr("fill", edgeColorScale(t))
    .attr("opacity", 0.3 + t * 0.6);
}}

function arrowMarker(weight) {{
  const bin = Math.max(0, Math.min(N_ARROW_BINS - 1, Math.floor(weight * N_ARROW_BINS)));
  return `url(#arrowhead-${{bin}})`;
}}

// ============================================================
// Layout: ELK assigns x/y. Each node's real model layer (L0,
// L1, ...) is locked via partitioning, so ELK never reassigns
// layers based on graph topology — important because skip
// connections (e.g. L0 -> L5) would otherwise confuse a
// longest-path ranker into drawing the wrong layer structure.
// ELK only solves what's actually unsolved: ordering nodes
// within each fixed layer to minimize edge crossings, and
// spline-routing edges around node geometry.
// ============================================================
let elkOk = false;
try {{
  if (typeof ELK !== "undefined") {{
    const elk = new ELK();

    const elkGraph = {{
      id: "root",
      layoutOptions: {{
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.partitioning.activate": "true",
        "elk.edgeRouting": "SPLINES",
        "elk.layered.spacing.nodeNodeBetweenLayers": "90",
        "elk.spacing.nodeNode": "24",
        "elk.spacing.edgeNode": "12",
      }},
      // Only in-circuit nodes take part in the ELK layout: pruned nodes
      // carry no edges, and ELK piles degree-0 nodes into a detached band
      // that reads as a separate structure. They are placed manually
      // below their layer column after the layout instead.
      children: GRAPH.nodes.filter(n => n.in_circuit).map(n => {{
        const r = nodeRadius(n);
        return {{
          id: n.id,
          width: r * 2 + 6,
          height: r * 2 + 18,  // extra room below for the label
          layoutOptions: {{ "elk.partitioning.partition": String(n.layer) }},
        }};
      }}),
      edges: GRAPH.edges.map(e => ({{
        id: e.source + ">" + e.target,
        sources: [e.source],
        targets: [e.target],
      }})),
    }};

    const elkResult = await elk.layout(elkGraph);

    // Write ELK's computed positions back onto GRAPH.nodes as node-center
    // coordinates (ELK returns each node's top-left box corner).
    const elkById = new Map(elkResult.children.map(c => [c.id, c]));
    GRAPH.nodes.forEach(n => {{
      const box = elkById.get(n.id);
      if (box) {{
        n.x = box.x + box.width / 2;
        n.y = box.y + nodeRadius(n);
      }}
    }});
    elkOk = true;
  }}
}} catch (err) {{
  console.warn("ELK layout failed, falling back to embedded layout:", err);
}}

if (!elkOk) {{
  // The payload ships the Python-computed layout (x_spacing=1.8, y_spacing
  // =0.72 data units); scale it up to pixel space so the file still renders
  // when the ELK CDN is unreachable (e.g. fully offline). This layout
  // already places pruned nodes within their layer clusters.
  const FALLBACK_SCALE = 70;
  GRAPH.nodes.forEach(n => {{
    n.x = n.x * FALLBACK_SCALE;
    n.y = n.y * FALLBACK_SCALE;
  }});
}} else {{
  // Place each layer's pruned nodes in a compact column directly beneath
  // that layer's circuit nodes, so kept and removed components read as
  // one structure instead of a detached band.
  const circuitNodesAll = GRAPH.nodes.filter(n => n.in_circuit);
  const layerXCenters = new Map();
  d3.group(circuitNodesAll, n => n.layer).forEach((nodes, layer) => {{
    layerXCenters.set(layer, d3.mean(nodes, n => n.x));
  }});
  // Linear fit over known layer centers, for layers whose every node
  // was pruned.
  const knownLayers = Array.from(layerXCenters.keys()).sort((a, b) => a - b);
  const xOfLayer = (layer) => {{
    if (layerXCenters.has(layer)) return layerXCenters.get(layer);
    if (knownLayers.length < 2) return (layerXCenters.get(knownLayers[0]) || 0);
    const l0 = knownLayers[0], l1 = knownLayers[knownLayers.length - 1];
    const x0 = layerXCenters.get(l0), x1 = layerXCenters.get(l1);
    return x0 + (x1 - x0) * (layer - l0) / (l1 - l0);
  }};
  const prunedYBase = d3.max(circuitNodesAll, n => n.y) + 55;
  d3.group(GRAPH.nodes.filter(n => !n.in_circuit), n => n.layer)
    .forEach((nodes, layer) => {{
      nodes.sort((a, b) => ((a.head ?? -1) - (b.head ?? -1)));
      nodes.forEach((n, i) => {{
        n.x = xOfLayer(layer);
        n.y = prunedYBase + i * 13;
      }});
    }});
}}

// ---- Coordinate transform: pixel-space (from ELK) → screen-space ----
const xs = GRAPH.nodes.map(n => n.x);
const ys = GRAPH.nodes.map(n => n.y);
const xMin = Math.min(...xs), xMax = Math.max(...xs);
const yMin = Math.min(...ys), yMax = Math.max(...ys);

// We use D3 zoom to handle all coordinate mapping; the base transform
// maps ELK's pixel-space layout to an initial "fit" view.
function getInitialTransform(svgW, svgH) {{
  const padding = 80;
  const dataW = xMax - xMin || 1;
  const dataH = yMax - yMin || 1;
  const scaleX = (svgW - padding * 2) / dataW;
  const scaleY = (svgH - padding * 2) / dataH;
  // ELK already lays out in real pixel units, so this should land near 1.
  // Capped defensively, not relied on, in case of a very sparse graph.
  const scale  = Math.min(scaleX, scaleY, 3);
  const tx = padding - xMin * scale + (svgW - padding * 2 - dataW * scale) / 2;
  const ty = padding - yMin * scale + (svgH - padding * 2 - dataH * scale) / 2;
  return d3.zoomIdentity.translate(tx, ty).scale(scale);
}}

// ---- Zoom behaviour ----
let currentTransform = d3.zoomIdentity;
const zoom = d3.zoom()
  .scaleExtent([0.05, 20])
  .on("zoom", (event) => {{
    currentTransform = event.transform;
    rootG.attr("transform", currentTransform);
  }});

svg.call(zoom);

function resetZoom(animate = true) {{
  const svgEl = document.getElementById("graph-svg");
  const t = getInitialTransform(svgEl.clientWidth, svgEl.clientHeight);
  if (animate) {{
    svg.transition().duration(450).call(zoom.transform, t);
  }} else {{
    svg.call(zoom.transform, t);
  }}
}}

document.getElementById("reset-zoom").addEventListener("click", () => resetZoom(true));

// ---- Initial fit ----
// Fit instantly (no animation) so automated capture (print-to-PDF /
// screenshot) sees the settled view, then raise a readiness flag that
// headless renderers can wait on instead of guessing a timeout. The
// double rAF ensures the SVG has its final client size before we fit.
requestAnimationFrame(() => requestAnimationFrame(() => {{
  resetZoom(false);
  markReady();

  // Re-fit whenever the SVG's box changes: window resize, and — key for
  // export — the relayout when a browser switches to print media, which
  // would otherwise leave the graph fitted to the old (screen) size and
  // clip it in the printed page. resetZoom only edits the inner <g>
  // transform, not the observed element, so this never loops.
  const _ro = new ResizeObserver(() => {{
    if (window.__ckGraphReady) resetZoom(false);
  }});
  _ro.observe(document.getElementById("graph-svg"));
  window.addEventListener("beforeprint", () => resetZoom(false));
}}));

// ---- Draw layer labels ----
(function drawLayerLabels() {{
  const layerXs = d3.rollup(
    GRAPH.nodes,
    group => group[0].x,
    d => d.layer,
  );
  // Place below all nodes (pixel-space offset, post-ELK layout)
  const labelY = yMax + 24;

  layerXs.forEach((x, layer) => {{
    layerLabelsG.append("text")
      .attr("class", "layer-label")
      .attr("x", x)
      .attr("y", labelY)
      .text(`L${{layer}}`);
  }});
}})();

// ---- Draw background (pruned) nodes ----
const bgNodes = GRAPH.nodes.filter(n => !n.in_circuit);
nodesBgG.selectAll(".node-bg")
  .data(bgNodes)
  .join("circle")
  .attr("class", "node-bg")
  .attr("cx", d => d.x)
  .attr("cy", d => d.y)
  .attr("r", 3.5)
  .attr("fill", CK_THEME.palette.background_node)
  .attr("stroke", CK_THEME.palette.background_node_stroke)
  .attr("stroke-width", 0.6)
  .attr("opacity", 0.45)
  .on("mouseover", (event, d) => showNodeTooltip(event, d))
  .on("mousemove", moveTooltip)
  .on("mouseout", hideTooltip);

// ---- Draw circuit nodes ----
const circuitNodes = GRAPH.nodes.filter(n => n.in_circuit);
const circleSelection = nodesCircuitG.selectAll(".node-circuit")
  .data(circuitNodes, d => d.id)
  .join("circle")
  .attr("class", "node-circuit")
  .attr("cx", d => d.x)
  .attr("cy", d => d.y)
  .attr("r", d => nodeRadius(d))
  .attr("fill", d => nodeColor(d.type))
  .attr("stroke", CK_THEME.palette.circuit_stroke)
  .attr("stroke-width", 1.2)
  .attr("opacity", d => 0.55 + d.log_norm_score * 0.45)
  .style("cursor", "pointer")
  .on("mouseover", (event, d) => {{
    showNodeTooltip(event, d);
    highlightNode(d.id);
  }})
  .on("mousemove", moveTooltip)
  .on("mouseout", (event, d) => {{
    hideTooltip();
    clearHighlight();
  }});

// ---- Draw node labels ----
labelsG.selectAll(".node-label")
  .data(circuitNodes, d => d.id)
  .join("text")
  .attr("class", "node-label")
  .attr("x", d => d.x)
  .attr("y", d => d.y - nodeRadius(d) - 2)
  .attr("text-anchor", "middle")
  .text(d => d.id);

// ---- Draw edges ----
let currentThreshold = {default_threshold};

function redrawEdges(threshold) {{
  currentThreshold = threshold;
  const visible = GRAPH.edges.filter(e => e.normalized_weight >= threshold);

  const edgeSel = edgesG.selectAll(".edge")
    .data(visible, d => d.source + ">" + d.target);

  edgeSel.join(
    enter => enter.append("line")
      .attr("class", "edge")
      .call(applyEdgeAttrs),
    update => update.call(applyEdgeAttrs),
    exit => exit.remove(),
  );

  // Invisible wide companion lines that provide a usable hover hit area
  // (the visible edges are sub-pixel thin and have pointer-events: none).
  const hitSel = edgesHitG.selectAll(".edge-hit")
    .data(visible, d => d.source + ">" + d.target);

  hitSel.join(
    enter => enter.append("line")
      .attr("class", "edge-hit")
      .call(applyEdgeEndpoints)
      .on("mouseover", (event, d) => showEdgeTooltip(event, d))
      .on("mousemove", moveTooltip)
      .on("mouseout", hideTooltip),
    update => update.call(applyEdgeEndpoints),
    exit => exit.remove(),
  );

  updateStats(visible.length);
}}

function applyEdgeEndpoints(sel) {{
  sel
    .attr("x1", d => (nodeById.get(d.source) || {{}}).x || 0)
    .attr("y1", d => (nodeById.get(d.source) || {{}}).y || 0)
    .attr("x2", d => (nodeById.get(d.target) || {{}}).x || 0)
    .attr("y2", d => (nodeById.get(d.target) || {{}}).y || 0);
}}

function applyEdgeAttrs(sel) {{
  sel
    .call(applyEdgeEndpoints)
    .attr("stroke", d => edgeColorScale(d.normalized_weight))
    .attr("stroke-width", d => edgeWidthScale(d.normalized_weight))
    .attr("opacity", d => 0.25 + d.normalized_weight * 0.65)
    .attr("marker-end", d => arrowMarker(d.normalized_weight));
}}

redrawEdges(currentThreshold);

// ============================================================
// Interactions: threshold slider
// ============================================================
const slider = document.getElementById("threshold-slider");
const thresholdDisplay = document.getElementById("threshold-value");

slider.addEventListener("input", () => {{
  const t = parseFloat(slider.value);
  thresholdDisplay.textContent = t.toFixed(2);
  redrawEdges(t);
}});

// ============================================================
// Interactions: toggle background nodes
// ============================================================
document.getElementById("toggle-background").addEventListener("change", function() {{
  const opacity = this.checked ? 0.45 : 0;
  nodesBgG.selectAll(".node-bg")
    // Hidden nodes must not keep capturing hover events / tooltips.
    .style("pointer-events", this.checked ? "auto" : "none")
    .transition().duration(200)
    .attr("opacity", opacity);
}});

// ============================================================
// Interactions: node hover highlight
// ============================================================
function highlightNode(nodeId) {{
  // Collect neighbour ids from visible edges
  const visibleEdges = GRAPH.edges.filter(e => e.normalized_weight >= currentThreshold);
  const connectedIds = new Set();
  const connectedEdgeKeys = new Set();

  visibleEdges.forEach(e => {{
    if (e.source === nodeId) {{
      connectedIds.add(e.target);
      connectedEdgeKeys.add(e.source + ">" + e.target);
    }}
    if (e.target === nodeId) {{
      connectedIds.add(e.source);
      connectedEdgeKeys.add(e.source + ">" + e.target);
    }}
  }});

  // Dim non-connected circuit nodes
  nodesCircuitG.selectAll(".node-circuit")
    .transition().duration(120)
    .attr("opacity", d => {{
      if (d.id === nodeId) return 1;
      if (connectedIds.has(d.id)) return 0.85;
      return 0.12;
    }});

  // Dim non-connected labels
  labelsG.selectAll(".node-label")
    .transition().duration(120)
    .attr("opacity", d => {{
      if (d.id === nodeId) return 1;
      if (connectedIds.has(d.id)) return 0.85;
      return 0.08;
    }});

  // Dim non-connected edges
  edgesG.selectAll(".edge")
    .transition().duration(120)
    .attr("opacity", d => {{
      const key = d.source + ">" + d.target;
      return connectedEdgeKeys.has(key) ? 1 : 0.04;
    }})
    .attr("stroke-width", d => {{
      const key = d.source + ">" + d.target;
      return connectedEdgeKeys.has(key)
        ? edgeWidthScale(d.normalized_weight) * 2.5
        : edgeWidthScale(d.normalized_weight);
    }});
}}

function clearHighlight() {{
  nodesCircuitG.selectAll(".node-circuit")
    .transition().duration(180)
    .attr("opacity", d => 0.55 + d.log_norm_score * 0.45);

  labelsG.selectAll(".node-label")
    .transition().duration(180)
    .attr("opacity", 1);

  edgesG.selectAll(".edge")
    .transition().duration(180)
    .attr("opacity", d => 0.25 + d.normalized_weight * 0.65)
    .attr("stroke-width", d => edgeWidthScale(d.normalized_weight));
}}

// ============================================================
// Tooltip
// ============================================================
const tooltip = document.getElementById("tooltip");

function showNodeTooltip(event, node) {{
  const statusClass  = node.in_circuit ? "tt-status-in" : "tt-status-out";
  const statusText   = node.in_circuit ? "✓ In circuit" : "✗ Pruned";
  const headInfo     = node.head !== null ? `<div class="tt-row">
    <span class="tt-key">Head</span><span>${{node.head}}</span></div>` : "";
  const typeColor    = nodeColor(node.type);
  const typeLabel    = node.type.replace("_", " ");

  tooltip.innerHTML = `
    <div class="tt-name">${{node.id}}</div>
    <div><span class="tt-badge" style="background:${{typeColor}}">${{typeLabel}}</span></div>
    <div class="tt-row">
      <span class="tt-key">Layer</span><span>${{node.layer}}</span>
    </div>
    ${{headInfo}}
    <div class="tt-row">
      <span class="tt-key">Score</span><span>${{fmt(node.raw_score, 6)}}</span>
    </div>
    <div class="tt-row">
      <span class="tt-key">Rank</span>
      <span>#${{node.rank}} / ${{GRAPH.nodes.length}}</span>
    </div>
    <div class="tt-row">
      <span class="tt-key">Status</span>
      <span class="${{statusClass}}">${{statusText}}</span>
    </div>`;

  tooltip.classList.add("visible");
  moveTooltip(event);
}}

function showEdgeTooltip(event, edge) {{
  tooltip.innerHTML = `
    <div class="tt-name" style="font-size:11px">
      ${{edge.source}} → ${{edge.target}}
    </div>
    <div class="tt-row">
      <span class="tt-key">Strength</span>
      <span>${{fmt(edge.normalized_weight, 3)}}</span>
    </div>
    <div class="tt-row">
      <span class="tt-key">Raw weight</span>
      <span>${{fmt(edge.raw_weight, 8)}}</span>
    </div>`;
  tooltip.classList.add("visible");
  moveTooltip(event);
}}

function moveTooltip(event) {{
  const x = event.clientX + 14;
  const y = event.clientY - 10;
  const ttW = tooltip.offsetWidth;
  const ttH = tooltip.offsetHeight;
  const winW = window.innerWidth;
  const winH = window.innerHeight;
  tooltip.style.left = (x + ttW > winW ? x - ttW - 28 : x) + "px";
  tooltip.style.top  = (y + ttH > winH ? y - ttH      : y) + "px";
}}

function hideTooltip() {{
  tooltip.classList.remove("visible");
}}

}} // end initGraph()

</script>
</body>
</html>
"""
