"""
CircuitKit Visualization Design System.

Single source of truth for colors, typography, and layout constants used
across all CircuitKit visualizers (Plotly notebook widgets and D3.js HTML
exports). Any new visualizer should import from here rather than hardcoding
colors or layout values.

Usage::

    from circuitkit.visualize.theme import THEME, get_plotly_layout, get_node_color

    fig.update_layout(**get_plotly_layout(title="My Circuit"))
    color = get_node_color("attn_head")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Palette:
    """
    Immutable color palette for CircuitKit visualizations.

    Designed around a dark-slate / warm-amber research aesthetic: readable on
    white backgrounds in notebooks, accessible at WCAG AA contrast ratios,
    and distinguishable when printed in grayscale.
    """

    # --- Node type colors ---
    attn_head: str = "#1D4ED8"      # deep blue    – attention heads (6.7:1 on white / white text)
    mlp: str = "#F77F00"            # amber-orange – MLP sublayers
    residual: str = "#2DC653"       # forest-green – residual stream nodes
    embed: str = "#E63946"          # signal-red   – embed / unembed nodes
    unknown: str = "#6C757D"        # neutral gray – unrecognised types

    # --- Circuit vs. background distinction ---
    circuit_stroke: str = "#212529"     # near-black border on in-circuit nodes
    background_node: str = "#CED4DA"    # light gray for pruned/out-of-circuit nodes
    background_node_stroke: str = "#ADB5BD"

    # --- Edge colors ---
    edge_low: str = "#DEE2E6"       # faint gray  – low-importance edges
    edge_high: str = "#C1121F"      # deep crimson – high-importance edges
    edge_background: str = "#E9ECEF"  # nearly invisible – background model graph

    # --- Canvas / chrome ---
    bg: str = "#FFFFFF"             # pure white canvas
    bg_paper: str = "#F8F9FA"       # off-white paper (panel backgrounds)
    grid: str = "#E9ECEF"           # very light grid lines
    text_primary: str = "#212529"   # near-black primary text
    text_secondary: str = "#6C757D" # muted secondary text
    accent: str = "#1D4ED8"         # brand accent (matches attn_head)
    accent_gradient_end: str = "#7B2D8B"  # for gallery header gradient

    # --- Colorscales (Plotly format) ---
    @property
    def edge_colorscale(self):
        """Plotly colorscale from faint gray to deep crimson."""
        return [
            [0.0, self.edge_low],
            [0.5, "#E85D04"],   # mid-orange
            [1.0, self.edge_high],
        ]

    @property
    def importance_colorscale(self):
        """Plotly colorscale for node/score heatmaps (white → indigo)."""
        return [
            [0.0, "#FFFFFF"],
            [0.5, "#90A8F8"],
            [1.0, self.attn_head],
        ]


PALETTE = _Palette()


# ---------------------------------------------------------------------------
# Node-type color lookup
# ---------------------------------------------------------------------------

_NODE_TYPE_COLORS: Dict[str, str] = {
    "attn_head": PALETTE.attn_head,
    "attn_in":   PALETTE.attn_head,
    "attn_out":  "#1E3A8A",          # darker blue variant (navy-blue)
    "mlp":       PALETTE.mlp,
    "mlp_in":    PALETTE.mlp,
    "mlp_out":   "#C96300",          # darker amber variant
    "residual":  PALETTE.residual,
    "embed":     PALETTE.embed,
    "unknown":   PALETTE.unknown,
}


def get_node_color(node_type: str) -> str:
    """
    Return the canonical hex color for a node type.

    Args:
        node_type: One of ``"attn_head"``, ``"mlp"``, ``"residual"``,
            ``"embed"``, ``"unknown"`` (and their ``_in`` / ``_out`` variants).

    Returns:
        A hex color string (e.g. ``"#1D4ED8"``).
    """
    return _NODE_TYPE_COLORS.get(node_type, PALETTE.unknown)


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------

#: Font stack used in all Plotly figures.
FONT_FAMILY = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
    "Roboto, Helvetica, Arial, sans-serif"
)

#: Font stack for monospaced node labels (e.g. "A0.1", "MLP 3").
FONT_MONO = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace"


# ---------------------------------------------------------------------------
# Plotly layout template
# ---------------------------------------------------------------------------

def get_plotly_layout(
    title: Optional[str] = None,
    width: int = 1200,
    height: int = 700,
    margin: Optional[Dict[str, int]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Return a consistent Plotly ``update_layout`` kwargs dict.

    All CircuitKit Plotly figures should call::

        fig.update_layout(**get_plotly_layout(title="My Circuit", height=800))

    Args:
        title: Optional figure title string.
        width: Figure width in pixels.
        height: Figure height in pixels.
        margin: Override default margins ``{"b": 40, "l": 20, "r": 20, "t": 60}``.
        **overrides: Any extra ``update_layout`` keys (merged last, so they win).

    Returns:
        Dict suitable for ``fig.update_layout(**...)``.
    """
    _margin = {"b": 40, "l": 20, "r": 20, "t": 60}
    if margin:
        _margin.update(margin)

    layout: Dict[str, Any] = {
        "width": width,
        "height": height,
        "margin": _margin,
        "plot_bgcolor": PALETTE.bg,
        "paper_bgcolor": PALETTE.bg,
        "font": {
            "family": FONT_FAMILY,
            "color": PALETTE.text_primary,
            "size": 12,
        },
        "hoverlabel": {
            "bgcolor": PALETTE.bg_paper,
            "bordercolor": PALETTE.grid,
            "font": {"family": FONT_FAMILY, "size": 12, "color": PALETTE.text_primary},
        },
        "showlegend": False,
        "hovermode": "closest",
        "xaxis": {
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "color": PALETTE.text_secondary,
        },
        "yaxis": {
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "color": PALETTE.text_secondary,
        },
    }

    if title is not None:
        layout["title"] = {
            "text": title,
            "font": {
                "family": FONT_FAMILY,
                "size": 16,
                "color": PALETTE.text_primary,
            },
            "x": 0.5,
            "xanchor": "center",
        }

    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# D3 theme export
# ---------------------------------------------------------------------------

def get_d3_theme() -> Dict[str, Any]:
    """
    Return the design system as a plain JSON-serializable dict.

    Embedded as a ``window.CK_THEME`` constant in the D3.js HTML template
    so that all styling decisions in JavaScript reference the same values.

    Returns:
        Dict with keys: ``palette``, ``font``, ``node_colors``.
    """
    return {
        "palette": {
            "attn_head": PALETTE.attn_head,
            "mlp": PALETTE.mlp,
            "residual": PALETTE.residual,
            "embed": PALETTE.embed,
            "unknown": PALETTE.unknown,
            "circuit_stroke": PALETTE.circuit_stroke,
            "background_node": PALETTE.background_node,
            "background_node_stroke": PALETTE.background_node_stroke,
            "edge_low": PALETTE.edge_low,
            "edge_high": PALETTE.edge_high,
            "edge_background": PALETTE.edge_background,
            "bg": PALETTE.bg,
            "bg_paper": PALETTE.bg_paper,
            "grid": PALETTE.grid,
            "text_primary": PALETTE.text_primary,
            "text_secondary": PALETTE.text_secondary,
            "accent": PALETTE.accent,
            "accent_gradient_end": PALETTE.accent_gradient_end,
        },
        "font": {
            "family": FONT_FAMILY,
            "mono": FONT_MONO,
        },
        "node_colors": _NODE_TYPE_COLORS,
    }


# ---------------------------------------------------------------------------
# Gallery CSS helper
# ---------------------------------------------------------------------------

def get_gallery_css() -> str:
    """
    Return the CSS block for gallery.py's HTML header.

    Replaces hardcoded color strings in the gallery template with theme values.

    Returns:
        A ``<style>...</style>`` block as a string.
    """
    p = PALETTE
    return f"""<style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: {FONT_FAMILY};
            background-color: {p.bg_paper};
            line-height: 1.6;
            color: {p.text_primary};
        }}

        header {{
            background: linear-gradient(135deg, {p.accent} 0%, {p.accent_gradient_end} 100%);
            color: #FFFFFF;
            padding: 2rem;
            text-align: center;
        }}

        header h1 {{ font-size: 2.5em; margin-bottom: 0.5rem; }}
        .subtitle {{ font-size: 0.9em; opacity: 0.9; }}

        .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}

        .search-bar {{ margin: 2rem 0; display: flex; gap: 1rem; }}
        .search-bar input {{
            flex: 1; padding: 0.75rem;
            border: 1px solid {p.grid}; border-radius: 4px; font-size: 1em;
        }}
        .search-bar button {{
            padding: 0.75rem 1.5rem; background: {p.accent};
            color: #FFFFFF; border: none; border-radius: 4px;
            cursor: pointer; font-weight: bold;
        }}
        .search-bar button:hover {{ background: {p.accent_gradient_end}; }}

        .filter-tags {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0; }}
        .tag {{
            padding: 0.4rem 0.8rem; background: {p.grid};
            border-radius: 20px; cursor: pointer; font-size: 0.9em; transition: all 0.3s;
        }}
        .tag:hover, .tag.active {{ background: {p.accent}; color: #FFFFFF; }}

        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 2rem; margin: 2rem 0;
        }}
        .gallery-item {{
            background: {p.bg}; border-radius: 8px; overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: all 0.3s;
        }}
        .gallery-item:hover {{
            box-shadow: 0 4px 16px rgba(0,0,0,0.16); transform: translateY(-4px);
        }}
        .item-header {{
            background: {p.bg_paper}; padding: 1.5rem;
            border-bottom: 1px solid {p.grid};
        }}
        .item-title {{ font-size: 1.3em; font-weight: bold; margin-bottom: 0.5rem; }}
        .item-type {{
            display: inline-block; padding: 0.3rem 0.8rem;
            background: {p.accent}; color: #FFFFFF;
            border-radius: 20px; font-size: 0.8em; font-weight: bold; margin-bottom: 0.5rem;
        }}
        .item-description {{ font-size: 0.95em; color: {p.text_secondary}; margin: 0.5rem 0; }}
    </style>"""
