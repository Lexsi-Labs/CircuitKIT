"""
Visualization gallery generator for comprehensive circuit analysis.

Auto-generates an interactive HTML gallery with:
- Circuit graphs (all discovered circuits)
- Saliency comparisons
- Transfer matrix heatmaps
- Example evaluations
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .theme import get_gallery_css


class GalleryGenerator:
    """
    Generate interactive HTML gallery of circuit visualizations.

    Collects visualizations from multiple sources and creates a
    single-page HTML gallery with navigation and search capabilities.
    """

    def __init__(self, output_dir: str = "circuit_gallery"):
        """
        Initialize gallery generator.

        Args:
            output_dir: Directory to store gallery and assets.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gallery_items: List[Dict[str, Any]] = []
        self.metadata = {
            "title": "CircuitKit Visualization Gallery",
            "generated": datetime.now().isoformat(),
            "items": 0,
        }

    def add_circuit_graph(
        self,
        name: str,
        description: str,
        html_figure: str,
        circuit_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Add a circuit graph visualization to gallery.

        Args:
            name: Display name for the circuit.
            description: Description of the circuit.
            html_figure: Plotly figure HTML string.
            circuit_data: Optional circuit structure data.
            metadata: Optional metadata (algorithm, task, model, etc.).
        """
        item = {
            "type": "circuit_graph",
            "name": name,
            "description": description,
            "html": html_figure,
            "circuit_data": circuit_data or {},
            "metadata": metadata or {},
        }
        self.gallery_items.append(item)

    def add_saliency_comparison(
        self,
        name: str,
        description: str,
        html_figures: Dict[str, str],
        tokens: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Add saliency comparison visualizations.

        Args:
            name: Display name.
            description: Description.
            html_figures: Dict mapping visualization name to HTML string.
                Expected keys: 'layer_heatmaps', 'aggregate', 'comparison'.
            tokens: Optional token list.
            metadata: Optional metadata.
        """
        item = {
            "type": "saliency_comparison",
            "name": name,
            "description": description,
            "html_figures": html_figures,
            "tokens": tokens or [],
            "metadata": metadata or {},
        }
        self.gallery_items.append(item)

    def add_feature_attribution(
        self,
        name: str,
        description: str,
        html_figures: Dict[str, str],
        top_nodes: Optional[List[Tuple[str, float]]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Add feature attribution visualizations.

        Args:
            name: Display name.
            description: Description.
            html_figures: Dict mapping visualization name to HTML string.
                Expected keys: 'importance_bar', 'network', 'comparison'.
            top_nodes: Optional list of (node, score) tuples.
            metadata: Optional metadata.
        """
        item = {
            "type": "feature_attribution",
            "name": name,
            "description": description,
            "html_figures": html_figures,
            "top_nodes": top_nodes or [],
            "metadata": metadata or {},
        }
        self.gallery_items.append(item)

    def add_comparison_dashboard(
        self,
        name: str,
        description: str,
        html_figures: Dict[str, str],
        comparison_type: str = "stability",
        summary_stats: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Add comparison dashboard visualizations.

        Args:
            name: Display name.
            description: Description.
            html_figures: Dict mapping visualization name to HTML string.
                Expected keys: 'stability', 'correlation', 'robustness',
                'transfer', 'distribution'.
            comparison_type: Type of comparison.
            summary_stats: Optional summary statistics.
            metadata: Optional metadata.
        """
        item = {
            "type": "comparison_dashboard",
            "name": name,
            "description": description,
            "html_figures": html_figures,
            "comparison_type": comparison_type,
            "summary_stats": summary_stats or {},
            "metadata": metadata or {},
        }
        self.gallery_items.append(item)

    def add_evaluation_report(
        self,
        name: str,
        description: str,
        evaluation_data: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Add evaluation report to gallery.

        Args:
            name: Display name.
            description: Description.
            evaluation_data: Dictionary with evaluation metrics and results.
            metadata: Optional metadata.
        """
        item = {
            "type": "evaluation_report",
            "name": name,
            "description": description,
            "evaluation_data": evaluation_data,
            "metadata": metadata or {},
        }
        self.gallery_items.append(item)

    def generate_html(self) -> str:
        """
        Generate the complete gallery HTML page.

        Returns:
            HTML string.
        """
        html_content = self._generate_header()
        html_content += self._generate_navigation()
        html_content += self._generate_items()
        html_content += self._generate_footer()

        return html_content

    def _generate_header(self) -> str:
        """Generate HTML header with styling."""
        timestamp = self.metadata["generated"]
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.metadata['title']}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            {get_gallery_css()}
        </head>
        <body>
            <header>
                <h1>⚡ {self.metadata['title']}</h1>
                <p class="subtitle">Circuit Discovery and Visualization Analysis</p>
                <p class="subtitle">Generated: {timestamp}</p>
            </header>
        """

    def _generate_navigation(self) -> str:
        """Generate navigation and search interface."""
        self.metadata["items"] = len(self.gallery_items)
        item_types = {}
        for item in self.gallery_items:
            item_type = item["type"]
            item_types[item_type] = item_types.get(item_type, 0) + 1

        type_tags = "".join(
            [
                f'<button class="tag" onclick="filterByType(\'{t}\')">{t.replace("_", " ").title()} ({count})</button>'
                for t, count in item_types.items()
            ]
        )

        return f"""
        <div class="container">
            <div class="search-bar">
                <input type="text" id="searchInput" placeholder="Search visualizations...">
                <button onclick="searchGallery()">Search</button>
            </div>

            <div class="filter-tags">
                <button class="tag active" onclick="filterByType('')">All ({len(self.gallery_items)})</button>
                {type_tags}
            </div>

            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{len(self.gallery_items)}</div>
                    <div class="stat-label">Visualizations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(set(item['type'] for item in self.gallery_items))}</div>
                    <div class="stat-label">Types</div>
                </div>
            </div>

            <div id="gallery" class="gallery-grid">
        """

    def _generate_items(self) -> str:
        """Generate gallery item HTML."""
        html = ""
        for idx, item in enumerate(self.gallery_items):
            html += self._generate_item_html(item, idx)
        html += "</div></div>"  # Close gallery grid and container
        return html

    def _generate_item_html(self, item: Dict[str, Any], idx: int) -> str:
        """Generate HTML for a single gallery item."""
        item_type = item["type"].replace("_", " ").title()
        name = item.get("name", "Untitled")
        description = item.get("description", "No description")

        metadata_html = ""
        if item.get("metadata"):
            for key, value in item["metadata"].items():
                metadata_html += f'<div class="metadata-item"><span class="metadata-label">{key}:</span> {value}</div>'

        content_html = ""
        if item["type"] == "circuit_graph":
            content_html = f'<div class="plotly-container" id="plot-{idx}"></div>'
        elif item["type"] in ("saliency_comparison", "feature_attribution", "comparison_dashboard"):
            for fig_name, fig_html in item.get("html_figures", {}).items():
                content_html += f"""
                <div style="margin: 1rem 0;">
                    <h4>{fig_name.replace("_", " ").title()}</h4>
                    <div class="plotly-container" id="plot-{idx}-{fig_name}"></div>
                </div>
                """
        elif item["type"] == "evaluation_report":
            eval_data = item.get("evaluation_data", {})
            content_html = f"<pre>{json.dumps(eval_data, indent=2)}</pre>"

        metadata_section = f'<div class="metadata">{metadata_html}</div>' if metadata_html else ""

        return f"""
        <div class="gallery-item" data-type="{item['type']}">
            <div class="item-header">
                <span class="item-type">{item_type}</span>
                <div class="item-title">{name}</div>
                <div class="item-description">{description}</div>
            </div>
            <div class="item-content">
                {content_html}
                {metadata_section}
            </div>
        </div>
        """

    def _generate_footer(self) -> str:
        """Generate HTML footer with scripts."""
        return """
            <footer>
                <p>&copy; 2024 CircuitKit. All visualizations generated automatically.</p>
            </footer>

            <script>
                function filterByType(type) {{
                    const items = document.querySelectorAll('.gallery-item');
                    items.forEach(item => {{
                        if (type === '' || item.getAttribute('data-type') === type) {{
                            item.classList.remove('hidden');
                        }} else {{
                            item.classList.add('hidden');
                        }}
                    }});
                }}

                function searchGallery() {{
                    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
                    const items = document.querySelectorAll('.gallery-item');
                    items.forEach(item => {{
                        const title = item.querySelector('.item-title').textContent.toLowerCase();
                        const description = item.querySelector('.item-description').textContent.toLowerCase();
                        if (title.includes(searchTerm) || description.includes(searchTerm)) {{
                            item.classList.remove('hidden');
                        }} else {{
                            item.classList.add('hidden');
                        }}
                    }});
                }}

                // Auto-refresh Plotly figures
                setTimeout(function() {{
                    const plots = document.querySelectorAll('[id^="plot-"]');
                    plots.forEach(plot => {{
                        try {{
                            Plotly.newPlot(plot.id, {{}}, {{}});
                        }} catch(e) {{
                            // Silently fail if Plotly data not available
                        }}
                    }});
                }}, 100);
            </script>
        </body>
        </html>
        """

    def save(self, output_file: Optional[str] = None) -> Path:
        """
        Save gallery to HTML file.

        Args:
            output_file: Output file path. If None, uses index.html in output_dir.

        Returns:
            Path to saved file.
        """
        if output_file is None:
            output_file = self.output_dir / "index.html"
        else:
            output_file = Path(output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        html_content = self.generate_html()
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_file

    def save_metadata(self) -> None:
        """Save gallery metadata to JSON."""
        # Synchronize the count before saving
        self.metadata["items"] = len(self.gallery_items)
        metadata_file = self.output_dir / "gallery_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
