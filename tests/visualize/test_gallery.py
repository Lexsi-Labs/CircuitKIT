"""
Tests for visualization gallery generator.
"""

import json
from pathlib import Path

import pytest

from circuitkit.visualize.gallery import GalleryGenerator


class TestGalleryGenerator:
    """Test gallery generator."""

    @pytest.fixture
    def gallery(self, tmp_path):
        """Create a gallery generator."""
        return GalleryGenerator(output_dir=str(tmp_path / "gallery"))

    def test_initialization(self, gallery):
        """Test gallery initialization."""
        assert gallery.output_dir is not None
        assert gallery.gallery_items == []

    def test_add_circuit_graph(self, gallery):
        """Test adding circuit graph."""
        gallery.add_circuit_graph(
            name="Test Circuit",
            description="A test circuit",
            html_figure="<div>test</div>",
        )
        assert len(gallery.gallery_items) == 1
        assert gallery.gallery_items[0]["type"] == "circuit_graph"

    def test_add_saliency_comparison(self, gallery):
        """Test adding saliency comparison."""
        gallery.add_saliency_comparison(
            name="Saliency Test",
            description="Test saliency comparison",
            html_figures={
                "layer_heatmaps": "<div>test1</div>",
                "aggregate": "<div>test2</div>",
                "comparison": "<div>test3</div>",
            },
        )
        assert len(gallery.gallery_items) == 1
        assert gallery.gallery_items[0]["type"] == "saliency_comparison"

    def test_add_feature_attribution(self, gallery):
        """Test adding feature attribution."""
        gallery.add_feature_attribution(
            name="Attribution Test",
            description="Test feature attribution",
            html_figures={
                "importance_bar": "<div>test</div>",
                "network": "<div>test2</div>",
            },
        )
        assert len(gallery.gallery_items) == 1
        assert gallery.gallery_items[0]["type"] == "feature_attribution"

    def test_add_comparison_dashboard(self, gallery):
        """Test adding comparison dashboard."""
        gallery.add_comparison_dashboard(
            name="Comparison Test",
            description="Test comparison dashboard",
            html_figures={
                "stability": "<div>test</div>",
                "correlation": "<div>test2</div>",
            },
            comparison_type="stability",
        )
        assert len(gallery.gallery_items) == 1
        assert gallery.gallery_items[0]["type"] == "comparison_dashboard"

    def test_add_evaluation_report(self, gallery):
        """Test adding evaluation report."""
        evaluation_data = {
            "patching_score": 0.8,
            "ablation_score": 0.75,
        }
        gallery.add_evaluation_report(
            name="Evaluation Test",
            description="Test evaluation report",
            evaluation_data=evaluation_data,
        )
        assert len(gallery.gallery_items) == 1
        assert gallery.gallery_items[0]["type"] == "evaluation_report"

    def test_generate_html(self, gallery):
        """Test HTML generation."""
        gallery.add_circuit_graph(
            name="Test Circuit",
            description="A test circuit",
            html_figure="<div>test</div>",
        )
        html = gallery.generate_html()
        assert isinstance(html, str)
        assert "Test Circuit" in html
        assert "circuit_gallery" in html.lower() or "CircuitKit" in html

    def test_save_html(self, gallery, tmp_path):
        """Test saving gallery to HTML."""
        gallery.add_circuit_graph(
            name="Test Circuit",
            description="A test circuit",
            html_figure="<div>test</div>",
        )
        output_file = tmp_path / "index.html"
        saved_path = gallery.save(str(output_file))

        assert saved_path.exists()
        with open(saved_path) as f:
            content = f.read()
            assert "Test Circuit" in content

    def test_save_default_location(self, gallery):
        """Test saving to default location."""
        gallery.add_circuit_graph(
            name="Test",
            description="Test",
            html_figure="<div>test</div>",
        )
        saved_path = gallery.save()
        assert saved_path.exists()

    def test_multiple_items(self, gallery):
        """Test adding multiple items."""
        for i in range(3):
            gallery.add_circuit_graph(
                name=f"Circuit {i}",
                description=f"Test circuit {i}",
                html_figure=f"<div>test{i}</div>",
            )

        assert len(gallery.gallery_items) == 3
        html = gallery.generate_html()
        assert "Circuit 0" in html
        assert "Circuit 1" in html
        assert "Circuit 2" in html

    def test_with_metadata(self, gallery):
        """Test adding items with metadata."""
        gallery.add_circuit_graph(
            name="Test",
            description="Test",
            html_figure="<div>test</div>",
            metadata={
                "algorithm": "eap",
                "task": "ioi",
                "model": "gpt2",
            },
        )
        item = gallery.gallery_items[0]
        assert item["metadata"]["algorithm"] == "eap"

    def test_with_circuit_data(self, gallery):
        """Test adding circuit graph with circuit data."""
        circuit_data = {
            "nodes": {"A0.0": {}, "MLP0": {}},
            "edges": [["A0.0", "MLP0"]],
        }
        gallery.add_circuit_graph(
            name="Test",
            description="Test",
            html_figure="<div>test</div>",
            circuit_data=circuit_data,
        )
        item = gallery.gallery_items[0]
        assert item["circuit_data"] == circuit_data

    def test_with_top_nodes(self, gallery):
        """Test adding feature attribution with top nodes."""
        top_nodes = [("A0.0", 0.8), ("A0.1", 0.6), ("MLP0", 0.7)]
        gallery.add_feature_attribution(
            name="Test",
            description="Test",
            html_figures={"importance_bar": "<div>test</div>"},
            top_nodes=top_nodes,
        )
        item = gallery.gallery_items[0]
        assert len(item["top_nodes"]) == 3

    def test_with_tokens(self, gallery):
        """Test adding saliency comparison with tokens."""
        tokens = ["The", "quick", "brown", "fox"]
        gallery.add_saliency_comparison(
            name="Test",
            description="Test",
            html_figures={"layer_heatmaps": "<div>test</div>"},
            tokens=tokens,
        )
        item = gallery.gallery_items[0]
        assert item["tokens"] == tokens

    def test_with_summary_stats(self, gallery):
        """Test adding comparison dashboard with summary stats."""
        summary_stats = {
            "circuit_1": {"mean": 0.5, "std": 0.1},
            "circuit_2": {"mean": 0.6, "std": 0.09},
        }
        gallery.add_comparison_dashboard(
            name="Test",
            description="Test",
            html_figures={"stability": "<div>test</div>"},
            summary_stats=summary_stats,
        )
        item = gallery.gallery_items[0]
        assert item["summary_stats"] == summary_stats

    def test_html_contains_search(self, gallery):
        """Test that HTML includes search functionality."""
        gallery.add_circuit_graph(
            name="Test",
            description="Test",
            html_figure="<div>test</div>",
        )
        html = gallery.generate_html()
        assert "search" in html.lower() or "filter" in html.lower()

    def test_html_contains_metadata_timestamp(self, gallery):
        """Test that HTML includes generation timestamp."""
        html = gallery.generate_html()
        assert "generated" in html.lower() or "Generated" in html

    def test_gallery_items_are_independent(self, gallery):
        """Test that items are independent."""
        gallery.add_circuit_graph(
            name="Circuit 1",
            description="Test 1",
            html_figure="<div>test1</div>",
        )
        gallery.add_saliency_comparison(
            name="Saliency 1",
            description="Test 2",
            html_figures={"layer": "<div>test2</div>"},
        )

        # Modifying one item shouldn't affect the other
        gallery.gallery_items[0]["name"] = "Modified"
        assert gallery.gallery_items[1]["name"] == "Saliency 1"

    def test_save_metadata(self, gallery, tmp_path):
        """Verify that gallery metadata is correctly persisted to JSON."""
        gallery.add_circuit_graph("Test", "Desc", "<div></div>")
        gallery.save_metadata()

        metadata_path = Path(gallery.output_dir) / "gallery_metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            data = json.load(f)
            assert data["items"] == 1  # Check if items were counted correctly
            assert "generated" in data  # Verify timestamp existence

    def test_nested_output_directory_creation(self, tmp_path):
        """Ensure the generator creates deeply nested directories if they don't exist."""
        nested_path = tmp_path / "reports" / "2024" / "circuit_analysis"
        gallery = GalleryGenerator(output_dir=str(nested_path))

        gallery.save()
        assert (nested_path / "index.html").exists()

    def test_resilience_to_missing_optional_data(self, gallery):
        """Ensure the generator handles None/Empty inputs for optional fields without crashing."""
        # Add an evaluation report with minimal data
        gallery.add_evaluation_report(
            name="Empty Test", description="Testing robustness", evaluation_data={}  # Minimal dict
        )

        html = gallery.generate_html()
        assert "Empty Test" in html
        # Ensure it doesn't render literal 'None' in the HTML metadata section
        assert "None" not in html

    def test_html_dependency_inclusion(self, gallery):
        """Verify that Plotly and CSS are correctly injected into the header."""
        html = gallery.generate_html()

        # Verify Plotly CDN is present
        assert "cdn.plot.ly" in html
        # Verify main structural tags are closed
        assert "</html>" in html
        assert "</body>" in html
        # Verify that each plotly-container has a unique ID to avoid rendering collisions
        gallery.add_circuit_graph("C1", "D1", "fig1")
        gallery.add_circuit_graph("C2", "D2", "fig2")
        html_multi = gallery.generate_html()
        assert 'id="plot-0"' in html_multi
        assert 'id="plot-1"' in html_multi

    def test_item_type_ui_formatting(self, gallery):
        """Verify snake_case types are converted to Title Case for display."""
        gallery.add_saliency_comparison("Test", "Desc", {"layer": "<div></div>"})
        html = gallery.generate_html()

        # Logically correct check:
        # 1. The visible label MUST be Title Case
        assert "Saliency Comparison" in html

        # 2. Use a specific selector-like search to avoid CSS collisions.
        # Look for the exact span structure used in _generate_item_html
        expected_span = '<span class="item-type">Saliency Comparison</span>'
        assert expected_span in html

        # 3. Verify that the snake_case version STILL EXISTS for JS filtering
        # (The test previously thought this was a bug, but it's a feature)
        assert 'data-type="saliency_comparison"' in html
