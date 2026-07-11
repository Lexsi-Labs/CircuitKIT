"""Validation: GalleryGenerator — multi-figure gallery on real GPT-2 IOI."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "07_gallery_generator"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()
    from circuitkit.visualize import (
        GalleryGenerator, FeatureSaliencyVisualizer, CircuitGraphVisualizer,
    )

    t0 = time.time()
    g = GalleryGenerator()

    # 1. feature attribution
    feat_v = FeatureSaliencyVisualizer(
        node_attributions=fx["node_scores"],
        node_names=list(fx["node_scores"].keys()),
    )
    feat_html = (out_dir / "feat_inline.html")
    feat_v.export_to_html(str(feat_html))
    g.add_feature_attribution(
        name="GPT-2 IOI EAP-IG attributions",
        description="Real EAP-IG node attributions on GPT-2 IOI",
        html_figures={"importance_bar": feat_html.read_text()[:50000]},
        top_nodes=feat_v.get_top_nodes(k=5),
    )

    # 2. circuit graph
    graph_v = CircuitGraphVisualizer(graph=fx["graph"], scores=fx["circuit_scores"])
    graph_html = out_dir / "graph_inline.html"
    graph_v.to_html(str(graph_html))
    g.add_circuit_graph(
        name="GPT-2 IOI circuit graph",
        description="Pruned 14 of 144 attention heads at α=0.1",
        html_figure=graph_html.read_text()[:50000],
        circuit_data={
            "n_nodes": len(fx["graph"]["nodes"]),
            "n_edges": len(fx["graph"]["edges"]),
        },
    )

    # finalize
    final_html = out_dir / "gallery.html"
    g.save(str(final_html))
    g.save_metadata()
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.visualize.GalleryGenerator",
        "input": {"n_sections": 2, "model": fx["meta"]["model"]},
        "output": {
            "html": str(final_html), "html_bytes": final_html.stat().st_size,
        },
        "metrics": {"wall_seconds": round(elapsed, 3)},
        "status": "WORKING" if final_html.stat().st_size > 1000 else "BROKEN",
    }
    write_status(out_dir, status)
    print()
    print(f"GalleryGenerator — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  Input:        2 sections (feature_attribution, circuit_graph)")
    print(f"  Output HTML:  {final_html.relative_to(Path.cwd())}  "
          f"({final_html.stat().st_size:,} bytes)")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Status:       {status['status']}")
    return 0 if status["status"] == "WORKING" else 1

if __name__ == "__main__":
    sys.exit(main())
