"""Validation: FeatureSaliencyVisualizer on real GPT-2 IOI node attributions."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "03_feature_saliency"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()
    from circuitkit.visualize import FeatureSaliencyVisualizer

    t0 = time.time()
    v = FeatureSaliencyVisualizer(
        node_attributions=fx["node_scores"],
        node_names=list(fx["node_scores"].keys()),
    )
    fig1 = v.plot_importance_bar()
    fig2 = v.plot_network_saliency()
    html_path = out_dir / "feature_saliency.html"
    v.export_to_html(str(html_path))
    json_path = out_dir / "feature_saliency.json"
    v.export_to_json(str(json_path))
    summary = v.get_attribution_summary()
    top = v.get_top_nodes(k=5)
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.visualize.FeatureSaliencyVisualizer",
        "input": {"n_nodes": len(fx["node_scores"])},
        "output": {
            "html": str(html_path), "html_bytes": html_path.stat().st_size,
            "json": str(json_path), "json_bytes": json_path.stat().st_size,
        },
        "metrics": {
            "wall_seconds": round(elapsed, 3),
            "top_5_nodes": [list(t) if isinstance(t, tuple) else t for t in top],
            "summary": summary,
        },
        "status": "WORKING" if html_path.stat().st_size > 1000 else "BROKEN",
    }
    write_status(out_dir, status)
    print()
    print(f"FeatureSaliencyVisualizer — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  Input:        {len(fx['node_scores'])} node attributions")
    print(f"  Top-3 nodes:  {top[:3]}")
    print(f"  Output HTML:  {html_path.relative_to(Path.cwd())}  "
          f"({html_path.stat().st_size:,} bytes)")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Status:       {status['status']}")
    return 0 if status["status"] == "WORKING" else 1

if __name__ == "__main__":
    sys.exit(main())
