"""Validation: ActivationSaliencyVisualizer on real GPT-2 IOI activations."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "02_activation_saliency"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()
    from circuitkit.visualize import ActivationSaliencyVisualizer

    t0 = time.time()
    v = ActivationSaliencyVisualizer(
        activations=fx["activations"], tokens=fx["tokens"],
        layer_names=list(fx["activations"].keys()),
    )
    figs = v.plot_layer_heatmaps()
    fig = v.plot_aggregate_heatmap()
    fig = v.plot_layer_comparison()
    html_path = out_dir / "saliency.html"
    v.export_to_html(str(html_path))
    json_path = out_dir / "saliency.json"
    v.export_to_json(str(json_path))
    summary = v.get_saliency_summary()
    top = v.get_top_tokens(layer=list(fx["activations"].keys())[0], k=5)
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.visualize.ActivationSaliencyVisualizer",
        "input": {"layers": len(fx["activations"]), "tokens": len(fx["tokens"])},
        "output": {
            "html": str(html_path), "html_bytes": html_path.stat().st_size,
            "json": str(json_path), "json_bytes": json_path.stat().st_size,
        },
        "metrics": {
            "wall_seconds": round(elapsed, 3),
            "n_layer_heatmaps": len(figs) if hasattr(figs, "__len__") else 1,
            "top_tokens_layer0": [list(t) if isinstance(t, tuple) else t for t in top],
        },
        "status": "WORKING" if html_path.stat().st_size > 1000 else "BROKEN",
    }
    write_status(out_dir, status)
    print()
    print(f"ActivationSaliencyVisualizer — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  Input:        {len(fx['activations'])} layers x {len(fx['tokens'])} tokens")
    print(f"  Output HTML:  {html_path.relative_to(Path.cwd())}  "
          f"({html_path.stat().st_size:,} bytes)")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Status:       {status['status']}")
    return 0 if status["status"] == "WORKING" else 1

if __name__ == "__main__":
    sys.exit(main())
