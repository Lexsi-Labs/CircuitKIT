"""Validation: ComparisonDashboard with 3 perturbed copies of the real circuit."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "05_comparison_dashboard"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()
    from circuitkit.visualize import ComparisonDashboard

    rng = np.random.default_rng(42)
    base = fx["node_scores"]
    circuits = {
        "eap-ig":  base,
        "eap-ig (seed=143)": {k: float(v + rng.normal(0, 0.05)) for k, v in base.items()},
        "eap-ig (seed=256)": {k: float(v + rng.normal(0, 0.10)) for k, v in base.items()},
    }

    t0 = time.time()
    c = ComparisonDashboard(circuits=circuits, comparison_type="stability")
    c.plot_stability_heatmap()
    c.plot_correlation_matrix()
    c.plot_distribution_comparison()
    html_path = out_dir / "comparison.html"
    c.export_to_html(str(html_path))
    json_path = out_dir / "comparison.json"
    c.export_to_json(str(json_path))
    stats = c.get_summary_stats()
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.visualize.ComparisonDashboard",
        "input": {
            "n_circuits": len(circuits),
            "labels": list(circuits.keys()),
        },
        "output": {
            "html": str(html_path), "html_bytes": html_path.stat().st_size,
            "json": str(json_path), "json_bytes": json_path.stat().st_size,
        },
        "metrics": {"wall_seconds": round(elapsed, 3)},
        "status": "WORKING" if html_path.stat().st_size > 1000 else "BROKEN",
    }
    write_status(out_dir, status)
    print()
    print(f"ComparisonDashboard — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  Input:        {len(circuits)} circuits "
          f"({list(circuits.keys())})")
    print(f"  Output HTML:  {html_path.relative_to(Path.cwd())}  "
          f"({html_path.stat().st_size:,} bytes)")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Status:       {status['status']}")
    return 0 if status["status"] == "WORKING" else 1

if __name__ == "__main__":
    sys.exit(main())
