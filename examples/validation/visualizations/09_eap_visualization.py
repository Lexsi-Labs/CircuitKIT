"""Validation: backends.eap.visualization color helpers."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "09_eap_visualization"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()
    from circuitkit.backends.eap import visualization as eap_viz

    t0 = time.time()
    # Exercise the exposed color helpers with synthetic-but-real-shaped inputs.
    color_a = eap_viz.get_color('q', 0.8)  # high-importance Q-edge
    color_b = eap_viz.get_color('v', 0.1)  # low-importance V-edge
    rand1 = eap_viz.generate_random_color('viridis')
    rand2 = eap_viz.generate_random_color('plasma')
    edge_types_n = len(eap_viz.EDGE_TYPE_COLORS)
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.backends.eap.visualization",
        "input": {"sample_scores": [0.8, 0.1]},
        "output": {},
        "metrics": {
            "wall_seconds": round(elapsed, 3),
            "color_for_0.8": str(color_a)[:60],
            "color_for_0.1": str(color_b)[:60],
            "edge_types_count": edge_types_n,
        },
        "status": "WORKING",
    }
    write_status(out_dir, status)
    print()
    print(f"eap.visualization helpers — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  EDGE_TYPE_COLORS: {edge_types_n} types")
    print(f"  get_color(0.8):   {color_a}")
    print(f"  get_color(0.1):   {color_b}")
    print(f"  Wall time:    {elapsed:.3f}s")
    print(f"  Status:       {status['status']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
