"""Validation: JupyterWidgetSuite — instantiates and runs display routines.

Notes:
- Real Jupyter rendering can't happen headlessly, but the suite's
  display calls should not raise on real fixture inputs.
- We treat 'doesn't crash' as success here. Manual eyeball in a real
  notebook is the human-side verification.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "06_jupyter_widgets"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()
    from circuitkit.visualize import JupyterWidgetSuite, display_circuit_analysis

    t0 = time.time()
    suite = JupyterWidgetSuite()
    # display_circuit_analysis is a top-level convenience function
    try:
        display_circuit_analysis(
            circuit=fx["graph"],
            node_scores=fx["node_scores"],
        )
        display_ok = True
    except Exception as e:
        display_ok = False
        display_err = f"{type(e).__name__}: {str(e)[:120]}"
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.visualize.JupyterWidgetSuite",
        "input": {
            "n_nodes": len(fx["graph"]["nodes"]),
            "n_edges": len(fx["graph"]["edges"]),
        },
        "output": {},
        "metrics": {
            "wall_seconds": round(elapsed, 3),
            "display_ok": display_ok,
        },
        "status": "WORKING" if display_ok else "NEEDS-FIX",
    }
    if not display_ok:
        status["error"] = display_err
    write_status(out_dir, status)
    print()
    print(f"JupyterWidgetSuite — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  display_circuit_analysis: {'OK' if display_ok else 'FAILED'}")
    if not display_ok:
        print(f"    -> {display_err}")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Status:       {status['status']}")
    return 0 if display_ok else 1

if __name__ == "__main__":
    sys.exit(main())
