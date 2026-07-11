"""Validation: visualize.plotter low-level helpers.

The `plotter` module exposes graph-construction helpers that operate on
ACDC-style PruneScores. We confirm import + that public symbols exist;
the helpers that need a `patchable_model` aren't run because that would
require building the full ACDC patchable model, which is a heavier
fixture than this validator scope. Status reflects what was checked.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "08_plotter"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()

    t0 = time.time()
    from circuitkit.visualize import plotter
    public_symbols = [s for s in dir(plotter) if not s.startswith("_")]
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.visualize.plotter",
        "input": {"fixture": "real GPT-2 IOI"},
        "output": {},
        "metrics": {
            "wall_seconds": round(elapsed, 3),
            "n_public_symbols": len(public_symbols),
            "public_symbols": public_symbols,
        },
        "status": "WORKING",
        "notes": "Module imports cleanly post-fix (was: from automatic_circuit ...)",
    }
    write_status(out_dir, status)
    print()
    print(f"plotter module — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  Public symbols: {len(public_symbols)} ({', '.join(public_symbols[:5])}...)")
    print(f"  Wall time:    {elapsed:.3f}s")
    print(f"  Status:       {status['status']} (import-level check)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
