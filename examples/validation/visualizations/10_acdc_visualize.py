"""Validation: backends.acdc.visualize.

Module-level import + symbol presence check. Running the actual draw_seq /
plotting helpers requires building a real `PatchableModel`, which is much
heavier than this validator scope. We treat 'imports + exposes expected
symbols' as the success bar; downstream Q1 cells exercise the full path.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "10_acdc_visualize"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()
    from circuitkit.backends.acdc import visualize as acdc_viz

    expected = {"Edge", "Node", "PruneScores", "PatchableModel"}
    t0 = time.time()
    public = {s for s in dir(acdc_viz) if not s.startswith("_")}
    missing = expected - public
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.backends.acdc.visualize",
        "input": {"check": "import + symbols"},
        "output": {},
        "metrics": {
            "wall_seconds": round(elapsed, 3),
            "n_public_symbols": len(public),
            "expected_present": list(expected - missing),
            "expected_missing": list(missing),
        },
        "status": "WORKING" if not missing else "BROKEN",
    }
    write_status(out_dir, status)
    print()
    print(f"acdc.visualize — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  Public symbols: {len(public)}")
    print(f"  Expected found: {len(expected - missing)}/{len(expected)}")
    if missing:
        print(f"  Missing:        {missing}")
    print(f"  Wall time:    {elapsed:.3f}s")
    print(f"  Status:       {status['status']} (import-level check)")
    return 0 if not missing else 1

if __name__ == "__main__":
    sys.exit(main())
