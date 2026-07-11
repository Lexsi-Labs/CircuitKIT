"""Validation: CircuitEditor on real GPT-2 IOI circuit."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "04_circuit_editor"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()
    from circuitkit.visualize import CircuitEditor

    t0 = time.time()
    editor = CircuitEditor(
        initial_circuit=fx["graph"],
        node_names=list(fx["graph"]["nodes"].keys()),
    )
    circuit = editor.get_circuit()
    changes = editor.get_changes()
    json_path = out_dir / "edited_circuit.json"
    editor.save_circuit(str(json_path))
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.visualize.CircuitEditor",
        "input": {
            "n_nodes": len(fx["graph"]["nodes"]),
            "n_edges": len(fx["graph"]["edges"]),
        },
        "output": {
            "json": str(json_path),
            "json_bytes": json_path.stat().st_size,
        },
        "metrics": {
            "wall_seconds": round(elapsed, 3),
            "n_changes_recorded": len(changes) if hasattr(changes, "__len__") else 0,
        },
        "status": "WORKING" if json_path.stat().st_size > 100 else "BROKEN",
    }
    write_status(out_dir, status)
    print()
    print(f"CircuitEditor — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  Input:        {len(fx['graph']['nodes'])} nodes, "
          f"{len(fx['graph']['edges'])} edges")
    print(f"  Output JSON:  {json_path.relative_to(Path.cwd())}  "
          f"({json_path.stat().st_size:,} bytes)")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Status:       {status['status']}")
    return 0 if status["status"] == "WORKING" else 1

if __name__ == "__main__":
    sys.exit(main())
