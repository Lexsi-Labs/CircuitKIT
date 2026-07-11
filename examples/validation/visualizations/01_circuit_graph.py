"""Validation: CircuitGraphVisualizer on real GPT-2 IOI circuit.

Consumes the cached real circuit from _common.py and renders the graph
both as HTML (for human eyeball) and as JSON (for downstream tools).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# import sibling _common.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "01_circuit_graph"


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()

    from circuitkit.visualize import CircuitGraphVisualizer

    t0 = time.time()
    v = CircuitGraphVisualizer(graph=fx["graph"], scores=fx["circuit_scores"])
    html_path = out_dir / "circuit_graph.html"
    v.to_html(str(html_path))
    json_path = out_dir / "circuit_graph_data.json"
    v.export_graph_data(str(json_path))
    degree_stats = v.get_node_degree_stats()
    top_k = v.get_top_k_nodes(k=5)
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.visualize.CircuitGraphVisualizer",
        "input": {
            "model": fx["meta"]["model"],
            "task": fx["meta"]["task"],
            "algorithm": fx["meta"]["algorithm"],
            "n_nodes": len(fx["node_scores"]),
            "n_edges": len(fx["graph"]["edges"]),
        },
        "output": {
            "html": str(html_path),
            "html_bytes": html_path.stat().st_size,
            "json": str(json_path),
            "json_bytes": json_path.stat().st_size,
        },
        "metrics": {
            "wall_seconds": round(elapsed, 3),
            "degree_stats": degree_stats,
            "top_5_nodes": [list(t) if isinstance(t, tuple) else t for t in top_k],
        },
        "status": "WORKING" if html_path.stat().st_size > 1000 else "BROKEN",
    }
    write_status(out_dir, status)

    print()
    print(f"CircuitGraphVisualizer — {fx['meta']['model']} {fx['meta']['task']} "
          f"{fx['meta']['algorithm']}")
    print(f"  Input:        {len(fx['node_scores'])} nodes, "
          f"{len(fx['graph']['edges'])} edges")
    print(f"  Output HTML:  {html_path.relative_to(Path.cwd())}  "
          f"({html_path.stat().st_size:,} bytes)")
    print(f"  Output JSON:  {json_path.relative_to(Path.cwd())}")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Status:       {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
