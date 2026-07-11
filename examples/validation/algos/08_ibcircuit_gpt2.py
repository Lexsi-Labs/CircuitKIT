"""Validation: IBCircuit (information-bottleneck circuit discovery) on GPT-2 IOI.

IBCircuit is the non-EAP-family discovery method in CircuitKit. It trains
per-component IB noise gates against a KL-vs-original objective rather
than computing a gradient × activation product. End-to-end real-model
run. No mocks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _algos_common import (  # noqa: E402
    run_algo, top_k_nodes, make_results_dir, write_status,
)

SCRIPT_NAME = "08_ibcircuit_gpt2"
ALGO = "ibcircuit"


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)

    try:
        # Short epochs for the smoke test; production sweeps use 500-1000.
        result = run_algo(
            ALGO, num_examples=32,
            extra_discovery={"num_epochs": 200, "scope": "heads"},
        )
    except Exception as exc:
        write_status(out_dir, {
            "script": SCRIPT_NAME, "module": f"discover_circuit[{ALGO}]",
            "status": "BROKEN", "error": repr(exc),
        })
        print(f"\n{ALGO} — BROKEN: {exc}")
        return 1

    top10 = top_k_nodes(result["node_scores"], k=10)

    status = {
        "script": SCRIPT_NAME,
        "module": f"discover_circuit[{ALGO}]",
        "input": {
            "model": result["model"], "task": result["task"],
            "algorithm": ALGO, "num_examples": 32, "num_epochs": 200,
        },
        "output": {
            "artifact": result["artifact_path"],
            "scores_json": result["scores_path"],
            "n_nodes_total": result["n_nodes_total"],
            "n_nodes_pruned": result["n_nodes_pruned"],
        },
        "metrics": {
            "wall_seconds": result["wall_seconds"],
            "top_10_nodes": top10,
        },
        "status": (
            "WORKING"
            if result["n_nodes_total"] > 0 and result["n_nodes_pruned"] > 0
            else "BROKEN"
        ),
    }
    write_status(out_dir, status)

    print(f"\n{ALGO} on {result['model']} {result['task']}")
    print(f"  Total nodes scored:   {result['n_nodes_total']}")
    print(f"  Pruned (sp=0.1):      {result['n_nodes_pruned']}")
    print(f"  Wall time:            {result['wall_seconds']}s")
    print(f"  Top-3:                {[(n['node'], round(n['score'], 4)) for n in top10[:3]]}")
    print(f"  Status:               {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
