"""Compare the 6 stable discovery algorithms shipped in this release.

Runs each algorithm in sequence on GPT-2 / IOI and prints the top-5 nodes
plus wall-clock time. Useful as a quick sanity check that every algorithm
is installed correctly and as a visual diff of what each one selects.

Algorithm notes (see docs/guides for details):
    - eap, eap-ig, eap-gp, acdc : standard EAP-family, need corrupted pairs
    - cdt                       : node-level only (enforced by `level="node"` here)
    - ibcircuit                 : does not need corrupted pairs; uses its own
                                  training loop (num_epochs)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from circuitkit.api import discover_circuit


ALGOS = ["eap", "eap-ig", "eap-gp", "acdc", "cdt", "ibcircuit"]


def discover(algo: str, num_examples: int = 16) -> dict:
    out_dir = Path("./circuits")
    out_dir.mkdir(exist_ok=True)
    artifact = out_dir / f"{algo}_ioi.pt"
    cfg = {
        "model": {"name": "gpt2", "precision": "bfloat16"},
        "discovery": {
            "algorithm": algo, "task": "ioi", "level": "node",
            "batch_size": 1,
            "data_params": {"num_examples": num_examples},
        },
        "pruning": {"target_sparsity": 0.3, "scope": "heads"},
        "output_path": str(artifact),
    }
    if algo in ("eap-ig", "eap-gp"):
        cfg["discovery"]["ig_steps"] = 3
    if algo == "ibcircuit":
        cfg["discovery"]["num_epochs"] = 1000
        cfg["discovery"]["scope"] = "heads"
    t = time.time()
    discover_circuit(cfg)
    sidecar = artifact.parent / (artifact.stem + "_scores.json")
    scores = json.loads(sidecar.read_text())["node_scores"]
    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t, 2),
        "top5": [n for n, _ in sorted(
            scores.items(), key=lambda kv: abs(kv[1]), reverse=True
        )[:5]],
    }


def main() -> None:
    print(f"{'algorithm':12s}  {'time':>7s}  top-5 nodes")
    print("-" * 80)
    for algo in ALGOS:
        try:
            r = discover(algo)
            print(f"{r['algorithm']:12s}  {r['wall_seconds']:>6.1f}s  {r['top5']}")
        except Exception as exc:  # noqa: BLE001
            print(f"{algo:12s}  ERR  {type(exc).__name__}: {str(exc)[:60]}")


if __name__ == "__main__":
    main()