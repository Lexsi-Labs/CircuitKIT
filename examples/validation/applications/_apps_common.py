"""Shared helpers for the applications validation suite.

For each application we compare the same set of discovery algorithms,
load the cached CircuitScores from the algo benchmark cache, apply the
application (pruning / LoRA healing / quantization tier / unlearning),
and measure a downstream effect on a real task. No mocks; no synthetic
discovery. If the cache has no scores for an (algo, task, model) triple,
the cell falls back to running discover_circuit() so the suite is
self-contained.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Library importable
_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Reuse the runner status writer
sys.path.insert(0, str(_REPO / "validation"))
from _common import make_results_dir, write_status  # noqa: E402

# Reuse the benchmark layer's discovery cache so we do not re-discover.
_BENCH_CACHE = _REPO / "validation" / "_cache" / "benchmark"
_ALGO_CACHE = _REPO / "validation" / "_cache" / "algos"
_APP_CACHE = _REPO / "validation" / "_cache" / "applications"
_APP_CACHE.mkdir(parents=True, exist_ok=True)


# Default algorithm matrix for apps cells. Mirrors the benchmark suite so
# the per-algo comparison table covers every algorithm CircuitKit ships.
DEFAULT_ALGOS: List[str] = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]


def get_or_run_discovery(
    algorithm: str, model: str, task: str,
    num_examples: int = 32, batch_size: int = 1, ig_steps: int = 3,
    target_sparsity: float = 0.1, scope: str = "heads",
    precision: str = "float32", ibcircuit_epochs: int = 200,
) -> Dict[str, Any]:
    """Return CircuitScores dict for (algo, model, task), running discovery
    if not already cached by either the algos or benchmark layers."""
    from circuitkit.api import discover_circuit
    from circuitkit.artifacts.scores import CircuitScores

    cell_id = f"{algorithm}_{task}_{model.replace('/', '_')}"
    candidates = [
        _BENCH_CACHE / f"{cell_id}_scores.json",
        _ALGO_CACHE / f"{cell_id}_scores.json",
    ]
    scores_path = next((p for p in candidates if p.exists()), None)

    if scores_path is None:
        artifact_path = _APP_CACHE / f"{cell_id}.pt"
        scores_path = artifact_path.parent / (artifact_path.stem + "_scores.json")
        discovery: Dict[str, Any] = {
            "algorithm": algorithm,
            "task": task,
            "level": "node",
            "batch_size": batch_size,
            "data_params": {"num_examples": num_examples},
            "model_name": model,
        }
        if algorithm in ("eap-ig", "eap-ig-activations"):
            discovery["ig_steps"] = ig_steps
        if algorithm == "eap-gp":
            discovery["ig_steps"] = max(3, ig_steps)
        if algorithm == "ibcircuit":
            discovery["scope"] = scope
            discovery["num_epochs"] = ibcircuit_epochs

        config = {
            "model": {"name": model, "precision": precision},
            "discovery": discovery,
            "pruning": {"target_sparsity": target_sparsity, "scope": scope},
            "output_path": str(artifact_path),
        }
        discover_circuit(config)

    cs = CircuitScores.from_json(scores_path)
    return {
        "scores": cs,
        "scores_path": str(scores_path),
        "node_scores": cs.node_scores,
    }


def top_k_node_set(node_scores: Dict[str, float], k: int) -> List[str]:
    return [n for n, _ in sorted(
        node_scores.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:k]]


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
