"""Shared helpers for the algo-layer validation suite.

Every script here runs `discover_circuit` end-to-end on real GPT-2 IOI with
one of the promoted top-level algorithm keys, parses the resulting
CircuitScores artifact, and writes a structured status JSON for the
runner. No mocks. No synthetic scores.

Why a separate common module from `validation/_common.py`:
- _common.py caches a single eap-ig fixture for the visualizations layer.
- Here we need to RUN discovery many times under different algorithm
  settings, so we expose a `run_algo()` helper that takes the algo name
  and returns parsed scores + timing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the library importable when running scripts standalone.
_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Re-use the runner's results-dir + status writer.
sys.path.insert(0, str(_REPO / "validation"))
from _common import make_results_dir, write_status  # noqa: E402

# Cache directory for algo discovery artifacts (one .pt + .json per algo run).
ALGO_CACHE = _REPO / "validation" / "_cache" / "algos"
ALGO_CACHE.mkdir(parents=True, exist_ok=True)


def run_algo(
    algorithm: str,
    *,
    model: str = "gpt2",
    task: str = "ioi",
    num_examples: int = 32,
    batch_size: int = 1,
    ig_steps: int = 2,
    target_sparsity: float = 0.1,
    scope: str = "heads",
    extra_discovery: Optional[Dict[str, Any]] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Run discover_circuit for the given algorithm; return parsed result.

    Returns dict with:
      - algorithm, model, task
      - pruned_nodes: List[str]
      - node_scores: Dict[str, float] (full attribution map)
      - artifact_path, scores_path
      - wall_seconds: float
      - n_nodes_total, n_nodes_pruned
    """
    from circuitkit.api import discover_circuit

    artifact_path = ALGO_CACHE / f"{algorithm}_{task}_{model.replace('/', '_')}.pt"
    scores_path = artifact_path.parent / (artifact_path.stem + "_scores.json")

    discovery_cfg = {
        "algorithm": algorithm,
        "task": task,
        "level": "node",
        "batch_size": batch_size,
        "data_params": {"num_examples": num_examples},
    }
    # eap-exact does NOT use ig_steps; eap-ig and eap-ig-activations do.
    if algorithm in ("eap-ig", "eap-ig-activations"):
        discovery_cfg["ig_steps"] = ig_steps
    if extra_discovery:
        discovery_cfg.update(extra_discovery)

    config = {
        "model": {"name": model, "precision": "float32"},
        "discovery": discovery_cfg,
        "pruning": {"target_sparsity": target_sparsity, "scope": scope},
        "output_path": str(artifact_path),
    }

    t0 = time.time()
    if force_rerun or not scores_path.exists():
        pruned_nodes = discover_circuit(config)
        wall = time.time() - t0
        # discover_circuit writes the .pt artifact + JSON sidecar.
    else:
        # Cached: still need pruned_nodes from the .pt file.
        import torch
        pruned_nodes = torch.load(artifact_path, weights_only=False, map_location="cpu")
        wall = 0.0

    # Parse the unified CircuitScores JSON (single source of truth).
    if not scores_path.exists():
        raise RuntimeError(
            f"discover_circuit did not write {scores_path}; "
            f"the algo dispatcher may not be saving CircuitScores."
        )
    scores_blob = json.loads(scores_path.read_text())
    node_scores: Dict[str, float] = {
        str(k): float(v) for k, v in scores_blob.get("node_scores", {}).items()
    }

    return {
        "algorithm": algorithm,
        "model": model,
        "task": task,
        "pruned_nodes": list(pruned_nodes) if not isinstance(pruned_nodes, dict) else [],
        "node_scores": node_scores,
        "artifact_path": str(artifact_path),
        "scores_path": str(scores_path),
        "wall_seconds": round(wall, 2),
        "n_nodes_total": len(node_scores),
        "n_nodes_pruned": len(pruned_nodes) if not isinstance(pruned_nodes, dict) else 0,
    }


def top_k_nodes(node_scores: Dict[str, float], k: int = 10) -> List[Dict[str, Any]]:
    """Return top-k nodes by absolute score (descending)."""
    sorted_items = sorted(node_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [{"node": n, "score": float(s)} for n, s in sorted_items[:k]]


def jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard overlap of two lists treated as sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def top_k_set(node_scores: Dict[str, float], k: int) -> List[str]:
    """Top-k node names by absolute score."""
    return [n for n, _ in sorted(
        node_scores.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:k]]
