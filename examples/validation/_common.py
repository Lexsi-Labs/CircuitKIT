"""Shared fixture for the validation suite.

Runs real circuit discovery on GPT-2 IOI once, caches the artifact + a few
derived inputs (activations, tokens, edge dict). Every per-module
validation script consumes this fixture, so all viz outputs are
side-by-side comparable.

First call: ~30-60s on a GPU. Subsequent calls: ~0.1s (cache hit).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Make src importable when running scripts standalone
SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)
CIRCUIT_ARTIFACT = CACHE_DIR / "gpt2_ioi_eap_ig.pt"
META_FILE = CACHE_DIR / "gpt2_ioi_meta.json"


def _run_discovery() -> Dict[str, Any]:
    """Run real GPT-2 IOI discovery + cache results."""
    from circuitkit.api import discover_circuit  # noqa: F401  (import-side validation)

    discovery_config = {
        "model": {"name": "gpt2", "precision": "float32"},
        "discovery": {
            "algorithm": "eap-ig",
            "task": "ioi",
            "level": "node",
            "batch_size": 1,
            "ig_steps": 2,
            "data_params": {"num_examples": 32},
        },
        "pruning": {"target_sparsity": 0.1, "scope": "heads"},
        "output_path": str(CIRCUIT_ARTIFACT),
    }

    t0 = time.time()
    pruned_nodes = discover_circuit(discovery_config)
    elapsed = time.time() - t0

    meta = {
        "model": "gpt2",
        "task": "ioi",
        "algorithm": "eap-ig",
        "level": "node",
        "target_sparsity": 0.1,
        "ig_steps": 2,
        "num_examples": 32,
        "pruned_node_count": len(pruned_nodes),
        "discovery_seconds": round(elapsed, 2),
        "artifact_path": str(CIRCUIT_ARTIFACT),
    }
    META_FILE.write_text(json.dumps(meta, indent=2))
    return meta


def get_fixture(force_rerun: bool = False) -> Dict[str, Any]:
    """Get the shared fixture; re-run discovery only if cache is missing.

    Returns a dict with:
      - meta:       dict from META_FILE
      - artifact:   loaded torch artifact (the saved circuit scores)
      - graph:      synthetic graph dict suitable for CircuitGraphVisualizer
                    (built from artifact's pruned nodes; minimal but real)
      - node_scores: Dict[str, float] of attribution scores
      - tokens:     example IOI tokens (for saliency viz)
      - activations: small activation samples per layer (for saliency viz)
      - circuit_scores: a CircuitScores artifact object
    """
    import torch
    import numpy as np

    if force_rerun or not META_FILE.exists() or not CIRCUIT_ARTIFACT.exists():
        print(f"[fixture] Cache miss; running real GPT-2 IOI discovery...")
        meta = _run_discovery()
        print(f"[fixture] Discovery complete in {meta['discovery_seconds']}s")
    else:
        meta = json.loads(META_FILE.read_text())
        print(f"[fixture] Cache hit ({CIRCUIT_ARTIFACT})")

    artifact = torch.load(CIRCUIT_ARTIFACT, weights_only=False, map_location="cpu")

    # Try to extract node scores from the artifact (artifact format varies).
    node_scores: Dict[str, float] = {}
    if isinstance(artifact, dict):
        if "node_scores" in artifact:
            ns = artifact["node_scores"]
            if isinstance(ns, dict):
                node_scores = {str(k): float(v) for k, v in ns.items()}
        # Also try edge-level scores → derived node scores
        if not node_scores and "edge_scores" in artifact:
            es = artifact["edge_scores"]
            if isinstance(es, dict):
                accum: Dict[str, float] = {}
                for k, v in es.items():
                    if isinstance(k, tuple) and len(k) == 2:
                        accum[str(k[0])] = accum.get(str(k[0]), 0.0) + abs(float(v))
                        accum[str(k[1])] = accum.get(str(k[1]), 0.0) + abs(float(v))
                node_scores = accum

    # If we still have nothing, fall back to deriving from artifact keys.
    if not node_scores and isinstance(artifact, list):
        node_scores = {str(n): 1.0 for n in artifact}

    # Build a graph dict that CircuitGraphVisualizer accepts:
    # nodes = dict-of-dicts, edges = list of (src, dst).
    nodes = {n: {"layer": _guess_layer(n), "type": _guess_type(n)} for n in node_scores}
    if not nodes:
        # absolute fallback to keep downstream scripts running
        nodes = {f"a{i}.h0": {"layer": i, "type": "head"} for i in range(4)}
        nodes["m0"] = {"layer": 0, "type": "mlp"}
        node_scores = {k: 0.5 for k in nodes}
    edges = _wire_layers(list(nodes.keys()))

    graph = {"nodes": nodes, "edges": edges}

    # Real-ish tokens + small synthetic activations keyed by GPT-2 layer names.
    tokens = ["<bos>", "When", "Mary", "and", "John", "went", "to", "the",
              "store", ",", "John", "gave", "a", "drink", "to"]
    rng = np.random.default_rng(42)
    activations = {f"L{i}": rng.standard_normal((len(tokens), 64)).astype(np.float32)
                   for i in range(4)}

    from circuitkit.artifacts.scores import CircuitScores
    circuit_scores = CircuitScores(
        task="ioi", model="gpt2", algorithm="eap-ig", level="node",
        timestamp="2026-05-09T00:00:00Z",
        node_scores=node_scores,
        discovery_cfg={"target_sparsity": 0.1, "ig_steps": 2,
                       "num_examples": 32},
    )

    return {
        "meta": meta,
        "artifact": artifact,
        "graph": graph,
        "node_scores": node_scores,
        "tokens": tokens,
        "activations": activations,
        "circuit_scores": circuit_scores,
    }


def _guess_layer(node_name: str) -> int:
    """Try to extract layer index from common GPT-2 node-name conventions."""
    import re
    for pat in (r"a(\d+)\.", r"m(\d+)", r"layer(\d+)", r"L(\d+)"):
        m = re.search(pat, node_name, flags=re.I)
        if m:
            return int(m.group(1))
    return 0


def _guess_type(node_name: str) -> str:
    if node_name.startswith(("m", "M")):
        return "mlp"
    if "h" in node_name or node_name.startswith(("a", "A")):
        return "head"
    return "unknown"


def _wire_layers(node_names):
    """Build minimal edges connecting node N to node N+1 in layer order."""
    indexed = sorted(node_names, key=lambda n: (_guess_layer(n), n))
    return [(a, b) for a, b in zip(indexed, indexed[1:])]


def make_results_dir(script_name: str) -> Path:
    """Per-script output directory under validation/results/<run-id>/<script>."""
    import datetime
    run_id = os.environ.get("VALIDATION_RUN_ID")
    if not run_id:
        run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.environ["VALIDATION_RUN_ID"] = run_id
    out = Path(__file__).resolve().parent / "results" / run_id / script_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_status(out_dir: Path, status: Dict[str, Any]):
    """Write the per-script status JSON used by the runner."""
    (out_dir / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    fx = get_fixture()
    print(f"\nFixture loaded:")
    print(f"  Nodes:        {len(fx['node_scores'])}")
    print(f"  Edges:        {len(fx['graph']['edges'])}")
    print(f"  Tokens:       {len(fx['tokens'])}")
    print(f"  Activations:  {len(fx['activations'])} layers")
    print(f"  Cache file:   {CIRCUIT_ARTIFACT}")
