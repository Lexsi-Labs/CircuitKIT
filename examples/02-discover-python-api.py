#!/usr/bin/env python3
"""
01 - Circuit discovery via the Python API.

Runs EAP-IG circuit discovery on GPT-2 with the built-in IOI task, then
loads the resulting artifact back from disk. Self-contained and CPU-friendly.

Run:
    python examples/02-discover-python-api.py
"""

import os

from circuitkit.api import discover_circuit, load_circuit

# Discovery is configured with a single dict (or a path to a YAML file).
# Required top-level keys: model, discovery, pruning.
CONFIG = {
    "model": {"name": "gpt2", "precision": "bfloat16"},
    "discovery": {
        "algorithm": "eap-ig",            # default, STABLE-tier algorithm
        "task": "ioi",                    # built-in task; data is auto-generated
        "level": "node",                  # "node" (heads/MLPs) or "neuron"
        "batch_size": 2,
        "ig_steps": 2,                    # integrated-gradients steps (small for demo)
        "data_params": {"num_examples": 16},
    },
    "pruning": {"target_sparsity": 0.2, "scope": "heads"},
    "output_path": "./results/example_circuit.pt",
}


def main():
    os.makedirs("./results", exist_ok=True)

    print("Running EAP-IG circuit discovery on GPT-2 / IOI ...")
    pruned_nodes = discover_circuit(CONFIG)
    # Node-level discovery returns a list of node names to prune.
    print(f"Discovery complete. {len(pruned_nodes)} nodes selected for pruning.")
    print("First 5 nodes:", pruned_nodes[:5])

    # discover_circuit also writes a `_scores.pt` / `_scores.json` side-car
    # next to the artifact -- evaluate_circuit and the applications use it.
    artifact = CONFIG["output_path"]
    scores = artifact.replace(".pt", "_scores.json")
    print(f"Artifact:       {artifact}")
    print(f"Scores side-car: {scores}")

    # load_circuit reads the artifact back from disk.
    reloaded = load_circuit(artifact)
    print(f"Re-loaded circuit with {len(reloaded)} nodes.")

    print("\nDone. Continue with 05-evaluate-faithfulness.py to score this circuit.")


if __name__ == "__main__":
    main()
