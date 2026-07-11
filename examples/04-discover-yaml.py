#!/usr/bin/env python3
"""
03 - Circuit discovery from a YAML task configuration.

Demonstrates running discovery on a custom CSV dataset described by a YAML
file (examples/simple_csv_task.yaml + examples/simple_task_sample.csv) -- no
Python data code required.

The same YAML can be used from the CLI:
    circuitkit discover-yaml --model gpt2 \\
        --task-yaml examples/simple_csv_task.yaml --algorithm eap-ig \\
        --level node --sparsity 0.2 --num-examples 8 \\
        --output ./results/example_yaml_circuit.pt

This script does the equivalent through the Python API by registering the
YAML-defined task and then calling discover_circuit.

Run:
    python examples/04-discover-yaml.py
"""

import os
from pathlib import Path

from circuitkit.api import discover_circuit
from circuitkit.tasks.yaml_loader import YAMLTaskLoader
from circuitkit.tasks.registry import register_task

YAML_PATH = os.path.join(os.path.dirname(__file__), "simple_csv_task.yaml")


def main():
    os.makedirs("./results", exist_ok=True)

    # Load the task spec described by the YAML file and register it so that
    # discover_circuit can refer to it by name.
    print(f"Loading task from {YAML_PATH}")
    task_spec = YAMLTaskLoader.load(Path(YAML_PATH))
    register_task(task_spec)
    print(f"Registered YAML task: '{task_spec.name}'")

    config = {
        "model": {"name": "gpt2", "precision": "bfloat16"},
        "discovery": {
            "algorithm": "eap-ig",
            "task": task_spec.name,        # the task defined by the YAML
            "level": "node",
            "batch_size": 2,
            "ig_steps": 2,
            "data_params": {"num_examples": 8},
        },
        "pruning": {"target_sparsity": 0.2, "scope": "heads"},
        "output_path": "./results/example_yaml_circuit.pt",
    }

    print("Running discovery on the YAML-defined task ...")
    pruned_nodes = discover_circuit(config)
    print(f"Discovery complete. {len(pruned_nodes)} nodes selected.")
    print("First 5 nodes:", pruned_nodes[:5])
    print("\nDone. Artifact written to ./results/example_yaml_circuit.pt")


if __name__ == "__main__":
    main()
