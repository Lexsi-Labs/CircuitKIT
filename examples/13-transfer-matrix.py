#!/usr/bin/env python3
"""Transfer matrix: discover circuits on one task and evaluate on others."""

from circuitkit import load_model
from circuitkit.evaluation.transfer import TransferMatrix

model = load_model("gpt2")

matrix = TransferMatrix(task_names=["ioi", "sva", "greater_than"])

numpy_matrix = matrix.build(
    model,
    discovery_cfg_template={
        "model": {"name": "gpt2", "precision": "float32"},
        "discovery": {"algorithm": "eap-ig", "data_params": {"num_examples": 16}},
        "pruning": {"target_sparsity": 0.3},
    },
    device="cpu",
)

analysis = matrix.analyze()
print("Transfer matrix (source rows → target columns):")
print(numpy_matrix)
print(f"\nBest transfer:  {analysis['best_transfer']}")
print(f"Worst transfer: {analysis['worst_transfer']}")

# Equivalent CLI command:
# circuitkit transfer-matrix --model gpt2 --tasks ioi,sva,greater_than
