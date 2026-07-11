#!/usr/bin/env python3
"""
08 - Loading and reusing a saved circuit.

Once you've run discovery, you don't need to re-run it to try different
prune sparsities, quantization settings, or finetuning fractions. This
script discovers once, then demonstrates two ways to reload and reuse
the saved artifact:

    A. ck.load_scores() -- flat API, iterate applications at different settings.
    B. Pipeline.from_artifact() -- Pipeline API, downstream pipeline operations.

Run:
    python examples/09-load-and-reuse.py
"""

import os

import torch
from transformer_lens import HookedTransformer

import circuitkit as ck
from circuitkit import Pipeline
from circuitkit.api import discover_circuit

OUTPUT_PATH = "./results/reuse_demo_circuit.pt"
CONFIG = {
    "model": {"name": "gpt2", "precision": "bfloat16"},
    "discovery": {
        "algorithm": "eap-ig", "task": "ioi", "level": "node",
        "batch_size": 2, "ig_steps": 2,
        "data_params": {"num_examples": 16},
    },
    "pruning": {"target_sparsity": 0.2, "scope": "both"},
    "output_path": OUTPUT_PATH,
}


def discover_once() -> None:
    os.makedirs("./results", exist_ok=True)
    print("=== Running discovery once to create an artifact ===")
    discover_circuit(CONFIG)


def section_a_flat_reuse() -> None:
    """A. ck.load_scores() -- iterate prune/selective_finetune at different settings."""
    print("\n=== A. ck.load_scores() ===")

    circuit = ck.load_scores(OUTPUT_PATH)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.bfloat16)

    for sparsity in (0.1, 0.3):
        ck.prune(model, circuit, sparsity=sparsity, scope="both")
        print(f"  Pruned at sparsity={sparsity}")

    for top_fraction in (0.1, 0.25):
        result = ck.selective_finetune(circuit, top_fraction=top_fraction, scope="both")
        print(f"  top_fraction={top_fraction}: "
              f"{len(result.attn)} attn + {len(result.mlp)} mlp components")


def section_b_pipeline_reuse() -> None:
    """B. Pipeline.from_artifact() -- downstream pipeline operations."""
    print("\n=== B. Pipeline.from_artifact() ===")

    pipe = Pipeline.from_artifact(
        OUTPUT_PATH, model_name="gpt2", task="ioi",
        output_dir="./results/reuse_demo",
    )
    pipe.prune(sparsity=0.3, scope="both")
    pipe.export("./results/reuse_demo/checkpoint")
    pipe.summary()


def main():
    discover_once()
    section_a_flat_reuse()
    section_b_pipeline_reuse()
    print("\nDone.")


if __name__ == "__main__":
    main()