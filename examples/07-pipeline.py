#!/usr/bin/env python3
"""
06 - The Pipeline class: stateful Discover -> Evaluate -> Intervene.

Pipeline carries model, circuit, and history across method calls and
supports chaining. Use it for multi-step workflows; use the flat
`circuitkit.quick` functions (see 01-quickstart.py) for one-shot calls.

Run:
    python examples/07-pipeline.py
"""

import os

from circuitkit import Pipeline

OUTPUT_DIR = "./results/pipeline_demo"


def section_a_from_scratch() -> str:
    """A. Build a Pipeline from scratch and run the full workflow."""
    print("=== A. Pipeline from scratch ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe = Pipeline("gpt2", task="ioi", output_dir=OUTPUT_DIR)
    pipe.discover(
        algorithm="eap-ig", level="node",
        sparsity=0.3, n_examples=16, batch_size=2, ig_steps=2, scope="both",
    )
    pipe.evaluate(pillars=["patching", "ablation"], n_examples=16)
    pipe.prune(sparsity=0.2, scope="both")
    pipe.export(os.path.join(OUTPUT_DIR, "checkpoint"))
    pipe.visualize(output=os.path.join(OUTPUT_DIR, "circuit.html"))
    pipe.summary()

    return pipe.artifact_path


def section_b_from_artifact(artifact_path: str) -> None:
    """B. Reload an existing artifact and apply a different sparsity."""
    print("\n=== B. Pipeline.from_artifact ===")

    pipe = Pipeline.from_artifact(
        artifact_path, model_name="gpt2", task="ioi",
        output_dir=OUTPUT_DIR,
    )
    pipe.prune(sparsity=0.4, scope="both")
    pipe.export(os.path.join(OUTPUT_DIR, "checkpoint_sparse"))
    pipe.summary()


def main():
    artifact_path = section_a_from_scratch()
    section_b_from_artifact(artifact_path)
    print("\nDone.")


if __name__ == "__main__":
    main()