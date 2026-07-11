#!/usr/bin/env python3
"""
01 - Quickstart: the full Discover -> Evaluate -> Prune -> Export loop.

The fastest path through CircuitKit using the flat `circuitkit` API
(`import circuitkit as ck`). Runs end-to-end on GPT-2 / CPU in a couple
of minutes.

Run:
    python examples/01-quickstart.py
"""

import os

import circuitkit as ck

OUTPUT_PATH = "./results/quickstart_circuit.pt"


def main():
    os.makedirs("./results", exist_ok=True)

    # 1. Load a model with the hook flags discovery needs.
    print("Step 1: loading model ...")
    model = ck.load_model("gpt2")

    # 2. Discover a circuit for the IOI task.
    print("Step 2: discovering circuit ...")
    circuit = ck.discover(
        model, "ioi",
        algorithm="eap-ig", level="node",
        n_examples=16, batch_size=2, ig_steps=2,
        sparsity=0.2, scope="both",
        output_path=OUTPUT_PATH,
    )
    print(f"Discovered {len(circuit)} nodes. Top 3: {circuit.top_nodes(3)}")

    # 3. Score the circuit's faithfulness (fast pillars only).
    print("\nStep 3: evaluating faithfulness ...")
    report = ck.faithfulness(model, circuit, "ioi", pillars=["patching", "ablation"])
    print(f"Patching score : {report.patching_score:.4f}")
    print(f"Ablation score : {report.ablation_score:.4f}")

    # 4. Prune the model down to the discovered circuit.
    print("\nStep 4: pruning model ...")
    pruned_model = ck.prune(model, circuit, sparsity=0.2, scope="both")
    print("Pruning complete (masked a copy; original model untouched).")

    # 5. Export the pruned model as a HF checkpoint.
    print("\nStep 5: exporting checkpoint ...")
    ckpt_path = ck.export_checkpoint(pruned_model, circuit, "./results/quickstart_checkpoint")
    print(f"Checkpoint written to {ckpt_path}")

    # 6. Visualize the circuit graph.
    print("\nStep 6: visualizing circuit ...")
    ck.visualize_circuit(circuit, mode="graph", output="./results/quickstart_circuit.html")
    print("Graph saved to ./results/quickstart_circuit.html")

    print("\nDone. See 01-08 for deeper dives into each step.")


if __name__ == "__main__":
    main()