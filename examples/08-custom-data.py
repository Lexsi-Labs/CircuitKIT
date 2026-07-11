#!/usr/bin/env python3
"""
07 - Bringing your own dataset: three ways to use custom data.

The custom-data contract (data.type = "template"):
    - path     : CSV file. Any column names are fine.
    - template : dict of Python .format()-style strings referencing CSV
                 columns by name, e.g. "{question}".
        - clean_prompt / clean_answer   : REQUIRED.
        - corrupt_prompt / corrupt_answer : OPTIONAL.
            Provide both for paired algorithms (EAP, EAP-IG, ACDC, ...).
            Omit both for clean-only algorithms (IBCircuit, CD-T) -- no
            corrupted counterpart is needed.

Uses the shipped simple_task_sample.csv (columns: question, answer,
corrupted) to demonstrate:

    A. Paired data via Pipeline.from_custom_data() -- clean + corrupt
       templates, suitable for EAP-IG and other paired algorithms.
    B. Clean-only data via Pipeline.from_custom_data() -- no corrupt
       templates, suitable for IBCircuit / CD-T.
    C. Dict-config via prepare_custom_task() -- full low-level control.

Full reference: docs/user-guide/custom-data.md

Run:
    python examples/08-custom-data.py
"""

import os

from circuitkit import Pipeline

CSV_PATH = os.path.join(os.path.dirname(__file__), "simple_task_sample.csv")
OUTPUT_DIR = "./results/custom_data_demo"


def section_a_paired() -> None:
    """A. Paired clean/corrupt data -- EAP-IG."""
    print("=== A. Paired data (Pipeline.from_custom_data, EAP-IG) ===")

    pipe = Pipeline.from_custom_data(
        "gpt2", CSV_PATH,
        clean_prompt="{question}", clean_answer="{answer}",
        corrupt_prompt="{corrupted}", corrupt_answer="{corrupt_answer}",
        task_name="custom_qa_paired",
        output_dir=OUTPUT_DIR,
    )
    pipe.discover(
        algorithm="eap-ig", level="node",
        sparsity=0.2, n_examples=8, batch_size=2, ig_steps=2,
    )
    print(f"Discovered {len(pipe.circuit)} nodes. "
          f"Top 3: {pipe.circuit.top_nodes(3)}")


def section_b_clean_only() -> None:
    """B. Clean-only data -- IBCircuit (no corrupt pairs needed)."""
    print("\n=== B. Clean-only data (Pipeline.from_custom_data, IBCircuit) ===")

    pipe = Pipeline.from_custom_data(
        "gpt2", CSV_PATH,
        clean_prompt="{question}", clean_answer="{answer}",
        # No corrupt_prompt / corrupt_answer -- IBCircuit doesn't need pairs.
        task_name="custom_qa_clean",
        output_dir=OUTPUT_DIR,
    )
    pipe.discover(
        algorithm="ibcircuit", level="node",
        sparsity=0.2, n_examples=8, batch_size=2,
        num_epochs=200, scope="heads",
        # Note: scope= here sets pruning.scope. If IBCircuit needs
        # discovery.scope set explicitly too (see
        # discovery/algorithm_comparison.py), use the dict-config path
        # (Section C) for full control over the discovery block.
    )
    print(f"Discovered {len(pipe.circuit)} nodes. "
          f"Top 3: {pipe.circuit.top_nodes(3)}")


def section_c_dict_config() -> None:
    """C. Low-level dict-config via prepare_custom_task() for full control."""
    print("\n=== C. Dict-config (prepare_custom_task) ===")

    from circuitkit import quick
    from circuitkit.api import discover_circuit, prepare_custom_task

    model = quick.load_model("gpt2", algorithm="eap-ig")

    config = {
        "data": {
            "type": "template",
            "path": CSV_PATH,
            "template": {
                "clean_prompt": "{question}",
                "clean_answer": "{answer}",
                "corrupt_prompt": "{corrupted}",
                "corrupt_answer": "{corrupt_answer}",
            },
        },
        "model": {"name": "gpt2", "precision": "bfloat16"},
        "discovery": {
            "algorithm": "eap-ig", "task": "",
            "level": "node", "batch_size": 2, "ig_steps": 2,
            "data_params": {"num_examples": 8},
        },
        "pruning": {"target_sparsity": 0.2, "scope": "both"},
        "output_path": os.path.join(OUTPUT_DIR, "custom_dict_config.pt"),
    }

    # prepare_custom_task registers the task and returns its name; the
    # discovery block's "task" key must be filled in before discover_circuit.
    task_name = prepare_custom_task(config, model, task_name="custom_qa_dict")
    config["discovery"]["task"] = task_name

    nodes = discover_circuit(config)
    print(f"Discovered {len(nodes)} nodes via dict-config.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    section_a_paired()
    section_b_clean_only()
    section_c_dict_config()
    print("\nDone.")


if __name__ == "__main__":
    main()