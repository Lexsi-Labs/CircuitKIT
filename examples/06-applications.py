#!/usr/bin/env python3
"""
05 - Acting on a circuit: pruning, quantization, selective finetuning.

Once a circuit is discovered, you can act on it through circuitkit's
applications. This script demonstrates the three v1.0 applications at
both API levels:

    Section A - Flat `circuitkit` API (recommended for most users)
        ck.prune(), ck.quantize(), ck.selective_finetune()

    Section B - Class-level API (power users / custom pipelines)
        StructuralPruner.prune(), circuit_quantize(), select_components()

It first discovers a circuit so every application has real circuit scores.

Run:
    python examples/06-applications.py
"""

import os

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

import circuitkit as ck
from circuitkit.api import discover_circuit

CIRCUIT_CFG = {
    "model": {"name": "gpt2", "precision": "bfloat16"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": "ioi",
        "level": "node",
        "batch_size": 2,
        "ig_steps": 2,
        "data_params": {"num_examples": 16},
    },
    "pruning": {"target_sparsity": 0.2, "scope": "heads"},
    "output_path": "./results/example_apps_circuit.pt",
}


def discover_demo_circuit() -> ck.Circuit:
    """Run discovery once and return the resulting Circuit."""
    os.makedirs("./results", exist_ok=True)
    print("=== Discovering a circuit for the applications to act on ===")
    discover_circuit(CIRCUIT_CFG)
    return ck.load_scores(CIRCUIT_CFG["output_path"])


# --------------------------------------------------------------------------- #
# Section A - Flat API                                                        #
# --------------------------------------------------------------------------- #

def flat_pruning(circuit: ck.Circuit) -> None:
    """A1. Structural pruning via ck.prune()."""
    print("\n=== A1. Pruning (ck.prune) ===")
    model = HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.bfloat16)
    ck.prune(model, circuit, sparsity=0.2, scope="both", inplace=False)
    print("Pruning complete (masked a copy; original model untouched).")


def flat_quantization(circuit: ck.Circuit) -> None:
    """A2. Circuit-guided quantization via ck.quantize()."""
    print("\n=== A2. Quantization (ck.quantize) ===")
    from circuitkit.applications.arch_utils import UnsupportedArchitectureError

    hf_model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.bfloat16)
    try:
        plan = ck.quantize(hf_model, circuit, high_fraction=0.3)
        print(f"Quantization plan: {sum(len(v) for v in plan.values())} layer "
              f"assignments across {list(plan.keys())}.")
    except UnsupportedArchitectureError as e:
        # GPT-2 uses transformers Conv1D layers, which the optimum-quanto backend
        # cannot quantize. Real quantization runs on nn.Linear models — see
        # examples/quantization/quantize-llama.py and quantize-qwen.py.
        print(f"Skipping quantization on GPT-2: {e}")


def flat_selective_finetune(circuit: ck.Circuit) -> None:
    """A3. Selective finetuning component selection via ck.selective_finetune().

    GPT-2's config doesn't expose the generic num_hidden_layers / etc.
    aliases that the auto-load path looks for, so we pass the architecture
    params explicitly (same values class_selective_finetune derives below).
    """
    print("\n=== A3. Selective finetuning (ck.selective_finetune) ===")
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained("gpt2")
    result = ck.selective_finetune(
        circuit, top_fraction=0.2, scope="both",
        n_layers=hf_cfg.n_layer,
        n_q_heads=hf_cfg.n_head,
        n_kv_heads=hf_cfg.n_head,
        head_dim=hf_cfg.n_embd // hf_cfg.n_head,
    )
    print(f"Selected {len(result.attn)} attention components, "
          f"{len(result.mlp)} MLP components for finetuning.")


# --------------------------------------------------------------------------- #
# Section B - Class-level API                                                 #
# --------------------------------------------------------------------------- #

def class_pruning(circuit: ck.Circuit) -> None:
    """B1. StructuralPruner.prune() directly."""
    print("\n=== B1. Pruning (StructuralPruner) ===")
    from circuitkit.applications.pruning import StructuralPruner

    model = HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.bfloat16)
    pruner = StructuralPruner()
    # inplace=False (default) masks a copy, leaving the original model untouched.
    pruner.prune(model, circuit.circuit_scores, sparsity=0.2, scope="both", inplace=False)
    print("Pruning complete via StructuralPruner directly.")


def class_quantization(circuit: ck.Circuit) -> None:
    """B2. circuit_quantize() directly -- the underlying CircuitKit logic
    behind ck.quantize(). Optional `low_weights`/`high_weights` qtype kwargs
    (e.g. optimum.quanto's qint4/qint8) can tune the precision tiers further
    if optimum-quanto is installed.
    """
    print("\n=== B2. Quantization (circuit_quantize) ===")
    import re
    from circuitkit.applications.arch_utils import UnsupportedArchitectureError
    from circuitkit.applications.quantization import circuit_quantize

    hf_model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.bfloat16)
    n_layers = hf_model.config.n_layer

    # Derive per-head / per-MLP scores from the circuit's node scores.
    q_head_scores = {}
    mlp_scores = {}
    for name, score in circuit.scores.items():
        attn = re.match(r"[Aa](\d+)\.[hH]?(\d+)$", name)
        mlp = re.match(r"MLP\s*(\d+)$", name)
        if attn:
            layer, head = int(attn.group(1)), int(attn.group(2))
            q_head_scores[(layer, head)] = float(score)
        elif mlp:
            mlp_scores[int(mlp.group(1))] = float(score)

    # Any layer not covered by node scores still needs an MLP score tensor.
    for layer in range(n_layers):
        mlp_scores.setdefault(layer, torch.zeros(3072))

    try:
        plan = circuit_quantize(
            hf_model, q_head_scores, mlp_scores, n_layers,
            high_fraction=0.3,
        )
        print(f"Quantization plan: {sum(len(v) for v in plan.values())} layer "
              f"assignments across {list(plan.keys())}.")
    except UnsupportedArchitectureError as e:
        # GPT-2's Conv1D layers aren't quantizable by optimum-quanto; real
        # quantization runs on nn.Linear models (see examples/quantization/).
        print(f"Skipping quantization on GPT-2: {e}")


def class_selective_finetune(circuit: ck.Circuit) -> None:
    """B3. select_components() directly, with explicit architecture params."""
    print("\n=== B3. Selective finetuning (select_components) ===")
    from pathlib import Path
    from transformers import AutoConfig
    from circuitkit.applications.selective_finetuning.score_loader import load_scores
    from circuitkit.applications.selective_finetuning.selector import select_components

    artifact = Path(circuit.artifact_path)
    scores_pt = artifact.parent / f"{artifact.stem}_scores.pt"
    head_scores, mlp_scores, metadata = load_scores(str(scores_pt), model_name="gpt2")

    hf_cfg = AutoConfig.from_pretrained("gpt2")
    result = select_components(
        head_scores, mlp_scores, metadata,
        top_frac=0.2, scope="both",
        n_layers=hf_cfg.n_layer,
        n_q_heads=hf_cfg.n_head,
        n_kv_heads=hf_cfg.n_head,
        head_dim=hf_cfg.n_embd // hf_cfg.n_head,
    )
    print(f"Selected {len(result.attn)} attention components, "
          f"{len(result.mlp)} MLP components for finetuning.")


def main():
    circuit = discover_demo_circuit()

    print("\n--- Section A: Flat circuitkit API ---")
    flat_pruning(circuit)
    flat_quantization(circuit)
    flat_selective_finetune(circuit)

    print("\n--- Section B: Class-level API ---")
    class_pruning(circuit)
    class_quantization(circuit)
    class_selective_finetune(circuit)

    print("\nAll applications ran successfully at both API levels.")


if __name__ == "__main__":
    main()