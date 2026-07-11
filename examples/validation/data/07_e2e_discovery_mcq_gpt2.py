"""End-to-end: real MMLU through MCQ adapter -> mcq_choice_swap ->
NormalizedTaskSpec -> circuitkit.api.discover_circuit on GPT-2 (cuda:0).

Proves the full data-layer chain produces an artifact a real EAP-IG
discovery pass can actually consume.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status, fetch_hf

SCRIPT = "07_e2e_discovery_mcq_gpt2"


def main() -> int:
    out = make_results_dir(SCRIPT)
    from circuitkit.data.adapters.mcq import MCQAdapter
    from circuitkit.data.corruption.mcq_choice_swap import MCQChoiceSwap
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task
    from circuitkit.api import discover_circuit

    raw = fetch_hf("cais/mmlu", "high_school_world_history",
                   split="test", take=20)
    ds = MCQAdapter().adapt(raw, name="mmlu_he", max_records=16)

    # Apply mcq_choice_swap to every record.
    strat = MCQChoiceSwap()
    ds.records = [strat.apply(r) for r in ds.records]
    n_paired = sum(1 for r in ds.records if r.is_paired)
    print(f"[fixture] {n_paired}/{len(ds)} records paired by mcq_choice_swap")
    if n_paired < 8:
        print(f"[fixture] WARN: too few paired records ({n_paired})")
    ds.records = [r for r in ds.records if r.is_paired]

    # Register a NormalizedTaskSpec for this dataset.
    spec = NormalizedTaskSpec(ds, name="mmlu_he_e2e",
                              cache_dir=str(out / "_cache"))
    register_task(spec)

    discovery_config = {
        "model": {"name": "gpt2", "precision": "float32"},
        "discovery": {
            "algorithm": "eap-ig",
            "task": spec.name,
            "level": "node",
            "batch_size": 1,
            "ig_steps": 2,
            "data_params": {"num_examples": len(ds)},
        },
        "pruning": {"target_sparsity": 0.1, "scope": "heads"},
        "output_path": str(out / "circuit.pt"),
    }
    print(f"[discovery] running EAP-IG on GPT-2 with {len(ds)} MMLU records ...")
    t0 = time.time()
    pruned = discover_circuit(discovery_config)
    elapsed = time.time() - t0

    artifact_path = out / "circuit.pt"
    artifact_size = artifact_path.stat().st_size if artifact_path.exists() else 0

    status = {
        "script": SCRIPT,
        "module": "data E2E -> circuitkit.api.discover_circuit",
        "input": {
            "source": "hf://cais/mmlu/high_school_world_history",
            "n_records": len(ds),
            "model": "gpt2",
            "algorithm": "eap-ig",
        },
        "output": {
            "artifact": str(artifact_path),
            "artifact_bytes": artifact_size,
        },
        "metrics": {
            "wall_seconds": round(elapsed, 1),
            "pruned_count": len(pruned) if hasattr(pruned, "__len__") else None,
        },
        "status": "WORKING" if artifact_size > 100 else "BROKEN",
    }
    write_status(out, status)
    print()
    print(f"E2E MMLU -> EAP-IG -> GPT-2 discovery:")
    print(f"  records: {len(ds)},  wall: {elapsed:.1f}s")
    print(f"  artifact bytes: {artifact_size:,}")
    print(f"  pruned heads:   {len(pruned) if hasattr(pruned, '__len__') else 'n/a'}")
    print(f"  status: {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
