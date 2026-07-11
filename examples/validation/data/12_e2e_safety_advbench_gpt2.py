"""End-to-end: AdvBench through SafetyPromptAdapter -> NormalizedTaskSpec
-> discover_circuit on GPT-2.

Validates the safety-data pathway: harmful prompts paired with refusal
vs compliance answer tokens (Arditi et al. 2024 refusal-direction
recipe), routed through the same NormalizedTaskSpec that handles MCQ /
TOFU / conversational. Runs EAP-IG end-to-end and reports artifact
size + pruned head count.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status, fetch_hf

SCRIPT = "12_e2e_safety_advbench_gpt2"


def main() -> int:
    out = make_results_dir(SCRIPT)
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task
    from circuitkit.api import discover_circuit

    # AdvBench has a single "train" split — fetch_hf falls back automatically.
    try:
        raw = fetch_hf("walledai/AdvBench", split="train", take=40)
    except Exception as exc:
        write_status(out, {
            "script": SCRIPT, "status": "BROKEN",
            "error": f"could not load walledai/AdvBench: {exc}",
        })
        print(f"AdvBench load failed: {exc}")
        return 1

    ds = SafetyPromptAdapter().adapt(raw, name="advbench_e2e",
                                     max_records=32,
                                     pairing_mode="harmful_vs_benign")
    n_paired = sum(1 for r in ds.records if r.is_paired)
    print(f"[fixture] {n_paired}/{len(ds)} AdvBench records paired "
          f"(harmful → refusal vs compliance)")
    if n_paired < 8:
        write_status(out, {
            "script": SCRIPT, "status": "BROKEN",
            "error": f"too few paired records: {n_paired}",
        })
        return 1
    ds.records = [r for r in ds.records if r.is_paired]

    spec = NormalizedTaskSpec(ds, name="advbench_e2e",
                              cache_dir=str(out / "_cache"))
    register_task(spec)

    cfg = {
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
    print(f"[discovery] EAP-IG on GPT-2 with {len(ds)} AdvBench records ...")
    t0 = time.time()
    try:
        pruned = discover_circuit(cfg)
    except Exception as exc:
        write_status(out, {
            "script": SCRIPT, "status": "BROKEN",
            "error": f"discover_circuit failed: {type(exc).__name__}: {exc}",
        })
        print(f"discovery failed: {exc}")
        return 1
    elapsed = time.time() - t0

    artifact = out / "circuit.pt"
    artifact_size = artifact.stat().st_size if artifact.exists() else 0

    status = {
        "script": SCRIPT,
        "module": "data E2E -> SafetyPromptAdapter -> discover_circuit",
        "input": {
            "source": "hf://walledai/AdvBench",
            "n_records": len(ds),
            "model": "gpt2",
            "algorithm": "eap-ig",
            "pair_recipe": "refusal (' I') vs compliance (' Sure')",
        },
        "output": {
            "artifact": str(artifact),
            "artifact_bytes": artifact_size,
        },
        "metrics": {
            "wall_seconds": round(elapsed, 1),
            "pruned_count": (len(pruned) if hasattr(pruned, "__len__") else None),
        },
        "status": "WORKING" if artifact_size > 100 else "BROKEN",
    }
    write_status(out, status)

    print(f"\nE2E AdvBench -> EAP-IG -> GPT-2:")
    print(f"  records:        {len(ds)}")
    print(f"  wall:           {elapsed:.1f}s")
    print(f"  artifact bytes: {artifact_size:,}")
    print(f"  pruned heads:   {len(pruned) if hasattr(pruned, '__len__') else 'n/a'}")
    print(f"  status:         {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
