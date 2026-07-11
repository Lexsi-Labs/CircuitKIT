"""End-to-end: real TOFU forget+retain through ForgetRetainAdapter ->
resample (peer prompts as corrupt) -> NormalizedTaskSpec -> discover_circuit
on GPT-2. Validates the unlearning-data pipeline runs through real EAP-IG
attribution end-to-end.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status, fetch_hf

SCRIPT = "08_e2e_discovery_tofu_gpt2"


def main() -> int:
    out = make_results_dir(SCRIPT)
    from circuitkit.data.adapters.forget_retain import ForgetRetainAdapter
    from circuitkit.data.corruption.resample import Resample
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task
    from circuitkit.api import discover_circuit

    forget = fetch_hf("locuslab/TOFU", "forget01", split="train", take=12)
    retain = fetch_hf("locuslab/TOFU", "retain99", split="train", take=12)
    ds = ForgetRetainAdapter().adapt(
        {"forget": forget, "retain": retain},
        name="tofu_e2e",
        source="hf://locuslab/TOFU",
    )
    # Pair forget records with retain records as "resample" counterfactuals.
    forget_recs = [r for r in ds.records if r.meta.get("split") == "forget"]
    retain_recs = [r for r in ds.records if r.meta.get("split") == "retain"]
    strat = Resample()
    paired = []
    for r in forget_recs:
        paired.append(strat.apply(r, pool=retain_recs))
    ds.records = [r for r in paired if r.is_paired]
    print(f"[fixture] {len(ds)} forget records paired against retain peers")

    spec = NormalizedTaskSpec(ds, name="tofu_e2e",
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
    print(f"[discovery] EAP-IG on GPT-2 with {len(ds)} TOFU records ...")
    t0 = time.time()
    pruned = discover_circuit(discovery_config)
    elapsed = time.time() - t0
    art = out / "circuit.pt"

    status = {
        "script": SCRIPT,
        "module": "data E2E TOFU -> discover_circuit",
        "input": {"source": "hf://locuslab/TOFU forget01+retain99",
                  "n_records": len(ds), "model": "gpt2",
                  "algorithm": "eap-ig"},
        "output": {"artifact": str(art),
                   "artifact_bytes": art.stat().st_size if art.exists() else 0},
        "metrics": {"wall_seconds": round(elapsed, 1),
                    "pruned_count": len(pruned) if hasattr(pruned, "__len__") else None},
        "status": "WORKING" if art.exists() and art.stat().st_size > 100 else "BROKEN",
    }
    write_status(out, status)
    print()
    print(f"E2E TOFU -> discovery: records={len(ds)}, "
          f"wall={elapsed:.1f}s, pruned_heads={status['metrics']['pruned_count']}, "
          f"status={status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
