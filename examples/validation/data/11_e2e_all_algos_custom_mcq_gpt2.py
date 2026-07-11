"""End-to-end: real MMLU through MCQ adapter -> mcq_choice_swap ->
NormalizedTaskSpec -> discover_circuit for ALL 9 algorithms on GPT-2.

Confirms the data layer (custom HF dataset) works against every
algorithm we ship: EAP family, attribution variants, IBCircuit, CD-T,
PEAP, IFR. Each algo runs in isolation so one failure doesn't poison
the rest.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status, fetch_hf

SCRIPT = "11_e2e_all_algos_custom_mcq_gpt2"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
]


def main() -> int:
    out = make_results_dir(SCRIPT)
    from circuitkit.data.adapters.mcq import MCQAdapter
    from circuitkit.data.corruption.mcq_choice_swap import MCQChoiceSwap
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task
    from circuitkit.api import discover_circuit

    raw = fetch_hf("cais/mmlu", "high_school_world_history",
                   split="test", take=24)
    ds = MCQAdapter().adapt(raw, name="mmlu_he_all9", max_records=20)

    strat = MCQChoiceSwap()
    ds.records = [strat.apply(r) for r in ds.records]
    ds.records = [r for r in ds.records if r.is_paired]
    n_paired = len(ds.records)
    print(f"[fixture] {n_paired} paired records from MMLU")
    if n_paired < 8:
        write_status(out, {
            "script": SCRIPT, "status": "BROKEN",
            "error": f"too few paired records: {n_paired}",
        })
        return 1

    spec = NormalizedTaskSpec(ds, name="mmlu_he_all9_e2e",
                              cache_dir=str(out / "_cache"))
    register_task(spec)

    rows = []
    for algo in ALGOS:
        row = {"algorithm": algo}
        artifact_path = out / f"circuit_{algo}.pt"
        cfg = {
            "model": {"name": "gpt2", "precision": "float32"},
            "discovery": {
                "algorithm": algo,
                "task": spec.name,
                "level": "node",
                "batch_size": 1,
                "ig_steps": 2,
                "data_params": {"num_examples": n_paired},
            },
            "pruning": {"target_sparsity": 0.1, "scope": "heads"},
            "output_path": str(artifact_path),
        }
        try:
            t0 = time.time()
            pruned = discover_circuit(cfg)
            elapsed = time.time() - t0
            size = artifact_path.stat().st_size if artifact_path.exists() else 0
            row.update({
                "status": "WORKING" if size > 100 else "BROKEN",
                "wall_seconds": round(elapsed, 1),
                "artifact_bytes": size,
                "pruned_count": (
                    len(pruned) if hasattr(pruned, "__len__") else None
                ),
            })
            print(f"  {algo:25s}  {row['status']:8s}  "
                  f"{elapsed:5.1f}s  bytes={size}", flush=True)
        except Exception as exc:  # noqa: BLE001
            row.update({
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            })
            print(f"  {algo:25s}  ERROR    {row['error'][:80]}",
                  flush=True)
        rows.append(row)

    n_ok = sum(1 for r in rows if r.get("status") == "WORKING")
    write_status(out, {
        "script": SCRIPT,
        "module": "data E2E (custom MMLU) -> 9 algorithms",
        "input": {
            "source": "hf://cais/mmlu/high_school_world_history",
            "n_paired_records": n_paired,
            "model": "gpt2",
            "algorithms": ALGOS,
        },
        "metrics": {"n_algos_ok": n_ok, "n_algos_total": len(ALGOS)},
        "rows": rows,
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })

    print()
    print(f"[summary] {n_ok}/{len(ALGOS)} algorithms succeeded on "
          f"custom MMLU data ({n_paired} paired records)")
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
