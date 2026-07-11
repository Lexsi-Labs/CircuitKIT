"""End-to-end: real CrowS-Pairs through PairwiseAdapter ->
NormalizedTaskSpec -> EAP-IG discover_circuit on Llama-3.2-1B (cuda:1).

CrowS-Pairs records are natively paired so no corruption strategy needed.
This validates the production-scale custom-data path on a real 1B Llama.
"""
from __future__ import annotations
import os, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status

SCRIPT = "09_e2e_discovery_crowspairs_llama1b"

CACHE = Path("/tmp/crows_pairs.csv")
URL = ("https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
       "data/crows_pairs_anonymized.csv")


def main() -> int:
    out = make_results_dir(SCRIPT)
    if not CACHE.exists():
        urllib.request.urlretrieve(URL, CACHE)

    from circuitkit.data.adapters.pairwise import PairwiseAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task
    from circuitkit.api import discover_circuit

    ds = PairwiseAdapter().adapt(str(CACHE),
                                 name="crows_pairs_e2e",
                                 source="github://nyu-mll/crows-pairs",
                                 max_records=16)
    # Filter records whose clean+corrupt last words are both single tokens
    # in the Llama tokenizer (otherwise NormalizedTaskSpec will skip them).
    print(f"[fixture] {len(ds)} CrowS-Pairs records ready (n_paired={ds.n_paired})")

    spec = NormalizedTaskSpec(ds, name="crows_e2e",
                              cache_dir=str(out / "_cache"))
    register_task(spec)

    # Pin to cuda:1 so the GPT-2 cache on cuda:0 stays warm
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    discovery_config = {
        "model": {
            "name": "meta-llama/Llama-3.2-1B",
            "precision": "bfloat16",
        },
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
    print(f"[discovery] EAP-IG on Llama-3.2-1B with {len(ds)} CrowS-Pairs records (cuda:1)...")
    t0 = time.time()
    pruned = discover_circuit(discovery_config)
    elapsed = time.time() - t0
    art = out / "circuit.pt"

    status = {
        "script": SCRIPT,
        "module": "data E2E CrowS-Pairs -> Llama-3.2-1B discover_circuit",
        "input": {"source": "github://nyu-mll/crows-pairs",
                  "n_records": len(ds), "model": "Llama-3.2-1B",
                  "algorithm": "eap-ig"},
        "output": {"artifact": str(art),
                   "artifact_bytes": art.stat().st_size if art.exists() else 0},
        "metrics": {"wall_seconds": round(elapsed, 1),
                    "pruned_count": len(pruned) if hasattr(pruned, "__len__") else None},
        "status": "WORKING" if art.exists() and art.stat().st_size > 100 else "BROKEN",
    }
    write_status(out, status)
    print()
    print(f"E2E CrowS-Pairs -> Llama-3.2-1B discovery: records={len(ds)}, "
          f"wall={elapsed:.1f}s, pruned={status['metrics']['pruned_count']}, "
          f"status={status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
