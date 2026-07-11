"""Validation: cross-method comparison of the 8 discovery algorithms.

Runs every working top-level algorithm key on the same GPT-2 IOI
fixture and reports:
  - per-method wall time
  - top-k node sets at k = 5, 10, 20
  - pairwise Jaccard overlap of those top-k sets
  - faithfulness proxy: Jaccard@10 against eap-exact (gold-standard
    leave-one-out attribution)

Includes both the 7 EAP-family methods (linear/IG/exact/AtP-GD/
EAP-GP/RelP) and the non-EAP IBCircuit (information-bottleneck
gating) so the comparison covers both gradient-based and
optimisation-based discovery.

End-to-end real-model run. No mocks.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _algos_common import (  # noqa: E402
    run_algo, top_k_set, jaccard, make_results_dir, write_status,
)

SCRIPT_NAME = "04_compare_eap_family_gpt2"
ALGOS = [
    "eap",
    "eap-ig",
    "eap-ig-activations",
    "eap-clean-corrupted",
    "eap-exact",
    "atp-gd",
    "eap-gp",
    "relp",
    "ibcircuit",
]


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)

    results = {}
    failures = []

    for algo in ALGOS:
        try:
            # eap-exact / atp-gd / eap-gp run with smaller num_examples for tractable wall time
            num_examples = 16 if algo in ("eap-exact", "atp-gd", "eap-gp") else 32
            extra = None
            if algo == "ibcircuit":
                # Short epochs for the smoke comparison; production sweeps use 500-1000.
                extra = {"num_epochs": 200, "scope": "heads"}
            results[algo] = run_algo(algo, num_examples=num_examples, extra_discovery=extra)
            print(f"  {algo:25s}  {results[algo]['wall_seconds']:>6.2f}s  "
                  f"{results[algo]['n_nodes_total']} nodes")
        except Exception as exc:
            failures.append({"algo": algo, "error": repr(exc)})
            print(f"  {algo:25s}  FAILED: {exc}")

    if not results:
        write_status(out_dir, {
            "script": SCRIPT_NAME, "module": "discover_circuit[eap-family]",
            "status": "BROKEN", "error": "all algorithms failed", "failures": failures,
        })
        return 1

    # Build per-k top-set summaries.
    ks = [5, 10, 20]
    top_sets = {algo: {k: top_k_set(r["node_scores"], k) for k in ks}
                for algo, r in results.items()}

    # Pairwise Jaccard at k=10 (canonical comparison).
    algos_run = list(results.keys())
    pairwise = {}
    for i, a in enumerate(algos_run):
        for b in algos_run[i + 1:]:
            pairwise[f"{a}__vs__{b}"] = round(
                jaccard(top_sets[a][10], top_sets[b][10]), 4
            )

    # Faithfulness proxy: Jaccard(method, eap-exact) at k=10.
    faithfulness_vs_exact = {}
    if "eap-exact" in results:
        ref = top_sets["eap-exact"][10]
        for a in algos_run:
            if a == "eap-exact":
                continue
            faithfulness_vs_exact[a] = round(jaccard(top_sets[a][10], ref), 4)

    # Wall-time table.
    timing = {algo: results[algo]["wall_seconds"] for algo in algos_run}

    status = {
        "script": SCRIPT_NAME,
        "module": "discover_circuit[eap-family-comparison]",
        "input": {
            "model": "gpt2", "task": "ioi",
            "algorithms": algos_run, "ks": ks,
        },
        "metrics": {
            "wall_seconds_per_algo": timing,
            "top_5_per_algo":  {a: top_sets[a][5]  for a in algos_run},
            "top_10_per_algo": {a: top_sets[a][10] for a in algos_run},
            "pairwise_jaccard_at_k10": pairwise,
            "faithfulness_vs_exact_at_k10": faithfulness_vs_exact,
        },
        "failures": failures,
        "status": "WORKING" if len(results) == len(ALGOS) else "NEEDS-FIX",
    }
    write_status(out_dir, status)

    print(f"\nEAP-family cross-method comparison on GPT-2 IOI")
    print(f"  Algorithms run:    {len(results)}/{len(ALGOS)}")
    print(f"  Pairwise Jaccard@10:")
    for pair, j in sorted(pairwise.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {pair:55s}  J={j}")
    if faithfulness_vs_exact:
        print(f"  Faithfulness vs eap-exact (Jaccard@10):")
        for a, j in sorted(faithfulness_vs_exact.items(), key=lambda kv: kv[1], reverse=True):
            print(f"    {a:25s}  J={j}")
    print(f"  Status:            {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
