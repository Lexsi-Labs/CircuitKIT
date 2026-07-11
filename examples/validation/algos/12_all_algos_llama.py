"""Validation: All 5 attribution algorithms on Llama-3.2-1B with IOI task.

Runs EAP, EAP-IG, AtP* (atp-gd), PEAP, and RelP end-to-end on
meta-llama/Llama-3.2-1B using the same IOI task used for GPT-2 benchmarks.
This validates that all algorithms are architecture-agnostic (not GPT-2 only).

Also validates custom-data IOI-type pairs to confirm NormalizedTaskSpec
handles non-GPT-2 tokenizers correctly.

Usage:
    CUDA_VISIBLE_DEVICES=1 python 12_all_algos_llama.py
    CUDA_VISIBLE_DEVICES=1 python 12_all_algos_llama.py --model meta-llama/Llama-3.2-3B
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _algos_common import (  # noqa: E402
    run_algo, top_k_nodes, make_results_dir, write_status,
)

SCRIPT_NAME = "12_all_algos_llama"
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
TASK = "ioi"

# The 5 algorithms to validate
ALGOS = [
    ("eap",    {"num_examples": 24}),
    ("eap-ig", {"num_examples": 24, "ig_steps": 2}),
    ("atp-gd", {"num_examples": 24}),
    ("peap",   {"num_examples": 16}),
    ("relp",   {"num_examples": 24}),
]


def _run_custom_ioi_discovery(model_name: str, out_dir: Path) -> Dict[str, Any]:
    """Run EAP-IG on custom IOI-type prompt pairs to validate NormalizedTaskSpec."""
    try:
        from circuitkit.data.normalized_task import NormalizedTaskSpec
        from circuitkit.data.normalized import (
            ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset,
        )
        from circuitkit.tasks.registry import register_task, is_task_registered
        from circuitkit.api import discover_circuit

        task_name = f"custom_ioi_llama_{model_name.replace('/', '_')}"
        if not is_task_registered(task_name):
            # Build minimal IOI-type contrastive pairs with DIFFERENT clean/corrupt prompts.
            # clean: S1 first (e.g. "Mary and John"), answer = IO = " Mary"
            # corrupt: S2 first (name-swapped), answer = IO = " John"
            # This gives non-zero activation_difference needed by EAP.
            pairs = [
                ("When Mary and John went to the store, John gave a book to",
                 "When John and Mary went to the store, Mary gave a book to",
                 " Mary", " John"),
                ("When Sarah and Tom went to the park, Tom gave flowers to",
                 "When Tom and Sarah went to the park, Sarah gave flowers to",
                 " Sarah", " Tom"),
                ("When Alice and Bob went to the office, Bob sent an email to",
                 "When Bob and Alice went to the office, Alice sent an email to",
                 " Alice", " Bob"),
                ("When Emma and Jack went to school, Jack gave a pencil to",
                 "When Jack and Emma went to school, Emma gave a pencil to",
                 " Emma", " Jack"),
                ("When Lisa and Mike went to the gym, Mike gave a towel to",
                 "When Mike and Lisa went to the gym, Lisa gave a towel to",
                 " Lisa", " Mike"),
                ("When Anna and Paul went to the beach, Paul gave a shell to",
                 "When Paul and Anna went to the beach, Anna gave a shell to",
                 " Anna", " Paul"),
                ("When Kate and David went to the cafe, David gave a coffee to",
                 "When David and Kate went to the cafe, Kate gave a coffee to",
                 " Kate", " David"),
                ("When Jess and Chris went to the mall, Chris gave a gift to",
                 "When Chris and Jess went to the mall, Jess gave a gift to",
                 " Jess", " Chris"),
            ]
            records = [
                ContrastiveRecord(
                    record_id=str(i),
                    clean_prompt=clean_p,
                    clean_answer=io_ans,
                    corrupt_prompt=corrupt_p,
                    corrupt_answer=s_ans,
                    target_field="indirect_object",
                    contrast_source=ContrastSource.GENERATED,
                )
                for i, (clean_p, corrupt_p, io_ans, s_ans) in enumerate(pairs)
            ]
            ds = NormalizedDataset(
                name=task_name,
                shape=DatasetShape.PAIRWISE,
                records=records,
                source="custom_ioi",
            )
            spec = NormalizedTaskSpec(ds, name=task_name,
                                     cache_dir=str(out_dir / "_cache"))
            register_task(spec)

        t0 = time.time()
        result = run_algo(
            "eap-ig",
            model=model_name,
            task=task_name,
            num_examples=8,
            batch_size=1,
            ig_steps=2,
            target_sparsity=0.1,
        )
        elapsed = round(time.time() - t0, 1)
        return {"status": "WORKING", "wall_s": elapsed, "n_nodes": result["n_nodes_total"]}
    except Exception as exc:
        return {"status": "BROKEN", "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--skip-custom", action="store_true",
                        help="Skip custom IOI data validation")
    args = parser.parse_args()

    out_dir = make_results_dir(SCRIPT_NAME)
    model = args.model
    task = args.task

    print(f"\nRunning all 5 algorithms on {model} / {task}")
    print("=" * 60)

    results: List[Dict[str, Any]] = []
    any_broken = False

    for algo, extra in ALGOS:
        print(f"\n[{algo}] starting ...")
        try:
            t0 = time.time()
            result = run_algo(
                algo,
                model=model,
                task=task,
                batch_size=1,
                **extra,
            )
            elapsed = round(time.time() - t0, 1)
            top3 = [(n["node"], round(n["score"], 4))
                    for n in top_k_nodes(result["node_scores"], k=3)]
            status = "WORKING" if result["n_nodes_pruned"] > 0 else "BROKEN"
            rec = {
                "algorithm": algo,
                "model": model,
                "task": task,
                "n_nodes_total": result["n_nodes_total"],
                "n_nodes_pruned": result["n_nodes_pruned"],
                "wall_s": elapsed,
                "top3": top3,
                "status": status,
            }
            print(f"  nodes scored: {result['n_nodes_total']}, pruned: {result['n_nodes_pruned']}, "
                  f"wall: {elapsed}s, top3: {top3}")
        except Exception as exc:
            rec = {"algorithm": algo, "model": model, "task": task,
                   "status": "BROKEN", "error": str(exc)}
            print(f"  BROKEN: {exc}")
            any_broken = True
        results.append(rec)
        if rec["status"] != "WORKING":
            any_broken = True

    # Custom IOI data test
    if not args.skip_custom:
        print(f"\n[custom IOI data] validating NormalizedTaskSpec with {model} ...")
        custom_result = _run_custom_ioi_discovery(model, out_dir)
        results.append({"algorithm": "custom_ioi_eap-ig", **custom_result})
        print(f"  custom IOI: {custom_result}")
        if custom_result["status"] != "WORKING":
            any_broken = True

    summary = {
        "script": SCRIPT_NAME,
        "model": model,
        "task": task,
        "results": results,
        "overall_status": "BROKEN" if any_broken else "WORKING",
    }
    write_status(out_dir, summary)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"\nOverall: {summary['overall_status']}")
    for r in results:
        print(f"  {r['algorithm']}: {r['status']}")

    return 0 if not any_broken else 1


if __name__ == "__main__":
    sys.exit(main())
