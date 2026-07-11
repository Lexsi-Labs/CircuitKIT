"""Validation: every corruption strategy on real records.

For each registered strategy, build an applicable record (real where
possible) and verify ``apply()`` produces a successful pair.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status

SCRIPT = "06_strategies_all"


def main() -> int:
    out = make_results_dir(SCRIPT)
    # force-load all strategies via auto_detect's import side-effect
    from circuitkit.data import auto_detect  # noqa: F401
    from circuitkit.data.corruption import STRATEGY_REGISTRY
    from circuitkit.data.normalized import (
        ContrastiveRecord, ContrastSource, NormalizedDataset, DatasetShape,
    )

    # Real Q1 IOI prompt (matches the validation/_common.py fixture style)
    ioi = ContrastiveRecord(
        record_id="ioi-0",
        clean_prompt="When Mary and John went to the store, John gave a drink to",
        clean_answer=" Mary",
    )
    math = ContrastiveRecord(
        record_id="math-0",
        clean_prompt="Q: 5 plus 7 equals\nA:",
        clean_answer=" 12",
    )
    bool_q = ContrastiveRecord(
        record_id="bool-0",
        clean_prompt="Is the sky blue?",
        clean_answer="yes",
    )
    neg_q = ContrastiveRecord(
        record_id="neg-0",
        clean_prompt="The cake is delicious and everyone is happy.",
        clean_answer=".",
    )
    # Real MMLU sample (via MCQAdapter)
    from datasets import load_dataset
    from circuitkit.data.adapters.mcq import MCQAdapter
    raw = list(load_dataset("cais/mmlu", "high_school_world_history",
                            split="test", streaming=True).take(5))
    mmlu_ds = MCQAdapter().adapt(raw, name="mmlu", max_records=5)
    mmlu_rec = mmlu_ds.records[0]

    fixtures = {
        "entity_swap":      ioi,
        "token_swap":       ioi,
        "mcq_choice_swap":  mmlu_rec,
        "final_answer_swap_numeric": math,
        "final_answer_swap_boolean": bool_q,
        "logical_negation":  neg_q,
        "resample":         ioi,  # needs pool
    }

    rows = []
    t0 = time.time()
    for label, record in fixtures.items():
        strat_name = label.split("_with_")[0].split("_numeric")[0].split("_boolean")[0]
        strat_cls = STRATEGY_REGISTRY.get(strat_name)
        if strat_cls is None:
            rows.append({"strategy": strat_name, "label": label,
                         "ok": False, "note": "not registered"})
            continue
        s = strat_cls()
        try:
            if strat_name == "resample":
                pool = [ioi, math, bool_q, neg_q]
                result = s.corrupt(record, pool=pool)
            else:
                result = s.corrupt(record)
            rows.append({
                "strategy": strat_name, "label": label,
                "ok": result.succeeded,
                "corrupt_prompt": (result.corrupt_prompt[:80] + "...")
                                  if result.corrupt_prompt else None,
                "corrupt_answer": result.corrupt_answer,
                "note": result.notes,
            })
        except Exception as e:
            rows.append({"strategy": strat_name, "label": label,
                         "ok": False, "note": f"{type(e).__name__}: {e}"})
    elapsed = time.time() - t0

    n_ok = sum(1 for r in rows if r["ok"])
    status = {
        "script": SCRIPT,
        "module": "data.corruption.*",
        "input": {"n_strategies_tested": len(rows)},
        "output": {},
        "metrics": {"wall_seconds": round(elapsed, 2),
                    "n_ok": n_ok, "n_total": len(rows),
                    "rows": rows},
        "status": "WORKING" if n_ok == len(rows) else "PARTIAL",
    }
    write_status(out, status)
    print()
    print(f"6 corruption strategies tested on real records:")
    for r in rows:
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"  {mark}  {r['label']:32}  -> {r.get('note', '')[:80]}")
    print(f"\n  {n_ok}/{len(rows)} working   status: {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
