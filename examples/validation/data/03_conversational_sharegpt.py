"""Validation: conversational adapter on real ShareGPT data."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status

SCRIPT = "03_conversational_sharegpt"


def main() -> int:
    out = make_results_dir(SCRIPT)
    from datasets import load_dataset
    from circuitkit.data.adapters.conversational import ConversationalAdapter
    from circuitkit.data.worthiness import evaluate_worthiness

    raw = list(load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split="train", streaming=True,
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json"
    ).take(20))

    t0 = time.time()
    ds = ConversationalAdapter().adapt(
        raw, name="sharegpt",
        source="hf://anon8231489123/ShareGPT_Vicuna_unfiltered",
        max_records=20, max_turns_per_conv=1,
    )
    rep = evaluate_worthiness(ds)
    elapsed = time.time() - t0

    ds.save_json(str(out / "preview.json"))
    rep.save_json(str(out / "worthiness.json"))
    status = {
        "script": SCRIPT, "module": "data.adapters.ConversationalAdapter",
        "input": {"source": "hf://anon8231489123/ShareGPT_Vicuna_unfiltered",
                  "n_loaded": len(ds)},
        "output": {"preview": str(out / "preview.json"),
                   "worthiness": str(out / "worthiness.json")},
        "metrics": {"wall_seconds": round(elapsed, 2),
                    "verdict": rep.verdict.value,
                    "n_paired": ds.n_paired},
        "status": "WORKING" if len(ds) > 0 else "BROKEN",
    }
    write_status(out, status)
    print(f"\nConversationalAdapter on ShareGPT: {len(ds)} records, "
          f"verdict={rep.verdict.value}")
    print(f"  status: {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
