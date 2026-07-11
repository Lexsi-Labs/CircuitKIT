"""Validation: MCQ adapter on three real datasets (MMLU + ARC + HellaSwag)."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status, fetch_hf

SCRIPT = "04_mcq_mmlu_arc_hellaswag"


def main() -> int:
    out = make_results_dir(SCRIPT)
    from circuitkit.data.adapters.mcq import MCQAdapter
    from circuitkit.data.worthiness import evaluate_worthiness
    from transformers import GPT2Tokenizer

    cases = [
        ("mmlu",      lambda: fetch_hf("cais/mmlu", "high_school_world_history",
                                       split="test", take=20)),
        ("arc-easy",  lambda: fetch_hf("allenai/ai2_arc", "ARC-Easy",
                                       split="test", take=20)),
        ("hellaswag", lambda: fetch_hf("Rowan/hellaswag", split="validation",
                                       take=20)),
    ]
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    summary = []
    t0 = time.time()
    for name, fetch in cases:
        raw = fetch()
        ds = MCQAdapter().adapt(raw, name=name, max_records=20)
        rep = evaluate_worthiness(ds, tokenizer=tok)
        ds.save_json(str(out / f"{name}_preview.json"))
        rep.save_json(str(out / f"{name}_worthiness.json"))
        summary.append({"dataset": name, "n": len(ds),
                        "verdict": rep.verdict.value})
    elapsed = time.time() - t0

    all_ok = all(s["n"] > 0 for s in summary)
    status = {
        "script": SCRIPT, "module": "data.adapters.MCQAdapter",
        "input": {"datasets": [s["dataset"] for s in summary]},
        "output": {"preview_dir": str(out)},
        "metrics": {"wall_seconds": round(elapsed, 2),
                    "per_dataset": summary},
        "status": "WORKING" if all_ok else "BROKEN",
    }
    write_status(out, status)
    print(f"\nMCQAdapter on 3 datasets:")
    for s in summary:
        print(f"  {s['dataset']:10}  n={s['n']:3}  verdict={s['verdict']}")
    print(f"  status: {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
