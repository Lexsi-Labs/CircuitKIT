"""Validation: instruction adapter on real Alpaca."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status, fetch_hf

SCRIPT = "02_instruction_alpaca"


def main() -> int:
    out = make_results_dir(SCRIPT)
    from circuitkit.data.adapters.instruction import InstructionAdapter
    from circuitkit.data.worthiness import evaluate_worthiness
    from transformers import GPT2Tokenizer

    raw = fetch_hf("tatsu-lab/alpaca", split="train", take=64)
    t0 = time.time()
    ds = InstructionAdapter().adapt(raw, name="alpaca",
                                    source="hf://tatsu-lab/alpaca",
                                    max_records=64)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    rep = evaluate_worthiness(ds, tokenizer=tok)
    elapsed = time.time() - t0

    ds.save_json(str(out / "preview.json"))
    rep.save_json(str(out / "worthiness.json"))

    status = {
        "script": SCRIPT, "module": "data.adapters.InstructionAdapter",
        "input": {"source": "hf://tatsu-lab/alpaca", "n_loaded": len(ds)},
        "output": {"preview": str(out / "preview.json"),
                   "worthiness": str(out / "worthiness.json")},
        "metrics": {"wall_seconds": round(elapsed, 2),
                    "n_paired": ds.n_paired,
                    "verdict": rep.verdict.value},
        "status": "WORKING" if len(ds) > 0 else "BROKEN",
    }
    write_status(out, status)
    print()
    print(f"InstructionAdapter on Alpaca: {len(ds)} records, "
          f"verdict={rep.verdict.value}")
    print(f"  status: {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
