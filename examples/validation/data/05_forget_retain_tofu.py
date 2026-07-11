"""Validation: forget_retain adapter on real TOFU forget01 + retain99."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status, fetch_hf

SCRIPT = "05_forget_retain_tofu"


def main() -> int:
    out = make_results_dir(SCRIPT)
    from circuitkit.data.adapters.forget_retain import ForgetRetainAdapter
    from circuitkit.data.worthiness import evaluate_worthiness

    forget = fetch_hf("locuslab/TOFU", "forget01", split="train", take=20)
    retain = fetch_hf("locuslab/TOFU", "retain99", split="train", take=20)
    raw = {"forget": forget, "retain": retain}

    t0 = time.time()
    ds = ForgetRetainAdapter().adapt(raw, name="tofu",
                                      source="hf://locuslab/TOFU")
    rep = evaluate_worthiness(ds)
    elapsed = time.time() - t0

    ds.save_json(str(out / "preview.json"))
    rep.save_json(str(out / "worthiness.json"))
    status = {
        "script": SCRIPT, "module": "data.adapters.ForgetRetainAdapter",
        "input": {"source": "hf://locuslab/TOFU forget01+retain99",
                  "n_forget": ds.meta.get("n_forget"),
                  "n_retain": ds.meta.get("n_retain")},
        "output": {"preview": str(out / "preview.json"),
                   "worthiness": str(out / "worthiness.json")},
        "metrics": {"wall_seconds": round(elapsed, 2),
                    "verdict": rep.verdict.value},
        "status": "WORKING" if (ds.meta.get("n_forget", 0) > 0
                                and ds.meta.get("n_retain", 0) > 0)
                  else "BROKEN",
    }
    write_status(out, status)
    print(f"\nForgetRetainAdapter on TOFU: {ds.meta['n_forget']} forget + "
          f"{ds.meta['n_retain']} retain, verdict={rep.verdict.value}")
    print(f"  status: {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
