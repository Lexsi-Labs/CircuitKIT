"""Validation: pairwise adapter on real CrowS-Pairs CSV."""
from __future__ import annotations
import json, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data_common import make_results_dir, write_status

SCRIPT = "01_pairwise_crowspairs"
CACHE = Path("/tmp/crows_pairs.csv")
URL = ("https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
       "data/crows_pairs_anonymized.csv")


def main() -> int:
    out = make_results_dir(SCRIPT)
    if not CACHE.exists():
        urllib.request.urlretrieve(URL, CACHE)
    from circuitkit.data.adapters.pairwise import PairwiseAdapter
    from circuitkit.data.worthiness import evaluate_worthiness
    from transformers import GPT2Tokenizer

    t0 = time.time()
    ds = PairwiseAdapter().adapt(str(CACHE), name="crows_pairs",
                                 source="github://nyu-mll/crows-pairs",
                                 max_records=128)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    rep = evaluate_worthiness(ds, tokenizer=tok,
                              expected_length_contract="preserve")
    elapsed = time.time() - t0

    ds.save_json(str(out / "preview.json"))
    rep.save_json(str(out / "worthiness.json"))

    status = {
        "script": SCRIPT, "module": "data.adapters.PairwiseAdapter",
        "input": {"source": URL, "n_loaded": len(ds)},
        "output": {"preview": str(out / "preview.json"),
                   "worthiness": str(out / "worthiness.json")},
        "metrics": {"wall_seconds": round(elapsed, 2),
                    "n_paired": ds.n_paired,
                    "verdict": rep.verdict.value,
                    "pillar_token_alignment_score":
                        next(c.score for c in rep.checks
                             if c.name == "token_alignment")},
        "status": "WORKING" if ds.n_paired > 0 else "BROKEN",
    }
    write_status(out, status)
    print()
    print(f"PairwiseAdapter on CrowS-Pairs: {len(ds)} records, "
          f"verdict={rep.verdict.value}")
    print(f"  token_alignment score: "
          f"{status['metrics']['pillar_token_alignment_score']:.0%}")
    print(f"  output: {out.relative_to(Path.cwd())}")
    print(f"  status: {status['status']}")
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
