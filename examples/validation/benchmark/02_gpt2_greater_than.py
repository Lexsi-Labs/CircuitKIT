"""Benchmark: all 9 algorithms on GPT-2 + greater_than.

`greater_than` is a complementary algorithmic task to IOI (Hanna et al.
2023) — predicting which year is greater. Tests whether the algorithms
generalize beyond the IOI circuit to a different mechanism.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_common import (  # noqa: E402
    ALL_ALGOS, run_benchmark_cell, render_summary, make_results_dir, write_status,
)

SCRIPT_NAME = "02_gpt2_greater_than"
MODEL = "gpt2"
TASK = "greater_than"


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    rows = []

    for algo in ALL_ALGOS:
        if algo in ("eap-exact", "atp-gd", "eap-gp"):
            num_examples = 16
        else:
            num_examples = 32

        row = run_benchmark_cell(
            algorithm=algo, model=MODEL, task=TASK,
            num_examples=num_examples, batch_size=1, ig_steps=3,
            target_sparsity=0.1, scope="heads",
        )
        rows.append(row)
        msg = (
            f"{algo:25s}  "
            f"discover={row.get('discovery_seconds', '?'):>6}s  "
            f"eval={row.get('eval_seconds', '?'):>6}s  "
            f"patch={row.get('patching')}  "
            f"abl={row.get('ablation')}"
        )
        if "error" in row:
            msg += f"  ERR: {row['error'][:80]}"
        print(msg, flush=True)

    md = render_summary(rows)
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    status = {
        "script": SCRIPT_NAME,
        "module": "benchmark[gpt2-greater_than]",
        "input": {"model": MODEL, "task": TASK, "n_algos": len(ALL_ALGOS)},
        "output": {
            "table_md": str(out_dir / "table.md"),
            "rows_json": str(out_dir / "rows.json"),
        },
        "metrics": {"n_algos_ok": n_ok, "n_algos_total": len(ALL_ALGOS)},
        "status": "WORKING" if n_ok == len(ALL_ALGOS) else "NEEDS-FIX",
    }
    write_status(out_dir, status)

    print(f"\n{MODEL}/{TASK}: {n_ok}/{len(ALL_ALGOS)} algorithms succeeded")
    print(md)
    return 0 if status["status"] == "WORKING" else 1


if __name__ == "__main__":
    sys.exit(main())
