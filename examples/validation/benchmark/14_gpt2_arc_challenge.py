"""Benchmark: all 9 algorithms on gpt2 + arc_challenge.

ARC-Challenge MCQ (allenai/ai2_arc) via NormalizedTaskSpec + MCQChoiceSwap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_common import (  # noqa: E402
    ALL_ALGOS, run_benchmark_cell, render_summary, make_results_dir, write_status,
)

SCRIPT_NAME = "14_gpt2_arc_challenge"
MODEL = "gpt2"
TASK = "arc_challenge"
NEEDS_CUSTOM_REGISTRATION = True


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    if NEEDS_CUSTOM_REGISTRATION:
        from _custom_tasks import register_all
        register_all()
    rows = []

    for algo in ALL_ALGOS:
        if algo == "eap-exact":
            num_examples = 8
        elif algo in ("atp-gd", "eap-gp"):
            num_examples = 12
        else:
            num_examples = 24

        row = run_benchmark_cell(
            algorithm=algo, model=MODEL, task=TASK,
            num_examples=num_examples, batch_size=1, ig_steps=3,
            target_sparsity=0.1, scope="heads",
            ibcircuit_epochs=200,
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
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "benchmark[gpt2-arc_challenge]",
        "input": {"model": MODEL, "task": TASK, "n_algos": len(ALL_ALGOS)},
        "output": {
            "table_md": str(out_dir / "table.md"),
            "rows_json": str(out_dir / "rows.json"),
        },
        "metrics": {"n_algos_ok": n_ok, "n_algos_total": len(ALL_ALGOS)},
        "status": "WORKING" if n_ok == len(ALL_ALGOS) else "NEEDS-FIX",
    })

    print(f"\n{MODEL}/{TASK}: {n_ok}/{len(ALL_ALGOS)} algorithms succeeded")
    print(md)
    return 0 if n_ok == len(ALL_ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
