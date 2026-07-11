"""Aggregate all 4 benchmark cells into one master table.

Reads `rows.json` from the four 0X_*.py result dirs of the current run,
groups by algorithm, and writes a single Markdown table cross-referenced
by (model, task). Run this AFTER scripts 01-04.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_common import make_results_dir, write_status, ALL_ALGOS  # noqa: E402

SCRIPT_NAME = "99_aggregate"

CELLS = [
    # IOI-type (synthetic, paired) tasks
    ("01_gpt2_ioi",            "gpt2",                   "ioi"),
    ("02_gpt2_greater_than",   "gpt2",                   "greater_than"),
    ("03_llama1b_ioi",         "meta-llama/Llama-3.2-1B", "ioi"),
    ("04_llama1b_greater_than","meta-llama/Llama-3.2-1B", "greater_than"),
    # Custom HF data (MMLU = 57-subject MCQ from real HuggingFace dataset)
    ("05_gpt2_mmlu",           "gpt2",                   "mmlu"),
    ("06_llama1b_mmlu",        "meta-llama/Llama-3.2-1B", "mmlu"),
]


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    run_id = os.environ.get("VALIDATION_RUN_ID")
    if not run_id:
        print("VALIDATION_RUN_ID not set; aggregator must be run via _runner.py")
        return 1
    base = Path(__file__).resolve().parent.parent / "results" / run_id

    cell_rows = {}
    for cell_dir, model, task in CELLS:
        rows_path = base / cell_dir / "rows.json"
        if not rows_path.exists():
            cell_rows[(model, task)] = None
            continue
        cell_rows[(model, task)] = json.loads(rows_path.read_text())

    # Master table: rows = algorithms, cols = (model, task) -> patching score.
    lines = ["# Algo Benchmark — paper-style 6-pillar (subset)", ""]
    header = ["Algorithm"]
    for _, m, t in CELLS:
        header.append(f"{m.split('/')[-1]} / {t}")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    def fmt_cell(row):
        if row is None:
            return "—"
        if "error" in row:
            return "ERR"
        p = row.get("patching")
        if p is None:
            return "—"
        try:
            return f"{float(p):.3f}"
        except (TypeError, ValueError):
            return "—"

    for algo in ALL_ALGOS:
        cells = [f"`{algo}`"]
        for cell_dir, m, t in CELLS:
            rows = cell_rows.get((m, t))
            if rows is None:
                cells.append("(skipped)")
                continue
            row = next((r for r in rows if r.get("algorithm") == algo), None)
            cells.append(fmt_cell(row))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Patching score = Pillar-1 faithfulness (higher is better).")
    lines.append("ERR = discovery or evaluation crashed for that cell.")
    lines.append("'—' = not produced this run.")

    md = "\n".join(lines) + "\n"
    (out_dir / "master_table.md").write_text(md)
    print(md)

    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "benchmark[aggregate]",
        "input": {"cells": [c[0] for c in CELLS]},
        "output": {"master_table": str(out_dir / "master_table.md")},
        "status": "WORKING",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
