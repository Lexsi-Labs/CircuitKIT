"""Aggregate bench + apps results across all runs into one master table.

Reads the most recent benchmark and applications result directories,
joins per-algorithm scores across (task, model, application), and
emits a single Markdown summary that's ready to paste into the paper.

Usage:
    python validation/master_aggregator.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "validation" / "results"


def _latest_run_with(prefix: str) -> Optional[Path]:
    """Find the most recent run dir that contains a script-name starting with prefix."""
    candidates = sorted(RESULTS_DIR.glob("*"), reverse=True)
    for run in candidates:
        if not run.is_dir():
            continue
        for child in run.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                return run
    return None


def _read_rows(run_dir: Path, script_glob: str) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for script_dir in sorted(run_dir.glob(script_glob)):
        if not script_dir.is_dir():
            continue
        rows_path = script_dir / "rows.json"
        if not rows_path.exists():
            continue
        try:
            rows = json.loads(rows_path.read_text())
            if isinstance(rows, list):
                out[script_dir.name] = rows
        except json.JSONDecodeError:
            continue
    return out


def _format_score(x: Any) -> str:
    if x is None or x == "":
        return "—"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        if isinstance(x, str) and "error" in x.lower():
            return "ERR"
        return str(x)[:8]


def main() -> int:
    bench_runs = sorted(RESULTS_DIR.glob("*"), reverse=True)
    bench_data: Dict[str, List[Dict[str, Any]]] = {}  # cell -> rows
    apps_data: Dict[str, List[Dict[str, Any]]] = {}

    seen_bench = set()
    seen_apps = set()
    for run in bench_runs:
        if not run.is_dir():
            continue
        for child in run.iterdir():
            if not child.is_dir():
                continue
            n = child.name
            if (n.startswith("0") or n.startswith("1")) and "_gpt2_" in n or "_llama" in n:
                if "lora_healing" in n or "knowledge_editing" in n or \
                        "pruning" in n or "unlearning" in n or "steering" in n:
                    if n not in seen_apps:
                        rp = child / "rows.json"
                        if rp.exists():
                            try:
                                apps_data[n] = json.loads(rp.read_text())
                                seen_apps.add(n)
                            except json.JSONDecodeError:
                                pass
                else:
                    if n not in seen_bench:
                        rp = child / "rows.json"
                        if rp.exists():
                            try:
                                bench_data[n] = json.loads(rp.read_text())
                                seen_bench.add(n)
                            except json.JSONDecodeError:
                                pass

    out_dir = REPO / "validation" / "results" / "master_aggregator"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Benchmark master table: algo (rows) x cell (cols), value = patching score
    by_algo_bench: Dict[str, Dict[str, str]] = defaultdict(dict)
    for cell, rows in sorted(bench_data.items()):
        for row in rows:
            algo = row.get("algorithm", "?")
            score = row.get("patching")
            if "error" in row and not score:
                by_algo_bench[algo][cell] = "ERR"
            else:
                by_algo_bench[algo][cell] = _format_score(score)

    # ----- Apps master table: algo x cell, value = primary metric
    def _apps_metric(row: Dict[str, Any]) -> str:
        if "error" in row:
            return "ERR"
        for k in ("circuit_logit_diff", "circuit_logit_gap",
                  "edit_success_rate", "forget_delta",
                  "healed_logit_diff"):
            if k in row and row[k] is not None:
                return _format_score(row[k])
        return "—"

    by_algo_apps: Dict[str, Dict[str, str]] = defaultdict(dict)
    for cell, rows in sorted(apps_data.items()):
        for row in rows:
            algo = row.get("algorithm", "?")
            by_algo_apps[algo][cell] = _apps_metric(row)

    all_algos = sorted(set(by_algo_bench.keys()) | set(by_algo_apps.keys()))
    bench_cells = sorted(bench_data.keys())
    apps_cells = sorted(apps_data.keys())

    lines = [
        "# Master comparison table",
        "",
        "All values are the per-algorithm primary metric for each cell. "
        "Benchmark rows show Pillar-1 patching score (logit-diff faithfulness, "
        "higher is better). Application rows show the cell's main metric: ",
        "circuit-pruned logit-diff for pruning cells; LoRA-healed logit-diff "
        "for soft-healing; ROME edit-success rate for knowledge editing; "
        "forget-NLL delta for unlearning. ERR = the cell crashed for that "
        "algorithm. '—' = not run.",
        "",
        "## Benchmark — discovery faithfulness (Pillar 1 patching)",
        "",
    ]

    if bench_cells:
        header = ["Algorithm"] + [c.replace("_gpt2_", " ").replace("_llama1b_", " L1 ").replace("_", " ") for c in bench_cells]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for algo in all_algos:
            row = [f"`{algo}`"]
            for c in bench_cells:
                row.append(by_algo_bench.get(algo, {}).get(c, "—"))
            lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Applications — circuit-guided downstream metrics", ""]
    if apps_cells:
        header = ["Algorithm"] + [c.replace("_gpt2_", " ").replace("_", " ") for c in apps_cells]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for algo in all_algos:
            row = [f"`{algo}`"]
            for c in apps_cells:
                row.append(by_algo_apps.get(algo, {}).get(c, "—"))
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("(no applications cells found in any run yet)")

    md = "\n".join(lines) + "\n"
    out_path = out_dir / "master_table.md"
    out_path.write_text(md)
    print(md)
    print(f"\nSaved to: {out_path}")

    # Also write JSON for machine consumption.
    summary = {
        "n_bench_cells": len(bench_cells),
        "n_apps_cells": len(apps_cells),
        "algorithms": all_algos,
        "bench": {a: by_algo_bench.get(a, {}) for a in all_algos},
        "apps": {a: by_algo_apps.get(a, {}) for a in all_algos},
    }
    (out_dir / "master_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
