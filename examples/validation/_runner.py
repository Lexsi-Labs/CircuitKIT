"""Run every validation script in sequence and aggregate results.

Usage:
    cd validation && python _runner.py [layer]

`layer` defaults to "visualizations". Future layers ("applications",
"algos", "data", "metrics") will live in sibling subfolders and be
selected via the same arg.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(layer: str = "visualizations") -> int:
    scripts = sorted((ROOT / layer).glob("[0-9]*_*.py"))
    if not scripts:
        print(f"No scripts in {ROOT / layer}")
        return 1

    # Stable run id shared across all child scripts
    import datetime
    run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    os.environ["VALIDATION_RUN_ID"] = run_id
    run_dir = ROOT / "results" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {layer} validation run {run_id} ===")
    print(f"Output: {run_dir}\n")

    rows = []
    t_total = time.time()
    for s in scripts:
        t0 = time.time()
        rc = subprocess.call([sys.executable, str(s)], env=os.environ.copy())
        elapsed = time.time() - t0
        # Pull this script's status.json
        status_dir = run_dir / s.stem
        status_file = status_dir / "status.json"
        if status_file.exists():
            status = json.loads(status_file.read_text())
        else:
            status = {"status": "BROKEN", "error": "no status.json produced"}
        rows.append({
            "script": s.stem,
            "exit_code": rc,
            "wall_seconds": round(elapsed, 2),
            "status": status.get("status", "UNKNOWN"),
            "module": status.get("module", "?"),
        })
        print()  # spacing between scripts

    # Write summary
    total_elapsed = time.time() - t_total
    summary = {
        "layer": layer,
        "run_id": run_id,
        "n_scripts": len(scripts),
        "n_working": sum(1 for r in rows if r["status"] == "WORKING"),
        "n_broken": sum(1 for r in rows if r["status"] == "BROKEN"),
        "n_needs_fix": sum(1 for r in rows if r["status"] == "NEEDS-FIX"),
        "wall_seconds_total": round(total_elapsed, 2),
        "rows": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    md = build_markdown(summary)
    (run_dir / "summary.md").write_text(md)

    print()
    print("=" * 60)
    print(f"Layer:   {layer}")
    print(f"Run id:  {run_id}")
    print(f"Total:   {summary['n_scripts']} scripts, "
          f"{summary['n_working']} WORKING, "
          f"{summary['n_broken']} BROKEN, "
          f"{summary['n_needs_fix']} NEEDS-FIX")
    print(f"Time:    {total_elapsed:.1f}s")
    print(f"Output:  {run_dir / 'summary.md'}")
    print("=" * 60)

    return 0 if summary["n_broken"] == 0 else 1


def build_markdown(summary: dict) -> str:
    lines = [
        f"# Validation summary — {summary['layer']}",
        "",
        f"- Run id: `{summary['run_id']}`",
        f"- Total: **{summary['n_scripts']}** scripts in **{summary['wall_seconds_total']}s**",
        f"- Working: {summary['n_working']}",
        f"- Needs fix: {summary['n_needs_fix']}",
        f"- Broken: {summary['n_broken']}",
        "",
        "| Script | Status | Module | Wall (s) |",
        "|---|---|---|---|",
    ]
    for r in summary["rows"]:
        lines.append(f"| `{r['script']}` | {r['status']} | "
                     f"`{r['module']}` | {r['wall_seconds']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    layer = sys.argv[1] if len(sys.argv) > 1 else "visualizations"
    sys.exit(main(layer))
