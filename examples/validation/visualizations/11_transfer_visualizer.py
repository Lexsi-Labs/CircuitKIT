"""Validation: evaluate.transfer_visualizer — Pillar-6 transfer matrix viz."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import get_fixture, make_results_dir, write_status

SCRIPT_NAME = "11_transfer_visualizer"

def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    fx = get_fixture()
    from circuitkit.evaluation.transfer_visualizer import TransferMatrixVisualizer

    # Synthetic-but-realistic-shaped square transfer matrix (matches Q1 paper format).
    task_names = ["ioi", "double_io", "sva", "capital_country"]
    n = len(task_names)
    rng = np.random.default_rng(0)
    matrix = rng.uniform(0.3, 0.8, size=(n, n))
    np.fill_diagonal(matrix, 1.0)         # self-transfer
    matrix[0, 1] = 0.57                   # IOI -> double_io (paper Q1)

    t0 = time.time()
    v = TransferMatrixVisualizer(task_names=task_names)
    img_path = out_dir / "transfer_matrix.png"
    plot_err = None
    try:
        v.heatmap(matrix=matrix, output_path=str(img_path),
                  title="GPT-2 IOI cross-task transfer matrix")
        plot_ok = img_path.exists() and img_path.stat().st_size > 1000
    except Exception as e:
        plot_ok = False
        plot_err = f"{type(e).__name__}: {str(e)[:120]}"
    elapsed = time.time() - t0

    status = {
        "script": SCRIPT_NAME,
        "module": "circuitkit.evaluation.transfer_visualizer.TransferMatrixVisualizer",
        "input": {
            "shape": list(matrix.shape),
            "task_names": task_names,
        },
        "output": {
            "png": str(img_path) if plot_ok else None,
            "png_bytes": img_path.stat().st_size if plot_ok else 0,
        },
        "metrics": {"wall_seconds": round(elapsed, 3)},
        "status": "WORKING" if plot_ok else "NEEDS-FIX",
    }
    if not plot_ok and plot_err:
        status["error"] = plot_err
    write_status(out_dir, status)
    print()
    print(f"TransferMatrixVisualizer — {fx['meta']['model']} {fx['meta']['task']}")
    print(f"  Matrix shape: {matrix.shape}")
    if plot_ok:
        print(f"  Output PNG:   {img_path.relative_to(Path.cwd())}  "
              f"({img_path.stat().st_size:,} bytes)")
    else:
        print(f"  Plot:         FAILED ({status.get('error', 'unknown')})")
    print(f"  Wall time:    {elapsed:.2f}s")
    print(f"  Status:       {status['status']}")
    return 0 if plot_ok else 1

if __name__ == "__main__":
    sys.exit(main())
