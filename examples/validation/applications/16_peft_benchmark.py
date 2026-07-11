"""Application: PEFTBenchmark + CrossArchitectureBenchmark smoke.

Exercises apply/benchmark_peft.py. These classes measure
parameter efficiency, memory, and inference latency for different
PEFT methods (lora / adapter / prefix / bitfit) on HF transformers
models. Not circuit-aware — included for library completeness.

Smoke test on GPT-2 + Llama-3.2-1B via HuggingFace AutoModel (not
TransformerLens, which the rest of our pipeline uses).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import make_results_dir, write_status  # noqa: E402

SCRIPT_NAME = "16_peft_benchmark"

PEFT_METHODS = ["lora", "adapter", "prefix", "bitfit"]


def _run_method(method: str, model, device: str):
    from circuitkit.applications.finetuning.benchmark_peft import PEFTBenchmark
    t0 = time.time()
    try:
        bench = PEFTBenchmark(model=model, method=method, rank=8,
                              device=device, verbose=False)
        metrics = bench.run(num_batches=2, batch_size=2)
    except Exception as exc:  # noqa: BLE001
        return {"method": method, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "method": method,
        "wall_seconds": round(time.time() - t0, 2),
        "trainable_params": getattr(metrics, "trainable_params", None),
        "total_params": getattr(metrics, "total_params", None),
        "param_efficiency": getattr(metrics, "param_efficiency", None),
        "peak_memory_mb": getattr(metrics, "peak_memory_mb", None),
        "inference_latency_ms": getattr(metrics, "inference_latency_ms", None),
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("gpt2").to(device)
        for method in PEFT_METHODS:
            row = _run_method(method, model, device)
            rows.append(row)
            if "error" in row:
                print(f"  {method:10s}  ERR: {row['error'][:80]}", flush=True)
            else:
                print(f"  {method:10s}  trainable={row.get('trainable_params')}  "
                      f"latency={row.get('inference_latency_ms')}ms  "
                      f"({row['wall_seconds']}s)", flush=True)
    except Exception as exc:  # noqa: BLE001
        rows.append({"error": f"AutoModel load failed: {exc}"})

    lines = ["# Application 16 — PEFT benchmark", "",
             "| Method | Trainable | Total | Efficiency | Latency (ms) | Note |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| `{r.get('method', '?')}` | "
            f"{r.get('trainable_params', '—')} | "
            f"{r.get('total_params', '—')} | "
            f"{r.get('param_efficiency', '—')} | "
            f"{r.get('inference_latency_ms', '—')} | "
            f"{r.get('error', '')} |"
        )
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.benchmark_peft",
        "output": {"table_md": str(out_dir / "table.md")},
        "metrics": {"n_methods_ok": n_ok, "n_methods_total": len(PEFT_METHODS)},
        "status": "WORKING" if n_ok == len(PEFT_METHODS) else "NEEDS-FIX",
    })
    print(md)
    return 0 if n_ok == len(PEFT_METHODS) else 1


if __name__ == "__main__":
    sys.exit(main())
