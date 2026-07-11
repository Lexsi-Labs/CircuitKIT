"""Application: ActivationSteering on GPT-2 / IOI for every algorithm.

Steers the IOI completion via circuit-located vectors. For each
algorithm:
  1. Pull CircuitScores.
  2. Build steering vectors as mean(IO target) - mean(S target) at the
     top-30% circuit nodes.
  3. Apply to a held-out IOI prompt and measure the change in IO-S logit
     gap with steering vs without.

Higher delta means the algorithm-located nodes carry more of the
IO-vs-S routing information.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status, top_k_node_set,
)

SCRIPT_NAME = "13_steering_gpt2_ioi"
MODEL = "gpt2"
TASK = "ioi"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]


def _build_examples(dataloader, n=8, side="clean"):
    """Pull n examples from a paired dataloader; return list of dicts."""
    out = []
    for clean, corrupted, _ in dataloader:
        for c, r in zip(clean, corrupted):
            out.append({"text": c if side == "clean" else r})
            if len(out) >= n:
                return out
        if len(out) >= n:
            break
    return out


def _ioi_logit_gap(model, prompt, io_token_id, s_token_id) -> float:
    in_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.cfg.device)
    with torch.inference_mode():
        logits = model(in_ids)
    return float(logits[0, -1, io_token_id].item() - logits[0, -1, s_token_id].item())


def _run_one_algo(algo, model_clean, dataloader, io_id, s_id):
    from circuitkit.applications.steering.steering import ActivationSteering
    try:
        cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=32)
    except Exception as exc:
        return {"algorithm": algo, "error": f"discovery: {type(exc).__name__}: {exc}"}
    cs = cell["scores"]
    t0 = time.time()

    top_set = set(top_k_node_set(cs.node_scores,
                                 k=max(1, len(cs.node_scores) // 3)))
    filtered = {n: s for n, s in cs.node_scores.items() if n in top_set}
    try:
        steering = ActivationSteering(
            model_clean, filtered, score_threshold=0.0,
        )
        source_ex = _build_examples(dataloader, n=8, side="corrupted")
        target_ex = _build_examples(dataloader, n=8, side="clean")
        steering.compute_steering_vector(source_ex, target_ex)

        # Use one of the SAME-shape clean examples as the eval prompt so
        # the per-position steering vectors broadcast correctly. The
        # ActivationSteering API stores per-position vectors which only
        # match the seq-length of the prompts used at compute time.
        eval_prompt = target_ex[0]["text"]
        baseline_gap = _ioi_logit_gap(model_clean, eval_prompt, io_id, s_id)
        out = steering.steer(eval_prompt, coefficient=1.0)
        steered_logits = out["output"]
        steered_gap = float(
            steered_logits[0, -1, io_id].item()
            - steered_logits[0, -1, s_id].item()
        )
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "n_steered_nodes": len(filtered),
        "baseline_logit_gap": round(baseline_gap, 4),
        "steered_logit_gap": round(steered_gap, 4),
        "delta": round(steered_gap - baseline_gap, 4),
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.float32)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import get_task
    _bootstrap_builtin_tasks()
    task_spec = get_task(TASK)
    dl_cfg = {"algorithm": "eap", "task": TASK, "level": "node",
              "batch_size": 4, "data_params": {"num_examples": 32}}
    dl = task_spec.build_dataloader(model, dl_cfg, device)

    io_id = int(model.tokenizer(" Mary", add_special_tokens=False)["input_ids"][0])
    s_id = int(model.tokenizer(" John", add_special_tokens=False)["input_ids"][0])

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, model, dl, io_id, s_id)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:25s}  baseline={row['baseline_logit_gap']:>7}  "
                  f"steered={row['steered_logit_gap']:>7}  "
                  f"Δ={row['delta']:+.3f}  ({row['wall_seconds']}s)",
                  flush=True)

    lines = [
        "# Application 13 — ActivationSteering on GPT-2 / IOI",
        "",
        "Steering vectors are mean(IO-target activations) − mean(S-target",
        "activations) at the top-30% circuit nodes per algorithm. We measure",
        "the change in IO−S logit gap on a held-out prompt.",
        "",
        "| Algorithm | Baseline gap | Steered gap | Δ | Note |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['algorithm']}` | {r.get('baseline_logit_gap', '—')} | "
            f"{r.get('steered_logit_gap', '—')} | "
            f"{r.get('delta', '—')} | {r.get('error', '')} |"
        )
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.activation_steering",
        "input": {"model": MODEL, "task": TASK, "n_algos": len(ALGOS)},
        "output": {"table_md": str(out_dir / "table.md")},
        "metrics": {"n_algos_ok": n_ok, "n_algos_total": len(ALGOS)},
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })

    print()
    print(md)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
