"""C-ΔΘ hyperparameter grid: k-coefficient sweep × top-k% head slice
on GPT-2 / IOI.

Produces the paper's figure showing the trade-off between target Δ
and side-effect Δ as we vary:
  k         : coefficient applied to the steering vector
  top_k_frac: fraction of circuit-located heads we fine-tune

We hold the algorithm fixed (eap-ig — fastest reliable discovery) and
sweep the two main C-ΔΘ knobs.
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
    get_or_run_discovery, make_results_dir, write_status,
)

SCRIPT_NAME = "23_ctheta_k_grid_gpt2"
MODEL = "gpt2"
TASK = "ioi"
ALGO = "eap-ig"

K_GRID = [0.5, 1.0, 2.0, 4.0]
TOP_K_FRAC_GRID = [0.025, 0.05, 0.10]


def _ioi_loss(model, batch, *, target="positive"):
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    clean, corrupted, label = batch
    clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
        model, clean, corrupted,
        pair_padding_side=getattr(batch, "pair_padding_side", None),
    )
    logits = model(clean_tokens, attention_mask=attn_mask)
    losses = []
    for i in range(logits.shape[0]):
        last = input_lengths[i] - 1
        io_id = int(label[i, 0].item())
        s_id = int(label[i, 1].item())
        diff = logits[i, last, io_id] - logits[i, last, s_id]
        losses.append(diff if target == "negative" else -diff)
    return sum(losses) / max(1, len(losses))


def _eval_logit_diff(model, dataloader) -> float:
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    diffs = []
    for clean, corrupted, label in dataloader:
        clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
            model, clean, corrupted,
            pair_padding_side=getattr(dataloader, "pair_padding_side", None),
        )
        with torch.inference_mode():
            logits = model(clean_tokens, attention_mask=attn_mask)
        for i in range(logits.shape[0]):
            last = input_lengths[i] - 1
            a = int(label[i, 0].item())
            b = int(label[i, 1].item())
            diffs.append(logits[i, last, a].item() - logits[i, last, b].item())
    return sum(diffs) / max(1, len(diffs))


def _run_one_point(model_clean, cs, pos_dl, neg_dl, eval_dl, side_dl,
                   k: float, top_k_frac: float):
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering
    t0 = time.time()
    base_target = _eval_logit_diff(model_clean, eval_dl)
    base_side = _eval_logit_diff(model_clean, side_dl)
    try:
        cws = CircuitWeightSteering(model_clean, cs.node_scores,
                                    top_k_frac=top_k_frac)
        cws.fine_tune_positive(
            pos_dl, lambda m, b: _ioi_loss(m, b, target="positive"),
            n_steps=15, lr=1e-5,
        )
        cws.fine_tune_negative(
            neg_dl, lambda m, b: _ioi_loss(m, b, target="negative"),
            n_steps=15, lr=1e-5,
        )
        cws.compute_steering_vector()
        steered = cws.apply_steering(model_clean, k=k)
        post_target = _eval_logit_diff(steered, eval_dl)
        post_side = _eval_logit_diff(steered, side_dl)
        del steered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        return {"k": k, "top_k_frac": top_k_frac,
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return {
        "k": k,
        "top_k_frac": top_k_frac,
        "n_heads": len(cws.head_names),
        "param_count": cws.parameter_count(),
        "steering_norm": round(cws.total_steering_norm(), 4),
        "target_change": round(post_target - base_target, 4),
        "side_effect": round(post_side - base_side, 4),
        "wall_seconds": round(time.time() - t0, 2),
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
    base = {"algorithm": ALGO, "task": TASK, "level": "node",
            "batch_size": 4, "data_params": {"num_examples": 32, "seed": 42}}
    pos_dl = get_task(TASK).build_dataloader(model, base, device)
    neg_dl = get_task(TASK).build_dataloader(
        model, {**base, "data_params": {"num_examples": 32, "seed": 99}}, device,
    )
    eval_dl = get_task(TASK).build_dataloader(
        model, {**base, "data_params": {"num_examples": 32, "seed": 1234}}, device,
    )
    side = {"algorithm": ALGO, "task": "gender_bias", "level": "node",
            "batch_size": 4, "data_params": {"num_examples": 16}}
    side_dl = get_task("gender_bias").build_dataloader(model, side, device)

    cell = get_or_run_discovery(ALGO, MODEL, TASK, num_examples=32)
    cs = cell["scores"]

    rows = []
    for top_k_frac in TOP_K_FRAC_GRID:
        for k in K_GRID:
            row = _run_one_point(model, cs, pos_dl, neg_dl, eval_dl, side_dl,
                                 k=k, top_k_frac=top_k_frac)
            rows.append(row)
            if "error" in row:
                print(f"  top_k={top_k_frac}  k={k}  ERR: {row['error'][:80]}",
                      flush=True)
            else:
                print(f"  top_k={top_k_frac}  k={k}  Δtarget="
                      f"{row['target_change']:+.3f}  side="
                      f"{row['side_effect']:+.3f}  ({row['wall_seconds']}s)",
                      flush=True)

    md = ["# Application 23 — C-ΔΘ k × top_k_frac Hyperparameter Grid (GPT-2 / IOI)",
          "",
          f"Algorithm: `{ALGO}`. n_steps=15, lr=1e-5.",
          "",
          "| top_k_frac | k | Δ target | Side-effect | |w_b| | Heads | Params | Wall |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['top_k_frac']} | {r['k']} | ERR | — | — | — | — | — |")
        else:
            md.append(
                f"| {r['top_k_frac']} | {r['k']} | {r['target_change']:+.3f} | "
                f"{r['side_effect']:+.3f} | {r['steering_norm']} | "
                f"{r['n_heads']} | {r['param_count']} | {r['wall_seconds']}s |"
            )
    (out_dir / "table.md").write_text("\n".join(md) + "\n")
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.weight_steering (k × top_k grid)",
        "input": {"model": MODEL, "task": TASK, "algo": ALGO,
                  "k_grid": K_GRID, "top_k_frac_grid": TOP_K_FRAC_GRID},
        "metrics": {"n_points_ok": n_ok, "n_points_total": len(rows)},
        "status": "WORKING" if n_ok == len(rows) else "NEEDS-FIX",
    })
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
