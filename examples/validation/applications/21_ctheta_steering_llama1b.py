"""C-ΔΘ contrastive weight steering on Llama-3.2-1B / IOI.

Scales the GPT-2 C-ΔΘ result (cell 20) to a 1B-parameter Llama
backbone. Grouped-query attention is handled by CircuitWeightSteering
via the head_idx → kv_head_idx floor-div mapping in
apply/weight_steering.py.

We compare two algorithms on Llama-1B (eap, eap-ig) rather than the
full 12-algo matrix — the goal is to demonstrate scaling, not the
per-algorithm comparison (which already lives in cell 20 for GPT-2).
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

SCRIPT_NAME = "21_ctheta_steering_llama1b"
MODEL = "meta-llama/Llama-3.2-1B"
TASK = "ioi"
ALGOS = ["eap", "eap-ig"]


def _ioi_loss(model, batch, *, target="positive"):
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    clean, corrupted, label = batch
    pair_padding = getattr(batch, "pair_padding_side", None)
    clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
        model, clean, corrupted, pair_padding_side=pair_padding,
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


def _run_algo(algo, model_clean, pos_dl, neg_dl, eval_dl, side_dl, k=2.0):
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering
    try:
        cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=16,
                                    precision="bfloat16")
    except Exception as exc:
        return {"algorithm": algo,
                "error": f"discovery: {type(exc).__name__}: {str(exc)[:200]}"}
    cs = cell["scores"]
    t0 = time.time()
    base_target = _eval_logit_diff(model_clean, eval_dl)
    base_side = _eval_logit_diff(model_clean, side_dl)
    try:
        cws = CircuitWeightSteering(model_clean, cs.node_scores, top_k_frac=0.05)
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
        return {"algorithm": algo,
                "error": f"steering: {type(exc).__name__}: {str(exc)[:200]}"}
    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "n_heads": len(cws.head_names),
        "param_count": cws.parameter_count(),
        "steering_norm": round(cws.total_steering_norm(), 4),
        "k": k,
        "base_target_diff": round(base_target, 4),
        "post_target_diff": round(post_target, 4),
        "target_change": round(post_target - base_target, 4),
        "base_side_diff": round(base_side, 4),
        "post_side_diff": round(post_side, 4),
        "side_effect": round(post_side - base_side, 4),
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = HookedTransformer.from_pretrained(
            MODEL, device=device, dtype=torch.bfloat16,
        )
    except Exception as exc:
        write_status(out_dir, {
            "script": SCRIPT_NAME, "status": "BROKEN",
            "error": f"could not load {MODEL}: {exc}",
        })
        return 1
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import get_task
    _bootstrap_builtin_tasks()
    base = {"algorithm": "eap", "task": TASK, "level": "node",
            "batch_size": 2, "data_params": {"num_examples": 16, "seed": 42}}
    pos_dl = get_task(TASK).build_dataloader(model, base, device)
    neg_dl = get_task(TASK).build_dataloader(
        model, {**base, "data_params": {"num_examples": 16, "seed": 99}}, device,
    )
    eval_dl = get_task(TASK).build_dataloader(
        model, {**base, "data_params": {"num_examples": 16, "seed": 1234}}, device,
    )
    side = {"algorithm": "eap", "task": "gender_bias", "level": "node",
            "batch_size": 2, "data_params": {"num_examples": 8}}
    side_dl = get_task("gender_bias").build_dataloader(model, side, device)

    rows = []
    for algo in ALGOS:
        row = _run_algo(algo, model, pos_dl, neg_dl, eval_dl, side_dl)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:10s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:10s}  Δtarget={row['target_change']:+.3f}  "
                  f"side={row['side_effect']:+.3f}  |w_b|={row['steering_norm']}  "
                  f"({row['wall_seconds']}s)", flush=True)

    md = ["# Application 21 — C-ΔΘ Contrastive Weight Steering on Llama-3.2-1B / IOI",
          "",
          "Scales the GPT-2 / IOI C-ΔΘ result (cell 20) to a 1B-parameter Llama "
          "backbone. GQA head-to-kv-head mapping handled by the floor-div in "
          "CircuitWeightSteering.",
          "",
          "| Algorithm | Δ target | Side-effect | |w_b| | Params | Wall |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| `{r['algorithm']}` | ERR | — | — | — | — |")
        else:
            md.append(
                f"| `{r['algorithm']}` | {r['target_change']:+.3f} | "
                f"{r['side_effect']:+.3f} | {r.get('steering_norm', '—')} | "
                f"{r['param_count']} | {r['wall_seconds']}s |"
            )
    (out_dir / "table.md").write_text("\n".join(md) + "\n")
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.weight_steering",
        "input": {"model": MODEL, "task": TASK, "n_algos": len(ALGOS), "k": 2.0},
        "metrics": {"n_algos_ok": n_ok, "n_algos_total": len(ALGOS)},
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
