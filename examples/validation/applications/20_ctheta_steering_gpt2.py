"""C-ΔΘ contrastive weight steering vs MLP2 LoRA baseline (GPT-2 / IOI).

Mirrors /AdityaKasliwal/ckit_theta/examples/weight_steering_experiment.py:

  Pipeline A (per-algorithm circuit-guided contrastive weight steering):
    1. Discover circuit C on the source task.
    2. Fine-tune two copies of the model on the per-head W_Q/K/V/O
       slices of the top-k% nodes — one on the POSITIVE behaviour
       loss, one on the NEGATIVE loss.
    3. Compute w_b = θ_pos − θ_neg per slice.
    4. Apply steered = target + k · w_b.
    5. Eval target metric + side-effect metric.

  Pipeline B (architecture-prior baseline):
    LoRA on layer-2 MLP only, fine-tuned on the positive loss.
"""
from __future__ import annotations

import copy
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

SCRIPT_NAME = "20_ctheta_steering_gpt2"
MODEL = "gpt2"
TASK = "ioi"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]


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


def _pipeline_a_circuit_steering(algo, model_clean, pos_dl, neg_dl, eval_dl,
                                 side_dl, k=2.0, n_steps=20, lr=1e-5):
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering
    cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=32)
    cs = cell["scores"]
    t0 = time.time()
    base_target = _eval_logit_diff(model_clean, eval_dl)
    base_side = _eval_logit_diff(model_clean, side_dl)

    cws = CircuitWeightSteering(model_clean, cs.node_scores, top_k_frac=0.05)
    cws.fine_tune_positive(
        pos_dl, lambda m, b: _ioi_loss(m, b, target="positive"),
        n_steps=n_steps, lr=lr,
    )
    cws.fine_tune_negative(
        neg_dl, lambda m, b: _ioi_loss(m, b, target="negative"),
        n_steps=n_steps, lr=lr,
    )
    cws.compute_steering_vector()
    steered = cws.apply_steering(model_clean, k=k)
    post_target = _eval_logit_diff(steered, eval_dl)
    post_side = _eval_logit_diff(steered, side_dl)
    del steered
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
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


def _pipeline_b_mlp2_lora(model_clean, pos_dl, eval_dl, side_dl,
                          n_steps=20, lr=1e-4):
    from circuitkit.applications.finetuning.soft_healing import CircuitLoRA
    t0 = time.time()
    model = copy.deepcopy(model_clean)
    surrogate = {"MLP 2": 1.0}
    lora = CircuitLoRA(model, surrogate, lora_rank=4, lora_alpha=8.0,
                       score_threshold=0.0).to(model.cfg.device)
    base_target = _eval_logit_diff(model, eval_dl)
    base_side = _eval_logit_diff(model, side_dl)

    optim = torch.optim.AdamW(lora.parameters(), lr=lr)
    hooks = lora.get_lora_hooks()
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    pair_padding = getattr(pos_dl, "pair_padding_side", None)
    step = 0
    model.train()
    while step < n_steps:
        for clean, corrupted, label in pos_dl:
            if step >= n_steps:
                break
            clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
                model, clean, corrupted, pair_padding_side=pair_padding,
            )
            with model.hooks(fwd_hooks=hooks):
                logits = model(clean_tokens, attention_mask=attn_mask)
            losses = []
            for i in range(logits.shape[0]):
                last = input_lengths[i] - 1
                io = int(label[i, 0].item())
                sid = int(label[i, 1].item())
                losses.append(-(logits[i, last, io] - logits[i, last, sid]))
            loss = sum(losses) / max(1, len(losses))
            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1
    model.eval()

    def _eval_with_hooks(dl):
        diffs = []
        for clean, corrupted, label in dl:
            clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
                model, clean, corrupted,
                pair_padding_side=getattr(dl, "pair_padding_side", None),
            )
            with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
                logits = model(clean_tokens, attention_mask=attn_mask)
            for i in range(logits.shape[0]):
                last = input_lengths[i] - 1
                a = int(label[i, 0].item())
                b = int(label[i, 1].item())
                diffs.append(logits[i, last, a].item() - logits[i, last, b].item())
        return sum(diffs) / max(1, len(diffs))

    post_target = _eval_with_hooks(eval_dl)
    post_side = _eval_with_hooks(side_dl)
    return {
        "wall_seconds": round(time.time() - t0, 2),
        "param_count": sum(p.numel() for p in lora.parameters() if p.requires_grad),
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
    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.float32)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import get_task
    _bootstrap_builtin_tasks()
    pos_cfg = {"algorithm": "eap", "task": TASK, "level": "node",
               "batch_size": 4, "data_params": {"num_examples": 32, "seed": 42}}
    neg_cfg = {**pos_cfg, "data_params": {"num_examples": 32, "seed": 99}}
    eval_cfg = {**pos_cfg, "data_params": {"num_examples": 32, "seed": 1234}}
    side_cfg = {"algorithm": "eap", "task": "gender_bias", "level": "node",
                "batch_size": 4, "data_params": {"num_examples": 16}}
    pos_dl = get_task(TASK).build_dataloader(model, pos_cfg, device)
    neg_dl = get_task(TASK).build_dataloader(model, neg_cfg, device)
    eval_dl = get_task(TASK).build_dataloader(model, eval_cfg, device)
    side_dl = get_task("gender_bias").build_dataloader(model, side_cfg, device)

    print("Running MLP2 LoRA baseline...")
    mlp2 = _pipeline_b_mlp2_lora(model, pos_dl, eval_dl, side_dl)
    print(f"  baseline target_change={mlp2['target_change']:+.3f}  "
          f"side_effect={mlp2['side_effect']:+.3f}  "
          f"params={mlp2['param_count']}  ({mlp2['wall_seconds']}s)")

    rows = []
    for algo in ALGOS:
        try:
            row_a = _pipeline_a_circuit_steering(
                algo, model, pos_dl, neg_dl, eval_dl, side_dl, k=2.0,
            )
        except Exception as exc:  # noqa: BLE001
            row_a = {"error": f"{type(exc).__name__}: {exc}"}
        row = {"algorithm": algo, **row_a,
               "mlp2_target_change": mlp2["target_change"],
               "mlp2_side_effect": mlp2["side_effect"],
               "mlp2_param_count": mlp2["param_count"]}
        rows.append(row)
        if "error" in row_a:
            print(f"  {algo:25s}  ERR: {row_a['error'][:80]}", flush=True)
        else:
            print(f"  {algo:25s}  Δtarget={row_a['target_change']:+.3f}  "
                  f"side={row_a['side_effect']:+.3f}  "
                  f"|w_b|={row_a['steering_norm']}  "
                  f"params={row_a['param_count']}  "
                  f"({row_a['wall_seconds']}s)", flush=True)

    md = ["# Application 20 — C-ΔΘ Contrastive Weight Steering vs MLP2 LoRA",
          "", "Pipeline A: per-algorithm — fine-tune circuit-located heads' "
          "W_Q/K/V/O on positive vs negative IOI, then steered = target + "
          "k · (θ_pos − θ_neg).",
          "Pipeline B: shared baseline — LoRA on layer-2 MLP only, no circuit "
          "knowledge.",
          "",
          f"**MLP2 LoRA baseline:** target Δ={mlp2['target_change']:+.3f}, "
          f"side-effect={mlp2['side_effect']:+.3f}, params={mlp2['param_count']}",
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
    md_text = "\n".join(md) + "\n"
    (out_dir / "table.md").write_text(md_text)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.weight_steering",
        "input": {"model": MODEL, "task": TASK, "n_algos": len(ALGOS), "k": 2.0},
        "metrics": {"mlp2_target_change": mlp2["target_change"],
                    "mlp2_side_effect": mlp2["side_effect"],
                    "n_algos_ok": n_ok, "n_algos_total": len(ALGOS)},
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })
    print()
    print(md_text)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
