"""Application: LoRA-based soft healing on top of structural pruning.

For each discovery algorithm:
  1. Prune the lowest-scored 30% of nodes (weight zeroing).
  2. Attach CircuitLoRA targeting the SURVIVING circuit components.
  3. Train one short LoRA pass on IOI (clean text only).
  4. Compare IOI logit-diff: pruned vs pruned+LoRA.

The metric: recovery = (post_lora - pruned) / (baseline - pruned).
- 0.0 means LoRA recovered nothing.
- 1.0 means LoRA fully restored the baseline.
- Negative means LoRA hurt.

Why this matters: the "good" pruning targets are nodes the algorithm
ranked as low-importance. The LoRA targets the kept (high-importance)
components. If the algorithm's ranking is informative, the LoRA can
heal the model by adapting only the surviving components.
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

SCRIPT_NAME = "02_lora_healing_gpt2_ioi"
MODEL = "gpt2"
TASK = "ioi"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]


def _ioi_logit_diff(model, dataloader) -> float:
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
            io_id, s_id = int(label[i, 0].item()), int(label[i, 1].item())
            diffs.append(logits[i, last, io_id].item() - logits[i, last, s_id].item())
    return sum(diffs) / max(1, len(diffs))


def _lora_logit_diff(model, lora, dataloader) -> float:
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    diffs = []
    hooks = lora.get_lora_hooks()
    for clean, corrupted, label in dataloader:
        clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
            model, clean, corrupted,
            pair_padding_side=getattr(dataloader, "pair_padding_side", None),
        )
        with torch.inference_mode():
            with model.hooks(fwd_hooks=hooks):
                logits = model(clean_tokens, attention_mask=attn_mask)
        for i in range(logits.shape[0]):
            last = input_lengths[i] - 1
            io_id, s_id = int(label[i, 0].item()), int(label[i, 1].item())
            diffs.append(logits[i, last, io_id].item() - logits[i, last, s_id].item())
    return sum(diffs) / max(1, len(diffs))


def _bottom_k(node_scores, sparsity):
    n = max(1, int(len(node_scores) * sparsity))
    return [k for k, _ in sorted(node_scores.items(), key=lambda kv: abs(kv[1]))[:n]]


def _train_one_epoch(model, lora, dataloader, lr=5e-4):
    """Single short training pass on IOI-clean cross-entropy."""
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    optim = torch.optim.AdamW(lora.parameters(), lr=lr)
    hooks = lora.get_lora_hooks()
    losses = []
    for clean, corrupted, label in dataloader:
        clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
            model, clean, corrupted,
            pair_padding_side=getattr(dataloader, "pair_padding_side", None),
        )
        with model.hooks(fwd_hooks=hooks):
            logits = model(clean_tokens, attention_mask=attn_mask)
        # Per-example loss = -(io - s) — push the model toward IO.
        loss = 0.0
        for i in range(logits.shape[0]):
            last = input_lengths[i] - 1
            io_id, s_id = int(label[i, 0].item()), int(label[i, 1].item())
            loss = loss - (logits[i, last, io_id] - logits[i, last, s_id])
        loss = loss / logits.shape[0]
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def _run_one_algo(algo: str, dataloader, model_clean: HookedTransformer,
                  train_dl=None):
    from circuitkit.applications.pruning.weight_pruner import apply_node_pruning_to_weights
    from circuitkit.applications.finetuning.soft_healing import CircuitLoRA

    cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=32)
    cs = cell["scores"]
    t0 = time.time()

    nodes_to_prune = _bottom_k(cs.node_scores, sparsity=0.3)
    pruned = copy.deepcopy(model_clean)
    try:
        apply_node_pruning_to_weights(pruned, nodes_to_prune)
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo, "error": f"prune: {exc}"}

    pruned_diff = _ioi_logit_diff(pruned, dataloader)

    # Attach LoRA targeting the SURVIVING (high-scored) components.
    surviving = {n: s for n, s in cs.node_scores.items() if n not in nodes_to_prune}
    try:
        lora = CircuitLoRA(
            model=pruned, circuit_scores=surviving,
            lora_rank=4, lora_alpha=8.0,
            score_threshold=0.0,
        ).to(pruned.cfg.device)
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo, "error": f"lora_init: {exc}"}

    try:
        # Train on the held-out training dataloader; evaluate on a
        # disjoint dataloader passed in by main.
        _train_dl = train_dl if train_dl is not None else dataloader
        train_loss = _train_one_epoch(pruned, lora, _train_dl, lr=5e-4)
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo, "error": f"lora_train: {exc}"}

    healed_diff = _lora_logit_diff(pruned, lora, dataloader)
    del pruned, lora
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "pruned_logit_diff": round(pruned_diff, 4),
        "healed_logit_diff": round(healed_diff, 4),
        "train_loss": round(train_loss, 4),
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
    # Train/eval split: build TWO disjoint dataloaders with different
    # seeds. The earlier version of this cell used ONE dataloader for both
    # LoRA training and evaluation, which inflated recovery to >100%
    # (the LoRA was overfitting to the eval-time logit-diff metric on
    # the same examples it was trained on). Split fixes that.
    train_cfg = {
        "algorithm": "eap", "task": TASK, "level": "node", "batch_size": 4,
        "data_params": {"num_examples": 64, "seed": 42},
    }
    eval_cfg = {
        "algorithm": "eap", "task": TASK, "level": "node", "batch_size": 4,
        "data_params": {"num_examples": 64, "seed": 1234},
    }
    train_dl = task_spec.build_dataloader(model, train_cfg, device)
    dataloader = task_spec.build_dataloader(model, eval_cfg, device)

    baseline = _ioi_logit_diff(model, dataloader)
    print(f"GPT-2 baseline IOI logit-diff: {baseline:.4f}\n", flush=True)

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, dataloader, model, train_dl=train_dl)
        row["baseline_logit_diff"] = round(baseline, 4)
        if "error" not in row and row.get("pruned_logit_diff") is not None:
            denom = baseline - row["pruned_logit_diff"]
            recov = ((row["healed_logit_diff"] - row["pruned_logit_diff"]) /
                     denom if abs(denom) > 1e-6 else 0.0)
            row["recovery_pct"] = round(recov * 100, 1)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(
                f"  {algo:25s}  pruned={row['pruned_logit_diff']:>7}  "
                f"healed={row['healed_logit_diff']:>7}  "
                f"recovery={row.get('recovery_pct')}%  "
                f"({row['wall_seconds']}s)",
                flush=True,
            )

    lines = [
        f"# Application 2 — LoRA Healing on Pruned GPT-2 / IOI",
        "",
        f"Baseline (no pruning) IOI logit-diff: **{baseline:.4f}**",
        "",
        "| Algorithm | Pruned | Healed | Recovery | Note |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        note = r.get("error", "")
        lines.append(
            f"| `{r['algorithm']}` | "
            f"{r.get('pruned_logit_diff', '—')} | "
            f"{r.get('healed_logit_diff', '—')} | "
            f"{r.get('recovery_pct', '—')}% | "
            f"{note} |"
        )
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.lora_healing",
        "input": {"model": MODEL, "task": TASK, "n_algos": len(ALGOS)},
        "output": {
            "table_md": str(out_dir / "table.md"),
            "rows_json": str(out_dir / "rows.json"),
        },
        "metrics": {
            "baseline_logit_diff": round(baseline, 4),
            "n_algos_ok": n_ok, "n_algos_total": len(ALGOS),
        },
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })

    print()
    print(md)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
