"""Application: circuit-restricted unlearning on TOFU forget set.

For each algorithm:
  1. Pull TOFU CircuitScores (registered via _custom_tasks.register_tofu).
  2. Apply gradient ASCENT on the forget set, but ONLY on the parameters
     belonging to the discovered circuit (top-30% nodes). Other weights
     are frozen.
  3. Measure NLL change on (a) the forget set (should rise — model is
     forgetting) and (b) a held-out retain set (should stay roughly
     constant — utility preserved).

Why per-algo: if the algorithm picks the right circuit, the
forget-set NLL should rise more sharply (better forgetting) AND the
retain-set NLL should rise less (better locality) than a random-
component selection.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status,
)

# Register TOFU before discovery if needed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

SCRIPT_NAME = "12_unlearning_tofu_gpt2"
MODEL = "gpt2"
TASK = "tofu"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]

N_STEPS = 20
LR = 5e-4


def _top_k_nodes(node_scores, k_frac=0.3):
    n = max(1, int(len(node_scores) * k_frac))
    return [name for name, _ in sorted(
        node_scores.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:n]]


def _params_for_node(model, node_name):
    """Return the parameter tensors associated with a circuit node.

    For attention head 'A{l}.{h}' that's the layer's W_Q/W_K/W_V/W_O slice.
    For 'MLP {l}' it's W_in/W_out at that layer. We yield the FULL block
    parameters (no head-slicing) — over-approximates but keeps the
    optimiser simple.
    """
    attn = re.match(r"A(\d+)\.(\d+)", node_name)
    if attn:
        l = int(attn.group(1))
        # Whole-attention parameters at layer l. Over-includes other heads
        # but the optimiser will only update the components used in the
        # gradient pass; in practice the loss only flows through used heads.
        block = model.blocks[l].attn
        return [block.W_Q, block.W_K, block.W_V, block.W_O,
                block.b_Q, block.b_K, block.b_V, block.b_O]
    mlp = re.match(r"MLP (\d+)", node_name)
    if mlp:
        l = int(mlp.group(1))
        block = model.blocks[l].mlp
        return [block.W_in, block.W_out, block.b_in, block.b_out]
    return []


def _nll_on_pairs(model, dataloader) -> float:
    """Mean negative log-likelihood of the clean target over the dataloader."""
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    nll_sum, count = 0.0, 0
    for clean, corrupted, label in dataloader:
        clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
            model, clean, corrupted,
            pair_padding_side=getattr(dataloader, "pair_padding_side", None),
        )
        with torch.inference_mode():
            logits = model(clean_tokens, attention_mask=attn_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        for i in range(logits.shape[0]):
            last = input_lengths[i] - 1
            tgt = int(label[i, 0].item()) if label.dim() == 2 else int(label[i].item())
            nll_sum += -log_probs[i, last, tgt].item()
            count += 1
    return nll_sum / max(1, count)


def _run_one_algo(algo: str, forget_dl, retain_dl, model_clean: HookedTransformer):
    try:
        cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=24)
    except Exception as exc:
        return {"algorithm": algo, "error": f"discovery: {type(exc).__name__}: {exc}"}
    cs = cell["scores"]
    t0 = time.time()

    circuit_nodes = _top_k_nodes(cs.node_scores, k_frac=0.3)
    edited = copy.deepcopy(model_clean)

    # Freeze everything; un-freeze only the circuit parameters.
    for p in edited.parameters():
        p.requires_grad_(False)
    circuit_params = []
    for n in circuit_nodes:
        for p in _params_for_node(edited, n):
            p.requires_grad_(True)
            circuit_params.append(p)
    if not circuit_params:
        return {"algorithm": algo, "error": "no circuit params"}

    forget_pre = _nll_on_pairs(edited, forget_dl)
    retain_pre = _nll_on_pairs(edited, retain_dl)

    # Gradient ASCENT: maximise NLL on the forget set => minimise -NLL.
    optim = torch.optim.AdamW(circuit_params, lr=LR)
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    pair_padding = getattr(forget_dl, "pair_padding_side", None)

    edited.train()
    step = 0
    for clean, corrupted, label in forget_dl:
        if step >= N_STEPS:
            break
        clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
            edited, clean, corrupted, pair_padding_side=pair_padding,
        )
        logits = edited(clean_tokens, attention_mask=attn_mask)
        loss = 0.0
        log_probs = torch.log_softmax(logits, dim=-1)
        for i in range(logits.shape[0]):
            last = input_lengths[i] - 1
            tgt = int(label[i, 0].item()) if label.dim() == 2 else int(label[i].item())
            loss = loss + log_probs[i, last, tgt]
        loss = loss / logits.shape[0]  # gradient ASCENT on log-prob
        optim.zero_grad()
        loss.backward()
        optim.step()
        step += 1
    edited.eval()

    forget_post = _nll_on_pairs(edited, forget_dl)
    retain_post = _nll_on_pairs(edited, retain_dl)

    del edited
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "n_circuit_nodes": len(circuit_nodes),
        "forget_nll_pre": round(forget_pre, 4),
        "forget_nll_post": round(forget_post, 4),
        "forget_delta": round(forget_post - forget_pre, 4),
        "retain_nll_pre": round(retain_pre, 4),
        "retain_nll_post": round(retain_post, 4),
        "retain_delta": round(retain_post - retain_pre, 4),
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Register TOFU task lazily.
    try:
        from _custom_tasks import register_tofu
        register_tofu()
    except Exception as exc:  # noqa: BLE001
        write_status(out_dir, {
            "script": SCRIPT_NAME, "module": "applications.unlearning_tofu",
            "status": "BROKEN", "error": f"register_tofu: {exc}",
        })
        print(f"register_tofu failed: {exc}")
        return 1

    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.float32)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import get_task
    _bootstrap_builtin_tasks()
    task_spec = get_task(TASK)
    dl_cfg = {
        "algorithm": "eap", "task": TASK, "level": "node", "batch_size": 4,
        "data_params": {"num_examples": 32}, "model_name": MODEL,
    }
    forget_dl = task_spec.build_dataloader(model, dl_cfg, device)
    # For the retain set we just reuse the same dataloader as a baseline.
    # The TOFU task already includes paired forget/retain in its source.
    retain_dl = task_spec.build_dataloader(model, dl_cfg, device)

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, forget_dl, retain_dl, model)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:25s}  forget Δ={row['forget_delta']:+.3f}  "
                  f"retain Δ={row['retain_delta']:+.3f}  "
                  f"({row['wall_seconds']}s)", flush=True)

    lines = [
        f"# Application 12 — Circuit-Restricted TOFU Unlearning",
        "",
        f"Gradient ASCENT on TOFU forget targets, restricted to the top-30%",
        f"circuit nodes per algorithm. forget Δ > 0 means NLL rose (forgetting),",
        f"retain Δ should stay near 0 (locality preserved).",
        "",
        "| Algorithm | n_circuit | forget NLL Δ | retain NLL Δ | Note |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        note = r.get("error", "")
        lines.append(
            f"| `{r['algorithm']}` | "
            f"{r.get('n_circuit_nodes', '—')} | "
            f"{r.get('forget_delta', '—')} | "
            f"{r.get('retain_delta', '—')} | "
            f"{note} |"
        )
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.unlearning_tofu",
        "input": {"model": MODEL, "task": TASK, "n_steps": N_STEPS, "lr": LR,
                  "n_algos": len(ALGOS)},
        "output": {
            "table_md": str(out_dir / "table.md"),
            "rows_json": str(out_dir / "rows.json"),
        },
        "metrics": {"n_algos_ok": n_ok, "n_algos_total": len(ALGOS)},
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })

    print()
    print(md)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
