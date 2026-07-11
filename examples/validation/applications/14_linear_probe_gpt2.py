"""Application: train linear probes on circuit-located activations.

For each algorithm:
  1. Pull CircuitScores on a paired task (IOI).
  2. Pick the top-5 attention head nodes per algorithm.
  3. Train a linear probe per head on (activation -> IO/S label).
  4. Report mean probe accuracy across heads.

Higher accuracy means the algorithm-located heads carry linearly-
decodable IOI information.
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

SCRIPT_NAME = "14_linear_probe_gpt2"
MODEL = "gpt2"
TASK = "ioi"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]


def _collect_head_activations(model, dataloader, head_nodes, hook_pattern="z"):
    """For each (clean, corrupted, label) pair, cache hook_z at the named heads
    and return (X, y) for probe training. Labels: 0 for clean, 1 for corrupted."""
    import re
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    layer_head = []
    for n in head_nodes:
        m = re.match(r"A(\d+)\.(\d+)", n)
        if m:
            layer_head.append((int(m.group(1)), int(m.group(2))))
    if not layer_head:
        return None, None, []

    cache = {l: None for l, _ in layer_head}

    def make_hook(L):
        def h(act, hook):
            cache[L] = act.detach().clone()
            return act
        return h

    hooks = [(f"blocks.{l}.attn.hook_{hook_pattern}", make_hook(l)) for l in cache]

    X_per_node = {(l, h): [] for (l, h) in layer_head}
    y = []
    for clean, corrupted, _ in dataloader:
        for label, batch_text in [(0, clean), (1, corrupted)]:
            input_ids = model.tokenizer(batch_text, return_tensors="pt",
                                        padding=True).input_ids.to(model.cfg.device)
            with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
                _ = model(input_ids)
            for (l, h) in layer_head:
                # cache[l] has shape [batch, pos, n_heads, d_head]; take the head + last position
                last = input_ids.shape[1] - 1
                # Do NOT null cache[l] here — multiple heads at the same
                # layer share cache[l]; nulling it after the first head
                # made the second head crash with NoneType subscription.
                X_per_node[(l, h)].append(cache[l][:, last, h, :].cpu())
            y.extend([label] * len(batch_text))

    X = {(l, h): torch.cat(v, dim=0) for (l, h), v in X_per_node.items()}
    y_t = torch.tensor(y[:len(next(iter(X.values())))])
    return X, y_t, layer_head


def _train_eval_probe(X: torch.Tensor, y: torch.Tensor) -> float:
    """Quick L2-regularised linear classifier; report accuracy on the same set
    (not held-out — small smoke; documented as such)."""
    import torch.nn as nn
    n = X.shape[0]
    perm = torch.randperm(n)
    X = X[perm].float()
    y = y[perm].float()
    cls = nn.Linear(X.shape[1], 1)
    opt = torch.optim.AdamW(cls.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(40):
        logits = cls(X).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.inference_mode():
        preds = (cls(X).squeeze(-1) > 0).float()
        return float((preds == y).float().mean().item())


def _run_one_algo(algo, model, dataloader):
    cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=32)
    cs = cell["scores"]
    t0 = time.time()
    head_nodes = [n for n in top_k_node_set(cs.node_scores, k=5)
                  if n.startswith("A")][:5]
    if not head_nodes:
        return {"algorithm": algo, "error": "no attn heads in top-5"}
    try:
        X, y, layer_head = _collect_head_activations(model, dataloader, head_nodes)
        if not X:
            return {"algorithm": algo, "error": "could not parse heads"}
        accs = {}
        for (l, h), feats in X.items():
            accs[f"A{l}.{h}"] = _train_eval_probe(feats, y)
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "n_heads": len(accs),
        "mean_probe_accuracy": round(sum(accs.values()) / len(accs), 4),
        "per_head_acc": {k: round(v, 3) for k, v in accs.items()},
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.float32)
    model.cfg.use_attn_result = True

    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import get_task
    _bootstrap_builtin_tasks()
    task_spec = get_task(TASK)
    dl_cfg = {"algorithm": "eap", "task": TASK, "level": "node",
              "batch_size": 4, "data_params": {"num_examples": 32}}
    dl = task_spec.build_dataloader(model, dl_cfg, device)

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, model, dl)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:25s}  mean_acc={row['mean_probe_accuracy']:.3f}  "
                  f"({row['wall_seconds']}s)", flush=True)

    lines = [
        "# Application 14 — Linear probes on top-5 circuit attention heads",
        "",
        "For each algorithm we extract the top-5 attention-head nodes from",
        "the discovered circuit and train a linear probe at each head's",
        "hook_z output to discriminate clean (label 0) vs corrupted (label 1)",
        "IOI inputs. Higher mean accuracy means the algorithm-located heads",
        "carry linearly-decodable IOI information. (Train and eval share the",
        "set — smoke-quality benchmark; held-out version is a TODO.)",
        "",
        "| Algorithm | n_heads | Mean probe acc | Note |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['algorithm']}` | {r.get('n_heads', '—')} | "
            f"{r.get('mean_probe_accuracy', '—')} | {r.get('error', '')} |"
        )
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.linear_probe",
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
