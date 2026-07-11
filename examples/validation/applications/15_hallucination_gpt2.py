"""Application: hallucination probe smoke (LinearProbe + ProbeTrainer).

The full HallucinationDetector class wraps a CircuitArtifact + HF
arch_cfg and is HF-specific. For the per-algorithm smoke we exercise
the underlying probe building blocks directly (LinearProbe +
ProbeTrainer in apply/linear_probe.py): train a probe at the highest-
scored layer of each algorithm's circuit on a tiny factual /
hallucinated split and report mean accuracy.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status,
)

SCRIPT_NAME = "15_hallucination_gpt2"
MODEL = "gpt2"
TASK = "capital_country"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]

FACTUAL = [
    ("The capital of France is", " Paris"),
    ("The capital of Germany is", " Berlin"),
    ("The capital of Spain is", " Madrid"),
    ("The capital of Italy is", " Rome"),
    ("The capital of Japan is", " Tokyo"),
    ("The capital of Russia is", " Moscow"),
    ("The capital of Brazil is", " Brasília"),
    ("The capital of Egypt is", " Cairo"),
]
HALLUCINATED = [
    ("The capital of France is", " London"),
    ("The capital of Germany is", " Vienna"),
    ("The capital of Spain is", " Lisbon"),
    ("The capital of Italy is", " Athens"),
    ("The capital of Japan is", " Beijing"),
    ("The capital of Russia is", " Kiev"),
    ("The capital of Brazil is", " Lima"),
    ("The capital of Egypt is", " Tripoli"),
]


def _highest_scored_layer(node_scores) -> int:
    """Pick the layer with the highest summed attribution. Aggregates
    attention heads and MLPs per-layer."""
    import re as _re
    layer_sums: dict = {}
    for name, score in node_scores.items():
        m = _re.match(r"A(\d+)\.\d+", name) or _re.match(r"MLP (\d+)", name)
        if m:
            l = int(m.group(1))
            layer_sums[l] = layer_sums.get(l, 0.0) + abs(score)
    if not layer_sums:
        return -1
    return max(layer_sums.items(), key=lambda kv: kv[1])[0]


def _collect_features(model, examples, layer):
    """Return the resid_post[layer, last_pos, :] for each example."""
    feats = []
    for prompt, answer in examples:
        full = prompt + answer
        ids = model.tokenizer(full, return_tensors="pt").input_ids.to(model.cfg.device)
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                ids,
                names_filter=lambda n, _l=layer: n == f"blocks.{_l}.hook_resid_post",
            )
        feats.append(cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu())
    return torch.stack(feats)


def _run_one_algo(algo: str, model: HookedTransformer):
    cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=24)
    cs = cell["scores"]
    t0 = time.time()
    layer = _highest_scored_layer(cs.node_scores)
    if layer < 0:
        return {"algorithm": algo, "error": "no scored components in circuit"}
    try:
        from circuitkit.applications.common.linear_probe import LinearProbe

        # Collect features.
        x_pos = _collect_features(model, FACTUAL, layer)
        x_neg = _collect_features(model, HALLUCINATED, layer)
        X = torch.cat([x_pos, x_neg], dim=0).float()
        y = torch.tensor([1] * len(FACTUAL) + [0] * len(HALLUCINATED)).float()

        # Tiny probe trained for a few epochs.
        d = X.shape[1]
        probe = LinearProbe(input_dim=d, dropout=0.0)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-3)
        for _ in range(40):
            logits = probe(X).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.inference_mode():
            preds = (probe(X).squeeze(-1) > 0).float()
            acc = float((preds == y).float().mean().item())
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "probe_layer": layer,
        "probe_accuracy": round(acc, 4),
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.float32)
    model.cfg.use_attn_result = True

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, model)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:25s}  layer={row['probe_layer']}  "
                  f"acc={row['probe_accuracy']:.3f}  "
                  f"({row['wall_seconds']}s)", flush=True)

    lines = ["# Application 15 — Hallucination linear probes", "",
             "| Algorithm | Layer | Probe acc | Note |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['algorithm']}` | "
                     f"{r.get('probe_layer', '—')} | "
                     f"{r.get('probe_accuracy', '—')} | "
                     f"{r.get('error', '')} |")
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))
    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.linear_probe",
        "input": {"model": MODEL, "task": TASK},
        "output": {"table_md": str(out_dir / "table.md")},
        "metrics": {"n_algos_ok": n_ok, "n_algos_total": len(ALGOS)},
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })
    print(md)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
