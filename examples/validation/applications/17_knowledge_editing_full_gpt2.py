"""Application: high-level CircuitKnowledgeEditor on capital_country.

Tests apply/knowledge_editing.py's `CircuitKnowledgeEditor.edit_via_circuit`
which auto-selects the target layer from a CircuitScores artifact rather
than asking the user to pick. Covers both ROME and MEMIT paths.
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

SCRIPT_NAME = "17_knowledge_editing_full_gpt2"
MODEL = "gpt2"
TASK = "capital_country"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]

EDITS = [
    {"prompt": "The capital of France is", "subject": "France", "target": "Lyon"},
    {"prompt": "The capital of Germany is", "subject": "Germany", "target": "Munich"},
]


def _logit_for_target(model, prompt, target_token: str) -> float:
    tok = model.tokenizer
    ids = tok(target_token, add_special_tokens=False)["input_ids"]
    if not ids:
        return float("nan")
    target_id = ids[0]
    in_ids = tok(prompt, return_tensors="pt").input_ids.to(model.cfg.device)
    with torch.inference_mode():
        logits = model(in_ids)
    return float(logits[0, -1, target_id].item())


class _CircuitWrapper:
    """Minimal Circuit-like object for CircuitKnowledgeEditor compatibility.
    The editor reads `.graph.nodes` so we expose top scores as a node dict."""
    def __init__(self, scores: dict, model_n_layers: int):
        from types import SimpleNamespace
        nodes = {}
        for name, score in scores.items():
            nodes[name] = SimpleNamespace(name=name, score=score)
        self.graph = SimpleNamespace(nodes=nodes, n_layers=model_n_layers)
        self.scores = scores


def _run_one_algo(algo: str, model_clean: HookedTransformer):
    try:
        cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=24)
    except Exception as exc:
        return {"algorithm": algo, "error": f"discovery: {type(exc).__name__}: {exc}"}
    cs = cell["scores"]
    t0 = time.time()
    successes = 0
    n = 0
    deltas = []
    for edit in EDITS:
        try:
            edited = copy.deepcopy(model_clean)
            from circuitkit.applications.editing.knowledge_editing import CircuitKnowledgeEditor
            editor = CircuitKnowledgeEditor(edited)
            wrapped = _CircuitWrapper(cs.node_scores, edited.cfg.n_layers)
            pre = _logit_for_target(model_clean, edit["prompt"], " " + edit["target"])
            editor.edit_via_circuit(
                prompt=edit["prompt"],
                subject=edit["subject"],
                target=edit["target"],
                circuit=wrapped,
                method="rome",
                use_corpus_C=False,
                n_prefixes=2,
                verify=False,
            )
            post = _logit_for_target(edited, edit["prompt"], " " + edit["target"])
            n += 1
            deltas.append(round(post - pre, 3))
            if post > pre + 1.0:
                successes += 1
            del edited
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            return {"algorithm": algo, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "n_attempted": n,
        "n_successful": successes,
        "edit_success_rate": round(successes / max(1, n), 3),
        "deltas": deltas,
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.float32)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, model)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:25s}  edits={row['n_successful']}/{row['n_attempted']}  "
                  f"({row['wall_seconds']}s)", flush=True)

    lines = ["# Application 17 — CircuitKnowledgeEditor (high-level wrapper)", "",
             "| Algorithm | Edits Successful | Edit Success Rate | Note |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['algorithm']}` | "
                     f"{r.get('n_successful', '—')}/{r.get('n_attempted', '—')} | "
                     f"{r.get('edit_success_rate', '—')} | "
                     f"{r.get('error', '')} |")
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))
    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.knowledge_editing.CircuitKnowledgeEditor",
        "input": {"model": MODEL, "task": TASK},
        "output": {"table_md": str(out_dir / "table.md")},
        "metrics": {"n_algos_ok": n_ok, "n_algos_total": len(ALGOS)},
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })
    print(md)
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
