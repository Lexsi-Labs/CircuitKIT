"""Application: ROME knowledge editing on GPT-2, target layer chosen
from each algorithm's circuit.

For each algorithm:
  1. Pull CircuitScores from cache (capital_country task -- factual recall).
  2. Select the highest-scored MLP layer as the ROME target layer.
  3. Apply ROME to edit a batch of facts via EasyEdit (zjunlp/EasyEdit).
  4. Report EasyEdit's built-in rewrite_acc / rephrase_acc / locality_acc.

Uses EasyEdit (zjunlp/EasyEdit) for ROME -- cloned to
validation/_vendor/EasyEdit if not already present.

The metric is per-algo edit-success: did ROME produce a working
weight update at the layer the algorithm picked? If algorithms
correctly identify the relevant MLPs, ROME should succeed on more
edits with their layer choices than a random middle-layer baseline.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status,
)

SCRIPT_NAME = "11_knowledge_editing_gpt2"
MODEL = "gpt2"
TASK = "capital_country"

_VENDOR = Path(__file__).resolve().parents[2] / "validation" / "_vendor"
_EASYEDIT_DIR = _VENDOR / "EasyEdit"
_EASYEDIT_REPO = "https://github.com/zjunlp/EasyEdit.git"

ALGOS = [
    "eap", "eap-ig", "eap-ig-activations", "eap-clean-corrupted",
    "relp", "atp-gd", "eap-gp", "eap-exact", "ibcircuit",
    "peap", "cdt", "eap-ifr",
]

EDITS = [
    {
        "edit_prompt": "The capital of France is",
        "subject": "France", "target": "Lyon",
        "paraphrase": "France's capital city is",
        "locality": "The capital of Italy is",
        "locality_target": "Rome",
    },
    {
        "edit_prompt": "The capital of Germany is",
        "subject": "Germany", "target": "Munich",
        "paraphrase": "Germany's capital city is",
        "locality": "The capital of Spain is",
        "locality_target": "Madrid",
    },
    {
        "edit_prompt": "The capital of Spain is",
        "subject": "Spain", "target": "Barcelona",
        "paraphrase": "Spain's capital city is",
        "locality": "The capital of Italy is",
        "locality_target": "Rome",
    },
    {
        "edit_prompt": "The capital of Italy is",
        "subject": "Italy", "target": "Milan",
        "paraphrase": "Italy's capital city is",
        "locality": "The capital of France is",
        "locality_target": "Paris",
    },
]


def _ensure_easyedit() -> bool:
    if _EASYEDIT_DIR.exists():
        if str(_EASYEDIT_DIR) not in sys.path:
            sys.path.insert(0, str(_EASYEDIT_DIR))
        return True
    _VENDOR.mkdir(parents=True, exist_ok=True)
    print(f"Cloning EasyEdit to {_EASYEDIT_DIR} ...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", _EASYEDIT_REPO, str(_EASYEDIT_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  git clone failed: {result.stderr}")
        return False
    sys.path.insert(0, str(_EASYEDIT_DIR))
    req = _EASYEDIT_DIR / "requirements.txt"
    if req.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
             "-r", str(req)],
            check=False,
        )
    print("  EasyEdit ready.")
    return True


def _highest_scored_mlp_layer(node_scores) -> int:
    mlp_scores = [(int(name.split()[1]), abs(score))
                  for name, score in node_scores.items()
                  if name.startswith("MLP ")]
    if mlp_scores:
        mlp_scores.sort(key=lambda x: x[1], reverse=True)
        return mlp_scores[0][0]
    import re as _re
    layer_sums: dict = {}
    for name, score in node_scores.items():
        m = _re.match(r"A(\d+)\.\d+", name)
        if m:
            layer_sums[int(m.group(1))] = (
                layer_sums.get(int(m.group(1)), 0.0) + abs(score)
            )
    if not layer_sums:
        return -1
    return max(layer_sums.items(), key=lambda kv: kv[1])[0]


def _make_gpt2_hparams(target_layer: int):
    """Build ROMEHyperParams for GPT-2 small (12 layers).

    EasyEdit ships a gpt2-xl.yaml but not a gpt2-small one. The module
    templates are identical; only v_loss_layer (last layer = 11) differs.
    """
    from easyeditor import ROMEHyperParams
    gpt2_xl_yaml = _EASYEDIT_DIR / "hparams" / "ROME" / "gpt2-xl.yaml"
    hparams = ROMEHyperParams.from_hparams(str(gpt2_xl_yaml))
    hparams.model_name = MODEL
    hparams.layers = [target_layer]
    hparams.v_loss_layer = 11
    hparams.device = 0 if torch.cuda.is_available() else "cpu"
    hparams.fp16 = False
    hparams.stats_dir = str(_EASYEDIT_DIR / "data" / "stats")
    return hparams


def _run_one_algo(algo: str, editor_cache: dict) -> dict:
    """Run ROME editing for one algorithm using EasyEdit.

    Reuses an editor instance cached per target_layer to avoid reloading
    the HuggingFace model. The editor uses keep_original_weight via
    sequential_edit=False so each algo's edits are independent.
    """
    from easyeditor import BaseEditor

    t0 = time.time()
    try:
        cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=24)
    except Exception as exc:
        return {"algorithm": algo, "error": f"discovery: {type(exc).__name__}: {exc}"}
    cs = cell["scores"]

    target_layer = _highest_scored_mlp_layer(cs.node_scores)
    if target_layer < 0:
        return {"algorithm": algo, "error": "no MLP node in circuit"}

    if target_layer not in editor_cache:
        hparams = _make_gpt2_hparams(target_layer)
        editor_cache[target_layer] = BaseEditor.from_hparams(hparams)
    editor = editor_cache[target_layer]
    editor.hparams.layers = [target_layer]

    prompts = [e["edit_prompt"] for e in EDITS]
    subjects = [e["subject"] for e in EDITS]
    target_new = [e["target"] for e in EDITS]
    rephrase_prompts = [e["paraphrase"] for e in EDITS]
    locality_inputs = {
        "neighborhood": {
            "prompt": [e["locality"] for e in EDITS],
            "ground_truth": [e["locality_target"] for e in EDITS],
        }
    }

    try:
        metrics, _, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            subject=subjects,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            sequential_edit=False,
            keep_original_weight=True,
        )
    except Exception as exc:
        return {"algorithm": algo, "target_layer": target_layer,
                "error": f"edit: {type(exc).__name__}: {exc}",
                "wall_seconds": round(time.time() - t0, 2)}

    edit_succ = 0
    para_succ = 0
    locality_pres = 0
    details = []
    for edit, m in zip(EDITS, metrics):
        post = m.get("post", {})
        rewrite_acc = float((post.get("rewrite_acc") or [0])[0])
        rephrase_acc = float((post.get("rephrase_acc") or [0])[0])
        nbr_acc = post.get("locality", {}).get("neighborhood_acc", [1.0])
        loc_acc = float(nbr_acc[0]) if nbr_acc else 1.0

        if rewrite_acc > 0.5:
            edit_succ += 1
        if rephrase_acc > 0.5:
            para_succ += 1
        if loc_acc > 0.5:
            locality_pres += 1
        details.append({
            "edit_prompt": edit["edit_prompt"], "target": edit["target"],
            "rewrite_acc": round(rewrite_acc, 3),
            "rephrase_acc": round(rephrase_acc, 3),
            "locality_acc": round(loc_acc, 3),
        })

    n = len(EDITS)
    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "target_layer": target_layer,
        "n_edits_attempted": n,
        "edit_success": edit_succ,
        "paraphrase_success": para_succ,
        "locality_preserved": locality_pres,
        "edit_success_rate": round(edit_succ / n, 3),
        "paraphrase_rate": round(para_succ / n, 3),
        "locality_rate": round(locality_pres / n, 3),
        "details": details,
    }


def main() -> int:
    if not _ensure_easyedit():
        print("ERROR: could not clone EasyEdit -- aborting.")
        return 1

    out_dir = make_results_dir(SCRIPT_NAME)
    editor_cache: dict = {}
    rows: List[dict] = []

    for algo in ALGOS:
        row = _run_one_algo(algo, editor_cache)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:25s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:25s}  layer={row['target_layer']}  "
                  f"edit={row['edit_success']}/{row['n_edits_attempted']}  "
                  f"para={row['paraphrase_success']}/{row['n_edits_attempted']}  "
                  f"loc={row['locality_preserved']}/{row['n_edits_attempted']}  "
                  f"({row['wall_seconds']}s)", flush=True)

    lines = [
        "# Application 11 -- ROME Knowledge Editing (EasyEdit)",
        "",
        f"For each algorithm, ROME's target_layer is the highest-scored MLP",
        f"in that algorithm's `{TASK}` circuit. Three metrics from EasyEdit:",
        "",
        "- **Edit**: rewrite_acc > 0.5 on the edit prompt.",
        "- **Paraphrase**: rephrase_acc > 0.5 (generalisation).",
        "- **Locality**: neighborhood_acc > 0.5 (unrelated fact preserved).",
        "",
        "| Algorithm | Layer | Edit | Paraphrase | Locality | Note |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        note = r.get("error", "")
        n = r.get("n_edits_attempted", "?")
        lines.append(
            f"| `{r['algorithm']}` | "
            f"{r.get('target_layer', '?')} | "
            f"{r.get('edit_success', '?')}/{n} | "
            f"{r.get('paraphrase_success', '?')}/{n} | "
            f"{r.get('locality_preserved', '?')}/{n} | "
            f"{note} |"
        )
    md = "\n".join(lines) + "\n"
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.rome_knowledge_editing",
        "input": {"model": MODEL, "task": TASK, "n_facts": len(EDITS),
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
