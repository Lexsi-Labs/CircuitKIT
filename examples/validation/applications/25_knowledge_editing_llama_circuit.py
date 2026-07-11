"""Circuit-conditioned ROME knowledge editing on Llama-3.2-3B-Instruct.

For each discovery method, pull the CircuitScores artifact, pick the
highest-scored MLP layer as the ROME target layer, run a small batch
of edits, and measure edit-success + paraphrase + locality. Compare
to a random-middle-layer baseline at the same edit budget.

Mirrors 11_knowledge_editing_gpt2.py but on Llama-3.2-3B (production
scale matching workshop Q2/Q3/Q4).

Uses EasyEdit (zjunlp/EasyEdit) for ROME editing -- cloned to
validation/_vendor/EasyEdit if not already present.

Headline metric per (method, seed): mean of
  edit_succ * paraphrase_succ * (1 - locality_drift)
which is high only when the edit takes, generalizes, and leaves
unrelated facts untouched.
"""
from __future__ import annotations

import argparse
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

_VENDOR = Path(__file__).resolve().parents[2] / "validation" / "_vendor"
_EASYEDIT_DIR = _VENDOR / "EasyEdit"
_EASYEDIT_REPO = "https://github.com/zjunlp/EasyEdit.git"
_ROME_HPARAMS_YAML = _EASYEDIT_DIR / "hparams" / "ROME" / "llama3.2-3b.yaml"


def _ensure_easyedit() -> bool:
    """Clone EasyEdit to validation/_vendor/EasyEdit if not present.

    Returns True if EasyEdit is importable after setup.
    """
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


SCRIPT_NAME = "25_knowledge_editing_llama_circuit"
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TASK = "capital_country"
SEEDS = [42, 143, 256]
METHODS = ["eap", "eap-ig", "eap-clean-corrupted", "atp-gd",
           "eap-gp", "relp", "eap-exact"]

EDITS = [
    {"edit_prompt": "The capital of France is", "subject": "France",
     "target": "Lyon", "paraphrase": "France's capital city is",
     "locality": "The capital of Italy is", "locality_target": "Rome"},
    {"edit_prompt": "The capital of Germany is", "subject": "Germany",
     "target": "Munich", "paraphrase": "Germany's capital city is",
     "locality": "The capital of Spain is", "locality_target": "Madrid"},
    {"edit_prompt": "The capital of Spain is", "subject": "Spain",
     "target": "Barcelona", "paraphrase": "Spain's capital city is",
     "locality": "The capital of Italy is", "locality_target": "Rome"},
    {"edit_prompt": "The capital of Italy is", "subject": "Italy",
     "target": "Milan", "paraphrase": "Italy's capital city is",
     "locality": "The capital of France is", "locality_target": "Paris"},
    {"edit_prompt": "The capital of Portugal is", "subject": "Portugal",
     "target": "Porto", "paraphrase": "Portugal's capital city is",
     "locality": "The capital of Greece is", "locality_target": "Athens"},
    {"edit_prompt": "The capital of Greece is", "subject": "Greece",
     "target": "Thessaloniki", "paraphrase": "Greece's capital city is",
     "locality": "The capital of Portugal is", "locality_target": "Lisbon"},
]


def _highest_mlp_layer(node_scores: Dict[str, float]) -> int:
    """Return the index of the highest-scored MLP layer."""
    mlp_layers = [(int(name.split()[-1]), float(score))
                  for name, score in node_scores.items()
                  if name.startswith("MLP ")]
    if not mlp_layers:
        return -1
    return max(mlp_layers, key=lambda lv: lv[1])[0]


def _run_rome_easyedit(
    model_name: str,
    target_layer: int,
    edits: List[dict],
    device_id: int = 0,
) -> List[Dict]:
    """Apply ROME for all edits via EasyEdit and return per-edit metrics.

    EasyEdit's BaseEditor loads a HuggingFace model internally, applies
    ROME at the requested layer, and computes pre/post metrics in one pass.
    Sequential editing is used: each edit is applied on top of the previous.
    """
    from easyeditor import ROMEHyperParams, BaseEditor

    hparams = ROMEHyperParams.from_hparams(str(_ROME_HPARAMS_YAML))
    hparams.model_name = model_name
    hparams.layers = [target_layer]
    hparams.device = device_id if torch.cuda.is_available() else "cpu"
    hparams.fp16 = False
    hparams.stats_dir = str(_EASYEDIT_DIR / "data" / "stats")

    prompts = [e["edit_prompt"] for e in edits]
    subjects = [e["subject"] for e in edits]
    target_new = [e["target"] for e in edits]
    rephrase_prompts = [e["paraphrase"] for e in edits]
    locality_inputs = {
        "neighborhood": {
            "prompt": [e["locality"] for e in edits],
            "ground_truth": [e["locality_target"] for e in edits],
        }
    }

    editor = BaseEditor.from_hparams(hparams)
    metrics, _, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        sequential_edit=True,
        keep_original_weight=False,
    )
    return metrics


def _parse_easyedit_metrics(
    edits: List[dict], metrics_list: List[Dict]
) -> tuple:
    """Parse EasyEdit per-edit metrics into per_edit list + aggregate rates."""
    per_edit = []
    for edit, m in zip(edits, metrics_list):
        post = m.get("post", {})
        rewrite_acc = post.get("rewrite_acc", [0.0])
        rephrase_acc = post.get("rephrase_acc", [0.0])
        nbr_acc = post.get("locality", {}).get("neighborhood_acc", [1.0])

        edit_succ = float(rewrite_acc[0]) if rewrite_acc else 0.0
        para_succ = float(rephrase_acc[0]) if rephrase_acc else 0.0
        loc_acc = float(nbr_acc[0]) if nbr_acc else 1.0

        per_edit.append({
            "edit_id": edit["subject"],
            "edit_success": edit_succ,
            "paraphrase_success": para_succ,
            "locality_preserved": loc_acc,
        })

    edit_succ_rate = sum(e["edit_success"] for e in per_edit) / len(per_edit)
    para_succ_rate = sum(e["paraphrase_success"] for e in per_edit) / len(per_edit)
    loc_preserved = sum(e["locality_preserved"] for e in per_edit) / len(per_edit)
    locality_drift = 1.0 - loc_preserved

    return per_edit, edit_succ_rate, para_succ_rate, locality_drift


def _cell(method: str, model_name: str, seed: int,
          out_dir: Path) -> Dict[str, object]:
    t0 = time.time()
    cell = {"method": method, "seed": seed}

    try:
        discovery = get_or_run_discovery(
            algorithm=method, model=model_name, task=TASK,
            num_examples=32, batch_size=1,
        )
        node_scores = discovery["node_scores"]
        target_layer = _highest_mlp_layer(node_scores)
        cell["target_layer"] = target_layer

        if target_layer < 0:
            cell["error"] = "no MLP node in circuit"
            cell["wall_s"] = round(time.time() - t0, 2)
            return cell

        if not _ensure_easyedit():
            cell["error"] = "EasyEdit clone failed"
            cell["wall_s"] = round(time.time() - t0, 2)
            return cell

        metrics_list = _run_rome_easyedit(model_name, target_layer, EDITS)
        per_edit, edit_succ, para_succ, loc_drift = _parse_easyedit_metrics(
            EDITS, metrics_list
        )

        cell["per_edit"] = per_edit
        cell["edit_success_rate"] = round(edit_succ, 4)
        cell["paraphrase_success_rate"] = round(para_succ, 4)
        cell["locality_drift"] = round(loc_drift, 4)
        cell["composite"] = round(
            edit_succ * para_succ * max(0.0, 1.0 - loc_drift), 4
        )
    except Exception as exc:
        cell["error"] = f"{type(exc).__name__}: {exc}"

    cell["wall_s"] = round(time.time() - t0, 2)
    return cell


def _random_baseline(model_name: str, n_seeds: int = 3) -> Dict[str, object]:
    """Random-middle-layer baseline at the same edit budget."""
    import random
    from transformer_lens import HookedTransformer

    out = []
    for seed in range(n_seeds):
        random.seed(seed)
        tmp_model = HookedTransformer.from_pretrained(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float32,
        )
        n_layers = tmp_model.cfg.n_layers
        target_layer = random.randint(n_layers // 4, 3 * n_layers // 4)
        del tmp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            metrics_list = _run_rome_easyedit(model_name, target_layer, EDITS)
            _, edit_succ, _, _ = _parse_easyedit_metrics(EDITS, metrics_list)
        except Exception as exc:
            edit_succ = 0.0
            print(f"  baseline seed={seed} error: {exc}")

        out.append({"seed": seed, "target_layer": target_layer,
                    "edit_success_rate": round(edit_succ, 4)})
    return {"label": "random_middle_layer", "cells": out}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip_random_baseline", action="store_true")
    args = parser.parse_args()

    if not _ensure_easyedit():
        print("ERROR: could not clone EasyEdit -- aborting.")
        return 1

    out_dir = make_results_dir(SCRIPT_NAME)
    rows: List[Dict[str, object]] = []
    for method in args.methods:
        for seed in args.seeds:
            cell = _cell(method, args.model, seed, out_dir)
            rows.append(cell)
            print(f"[{method} seed={seed}] target_L={cell.get('target_layer')} "
                  f"composite={cell.get('composite')} {cell.get('error') or 'ok'}")

    baseline = None if args.skip_random_baseline else _random_baseline(args.model)

    payload = {"script": SCRIPT_NAME, "model": args.model, "task": TASK,
               "n_edits": len(EDITS), "method_seed_rows": rows,
               "random_baseline": baseline}
    write_status(out_dir, payload)
    print(f"Wrote {out_dir / 'status.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
