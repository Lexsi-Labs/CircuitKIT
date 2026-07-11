"""Safety dataset measurements: Pillar 1-7 faithfulness on AdvBench + WMDP.

Runs the full CircuitKit faithfulness evaluation suite (Pillars 1-7) on two
safety-relevant datasets:
  - AdvBench (Zou et al. 2023)  -- refusal-circuit localisation
  - WMDP (Li et al. 2024)       -- dangerous-knowledge circuit localisation

For each dataset x algorithm combo:
  1. Discover circuit with EAP-IG (fast, good signal).
  2. Run run_full_faithfulness (Pillars 1-6 + Intervention Reliability).
  3. Save JSON results per dataset.

This script is designed to run standalone on GPU 1 (RTX PRO 6000, 97 GB).

Usage (CLI):
    CUDA_VISIBLE_DEVICES=1 python 28_safety_dataset_pillar_evals.py
    CUDA_VISIBLE_DEVICES=1 python 28_safety_dataset_pillar_evals.py --datasets advbench
    CUDA_VISIBLE_DEVICES=1 python 28_safety_dataset_pillar_evals.py --quick
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "validation"))

from _apps_common import make_results_dir, write_status  # noqa: E402

SCRIPT = "28_safety_dataset_pillar_evals"

# Datasets: (task_name, hf_path, hf_subset, hf_split, adapter_type)
DATASETS = {
    "advbench": ("advbench_pillar", "walledai/AdvBench", None, "train", "safety"),
    "wmdp":     ("wmdp_pillar",     "cais/wmdp",         "wmdp-bio", "test", "mcq"),
}

ALGO = "eap-ig"
MODEL = "gpt2"
NUM_EXAMPLES = 32
PILLARS_QUICK = ["patching", "ablation", "intervention_reliability"]
PILLARS_FULL = ["patching", "ablation", "baselines", "robustness",
                "stability", "generalization", "intervention_reliability"]


def _register_advbench(task_name: str, n: int = 48) -> Any:
    from datasets import load_dataset
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task, is_task_registered
    if is_task_registered(task_name):
        return
    raw = list(load_dataset("walledai/AdvBench", split="train").take(n))
    ds = SafetyPromptAdapter().adapt(
        raw, name=task_name, max_records=n,
        pairing_mode="harmful_vs_benign",
    )
    ds.records = [r for r in ds.records if r.is_paired]
    spec = NormalizedTaskSpec(ds, name=task_name,
                              cache_dir=str(_REPO / "validation" / "_cache" / task_name))
    register_task(spec)
    return spec


def _register_wmdp(task_name: str, n: int = 48) -> Any:
    from circuitkit.tasks.builtins.wmdp import build_wmdp_spec
    from circuitkit.tasks.registry import register_task, is_task_registered
    if is_task_registered(task_name):
        return
    spec = build_wmdp_spec(subset="wmdp-bio", split="test",
                           name=task_name, max_records=n)
    register_task(spec)
    return spec


def _run_dataset(
    dataset_key: str,
    quick: bool,
    out_dir: Path,
) -> Dict[str, Any]:
    task_name, hf_path, hf_subset, hf_split, adapter = DATASETS[dataset_key]
    pillars = PILLARS_QUICK if quick else PILLARS_FULL

    print(f"\n[{dataset_key}] Registering task '{task_name}' ...")
    try:
        if adapter == "safety":
            spec = _register_advbench(task_name)
        else:
            spec = _register_wmdp(task_name)
    except Exception as exc:
        return {"dataset": dataset_key, "status": "BROKEN",
                "error": f"task registration failed: {exc}"}

    print(f"[{dataset_key}] Running circuit discovery ({ALGO}, {NUM_EXAMPLES} examples) ...")
    t0 = time.time()
    try:
        import torch
        from circuitkit.api import discover_circuit, _reconstruct_circuit_graph
        from circuitkit.artifacts.scores import CircuitScores

        _cache = Path(_REPO) / "validation" / "_cache" / "applications"
        _cache.mkdir(parents=True, exist_ok=True)
        cell_id = f"{ALGO}_{task_name}_{MODEL.replace('/', '_')}"
        artifact_path = _cache / f"{cell_id}.pt"
        scores_path = _cache / f"{cell_id}_scores.json"

        disc_cfg = {
            "algorithm": ALGO,
            "task": task_name,
            "level": "node",
            "batch_size": 1,
            "ig_steps": 3,
            "data_params": {"num_examples": NUM_EXAMPLES},
            "model_name": MODEL,
        }
        pruning_cfg = {"target_sparsity": 0.1, "scope": "heads"}

        if not scores_path.exists():
            discover_circuit({
                "model": {"name": MODEL, "precision": "float32"},
                "discovery": disc_cfg,
                "pruning": pruning_cfg,
                "output_path": str(artifact_path),
            })

        cs = CircuitScores.from_json(scores_path)
        node_scores = cs.node_scores

    except Exception as exc:
        return {"dataset": dataset_key, "status": "BROKEN",
                "error": f"discovery failed: {exc}"}

    discovery_wall = round(time.time() - t0, 1)
    print(f"[{dataset_key}] Discovery done in {discovery_wall}s")

    print(f"[{dataset_key}] Running Pillars: {pillars} ...")
    t1 = time.time()
    try:
        from transformer_lens import HookedTransformer
        from circuitkit.evaluation.full import run_full_faithfulness
        from circuitkit.tasks.registry import get_task

        task_spec = get_task(task_name)
        tl_model = HookedTransformer.from_pretrained(MODEL)
        tl_model.cfg.use_attn_result = True
        tl_model.cfg.use_split_qkv_input = True
        tl_model.cfg.use_hook_mlp_in = True
        tl_model.eval()

        disc_cfg_eval = {
            "algorithm": ALGO,
            "task": task_name,
            "level": "node",
            "batch_size": 4,
            "data_params": {"num_examples": NUM_EXAMPLES},
            "model_name": MODEL,
        }
        pruning_cfg_eval = {"target_sparsity": 0.1, "scope": "heads"}

        # Reconstruct Graph with in_graph flags from the CircuitScores
        scores_data = {"node_scores": node_scores}
        graph = _reconstruct_circuit_graph(
            model=tl_model,
            scores_data=scores_data,
            discovery_cfg=disc_cfg_eval,
            pruning_cfg=pruning_cfg_eval,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        report = run_full_faithfulness(
            model=tl_model,
            graph=graph,
            task_spec=task_spec,
            discovery_cfg=disc_cfg_eval,
            metric_fn=task_spec.metric_fn(),
            pillars=pillars,
            n_stability_runs=2,
            corruption_variants=["paraphrase", "entity_swap"],
            baseline_types=["random", "magnitude"],
        )
        pillar_wall = round(time.time() - t1, 1)

        # Capture the scalar pillar scores from the report. Iterate the actual
        # dataclass fields — an older `dir()` + startswith("p") heuristic matched
        # only `patching_score` and silently dropped `ablation_score` and the
        # rest. Dict-valued pillars (stability, robustness, ...) are skipped by
        # the int/float filter, as before.
        scores = {}
        for field in dataclasses.fields(report):
            val = getattr(report, field.name, None)
            if isinstance(val, (int, float)) or val is None:
                scores[field.name] = val

        result = {
            "dataset": dataset_key,
            "task_name": task_name,
            "algorithm": ALGO,
            "model": MODEL,
            "n_examples": NUM_EXAMPLES,
            "pillars_run": pillars,
            "discovery_wall_s": discovery_wall,
            "pillar_eval_wall_s": pillar_wall,
            "scores": scores,
            "status": "WORKING",
        }
    except Exception as exc:
        result = {
            "dataset": dataset_key,
            "status": "BROKEN",
            "error": f"pillar eval failed: {type(exc).__name__}: {str(exc)[:300]}",
        }

    out_path = out_dir / f"{dataset_key}_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[{dataset_key}] Results saved to {out_path} (status={result['status']})")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Safety dataset Pillar 1-7 evals")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                        default=list(DATASETS.keys()),
                        help="Which datasets to evaluate")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick subset of pillars (patching, ablation, intervention_reliability)")
    args = parser.parse_args()

    out = make_results_dir(SCRIPT)
    all_results = []
    any_broken = False

    for ds in args.datasets:
        res = _run_dataset(ds, quick=args.quick, out_dir=out)
        all_results.append(res)
        if res.get("status") != "WORKING":
            any_broken = True
            print(f"  ERROR [{ds}]: {res.get('error', 'unknown')}")

    summary = {
        "script": SCRIPT,
        "datasets": args.datasets,
        "quick": args.quick,
        "results": all_results,
        "overall_status": "BROKEN" if any_broken else "WORKING",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_status(out, summary)

    print(f"\nSummary: {summary['overall_status']}")
    for r in all_results:
        print(f"  {r['dataset']}: {r.get('status')} | scores={r.get('scores', {})}")

    return 0 if not any_broken else 1


if __name__ == "__main__":
    sys.exit(main())
