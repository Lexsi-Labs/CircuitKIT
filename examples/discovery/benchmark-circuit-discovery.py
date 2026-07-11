"""
benchmark_circuit_discovery.py

Multi-algorithm circuit discovery benchmark with full 6-pillar faithfulness evaluation.

Runs discovery (EAP, EAP-IG, IBCircuit, etc.) on a given model/task, then evaluates
the discovered circuit using the full faithfulness framework (run_full_faithfulness).
Results are saved as detailed JSON reports and a summary CSV.

Usage examples:
    # Built-in task
    python benchmark_circuit_discovery.py \
        --model gpt2 --algo eap --level neuron --task ioi --seed 42

    # Custom CSV data (paired)
    python benchmark_circuit_discovery.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --algos eap eap-ig \
        --custom-data jailbreak_binary.csv \
        --clean-prompt "{system_jailbreak}\n\nUser: {benign_req}\nAssistant:" \
        --corrupt-prompt "{system_jailbreak}\n\nUser: {harmful_req}\nAssistant:" \
        --clean-answer "{clean_ans}" \
        --corrupt-answer "{corrupt_ans}" \
        --level neuron --target-sparsity 0.3

Output structure (under --results-dir, default circuitkit/results/):
    benchmark_{algo}_{level}_{model}_{task}_sp{sparsity}_seed{seed}_{ts}.pt
    benchmark_{algo}_{level}_{model}_{task}_sp{sparsity}_seed{seed}_{ts}_scores.pt
    benchmark_{algo}_{level}_{model}_{task}_sp{sparsity}_seed{seed}_{ts}_faithfulness.json
    benchmark_summary_{model}_{task}_{level}_{ts}.csv
    benchmark_report_{model}_{task}_{level}_{ts}.txt
"""

import csv
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from circuitkit.api import discover_circuit, evaluate_circuit, prepare_custom_task


# Algorithms that do not require a corrupted counterpart
_UNPAIRED_ALGOS = {"ibcircuit", "cdt"}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    default_results_dir = os.path.join(project_root, "results")

    parser = argparse.ArgumentParser(
        description="Multi-algorithm circuit discovery benchmark with 6-pillar faithfulness eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--precision", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--mlp-hook", type=str, default="mlp_out",
                        choices=["mlp_out", "post_act"])

    # ── Sweep axes ────────────────────────────────────────────────────────────
    _algo_choices = [
        "acdc", "eap", "eap-ig", "ibcircuit",
        "eap-ig-activations", "eap-clean-corrupted", "eap-exact",
        "atp-gd", "eap-gp", "relp", "peap", "eap-ifr", "cdt",
    ]
    parser.add_argument("--algos", type=str, nargs="+",
                        default=["eap", "eap-ig", "ibcircuit"],
                        choices=_algo_choices)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])

    # Single-run shortcuts
    parser.add_argument("--algo", type=str, default=None, choices=_algo_choices,
                        help="Single algorithm (shorthand for --algos with one value)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (shorthand for --seeds with one value)")

    # ── Task & level ──────────────────────────────────────────────────────────
    parser.add_argument("--task", type=str, default="ioi")
    parser.add_argument("--target-task", type=str, default=None)
    parser.add_argument("--level", type=str, default="neuron",
                        choices=["node", "neuron"])
    parser.add_argument("--scope", type=str, default="both",
                        choices=["heads", "mlp", "both"])

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--eval-num-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--padding-side", type=str, default=None,
                        choices=["left", "right"])

    # ── MMLU-specific ─────────────────────────────────────────────────────────
    parser.add_argument("--subjects", type=str, nargs="+", default=None)
    parser.add_argument("--samples-per-subject", type=int, default=20)

    # ── WMDP-specific ─────────────────────────────────────────────────────────
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        choices=["wmdp-bio", "wmdp-chem", "wmdp-cyber"])
    parser.add_argument("--target-configs", type=str, nargs="+", default=None,
                        choices=["wmdp-bio", "wmdp-chem", "wmdp-cyber"])
    parser.add_argument("--samples-per-config", type=int, default=None)

    # ── BoolQ-specific ────────────────────────────────────────────────────────
    parser.add_argument("--cache-dir", type=str, default=None)

    # ── GLUE-specific ─────────────────────────────────────────────────────────
    parser.add_argument("--glue-task", type=str, default="sst2",
                        choices=["mrpc", "qqp", "sst2", "rte", "cola"])
    parser.add_argument("--glue-split", type=str, default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--samples-per-split", type=int, default=None)

    # ── Custom data ──────────────────────────────────────────────────────────
    parser.add_argument("--custom-data", type=str, default=None,
                        help="Path to a custom CSV. Overrides --task for data loading. "
                             "Use --clean-prompt / --corrupt-prompt / --clean-answer / "
                             "--corrupt-answer to define the template fields.")
    parser.add_argument("--clean-prompt", type=str, default=None)
    parser.add_argument("--corrupt-prompt", type=str, default=None)
    parser.add_argument("--clean-answer", type=str, default=None)
    parser.add_argument("--corrupt-answer", type=str, default=None)

    # ── Pruning ───────────────────────────────────────────────────────────────
    parser.add_argument("--target-sparsity", type=float, default=0.3)

    # ── IBCircuit hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha-loss", type=str, default="kl", choices=["kl", "ce"])
    parser.add_argument("--log-interval", type=int, default=100)

    # ── EAP-IG hyperparameters ────────────────────────────────────────────────
    parser.add_argument("--ig-steps", type=int, default=5)

    # ── Faithfulness eval ─────────────────────────────────────────────────────
    parser.add_argument("--pillars", type=str, nargs="+", default=None,
                        choices=["patching", "ablation", "baselines",
                                 "robustness", "stability", "generalization"])
    parser.add_argument("--n-stability-runs", type=int, default=5)

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--results-dir", type=str, default=default_results_dir)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Single-run shortcuts
    if args.algo is not None:
        args.algos = [args.algo]
    if args.seed is not None:
        args.seeds = [args.seed]

    # Fallback for eval examples
    if args.eval_num_examples is None:
        args.eval_num_examples = args.num_examples

    # Task-specific defaults for target_task
    if args.task == "ioi" and args.target_task is None:
        args.target_task = "double_io"
    if args.task in ("boolq", "glue") and args.target_task is None:
        args.target_task = None
    if args.task == "wmdp":
        if args.target_task is None:
            args.target_task = "wmdp"
        if args.configs is None:
            args.configs = ["wmdp-cyber"]
        if args.target_configs is None:
            args.target_configs = ["wmdp-bio"]

    # Validate custom data: template fields required for paired algorithms
    if args.custom_data:
        paired_algos = [a for a in args.algos if a not in _UNPAIRED_ALGOS]
        if paired_algos and not all([
            args.clean_prompt, args.corrupt_prompt,
            args.clean_answer, args.corrupt_answer,
        ]):
            parser.error(
                f"Algorithms {paired_algos} require paired data. "
                "Provide --clean-prompt, --corrupt-prompt, --clean-answer, --corrupt-answer."
            )

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Task registration (custom data only)
# ─────────────────────────────────────────────────────────────────────────────

def register_custom_tasks(args) -> Dict[str, str]:
    """
    Pre-register custom data tasks before the sweep begins.

    Returns a dict mapping algo -> resolved_task_name. Paired algorithms
    get a 'template' task; unpaired algorithms (ibcircuit, cdt) get a
    'clean_only' task. Both are registered once and reused across all seeds.

    For built-in tasks, returns {} and the sweep uses args.task directly.
    """
    if not args.custom_data:
        return {}

    import torch as t
    from transformer_lens import HookedTransformer

    device = "cuda" if t.cuda.is_available() else "cpu"
    dtype = getattr(t, args.precision)

    print(f"Loading model for custom task registration ({args.model})...")
    model = HookedTransformer.from_pretrained(args.model, device=device, dtype=dtype)

    task_map: Dict[str, str] = {}

    # Register paired task once (reused by all paired algorithms)
    paired_algos = [a for a in args.algos if a not in _UNPAIRED_ALGOS]
    if paired_algos:
        stem = os.path.splitext(os.path.basename(args.custom_data))[0]
        paired_name = f"custom:{stem}"
        cfg = {
            "data": {
                "type": "template",
                "path": args.custom_data,
                "template": {
                    "clean_prompt":   args.clean_prompt,
                    "corrupt_prompt": args.corrupt_prompt,
                    "clean_answer":   args.clean_answer,
                    "corrupt_answer": args.corrupt_answer,
                },
                **({"pair_padding_side": args.padding_side} if args.padding_side else {}),
            },
            "discovery": {"task": ""},
        }
        prepare_custom_task(cfg, model=model, task_name=paired_name)
        for algo in paired_algos:
            task_map[algo] = paired_name
        print(f"  Registered paired task '{paired_name}' for: {paired_algos}")

    # Register clean_only task once (reused by ibcircuit, cdt)
    unpaired_algos = [a for a in args.algos if a in _UNPAIRED_ALGOS]
    if unpaired_algos:
        stem = os.path.splitext(os.path.basename(args.custom_data))[0]
        unpaired_name = f"custom:{stem}:unpaired"
        cfg = {
            "data": {
                "type": "template",
                "path": args.custom_data,
                "template": {
                    "clean_prompt":  args.clean_prompt,
                    "clean_answer":  args.clean_answer,
                },
            },
            "discovery": {
                "algorithm": unpaired_algos[0],  # ibcircuit or cdt — both are clean-only algos
                "task": "",
            },
        }
        prepare_custom_task(cfg, model=model, task_name=unpaired_name)
        for algo in unpaired_algos:
            task_map[algo] = unpaired_name
        print(f"  Registered clean_only task '{unpaired_name}' for: {unpaired_algos}")

    del model
    if t.cuda.is_available():
        t.cuda.empty_cache()

    print()
    return task_map


# ─────────────────────────────────────────────────────────────────────────────
# Config builder
# ─────────────────────────────────────────────────────────────────────────────

def build_config(
    args,
    algo: str,
    seed: int,
    output_path: str,
    task_name: str,
) -> Dict[str, Any]:
    """
    Build a complete CircuitKit config dict for one (algo, seed) combination.

    task_name is the already-resolved task name — a registered built-in name
    (e.g. 'ioi') or the name returned by register_custom_tasks() for custom
    data (e.g. 'custom:jailbreak_binary'). No "data" block is included; the
    task is already in the registry.
    """
    discovery_intervention = "mean" if algo == "ibcircuit" else "patching"

    # Task-specific data_params
    if args.task == "mmlu":
        data_block = {
            "model_name": args.model,
            "subjects":   args.subjects,
            "samples_per_subject": args.samples_per_subject,
        }
    elif args.task == "wmdp":
        data_block = {
            "model_name":         args.model,
            "configs":            args.configs,
            "samples_per_config": args.samples_per_config or args.num_examples,
            "seed":               seed,
        }
    elif args.task == "glue":
        data_block = {
            "data_params": {"num_examples": args.num_examples, "seed": seed},
            "glue_task":         args.glue_task,
            "split":             args.glue_split,
            "samples_per_split": args.samples_per_split or args.num_examples,
        }
    elif args.task == "boolq":
        data_block = {
            "data_params": {"num_examples": args.num_examples, "seed": seed},
            **({"cache_dir": args.cache_dir} if args.cache_dir else {}),
        }
    else:
        # Covers: ioi, custom data, and any other task with standard data_params
        data_block = {
            "data_params": {"num_examples": args.num_examples, "seed": seed},
        }

    discovery_base = {
        "algorithm":    algo,
        "intervention": discovery_intervention,
        "task":         task_name,
        "scope":        args.scope,
        "level":        args.level,
        "batch_size":   args.batch_size,
        "mlp_hook":     args.mlp_hook,
        "evaluate":     False,
        "verbose":      args.verbose,
        **data_block,
        **({"pair_padding_side": args.padding_side} if args.padding_side else {}),
    }

    _ig_steps_algos = {"eap-ig", "eap-ig-activations", "eap-gp"}
    if algo == "eap-ig":
        discovery_base["method"] = "EAP-IG-inputs"
        discovery_base["ig_steps"] = args.ig_steps
    elif algo in _ig_steps_algos:
        discovery_base["ig_steps"] = args.ig_steps
    elif algo == "ibcircuit":
        discovery_base.update({
            "num_epochs":    args.num_epochs,
            "learning_rate": args.learning_rate,
            "alpha":         args.alpha,
            "beta":          args.beta,
            "alpha_loss":    args.alpha_loss,
            "log_interval":  args.log_interval,
            "mask_type":     "sigmoid",
        })

    eval_block: Dict[str, Any] = {
        "num_examples":           args.eval_num_examples,
        "seed":                   seed,
        "full_faithfulness_eval": True,
        "n_stability_runs":       args.n_stability_runs,
    }
    if args.pillars is not None:
        eval_block["pillars"] = args.pillars
    if args.target_task:
        eval_block["target_task"] = args.target_task
    if getattr(args, "target_configs", None):
        eval_block["target_configs"] = args.target_configs

    return {
        "model": {"name": args.model, "precision": args.precision},
        "discovery": discovery_base,
        "pruning": {
            "target_sparsity": args.target_sparsity,
            "scope":           args.scope,
            "intervention":    "mean",
            "random":          True,
        },
        "output_path": output_path,
        "eval": eval_block,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_run_stem(
    algo: str, level: str, model: str, task: str,
    sparsity: float, seed: int, timestamp: str,
) -> str:
    model_safe = model.split("/")[-1]
    return f"benchmark_{algo}_{level}_{model_safe}_{task}_sp{sparsity:.1f}_seed{seed}_{timestamp}"


def save_faithfulness_report(report, path: str) -> None:
    def _serialise(obj):
        if isinstance(obj, dict):
            return {k: _serialise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialise(v) for v in obj]
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
        except ImportError:
            pass
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    if hasattr(report, "to_dict"):
        data = report.to_dict()
    elif hasattr(report, "__dict__"):
        data = report.__dict__
    else:
        data = report

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialise(data), f, indent=2)


def extract_summary_row(
    algo: str, seed: int, report, run_time_s: float, artifact_stem: str,
) -> Dict[str, Any]:
    # evaluate_circuit returns a FaithfulnessReport for every path as of 1.0.
    return {
        "algo":           algo,
        "seed":           seed,
        "patching_score": getattr(report, "patching_score", None),
        "ablation_score": getattr(report, "ablation_score", None),
        "random_avg":     report.metadata.get("random_avg") if getattr(report, "metadata", None) else None,
        "run_time_s":     round(run_time_s, 1),
        "artifact_stem":  artifact_stem,
    }


def save_summary_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_text_report(
    rows: List[Dict], args, timestamp: str, path: str, errors: List[Dict],
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    pillars_str = ", ".join(args.pillars) if args.pillars else "all 6"
    display_task = args.task if not args.custom_data else f"custom:{os.path.splitext(os.path.basename(args.custom_data))[0]}"

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CIRCUITKIT CIRCUIT DISCOVERY BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp : {timestamp}\n")
        f.write(f"Model     : {args.model}\n")
        f.write(f"Task      : {display_task}\n")
        f.write(f"Level     : {args.level}\n")
        f.write(f"Scope     : {args.scope}\n")
        f.write(f"Sparsity  : {args.target_sparsity:.1%}\n")
        f.write(f"Algos     : {', '.join(args.algos)}\n")
        f.write(f"Seeds     : {', '.join(str(s) for s in args.seeds)}\n")
        f.write(f"Discovery examples : {args.num_examples}\n")
        f.write(f"Eval examples      : {args.eval_num_examples}\n")
        f.write(f"Pillars   : {pillars_str}\n\n")

        f.write("-" * 80 + "\n")
        f.write("PER-RUN RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Algo':<12} {'Seed':>6} {'Patching':>10} {'Ablation':>10} {'Time(s)':>9}\n")
        f.write("-" * 50 + "\n")
        for row in rows:
            p = f"{row['patching_score']:.4f}" if row["patching_score"] is not None else "  N/A  "
            a = f"{row['ablation_score']:.4f}" if row["ablation_score"] is not None else "  N/A  "
            f.write(f"{row['algo']:<12} {row['seed']:>6} {p:>10} {a:>10} {row['run_time_s']:>9.1f}\n")

        if errors:
            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("ERRORS\n")
            f.write("-" * 80 + "\n")
            for err in errors:
                f.write(f"  algo={err['algo']} seed={err['seed']}: {err['error']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Configure logging once, at the start, based on --verbose.
    # Never reconfigured inside the sweep loop.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.split("/")[-1]
    display_task = (
        f"custom:{os.path.splitext(os.path.basename(args.custom_data))[0]}"
        if args.custom_data else args.task
    )
    os.makedirs(args.results_dir, exist_ok=True)

    total_runs = len(args.algos) * len(args.seeds)
    pillars_display = ", ".join(args.pillars) if args.pillars else "all 6"
    _expensive = {"eap-exact", "atp-gd", "eap-gp"}

    print("=" * 70)
    print("CIRCUITKIT CIRCUIT DISCOVERY BENCHMARK")
    print("=" * 70)
    print(f"Model    : {args.model}  ({args.precision})")
    print(f"Task     : {display_task}  |  Level: {args.level}  |  Scope: {args.scope}")
    print(f"Algos    : {', '.join(args.algos)}")
    print(f"Seeds    : {', '.join(str(s) for s in args.seeds)}")
    print(f"Sparsity : {args.target_sparsity:.1%}")
    print(f"Examples : {args.num_examples} (discovery)  /  {args.eval_num_examples} (eval)")
    print(f"Pillars  : {pillars_display}")
    print(f"  Total runs: {total_runs}")
    if _expensive & set(args.algos):
        print(f"  Note: {_expensive & set(args.algos)} are compute-intensive; "
              f"consider --num-examples 16 for these.")
    print(f"Results  : {args.results_dir}/")
    print("=" * 70)
    print()

    # Register custom tasks once before the sweep.
    # Returns {} for built-in tasks (no-op path).
    # Maps algo -> registered task name for custom data.
    task_map = register_custom_tasks(args)

    summary_rows: List[Dict] = []
    errors: List[Dict] = []

    for run_idx, (seed, algo) in enumerate(
        ((s, a) for s in args.seeds for a in args.algos), start=1
    ):
        # Resolve task name: custom data uses pre-registered name, built-in uses args.task
        task_name = task_map.get(algo, args.task)

        stem = make_run_stem(algo, args.level, args.model, display_task,
                             args.target_sparsity, seed, timestamp)
        output_path = os.path.join(args.results_dir, stem + ".pt")
        faithfulness_report_path = os.path.join(args.results_dir, stem + "_faithfulness.json")

        print(f"[{run_idx}/{total_runs}]  algo={algo}  seed={seed}")
        print(f"  Artifact  → {output_path}")
        print(f"  Report    → {faithfulness_report_path}")

        config = build_config(args, algo, seed, output_path, task_name)
        run_start = time.time()

        # ── Discovery ─────────────────────────────────────────────────────
        try:
            print("  [1/2] Running discovery...")
            pruned_artifact = discover_circuit(config)

            if isinstance(pruned_artifact, dict):
                mlp_n  = sum(len(v) for v in pruned_artifact.get("mlp",   {}).values())
                attn_n = sum(len(v) for v in pruned_artifact.get("heads", {}).values())
                print(f"        Pruned {mlp_n + attn_n} neurons  ({mlp_n} MLP, {attn_n} attn)")
            else:
                print(f"        Pruned {len(pruned_artifact)} nodes")

        except Exception as e:
            elapsed = time.time() - run_start
            print(f"  ✗ Discovery failed: {e}")
            errors.append({"algo": algo, "seed": seed, "phase": "discovery", "error": str(e)})
            summary_rows.append({
                "algo": algo, "seed": seed,
                "patching_score": None, "ablation_score": None, "random_avg": None,
                "run_time_s": round(elapsed, 1), "artifact_stem": stem,
            })
            print()
            continue

        # ── Faithfulness evaluation ────────────────────────────────────────
        try:
            print("  [2/2] Running 6-pillar faithfulness evaluation...")
            report = evaluate_circuit(config, pruned_artifact_path=output_path)

            save_faithfulness_report(report, faithfulness_report_path)
            print(f"        Report saved → {faithfulness_report_path}")

            elapsed = time.time() - run_start
            row = extract_summary_row(algo, seed, report, elapsed, stem)
            summary_rows.append(row)

            if row["patching_score"] is not None:
                print(f"        Patching : {row['patching_score']:.4f}")
            if row["ablation_score"] is not None:
                print(f"        Ablation : {row['ablation_score']:.4f}")
            print(f"        Run time : {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - run_start
            print(f"  ✗ Evaluation failed: {e}")
            errors.append({"algo": algo, "seed": seed, "phase": "eval", "error": str(e)})
            summary_rows.append({
                "algo": algo, "seed": seed,
                "patching_score": None, "ablation_score": None, "random_avg": None,
                "run_time_s": round(elapsed, 1), "artifact_stem": stem,
            })

        print()

    # ── Aggregate outputs ──────────────────────────────────────────────────────
    summary_stem = f"benchmark_summary_{model_safe}_{display_task}_{args.level}_{timestamp}"
    csv_path    = os.path.join(args.results_dir, summary_stem + ".csv")
    report_path = os.path.join(args.results_dir, summary_stem + ".txt")

    save_summary_csv(summary_rows, csv_path)
    save_text_report(summary_rows, args, timestamp, report_path, errors)

    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Runs completed : {len(summary_rows)}")
    print(f"Errors         : {len(errors)}")
    print(f"Summary CSV    : {csv_path}")
    print(f"Text report    : {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()