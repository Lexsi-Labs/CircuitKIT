"""Shared driver for the paper-style algo benchmark.

Each per-(model,task) script imports `run_benchmark_cell` and runs every
algorithm in ALL_ALGOS, then writes a per-cell JSON with discovery wall
time + 6-pillar faithfulness scores from evaluate_circuit().

What we measure (Pillars 1-6 from CircuitKit's run_full_faithfulness):
  Pillar 1 — Patching:        circuit reproduces clean-input behaviour
                              when out-of-circuit edges are ablated.
  Pillar 2 — Ablation:        complement (the in-circuit edges, when
                              ablated, break the behaviour).
  Pillar 3 — Stability:       Jaccard of top-k circuits across seeds.
  Pillar 4 — Robustness:      faithfulness under input corruption (paraphrase).
  Pillar 5 — Baselines:       comparison vs random-pruned circuit.
  Pillar 6 — Generalization:  faithfulness on a related task.

We default to fast pillars (1+2) for the smoke runs because pillars 3-6
either need multi-seed runs (3) or extra dataloaders (4-6) — those are
opt-in via the `pillars` argument.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the library importable.
_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Re-use the runner status writer.
sys.path.insert(0, str(_REPO / "validation"))
from _common import make_results_dir, write_status  # noqa: E402

# The full algorithm matrix this benchmark sweeps over.
# Order matters: cheaper algos first so a partial run still has coverage.
ALL_ALGOS: List[str] = [
    "eap",
    "eap-ig",
    "eap-ig-activations",
    "eap-clean-corrupted",
    "relp",
    "atp-gd",
    "eap-gp",
    "eap-exact",
    "ibcircuit",
    "peap",
    "cdt",
    "eap-ifr",
]

# A per-algo cache for discovery artifacts so re-running a benchmark cell
# (eg. after fixing a bug) doesn't redo discovery.
_BENCH_CACHE = _REPO / "validation" / "_cache" / "benchmark"
_BENCH_CACHE.mkdir(parents=True, exist_ok=True)


def _build_config(*, algorithm: str, model: str, task: str,
                  num_examples: int, batch_size: int,
                  ig_steps: int, target_sparsity: float, scope: str,
                  artifact_path: Path,
                  precision: str = "float32",
                  ibcircuit_epochs: int = 200) -> Dict[str, Any]:
    """Build a discover_circuit / evaluate_circuit config dict for one cell."""
    discovery: Dict[str, Any] = {
        "algorithm": algorithm,
        "task": task,
        "level": "node",
        "batch_size": batch_size,
        "data_params": {"num_examples": num_examples},
        # MMLU TaskSpec requires model_name for per-example tokenisation;
        # other tasks ignore it harmlessly.
        "model_name": model,
    }
    # MMLU TaskSpec ignores `num_examples` and instead samples per subject
    # across 57 subjects. Without this, smoke runs explode to >1000 examples.
    # Map num_examples → samples_per_subject (1 per subject for tiny smokes).
    if task == "mmlu":
        discovery["samples_per_subject"] = max(1, num_examples // 12)
    if algorithm in ("eap-ig", "eap-ig-activations"):
        discovery["ig_steps"] = ig_steps
    if algorithm == "eap-gp":
        discovery["ig_steps"] = max(3, ig_steps)  # paper default k=5
    if algorithm == "ibcircuit":
        discovery["scope"] = scope
        discovery["num_epochs"] = ibcircuit_epochs

    return {
        "model": {"name": model, "precision": precision},
        "discovery": discovery,
        "pruning": {"target_sparsity": target_sparsity, "scope": scope},
        "output_path": str(artifact_path),
        "eval": {
            "num_examples": num_examples,
            "seed": 42,
            "pillars": ["patching", "ablation"],
            "n_stability_runs": 3,
            "corruption_variants": ["paraphrase"],
        },
    }


def maybe_warn_worthiness(model_name: str, task: str) -> None:
    """If the data layer's worthiness validator has a recent verdict for this
    (task) pair, log it so the bench cell makes the data quality visible.
    Best-effort: silent if no validator output exists.
    """
    import json as _json
    import logging as _logging
    log = _logging.getLogger("benchmark.worthiness")
    repo = Path(__file__).resolve().parents[2]
    candidates = list((repo / "validation" / "results").glob(
        f"*/0[0-9]*{task}*/worthiness.json"
    ))
    if not candidates:
        return
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        rep = _json.loads(latest.read_text())
        verdict = rep.get("verdict", "?")
        if verdict in ("RED", "YELLOW"):
            log.warning(
                f"worthiness verdict {verdict} for task={task!r} "
                f"(from {latest.parent.name}). "
                f"Bench results may be unreliable. "
                f"Suggested fixes: {rep.get('suggested_fixes', [])[:1]}"
            )
    except (_json.JSONDecodeError, KeyError):
        pass


def run_benchmark_cell(
    *, algorithm: str, model: str, task: str,
    num_examples: int = 32, batch_size: int = 1, ig_steps: int = 3,
    target_sparsity: float = 0.1, scope: str = "heads",
    pillars: Optional[List[str]] = None,
    precision: str = "float32",
    ibcircuit_epochs: int = 200,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Run one benchmark cell: discover + evaluate. Returns the row dict.

    Failures are caught and reported; the row contains an "error" field.
    """
    from circuitkit.api import discover_circuit, evaluate_circuit

    cell_id = f"{algorithm}_{task}_{model.replace('/', '_')}"
    artifact_path = _BENCH_CACHE / f"{cell_id}.pt"
    scores_json = artifact_path.parent / (artifact_path.stem + "_scores.json")

    config = _build_config(
        algorithm=algorithm, model=model, task=task,
        num_examples=num_examples, batch_size=batch_size,
        ig_steps=ig_steps, target_sparsity=target_sparsity, scope=scope,
        artifact_path=artifact_path,
        precision=precision, ibcircuit_epochs=ibcircuit_epochs,
    )
    if pillars is not None:
        config["eval"]["pillars"] = pillars
        # When user requests pillars 3-6, run_full_faithfulness is invoked
        # which needs the explicit flag.
        if any(p in pillars for p in ("stability", "robustness", "baselines", "generalization")):
            config["eval"]["full_faithfulness_eval"] = True

    # Best-effort worthiness check — logs RED/YELLOW verdicts before discovery.
    maybe_warn_worthiness(model, task)

    row: Dict[str, Any] = {
        "algorithm": algorithm, "model": model, "task": task,
        "num_examples": num_examples, "target_sparsity": target_sparsity,
    }

    # Discovery
    t0 = time.time()
    try:
        if force_rerun or not scores_json.exists():
            discover_circuit(config)
        row["discovery_seconds"] = round(time.time() - t0, 2)
    except Exception as exc:
        row["discovery_seconds"] = round(time.time() - t0, 2)
        row["error"] = f"discovery: {type(exc).__name__}: {exc}"
        return row

    # Evaluation (six-pillar; here just patching+ablation by default)
    t0 = time.time()
    try:
        eval_result = evaluate_circuit(config, pruned_artifact_path=str(artifact_path))
        row["eval_seconds"] = round(time.time() - t0, 2)
        # evaluate_circuit always returns a FaithfulnessReport (.patching_score is
        # Pillar 1, .ablation_score is Pillar 2). The legacy `{baseline_avg,
        # circuit_avg, random_avg}` dict branch below is retained only as a
        # defensive fallback for artefacts produced by old out-of-tree callers.
        if hasattr(eval_result, "patching_score"):
            row["patching"] = float(eval_result.patching_score) if eval_result.patching_score is not None else None
            row["ablation"] = float(eval_result.ablation_score) if eval_result.ablation_score is not None else None
            row["stability"] = getattr(eval_result, "stability", None)
            row["robustness"] = getattr(eval_result, "robustness", None)
            row["baseline_comparison"] = getattr(eval_result, "baseline_comparison", None)
            row["generalization"] = getattr(eval_result, "generalization", None)
            row["random_avg"] = None  # not exposed in report path
        else:
            ba = eval_result.get("baseline_avg")
            ca = eval_result.get("circuit_avg")
            row["patching"] = float(ba) if ba is not None else None
            row["ablation"] = float(ca) if ca is not None else None
            row["random_avg"] = eval_result.get("random_avg")
        # Keep the original keys around for transparency / downstream tooling.
        row["baseline_avg"] = row.get("patching")
        row["circuit_avg"]  = row.get("ablation")
    except Exception as exc:
        row["eval_seconds"] = round(time.time() - t0, 2)
        row["error"] = f"eval: {type(exc).__name__}: {exc}"
        # Keep going — discovery succeeded so the row is partially useful.

    return row


def render_summary(rows: List[Dict[str, Any]]) -> str:
    """Markdown table for a single (model, task) sweep."""
    if not rows:
        return "(no rows)\n"

    lines = [
        "| Algorithm | Discover (s) | Eval (s) | Patching (P1) | Ablation (P2) | Note |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        def fmt(x, prec=3):
            if x is None or x == "":
                return "—"
            try:
                return f"{float(x):.{prec}f}"
            except (TypeError, ValueError):
                return str(x)
        note = r.get("error", "")
        if note and len(note) > 50:
            note = note[:47] + "..."
        lines.append(
            f"| `{r.get('algorithm')}` | "
            f"{fmt(r.get('discovery_seconds'), 1)} | "
            f"{fmt(r.get('eval_seconds'), 1)} | "
            f"{fmt(r.get('patching'))} | "
            f"{fmt(r.get('ablation'))} | "
            f"{note} |"
        )
    return "\n".join(lines) + "\n"
