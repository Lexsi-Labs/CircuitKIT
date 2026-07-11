"""
D8: run_full_faithfulness() Orchestrator

Orchestrates all 6 pillars of faithfulness evaluation in optimal order.
This is the single entry point for comprehensive circuit evaluation.

Pillar Order (by computational cost):
  1. Pillar 1 (Causal Patching) — fast, baseline-dependent
  2. Pillar 2 (Ablation) — fast, baseline-dependent
  5. Pillar 5 (Baseline Comparison) — moderate cost
  4. Pillar 4 (Robustness) — moderate cost
  3. Pillar 3 (Stability) — expensive, requires multiple discovery runs
  6. Pillar 6 (Generalization) — expensive, requires multiple task dataloaders
"""

import gc
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import torch as t
from circuitkit.utils.device import empty_cache
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from ..backends.eap.graph import Graph
from ..tasks.specs import TaskSpec
from .pillars import (
    Pillar1_CausalPatching,
    Pillar2_Ablation,
    Pillar3_Stability,
    Pillar4_Robustness,
    Pillar5_Baselines,
    Pillar6_Generalization,
)
from .report import FaithfulnessReport

logger = logging.getLogger(__name__)


def _log_gpu_mem(label: str, logger):
    """Log GPU memory stats at DEBUG level. No-op if CUDA unavailable."""
    import torch

    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free_reserved = reserved - allocated
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_total = total - reserved
    logger.debug(
        f"[GPU-MEM] {label}: "
        f"alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, "
        f"free_in_reserved={free_reserved:.2f}GB, free_total={free_total:.2f}GB"
    )


def _flush_gpu_memory():
    """Release unreferenced GPU memory between pillar evaluations."""
    gc.collect()
    if t.cuda.is_available():
        empty_cache()


def run_full_faithfulness(  # noqa: C901 - complex function, refactor out of scope for lint pass
    model: HookedTransformer,
    graph: Graph,
    task_spec: TaskSpec,
    discovery_cfg: Dict[str, Any],
    device: str = "auto",
    pillars: Optional[List[str]] = None,
    n_stability_runs: int = 5,
    metric_fn: Optional[Callable] = None,
    dataloader: Optional[DataLoader] = None,
    intervention_dataloader: Optional[DataLoader] = None,
    corruption_variants: Optional[List[str]] = None,
    baseline_types: Optional[List[str]] = None,
    target_task_spec: Optional[TaskSpec] = None,
    corruption_dataloaders: Optional[Dict[str, DataLoader]] = None,
    target_dataloader: Optional[DataLoader] = None,
    pruning_cfg: Optional[Dict[str, Any]] = None,
    n_reliability_seeds: int = 3,
) -> FaithfulnessReport:
    """
    Run all 6 pillars of faithfulness evaluation end-to-end.

    Orchestrates a complete faithfulness assessment by evaluating circuits
    across multiple dimensions: causal explanation, ablation robustness,
    stability under distribution shift, robustness to corruptions, baseline
    comparison, and cross-task generalization.

    Pillar order is optimized for cost: fast pillars first, expensive ones
    (stability, generalization) last. Allows skipping expensive pillars via
    the `pillars` parameter.

    Args:
        model: HookedTransformer with hooks configured (use_attn_result=True, etc.).
        graph: Discovered circuit graph with in_graph flags set on edges/nodes.
        task_spec: TaskSpec defining metric function and task details.
        discovery_cfg: Discovery configuration dict containing algorithm,
            task, level, scope, and other discovery parameters.
        device: Target device ("cuda" or "cpu"). Defaults to "cuda".
        pillars: Which pillars to compute. Defaults to all:
            ["patching", "ablation", "baselines", "robustness", "stability", "generalization"]
            Pass a subset to skip expensive pillars, e.g., ["patching", "ablation"].
        n_stability_runs: Number of discovery runs for Pillar 3 (Stability).
            Defaults to 5. More runs = better stability estimate but slower.
        metric_fn: Optional custom metric function. If None, uses task_spec.metric_fn.
        dataloader: Evaluation dataloader yielding (clean, corrupted, label) batches.
            If None, built from task_spec and discovery_cfg.
        intervention_dataloader: Dataloader for computing mean activations
            (required for mean/mean-positional ablation in Pillars 2, 4, 6).
        corruption_variants: List of corruption types for Pillar 4 (Robustness).
            Defaults to all five: paraphrase, entity_swap, distractor, role_swap, token_swap.
        baseline_types: Baseline methods for Pillar 5. Defaults to ["random", "magnitude"].
        target_task_spec: TaskSpec for target task in Pillar 6 (Generalization).
            If None, Pillar 6 is skipped even if requested.
        target_dataloader: Dataloader for target task in Pillar 6.

    Returns:
        FaithfulnessReport with scores for all computed pillars.
        Uncomputed pillars are set to None.

    Raises:
        ValueError: If graph or task_spec is invalid, or if pillar request is invalid.
        RuntimeError: If a pillar fails to complete.

    Example:
        >>> # Minimal: Run all 6 pillars
        >>> report = run_full_faithfulness(
        ...     model=model,
        ...     graph=graph,
        ...     task_spec=task_spec,
        ...     discovery_cfg=config['discovery'],
        ...     dataloader=dataloader,
        ... )
        >>> print(f"Patching: {report.patching_score:.4f}")
        >>> print(f"Ablation: {report.ablation_score:.4f}")

        >>> # Quick eval: Skip expensive pillars
        >>> report = run_full_faithfulness(
        ...     model=model,
        ...     graph=graph,
        ...     task_spec=task_spec,
        ...     discovery_cfg=config['discovery'],
        ...     dataloader=dataloader,
        ...     pillars=["patching", "ablation", "robustness"],
        ... )
    """

    # ────────────────────────────────────────────────────────────────────────
    # Setup & Validation
    # ────────────────────────────────────────────────────────────────────────

    if pillars is None:
        pillars = [
            "patching",
            "ablation",
            "baselines",
            "robustness",
            "stability",
            "generalization",
            "intervention_reliability",
        ]

    valid_pillars = {
        "patching",
        "ablation",
        "baselines",
        "robustness",
        "stability",
        "generalization",
        "intervention_reliability",
    }
    invalid = set(pillars) - valid_pillars
    if invalid:
        raise ValueError(f"Invalid pillars: {invalid}. Valid: {valid_pillars}")

    if metric_fn is None:
        # Faithfulness needs a REWARD-oriented, per-sample metric (loss=False,
        # mean=False) so clean > corrupt and the normalized ratio is well-defined.
        # task_spec.metric_fn() returns the loss-style *discovery* metric
        # (loss=True), which inverts the denominator and makes Pillars 1/2
        # spuriously report status='invalid'. _make_eval_metric forces the
        # reward orientation — the same resolution evaluate_circuit() already
        # uses (see api.py). Do NOT change task_spec.metric_fn's default: the
        # EAP selectors correctly rely on loss=True for gradient minimization.
        from ..api import _make_eval_metric

        metric_fn = _make_eval_metric(task_spec)

    if dataloader is None:
        logger.warning(
            "No dataloader provided; attempting to build from task_spec and discovery_cfg"
        )
        dataloader = task_spec.build_dataloader(model, discovery_cfg, device)

    if corruption_variants is None:
        corruption_variants = [
            "paraphrase",
            "entity_swap",
            "distractor",
            "role_swap",
            "token_swap",
            "logical_negation",
            "format_distractor",
            "position_shift",
        ]

    if "robustness" in pillars and corruption_dataloaders is None:
        corruption_dataloaders = {}

    if baseline_types is None:
        baseline_types = ["random", "magnitude"]

    # Resolve the real model name. discovery_cfg is the *discovery* sub-config
    # and does NOT carry a 'model' key (that lives in the top-level config), so
    # the authoritative source is the loaded HookedTransformer's cfg.model_name.
    _cfg_model_name = getattr(getattr(model, "cfg", None), "model_name", None)
    if isinstance(_cfg_model_name, str) and _cfg_model_name:
        model_name = _cfg_model_name
    else:
        # Fallbacks: an explicit override placed on discovery_cfg, then unknown.
        model_name = (
            discovery_cfg.get("model_name")
            or discovery_cfg.get("model", {}).get("name")
            or "unknown"
        )

    logger.info(f"Starting full faithfulness evaluation with pillars: {pillars}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Task: {discovery_cfg.get('task', 'unknown')}")
    logger.info(f"Algorithm: {discovery_cfg.get('algorithm', 'unknown')}")

    _log_gpu_mem("full: entry, before any pillars", logger)

    # Track timing
    timing = {}
    total_start = time.time()

    # Initialize report
    report = FaithfulnessReport(
        patching_score=None,
        ablation_score=None,
    )

    # ────────────────────────────────────────────────────────────────────────
    # PILLAR 1: Causal Patching (fast)
    # ────────────────────────────────────────────────────────────────────────

    if "patching" in pillars:
        logger.info("Computing Pillar 1: Causal Patching...")
        start = time.time()
        try:
            patching_result = Pillar1_CausalPatching.run(
                model=model,
                graph=graph,
                dataloader=dataloader,
                metric_fn=metric_fn,
                device=device,
                quiet=False,
            )
            # Headline score is the normalized 0-1 faithfulness ratio. It is
            # None when the pillar reports status='invalid' (inverted metric
            # direction: clean < corrupt) — the report renders that as "N/A".
            patching_score = patching_result["score"]
            report.patching_score = patching_score
            timing["patching"] = time.time() - start
            _p_score = f"{patching_score:.4f}" if patching_score is not None else "invalid"
            logger.info(
                f"  Patching faithfulness ratio: {_p_score} "
                f"(raw metric: {patching_result['raw_score']:.4f}) "
                f"({timing['patching']:.1f}s)"
            )
            _flush_gpu_memory()
        except Exception as e:
            logger.error(f"Pillar 1 failed: {e}")
            raise RuntimeError(f"Pillar 1 (Causal Patching) failed: {e}") from e

        _log_gpu_mem("full: after Pillar 1 (Patching)", logger)

    # ────────────────────────────────────────────────────────────────────────
    # PILLAR 2: Ablation (fast)
    # ────────────────────────────────────────────────────────────────────────

    if "ablation" in pillars:
        logger.info("Computing Pillar 2: Ablation...")
        start = time.time()
        try:
            pillar2_intervention = discovery_cfg.get("eval_intervention", "zero")
            if (
                pillar2_intervention in ("mean", "mean-positional")
                and intervention_dataloader is None
            ):
                logger.warning(
                    f"Ablation intervention '{pillar2_intervention}' requires "
                    f"intervention_dataloader (not provided). Falling back to 'zero'."
                )
                pillar2_intervention = "zero"

            ablation_result = Pillar2_Ablation.run(
                model=model,
                graph=graph,
                dataloader=dataloader,
                metric_fn=metric_fn,
                intervention=pillar2_intervention,
                intervention_dataloader=intervention_dataloader,
                device=device,
                quiet=False,
            )
            # Headline score is the normalized 0-1 faithfulness ratio. None
            # when status='invalid' (inverted metric direction) — see Pillar 1.
            ablation_score = ablation_result["score"]
            report.ablation_score = ablation_score
            timing["ablation"] = time.time() - start
            _a_score = f"{ablation_score:.4f}" if ablation_score is not None else "invalid"
            logger.info(
                f"  Ablation faithfulness ratio: {_a_score} "
                f"(raw metric: {ablation_result['raw_score']:.4f}) "
                f"({timing['ablation']:.1f}s)"
            )
            _flush_gpu_memory()
        except Exception as e:
            logger.error(f"Pillar 2 failed: {e}")
            raise RuntimeError(f"Pillar 2 (Ablation) failed: {e}") from e

        _log_gpu_mem("full: after Pillar 2 (Ablation)", logger)

    # ────────────────────────────────────────────────────────────────────────
    # PILLAR 5: Baselines (moderate)
    # ────────────────────────────────────────────────────────────────────────

    if "baselines" in pillars:
        logger.info("Computing Pillar 5: Baseline Comparison...")
        start = time.time()
        try:
            baseline_result = Pillar5_Baselines.run(
                model=model,
                graph=graph,
                dataloader=dataloader,
                metric_fn=metric_fn,
                baseline_types=baseline_types,
                device=device,
                quiet=False,
            )
            report.baseline_comparison = baseline_result
            timing["baselines"] = time.time() - start
            logger.info(f"  Baseline comparison complete ({timing['baselines']:.1f}s)")
            _flush_gpu_memory()
            if "summary" in baseline_result:
                logger.info(f"    {baseline_result['summary']}")
            _log_gpu_mem("full: after Pillar 5 (Baselines)", logger)
        except Exception as e:
            logger.error(f"Pillar 5 failed: {e}")
            raise RuntimeError(f"Pillar 5 (Baseline Comparison) failed: {e}") from e

    # ────────────────────────────────────────────────────────────────────────
    # PILLAR 4: Robustness (moderate)
    # ────────────────────────────────────────────────────────────────────────

    if "robustness" in pillars:
        logger.info(f"Computing Pillar 4: Robustness to corruptions ({corruption_variants})...")
        start = time.time()
        robustness_dict = {}

        for variant in corruption_variants:
            try:
                variant_dataloader = (
                    corruption_dataloaders.get(variant)
                    if corruption_dataloaders is not None
                    else None
                )

                pillar4_intervention = discovery_cfg.get("eval_intervention", "patching")
                if (
                    pillar4_intervention in ("mean", "mean-positional")
                    and intervention_dataloader is None
                ):
                    logger.warning(
                        f"Robustness intervention '{pillar4_intervention}' requires "
                        f"intervention_dataloader (not provided). Falling back to 'patching'."
                    )
                    pillar4_intervention = "patching"

                variant_result = Pillar4_Robustness.run(
                    model=model,
                    graph=graph,
                    original_dataloader=dataloader,
                    corruption_variant=variant,
                    corruption_dataloader=variant_dataloader,
                    metric_fn=metric_fn,
                    intervention=pillar4_intervention,
                    intervention_dataloader=intervention_dataloader,
                    device=device,
                    quiet=False,
                )
                robustness_dict[variant] = variant_result
                logger.info(f"    {variant}: {variant_result}")
            except Exception as e:
                logger.error(f"Robustness variant '{variant}' failed: {e}")
                robustness_dict[variant] = {"error": str(e)}

        report.robustness = robustness_dict
        timing["robustness"] = time.time() - start
        logger.info(f"  Robustness evaluation complete ({timing['robustness']:.1f}s)")
        _flush_gpu_memory()
        _log_gpu_mem("full: after Pillar 4 (Robustness)", logger)

    # ────────────────────────────────────────────────────────────────────────
    # PILLAR 3: Stability (expensive)
    # ────────────────────────────────────────────────────────────────────────

    if "stability" in pillars:
        _log_gpu_mem("full: before Pillar 3 (Stability)", logger)
        logger.info(f"Computing Pillar 3: Stability ({n_stability_runs} runs)...")
        start = time.time()
        try:
            stability_result = Pillar3_Stability.run(
                model=model,
                graph=graph,
                dataloader=dataloader,
                metric_fn=metric_fn,
                n_runs=n_stability_runs,
                seed_start=discovery_cfg.get("data_params", {}).get("seed", 42),
                device=device,
                quiet=False,
                task_spec=task_spec,
                discovery_cfg=discovery_cfg,
                sparsity=(pruning_cfg or {}).get("target_sparsity", 0.3),
            )
            report.stability = stability_result
            timing["stability"] = time.time() - start
            logger.info(f"  Stability evaluation complete ({timing['stability']:.1f}s)")
            if "mean_jaccard" in stability_result:
                logger.info(f"    Mean Jaccard: {stability_result['mean_jaccard']:.4f}")
                _flush_gpu_memory()
        except Exception as e:
            logger.error(f"Pillar 3 failed: {e}")
            raise RuntimeError(f"Pillar 3 (Stability) failed: {e}") from e

    # ────────────────────────────────────────────────────────────────────────
    # PILLAR 6: Generalization (expensive, optional)
    # ────────────────────────────────────────────────────────────────────────

    if "generalization" in pillars:
        if target_task_spec is None or target_dataloader is None:
            logger.warning(
                "Pillar 6 (Generalization) requested but target_task_spec or "
                "target_dataloader not provided. Skipping generalization."
            )
        else:
            logger.info("Computing Pillar 6: Generalization...")
            start = time.time()
            try:
                pillar6_intervention = discovery_cfg.get("eval_intervention", "patching")
                if (
                    pillar6_intervention in ("mean", "mean-positional")
                    and intervention_dataloader is None
                ):
                    logger.warning(
                        f"Generalization intervention '{pillar6_intervention}' requires "
                        f"intervention_dataloader (not provided). Falling back to 'patching'."
                    )
                    pillar6_intervention = "patching"
                gen_result = Pillar6_Generalization.run(
                    model=model,
                    graph=graph,
                    source_dataloader=dataloader,
                    target_dataloader=target_dataloader,
                    metric_fn=metric_fn,
                    source_task_name=discovery_cfg.get("task", "source"),
                    target_task_name=target_task_spec.name,
                    intervention=pillar6_intervention,
                    intervention_dataloader=intervention_dataloader,
                    device=device,
                    quiet=False,
                )
                report.generalization = gen_result
                timing["generalization"] = time.time() - start
                logger.info(
                    f"  Generalization evaluation complete ({timing['generalization']:.1f}s)"
                )
            except Exception as e:
                logger.error(f"Pillar 6 failed: {e}")
                raise RuntimeError(f"Pillar 6 (Generalization) failed: {e}") from e

    # ────────────────────────────────────────────────────────────────────────
    # PILLAR 7: Intervention Reliability (expensive, requires re-discovery)
    # ────────────────────────────────────────────────────────────────────────

    if "intervention_reliability" in pillars:
        logger.info(
            f"Computing Pillar 7: Intervention Reliability ({n_reliability_seeds} seeds)..."
        )
        start = time.time()
        try:
            from .pillars.intervention_reliability import run_intervention_reliability

            reliability_result = run_intervention_reliability(
                model=model,
                graph=graph,
                task_spec=task_spec,
                discovery_cfg=discovery_cfg,
                pruning_cfg=pruning_cfg or {},
                device=device,
                metric_fn=metric_fn,
                dataloader=dataloader,
                n_seeds=n_reliability_seeds,
            )
            report.intervention_reliability = reliability_result
            timing["intervention_reliability"] = time.time() - start
            idx = reliability_result.get("reliability_index")
            logger.info(
                f"  Intervention reliability complete ({timing['intervention_reliability']:.1f}s)"
                + (f" — index={idx:.4f}" if idx is not None else "")
            )
            _flush_gpu_memory()
        except Exception as e:
            logger.error(f"Pillar 7 failed: {e}")
            raise RuntimeError(f"Pillar 7 (Intervention Reliability) failed: {e}") from e

        _log_gpu_mem("full: after Pillar 7 (Intervention Reliability)", logger)

    # ────────────────────────────────────────────────────────────────────────
    # Finalize Report
    # ────────────────────────────────────────────────────────────────────────

    total_time = time.time() - total_start

    # Add metadata
    report.metadata = {
        "algorithm": discovery_cfg.get("algorithm", "unknown"),
        "model": model_name,
        "task": discovery_cfg.get("task", "unknown"),
        "level": discovery_cfg.get("level", "node"),
        "scope": discovery_cfg.get("scope", "unknown"),
        "sparsity": (pruning_cfg or {}).get(
            "target_sparsity", discovery_cfg.get("pruning", {}).get("target_sparsity", 0.0)
        ),
        "pillars_computed": pillars,
        "timestamp": time.time(),
        "total_duration_seconds": total_time,
        "per_pillar_duration_seconds": timing,
    }

    logger.info("=" * 70)
    logger.info("FAITHFULNESS EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info("Pillar timings:")
    for pillar, duration in timing.items():
        logger.info(f"  {pillar:15s}: {duration:7.1f}s")
    logger.info("=" * 70)

    return report
