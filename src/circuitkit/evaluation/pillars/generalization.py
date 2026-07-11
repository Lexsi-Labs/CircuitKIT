"""
Pillar 6: Cross-Dataset Generalization

Evaluates whether a circuit discovered on one task transfers to related tasks.
Measures circuit effectiveness when applied to different but semantically similar
problem domains, indicating the generality of learned mechanisms.

Core Concept:
- Circuit is discovered on source task
- Circuit is evaluated on target task (different data distribution)
- Score: How well the circuit explains behavior on the target task
- Comparison: Performance on source vs target tasks

High score (near 1.0): Circuit mechanism generalizes well across tasks
Low score (near 0.0): Circuit is task-specific; doesn't transfer
Intermediate (0.5): Circuit partially transfers; some task-specific components needed
"""

import logging
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from ...backends.eap.graph import Graph
from ..evaluate import evaluate_baseline, evaluate_graph
from .causal_patching import _DEGENERATE_DENOM_EPS, _faithfulness_ratio, _mean_metric
from circuitkit.utils.device import get_device, empty_cache

logger = logging.getLogger(__name__)

# Below this normalized source faithfulness, the transfer *ratio* F_target /
# F_source is undefined (0/0) — report status='invalid' rather than inf/nan.
# Reuses causal_patching's degenerate-denominator tolerance so both pillars'
# faithfulness sentinels agree on what counts as "effectively zero".
_DEGENERATE_FAITH_EPS = _DEGENERATE_DENOM_EPS


class Pillar6_Generalization:
    """
    Pillar 6: Cross-Dataset Generalization.

    Measures whether a circuit discovered on a source task transfers to
    related target tasks. Evaluates generalization by applying the circuit
    to different task datasets without retraining.

    This pillar answers: "Do the mechanisms in the source circuit explain
    behavior on other tasks? How general is the learned circuit?"

    Typical transfer matrix (3 tasks × 3 tasks):
    - Rows: Discovery task (IOI, SVA, GreaterThan)
    - Cols: Evaluation task
    - Values: Transfer ratio (target_score / source_score)
    """

    @staticmethod
    def run(
        model: HookedTransformer,
        graph: Graph,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        metric_fn,
        source_task_name: str = "source",
        target_task_name: str = "target",
        intervention: str = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
        renormalize: bool = False,
        device: str = "auto",
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """
        Run cross-task generalization evaluation on a circuit.

        Discovers a circuit on source task data, then evaluates it on
        target task data to measure transfer effectiveness.

        Args:
            model: HookedTransformer model with use_attn_result=True.
            graph: Circuit graph discovered on source task, with in_graph
                flags already set.
            source_dataloader: Source task evaluation dataset yielding
                (clean, corrupted, label) batches. Used to establish baseline.
            target_dataloader: Target task evaluation dataset with same format.
                Different task, different data distribution.
            metric_fn: Metric function with signature
                (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
            source_task_name: Name of source task (for logging). Defaults to "source".
            target_task_name: Name of target task (for logging). Defaults to "target".
            intervention: Ablation method for out-of-circuit edges:
                - 'patching': Replace with corrupted activations (default)
                - 'zero': Replace with zeros
                - 'mean': Replace with dataset mean activations
                - 'mean-positional': Replace with position-specific means
            intervention_dataloader: Dataset for computing mean activations
                (required if intervention is 'mean' or 'mean-positional').
            renormalize: If True, affine-normalize each task's raw metric against
                that task's own clean/corrupt baselines before forming the transfer
                ratio (see below). This is the recommended setting for raw signed
                metrics (e.g. logit-difference): it maps each score into a bounded
                [0, 1] faithfulness so the transfer ratio is well-defined even when
                the raw source score is non-positive. Adds two extra baseline passes
                per task. Defaults to False (legacy raw-ratio behaviour, which
                reports status='invalid' on a non-positive source score).
            device: Target device ("cuda" or "cpu"). Defaults to "cuda".
            quiet: Suppress progress bar. Defaults to False.

        Normalization (when renormalize=True):
            Following the normalized-faithfulness convention of Zhang & Nanda (2023,
            "Towards Best Practices of Activation Patching") and Pillar 1, each task's
            circuit score is mapped to

                F_task = (y_circuit - y_corrupt) / (y_clean - y_corrupt)  ∈ [0, 1]

            where y_clean / y_corrupt are that task's full-model / corrupt baselines.
            The bounded transfer ratio is then F_target / F_source, which measures how
            well the circuit's *recovered* faithfulness carries across tasks rather
            than ratioing two unbounded signed metrics. Miller et al. (2024,
            "Transformer Circuit Faithfulness Metrics are not Robust") motivates
            normalizing before comparing across evaluation conditions.

        Returns:
            Dict with keys:
            - 'source_score': Circuit performance on source task (raw metric).
            - 'target_score': Circuit performance on target task (raw metric).
            - 'transfer_ratio': Target/source ratio. When renormalize=True this is
                the bounded normalized ratio F_target / F_source; otherwise the raw
                target_score / source_score.
            - 'source_task': Source task name
            - 'target_task': Target task name
            - 'transfer_delta': Absolute difference (source_score - target_score)
            - 'relative_transfer_drop': Relative drop ((source - target) / source)
            When renormalize=True, additionally:
            - 'source_faithfulness' / 'target_faithfulness': F_source / F_target in [0, 1].
            - 'raw_transfer_ratio': the unbounded target_score / source_score, for reference.
            - 'source_clean_score' / 'source_corrupt_score' / 'target_clean_score' /
              'target_corrupt_score': the per-task baselines used for normalization.
            - 'normalized': True.

        Raises:
            AssertionError: If model.cfg.use_attn_result is False.
            ValueError: If graph is None or metric_fn is None.
        """
        if graph is None:
            raise ValueError("Graph cannot be None")

        if metric_fn is None:
            raise ValueError("metric_fn cannot be None")

        if not hasattr(model.cfg, "use_attn_result") or not model.cfg.use_attn_result:
            raise AssertionError(
                "Model must be configured with use_attn_result=True. "
                "Configure model with: model.cfg.use_attn_result = True"
            )

        if intervention in ["mean", "mean-positional"] and intervention_dataloader is None:
            raise ValueError(
                f"intervention={intervention!r} requires an "
                "'intervention_dataloader' to compute mean activations, "
                "but none was provided. Pass intervention_dataloader=<DataLoader>, "
                "or use intervention='zero' which needs no extra data."
            )

        logger.info(
            f"Pillar 6: Evaluating circuit transfer from '{source_task_name}' "
            f"to '{target_task_name}'..."
        )

        # Evaluate circuit on source task (baseline)
        try:
            logger.info(f"Evaluating circuit on source task: {source_task_name}...")
            source_scores = evaluate_graph(
                model=model,
                graph=graph,
                dataloader=source_dataloader,
                metrics=metric_fn,
                quiet=quiet,
                intervention=intervention,
                intervention_dataloader=intervention_dataloader,
                skip_clean=True,
            )

            # Convert to scalar if needed
            if isinstance(source_scores, list):
                source_scores = source_scores[0]

            if hasattr(source_scores, "cpu"):
                source_scores = source_scores.cpu()

            # Compute average score
            if source_scores.ndim == 0:
                score_source = float(source_scores.item())
            else:
                score_source = float(source_scores.mean().item())

            logger.info(f"Pillar 6 Source Score ({source_task_name}): {score_source:.4f}")

        except Exception as e:
            logger.error(f"Pillar 6 evaluation on source task failed: {e}")
            raise

        # Evaluate circuit on target task
        try:
            logger.info(f"Evaluating circuit on target task: {target_task_name}...")
            target_scores = evaluate_graph(
                model=model,
                graph=graph,
                dataloader=target_dataloader,
                metrics=metric_fn,
                quiet=quiet,
                intervention=intervention,
                intervention_dataloader=intervention_dataloader,
                skip_clean=True,
            )

            # Convert to scalar if needed
            if isinstance(target_scores, list):
                target_scores = target_scores[0]

            if hasattr(target_scores, "cpu"):
                target_scores = target_scores.cpu()

            # Compute average score
            if target_scores.ndim == 0:
                score_target = float(target_scores.item())
            else:
                score_target = float(target_scores.mean().item())

            logger.info(f"Pillar 6 Target Score ({target_task_name}): {score_target:.4f}")

        except Exception as e:
            logger.error(f"Pillar 6 evaluation on target task failed: {e}")
            raise

        # Compute transfer metrics
        transfer_delta = score_source - score_target

        # ── Renormalized transfer (recommended for raw signed metrics) ──────────
        # Affine-normalize each task's score against its OWN clean/corrupt
        # baselines (Zhang & Nanda 2023; same construction as Pillar 1), so the
        # transfer ratio compares two bounded [0, 1] faithfulness values instead
        # of ratioing two unbounded signed metrics. This makes the ratio
        # well-defined precisely where the raw path reports status='invalid'.
        if renormalize:
            source_clean = _mean_metric(
                evaluate_baseline(
                    model, source_dataloader, metric_fn, run_corrupted=False, quiet=quiet
                )
            )
            source_corrupt = _mean_metric(
                evaluate_baseline(
                    model, source_dataloader, metric_fn, run_corrupted=True, quiet=quiet
                )
            )
            target_clean = _mean_metric(
                evaluate_baseline(
                    model, target_dataloader, metric_fn, run_corrupted=False, quiet=quiet
                )
            )
            target_corrupt = _mean_metric(
                evaluate_baseline(
                    model, target_dataloader, metric_fn, run_corrupted=True, quiet=quiet
                )
            )

            f_source = _faithfulness_ratio(score_source, source_clean, source_corrupt)
            f_target = _faithfulness_ratio(score_target, target_clean, target_corrupt)
            faith_source = f_source["score"]
            faith_target = f_target["score"]

            raw_transfer_ratio = (score_target / score_source) if score_source != 0 else None

            base_result = {
                "source_score": score_source,
                "target_score": score_target,
                "transfer_delta": transfer_delta,
                "source_faithfulness": faith_source,
                "target_faithfulness": faith_target,
                "raw_transfer_ratio": raw_transfer_ratio,
                "source_clean_score": source_clean,
                "source_corrupt_score": source_corrupt,
                "target_clean_score": target_clean,
                "target_corrupt_score": target_corrupt,
                "normalized": True,
                "source_task": source_task_name,
                "target_task": target_task_name,
            }

            # An unusable clean/corrupt denominator on EITHER task — near-zero
            # (|clean - corrupt| ~ 0) or INVERTED (clean < corrupt, i.e. the
            # metric runs backwards on that task) — makes _faithfulness_ratio
            # return a 0.0 *sentinel* for "undefined", not a genuine
            # zero-faithfulness score. That sentinel is indistinguishable from
            # a real 0.0 once only `score` is kept, so it must be caught here —
            # via the flags, not via faith_* < eps — before it can be silently
            # divided into a transfer_ratio that looks like a valid result.
            def _bad(f):
                return f["degenerate_denominator"] or f["inverted_denominator"]

            if _bad(f_source) or _bad(f_target):
                degenerate_side = (
                    "source"
                    if _bad(f_source) and not _bad(f_target)
                    else ("target" if _bad(f_target) and not _bad(f_source) else "both")
                )
                cause = (
                    "inverted (clean < corrupt)"
                    if (f_source["inverted_denominator"] or f_target["inverted_denominator"])
                    else "near-identical"
                )
                logger.warning(
                    f"Pillar 6 ({source_task_name} -> {target_task_name}): {degenerate_side} task's "
                    f"clean/corrupt baselines are {cause}; normalized faithfulness is undefined "
                    f"there. Reporting status='invalid'."
                )
                return {
                    **base_result,
                    "transfer_ratio": None,
                    "relative_transfer_drop": None,
                    "status": "invalid",
                    "reason": (
                        f"normalized faithfulness undefined on the {degenerate_side} task: its clean/"
                        f"corrupt baseline gap is {cause} (a property of the task's metric direction "
                        f"or baselines, not of the circuit), so F_target / F_source cannot be computed."
                    ),
                }

            # F_source ~ 0 (and NOT from a degenerate denominator, excluded
            # above) means the circuit is genuinely no more faithful than the
            # corrupt baseline on its own discovery task, so a transfer
            # *ratio* is undefined (0/0). Report invalid but keep the bounded
            # per-task faithfulness values, which remain meaningful.
            if faith_source < _DEGENERATE_FAITH_EPS:
                logger.warning(
                    f"Pillar 6 ({source_task_name} -> {target_task_name}): normalized source "
                    f"faithfulness ~0 ({faith_source:.4f}); transfer_ratio undefined."
                )
                return {
                    **base_result,
                    "transfer_ratio": None,
                    "relative_transfer_drop": None,
                    "status": "invalid",
                    "reason": (
                        "normalized source faithfulness ~0: the circuit is no more faithful than "
                        "the corrupt baseline on its own task, so F_target / F_source is undefined."
                    ),
                }

            transfer_ratio = faith_target / faith_source
            relative_transfer_drop = (faith_source - faith_target) / faith_source

            logger.info(
                f"Pillar 6 Transfer (normalized, {source_task_name} -> {target_task_name}): "
                f"F_source={faith_source:.4f}, F_target={faith_target:.4f}, "
                f"transfer_ratio={transfer_ratio:.4f}"
            )
            return {
                **base_result,
                "transfer_ratio": transfer_ratio,
                "relative_transfer_drop": relative_transfer_drop,
            }

        # transfer_ratio = target/source only lies in [0, 1] (and preserves the
        # ratio = 1 - relative_drop identity) for a non-negative, bounded metric.
        # On a raw signed metric a non-positive source or negative target makes
        # the ratio meaningless, so report status='invalid' rather than
        # fabricating a 0.0.
        if score_source <= 0 or score_target < 0:
            logger.warning(
                f"Pillar 6 ({source_task_name} -> {target_task_name}): transfer_ratio undefined "
                f"(source={score_source:.4f}, target={score_target:.4f}); reporting status='invalid'."
            )
            return {
                "source_score": score_source,
                "target_score": score_target,
                "transfer_ratio": None,
                "transfer_delta": transfer_delta,
                "relative_transfer_drop": None,
                "status": "invalid",
                "reason": (
                    "transfer_ratio undefined: raw signed metric with non-positive source or negative "
                    "target score, so target/source does not lie in [0, 1] and 1 - relative_drop no "
                    "longer holds. Use a bounded metric for cross-task transfer."
                ),
                "source_task": source_task_name,
                "target_task": target_task_name,
            }

        transfer_ratio = score_target / score_source
        relative_transfer_drop = transfer_delta / score_source

        logger.info(
            f"Pillar 6 Transfer Summary ({source_task_name} -> {target_task_name}): "
            f"source={score_source:.4f}, target={score_target:.4f}, "
            f"transfer_ratio={transfer_ratio:.4f}, "
            f"relative_drop={relative_transfer_drop:.4f}"
        )

        return {
            "source_score": score_source,
            "target_score": score_target,
            "transfer_ratio": transfer_ratio,
            "transfer_delta": transfer_delta,
            "relative_transfer_drop": relative_transfer_drop,
            "source_task": source_task_name,
            "target_task": target_task_name,
        }

    @staticmethod
    def build_transfer_matrix(
        model: HookedTransformer,
        circuits: Dict[str, Graph],
        task_dataloaders: Dict[str, DataLoader],
        metric_fn,
        intervention: str = "patching",
        intervention_dataloaders: Optional[Dict[str, DataLoader]] = None,
        renormalize: bool = False,
        device: str = "auto",
        quiet: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build a transfer matrix across multiple source and target tasks.

        Evaluates how circuits discovered on each source task transfer to
        all target tasks. Useful for understanding which mechanisms are general
        and which are task-specific.

        Args:
            model: HookedTransformer model.
            circuits: Dict mapping task names to their discovered circuits.
                Example: {'ioi': graph_ioi, 'sva': graph_sva, 'gt': graph_gt}
            task_dataloaders: Dict mapping task names to evaluation dataloaders.
                Example: {'ioi': dl_ioi, 'sva': dl_sva, 'gt': dl_gt}
            metric_fn: Metric function.
            intervention: Ablation method for all evaluations.
            intervention_dataloaders: Optional dict mapping task names to
                intervention dataloaders for mean ablations.
            device: Target device.
            quiet: Suppress progress bar.

        Returns:
            Nested dict representing transfer matrix:
            {
                'ioi': {
                    'ioi': {'transfer_ratio': 1.0, ...},  # Self-transfer (baseline)
                    'sva': {'transfer_ratio': 0.78, ...},  # IOI circuit on SVA
                    'gt': {'transfer_ratio': 0.45, ...},   # IOI circuit on GT
                },
                'sva': {
                    'ioi': {'transfer_ratio': 0.82, ...},
                    'sva': {'transfer_ratio': 1.0, ...},
                    'gt': {'transfer_ratio': 0.52, ...},
                },
                ...
            }
        """
        logger.info("Building transfer matrix across tasks...")

        transfer_matrix = {}

        source_tasks = list(circuits.keys())
        target_tasks = list(task_dataloaders.keys())

        for source_task in source_tasks:
            transfer_matrix[source_task] = {}
            circuit = circuits[source_task]

            for target_task in target_tasks:
                try:
                    logger.info(f"Transfer: {source_task} -> {target_task}...")

                    # Get intervention dataloader if available
                    interv_dl = None
                    if intervention_dataloaders and target_task in intervention_dataloaders:
                        interv_dl = intervention_dataloaders[target_task]

                    result = Pillar6_Generalization.run(
                        model=model,
                        graph=circuit,
                        source_dataloader=task_dataloaders[source_task],
                        target_dataloader=task_dataloaders[target_task],
                        metric_fn=metric_fn,
                        source_task_name=source_task,
                        target_task_name=target_task,
                        intervention=intervention,
                        intervention_dataloader=interv_dl,
                        renormalize=renormalize,
                        device=device,
                        quiet=quiet,
                    )
                    transfer_matrix[source_task][target_task] = result

                except Exception as e:
                    logger.warning(
                        f"Failed to evaluate transfer from {source_task} to {target_task}: {e}"
                    )
                    transfer_matrix[source_task][target_task] = {"error": str(e)}

        return transfer_matrix

    @staticmethod
    def summarize_transfer_matrix(
        transfer_matrix: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Summarize cross-task transfer matrix with statistics.

        Computes aggregate statistics on the transfer matrix to identify:
        - Overall generalization trends
        - Which tasks are hardest/easiest to transfer to
        - Which circuits are most general
        - Average within-task vs cross-task transfer

        Args:
            transfer_matrix: Output from build_transfer_matrix().

        Returns:
            Dict with keys:
            - 'transfer_ratios_matrix': 2D matrix of transfer ratios
            - 'average_transfer_ratio': Overall average transfer ratio
            - 'within_task_transfer': Avg diagonal (self-transfer)
            - 'cross_task_transfer': Avg off-diagonal transfer
            - 'best_transfer': Tuple (source, target, ratio)
            - 'worst_transfer': Tuple (source, target, ratio)
            - 'most_general_source': Source task with highest avg transfer
            - 'hardest_target': Target task with lowest avg transfer
        """
        logger.info("Summarizing transfer matrix statistics...")

        # Extract tasks
        source_tasks = list(transfer_matrix.keys())
        target_tasks = list(transfer_matrix[source_tasks[0]].keys()) if source_tasks else []

        # Build transfer ratio matrix
        ratios_matrix = {}
        all_ratios = []
        diagonal_ratios = []
        off_diagonal_ratios = []

        for source_task in source_tasks:
            ratios_matrix[source_task] = {}
            for target_task in target_tasks:
                result = transfer_matrix[source_task][target_task]
                # Skip status='invalid' / missing results (transfer_ratio is
                # None or absent) so undefined ratios never reach sum()/mean.
                if result.get("transfer_ratio") is not None:
                    ratio = result["transfer_ratio"]
                    ratios_matrix[source_task][target_task] = ratio
                    all_ratios.append(ratio)

                    if source_task == target_task:
                        diagonal_ratios.append(ratio)
                    else:
                        off_diagonal_ratios.append(ratio)

        # Compute statistics
        avg_overall = sum(all_ratios) / len(all_ratios) if all_ratios else 0.0
        avg_within = sum(diagonal_ratios) / len(diagonal_ratios) if diagonal_ratios else 0.0
        avg_cross = (
            sum(off_diagonal_ratios) / len(off_diagonal_ratios) if off_diagonal_ratios else 0.0
        )

        # Find best and worst transfers
        best_transfer = None
        worst_transfer = None
        best_ratio = -1
        worst_ratio = 2.0

        for source_task in source_tasks:
            for target_task in target_tasks:
                if source_task != target_task:  # Skip diagonal
                    result = transfer_matrix[source_task][target_task]
                    if result.get("transfer_ratio") is not None:
                        ratio = result["transfer_ratio"]
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_transfer = (source_task, target_task, ratio)
                        if ratio < worst_ratio:
                            worst_ratio = ratio
                            worst_transfer = (source_task, target_task, ratio)

        # Find most general source and hardest target
        source_avg_transfer = {}
        target_avg_transfer = {}

        for source_task in source_tasks:
            source_ratios = [
                transfer_matrix[source_task][target]["transfer_ratio"]
                for target in target_tasks
                if transfer_matrix[source_task][target].get("transfer_ratio") is not None
                and source_task != target
            ]
            if source_ratios:
                source_avg_transfer[source_task] = sum(source_ratios) / len(source_ratios)

        for target_task in target_tasks:
            target_ratios = [
                transfer_matrix[source][target_task]["transfer_ratio"]
                for source in source_tasks
                if transfer_matrix[source][target_task].get("transfer_ratio") is not None
                and source != target_task
            ]
            if target_ratios:
                target_avg_transfer[target_task] = sum(target_ratios) / len(target_ratios)

        most_general = (
            max(source_avg_transfer.items(), key=lambda x: x[1]) if source_avg_transfer else None
        )
        hardest_target = (
            min(target_avg_transfer.items(), key=lambda x: x[1]) if target_avg_transfer else None
        )

        return {
            "transfer_ratios_matrix": ratios_matrix,
            "average_transfer_ratio": avg_overall,
            "within_task_transfer": avg_within,
            "cross_task_transfer": avg_cross,
            "best_transfer": best_transfer,
            "worst_transfer": worst_transfer,
            "most_general_source": most_general,
            "hardest_target": hardest_target,
            "source_avg_transfer": source_avg_transfer,
            "target_avg_transfer": target_avg_transfer,
        }
