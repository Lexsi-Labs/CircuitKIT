"""
Pillar 2: Faithfulness Under Ablation

Evaluates whether the discovered circuit is faithful to the model's learned
behavior by measuring performance degradation when out-of-circuit components
are ablated (removed or zeroed out).

Core Concept:
- Circuit nodes are kept, all other nodes are removed/zeroed
- Score: Degree to which circuit nodes are sufficient for the behavior
- Comparison: Circuit performance vs random removal baseline

The headline score is the normalized faithfulness ratio:

    F = (y_circuit - y_corrupt) / (y_clean - y_corrupt)

where ``y_clean`` is the metric on the full (unmodified) model, ``y_corrupt``
is the metric on the corrupt baseline, and ``y_circuit`` is the metric on the
ablated circuit. F = 1.0 means the ablated circuit fully recovers the clean
behavior; F = 0.0 means it performs no better than the corrupt baseline.

High score (near 1.0): Circuit nodes are sufficient for behavior
Low score (near 0.0): Circuit alone is insufficient; missing important components
"""

import logging
from typing import Literal, Optional

from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from ...backends.eap.graph import Graph
from ..evaluate import evaluate_baseline, evaluate_graph
from .causal_patching import _DEGENERATE_DENOM_EPS, _faithfulness_ratio, _mean_metric
from circuitkit.utils.device import get_device, empty_cache

logger = logging.getLogger(__name__)


class Pillar2_Ablation:
    """
    Pillar 2: Faithfulness Under Ablation.

    Measures whether the circuit is sufficient for generating the learned
    behavior by evaluating performance when out-of-circuit nodes are ablated
    (removed or set to zero).

    This pillar answers: "Are the circuit nodes sufficient to produce the
    behavior? What performance do we lose by removing everything else?"

    Supports multiple ablation modes:
    - 'zero': Replace out-of-circuit activations with zeros
    - 'mean': Replace with dataset mean activations
    - 'mean-positional': Replace with position-specific means
    """

    @staticmethod
    def run(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        intervention: Literal["zero", "mean", "mean-positional"] = "zero",
        intervention_dataloader: Optional[DataLoader] = None,
        device: str = "auto",
        quiet: bool = False,
    ) -> dict:
        """
        Run ablation evaluation on a circuit.

        Ablates out-of-circuit nodes and measures the resulting performance.
        The intervention method controls how ablated nodes are handled.

        The headline ``score`` is the normalized faithfulness ratio
        ``F = (y_circuit - y_corrupt) / (y_clean - y_corrupt)``, NOT the raw
        (possibly unbounded) metric. The raw ablated-circuit metric is still
        reported under ``raw_score`` for inspection.

        Args:
            model: HookedTransformer model with use_attn_result=True.
            graph: Circuit graph with in_graph flags set on edges/nodes.
            dataloader: Evaluation dataset yielding (clean, corrupted, label) batches.
            metric_fn: Metric function with signature
                (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
            intervention: How to handle ablated nodes:
                - 'zero': Replace with zeros (default)
                - 'mean': Replace with dataset mean activations
                - 'mean-positional': Replace with position-specific means
            intervention_dataloader: Dataset for computing mean activations
                (required if intervention is 'mean' or 'mean-positional').
            device: Target device ("cuda" or "cpu"). Defaults to "cuda".
            quiet: Suppress progress bar. Defaults to False.

        Returns:
            dict with keys:
            - 'score' (float): Normalized faithfulness ratio, clamped at 1.0.
                1.0 = ablated circuit fully recovers clean behavior;
                0.0 = circuit no better than the corrupt baseline.
                If the denominator is degenerate, 0.0 with the flag below set.
            - 'raw_score' (float): Mean raw metric on the ablated circuit
                (unbounded for loss-style metrics).
            - 'raw_ratio' (float): Unclamped faithfulness ratio.
            - 'clean_score' (float): Mean raw metric on the full (clean) model.
            - 'corrupt_score' (float): Mean raw metric on the corrupt baseline.
            - 'degenerate_denominator' (bool): True if |clean - corrupt| ~ 0,
                making the ratio undefined; 'score' is then a 0.0 sentinel.

        Raises:
            AssertionError: If model.cfg.use_attn_result is False.
            ValueError: If graph is None, or if mean intervention without dataloader.
        """
        if graph is None:
            raise ValueError("Graph cannot be None")

        if not hasattr(model.cfg, "use_attn_result") or not model.cfg.use_attn_result:
            raise AssertionError(
                "Model must be configured with use_attn_result=True. "
                "Configure model with: model.cfg.use_attn_result = True"
            )

        intervention = intervention.lower()
        if intervention in ["mean", "mean-positional"] and intervention_dataloader is None:
            raise ValueError(
                f"intervention={intervention!r} requires an "
                "'intervention_dataloader' to compute mean activations, "
                "but none was provided. Pass intervention_dataloader=<DataLoader>, "
                "or use intervention='zero' which needs no extra data."
            )

        if intervention not in ["zero", "mean", "mean-positional"]:
            raise ValueError(
                f"Invalid intervention: {intervention}. "
                f"Must be 'zero', 'mean', or 'mean-positional'"
            )

        logger.info(f"Pillar 2: Running ablation evaluation (intervention: {intervention})...")

        # Run evaluation with specified ablation method
        try:
            # Clean (full model) and corrupt baselines — needed to normalize the
            # raw metric into a 0-1 faithfulness ratio.
            clean_score = _mean_metric(
                evaluate_baseline(model, dataloader, metric_fn, run_corrupted=False, quiet=quiet)
            )
            corrupt_score = _mean_metric(
                evaluate_baseline(model, dataloader, metric_fn, run_corrupted=True, quiet=quiet)
            )

            circuit_scores = evaluate_graph(
                model=model,
                graph=graph,
                dataloader=dataloader,
                metrics=metric_fn,
                quiet=quiet,
                intervention=intervention,
                intervention_dataloader=intervention_dataloader,
                skip_clean=True,
            )
            raw_score = _mean_metric(circuit_scores)

            ratio = _faithfulness_ratio(raw_score, clean_score, corrupt_score)

            if ratio["degenerate_denominator"]:
                # |clean - corrupt| ~ 0: undefined ratio — surface as
                # status='invalid' (score=None), mirroring the inverted case and
                # Pillar 1, rather than a silent 0.0 sentinel.
                logger.warning(
                    "Pillar 2: clean and corrupt baselines are near-identical "
                    f"(|{clean_score:.4f} - {corrupt_score:.4f}| < {_DEGENERATE_DENOM_EPS}); "
                    "faithfulness ratio is undefined. Reporting status='invalid'."
                )
                return {
                    "score": None,
                    "raw_score": raw_score,
                    "raw_ratio": ratio["raw_ratio"],
                    "clean_score": clean_score,
                    "corrupt_score": corrupt_score,
                    "degenerate_denominator": True,
                    "inverted_denominator": False,
                    "status": "invalid",
                    "reason": (
                        "faithfulness ratio undefined: the clean and corrupt baselines are "
                        "near-identical (|clean - corrupt| ~ 0), so (circuit - corrupt)/(clean - "
                        "corrupt) has a degenerate denominator. Use a task/metric where the "
                        "corruption meaningfully separates clean from corrupt."
                    ),
                }

            if ratio["inverted_denominator"]:
                # clean < corrupt: inverted metric direction — see Pillar 1 for
                # the full rationale. status='invalid', not a clamped 0.0.
                logger.warning(
                    f"Pillar 2: clean score ({clean_score:.4f}) is BELOW the corrupt score "
                    f"({corrupt_score:.4f}) — the faithfulness metric is inverted for this "
                    "task/metric combination, so the normalized ratio is undefined. "
                    "Reporting status='invalid'. Check the metric's sign/direction "
                    "rather than the circuit."
                )
                return {
                    "score": None,
                    "raw_score": raw_score,
                    "raw_ratio": ratio["raw_ratio"],
                    "clean_score": clean_score,
                    "corrupt_score": corrupt_score,
                    "degenerate_denominator": False,
                    "inverted_denominator": True,
                    "status": "invalid",
                    "reason": (
                        "faithfulness ratio undefined: clean baseline is below the corrupt "
                        "baseline (inverted metric direction), so (circuit - corrupt)/(clean - "
                        "corrupt) flips its meaning and any clamped score would misattribute a "
                        "metric-direction problem to the circuit. Fix the metric's sign or use "
                        "a metric where clean > corrupt."
                    ),
                }

            result = {
                "score": ratio["score"],
                "raw_score": raw_score,
                "raw_ratio": ratio["raw_ratio"],
                "clean_score": clean_score,
                "corrupt_score": corrupt_score,
                "degenerate_denominator": ratio["degenerate_denominator"],
                "inverted_denominator": False,
            }

            logger.info(
                f"Pillar 2 Score (Ablation - {intervention} faithfulness ratio): "
                f"{result['score']:.4f} (raw metric: {raw_score:.4f}, "
                f"clean: {clean_score:.4f}, corrupt: {corrupt_score:.4f})"
            )
            return result

        except Exception as e:
            logger.error(f"Pillar 2 evaluation failed: {e}")
            raise

    @staticmethod
    def compare_interventions(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        intervention_dataloader: Optional[DataLoader],
        metric_fn,
        device: str = "auto",
        quiet: bool = False,
    ) -> dict:
        """
        Compare circuit performance across different ablation methods.

        Useful for understanding which ablation method is most appropriate
        for the circuit and dataset.

        Args:
            model: HookedTransformer model.
            graph: Circuit graph.
            dataloader: Evaluation dataset.
            intervention_dataloader: Dataset for mean ablations.
            metric_fn: Metric function.
            device: Target device.
            quiet: Suppress progress bar.

        Returns:
            Dict with keys for each intervention method, each mapping to the
            full result dict returned by :meth:`run` (headline normalized ratio
            under the ``score`` key):
            - 'zero': Result with zero ablation
            - 'mean': Result with mean ablation (if dataloader provided)
            - 'mean_positional': Result with positional mean ablation (if dataloader provided)
        """
        logger.info("Comparing ablation methods...")

        results = {}

        # Zero ablation
        results["zero"] = Pillar2_Ablation.run(
            model=model,
            graph=graph,
            dataloader=dataloader,
            metric_fn=metric_fn,
            intervention="zero",
            device=device,
            quiet=quiet,
        )

        # Mean ablation (if dataloader provided)
        if intervention_dataloader is not None:
            results["mean"] = Pillar2_Ablation.run(
                model=model,
                graph=graph,
                dataloader=dataloader,
                metric_fn=metric_fn,
                intervention="mean",
                intervention_dataloader=intervention_dataloader,
                device=device,
                quiet=quiet,
            )

            results["mean_positional"] = Pillar2_Ablation.run(
                model=model,
                graph=graph,
                dataloader=dataloader,
                metric_fn=metric_fn,
                intervention="mean-positional",
                intervention_dataloader=intervention_dataloader,
                device=device,
                quiet=quiet,
            )

        return results

    @staticmethod
    def compare_with_baseline(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        intervention: Literal["zero", "mean", "mean-positional"] = "zero",
        intervention_dataloader: Optional[DataLoader] = None,
        device: str = "auto",
        quiet: bool = False,
    ) -> dict:
        """
        Compare circuit ablation performance with baseline (full model).

        Args:
            model: HookedTransformer model.
            graph: Circuit graph.
            dataloader: Evaluation dataset.
            metric_fn: Metric function.
            intervention: Ablation method.
            intervention_dataloader: Dataset for mean ablations.
            device: Target device.
            quiet: Suppress progress bar.

        Returns:
            Dict with keys:
            - 'circuit_score': Circuit performance under ablation
            - 'baseline_score': Full model performance
            - 'sufficiency': Ratio (circuit_score / baseline_score), clamped
              to [0.0, 1.0]. ``None`` when the ratio is undefined (see below).
            - 'raw_sufficiency': The unclamped quotient, kept inspectable.
            When ``baseline_score <= 0`` (signed metric where the full model
            itself scores non-positive), the quotient flips or loses its
            meaning, so the dict carries ``sufficiency=None``,
            ``status='invalid'`` and a ``reason`` — the same convention as
            ``run()`` and Pillars 1/4/5/6.
        """
        logger.info(
            f"Computing Pillar 2 with baseline comparison (intervention: {intervention})..."
        )

        # Get baseline (full model) performance
        baseline_scores = evaluate_baseline(model, dataloader, metric_fn)
        if isinstance(baseline_scores, list):
            baseline_scores = baseline_scores[0]
        baseline_avg = float(baseline_scores.mean().item())

        # Get raw circuit performance under ablation
        run_result = Pillar2_Ablation.run(
            model=model,
            graph=graph,
            dataloader=dataloader,
            metric_fn=metric_fn,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            device=device,
            quiet=quiet,
        )
        circuit_score = run_result["raw_score"]

        # circuit/baseline is only interpretable when the baseline is positive
        # and non-negligible. The old `if baseline_avg > 0 else 0.0` returned a
        # 0.0 SENTINEL for a negative baseline, and the cap-above-only
        # min(x, 1.0) let a negative circuit score pass through as a negative
        # "sufficiency". Same fix as Pillar 1's compare_with_baseline.
        if baseline_avg <= _DEGENERATE_DENOM_EPS:
            logger.warning(
                f"Pillar 2 compare_with_baseline: baseline score ({baseline_avg:.4f}) is "
                "non-positive — the circuit/baseline ratio is undefined for a signed metric. "
                "Reporting status='invalid'."
            )
            return {
                "circuit_score": circuit_score,
                "baseline_score": baseline_avg,
                "sufficiency": None,
                "raw_sufficiency": None,
                "status": "invalid",
                "reason": (
                    "sufficiency ratio undefined: the full-model baseline is non-positive "
                    "(signed metric), so circuit_score / baseline_score flips or loses its "
                    "meaning. Check the metric's sign/direction rather than the circuit."
                ),
            }

        raw_sufficiency = circuit_score / baseline_avg

        return {
            "circuit_score": circuit_score,
            "baseline_score": baseline_avg,
            # Clamped both sides in the valid regime; raw quotient preserved.
            "sufficiency": max(0.0, min(raw_sufficiency, 1.0)),
            "raw_sufficiency": raw_sufficiency,
        }
