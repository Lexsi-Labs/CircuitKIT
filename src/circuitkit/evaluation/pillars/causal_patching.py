"""
Pillar 1: Causal Patching Validity

Evaluates whether the discovered circuit explains model behavior through
causal intervention. Uses interchange intervention (patching) to measure
how well the circuit preserves model performance.

Core Concept:
- Clean run: Model processes clean input normally
- Circuit ablation: All out-of-circuit edges are replaced with corrupted activations
- Score: How well the circuit predicts the model's behavior

The headline score is the normalized faithfulness ratio:

    F = (y_circuit - y_corrupt) / (y_clean - y_corrupt)

where ``y_clean`` is the metric on the full (unmodified) model, ``y_corrupt``
is the metric on the corrupt baseline, and ``y_circuit`` is the metric on the
patched circuit. F = 1.0 means the circuit fully recovers the clean behavior;
F = 0.0 means it performs no better than the corrupt baseline.

High score (near 1.0): Circuit captures the causal mechanisms driving behavior
Low score (near 0.0): Circuit is not faithful to actual model computations
"""

import logging

from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from ...backends.eap.graph import Graph
from ..evaluate import evaluate_baseline, evaluate_graph
from circuitkit.utils.device import get_device, empty_cache

logger = logging.getLogger(__name__)

# Below this absolute clean-vs-corrupt gap, the faithfulness ratio is
# undefined (division by ~zero) — we report a sentinel instead of inf/nan.
_DEGENERATE_DENOM_EPS = 1e-6


def _mean_metric(scores) -> float:
    """Reduce an evaluate_* return value (tensor / list of tensors) to a float mean."""
    if isinstance(scores, list):
        scores = scores[0]
    if hasattr(scores, "cpu"):
        scores = scores.cpu()
    if hasattr(scores, "ndim"):
        return float(scores.item()) if scores.ndim == 0 else float(scores.mean().item())
    return float(scores)


def _faithfulness_ratio(circuit: float, clean: float, corrupt: float) -> dict:
    """
    Compute the normalized faithfulness ratio F = (circuit - corrupt) / (clean - corrupt).

    Guards two undefined-denominator regimes rather than silently producing a
    misleading number:

    * near-zero (|clean - corrupt| ~ 0): the ratio is undefined — score 0.0
      with ``degenerate_denominator`` set; callers should surface
      status="invalid" (as Pillars 1/2 do), not a score.
    * inverted (clean < corrupt): the metric runs in the wrong direction for
      this normalization (e.g. a signed logit-diff where the "clean" input
      scores BELOW the corrupt baseline, as on WMDP with an inverted metric
      sign). Any ratio would flip its meaning — a circuit *better* than
      corrupt maps to a *negative* ratio — and the old ``max(0.0, ...)``
      clamp silently squashed that to 0.0, misreporting "no faithfulness"
      when the real issue is metric direction. Sentinel score 0.0 with
      ``inverted_denominator`` set; callers should surface status="invalid"
      (as Pillars 4/5/6 do for the analogous cases), not a score.

    The ratio is clamped to [0.0, 1.0] only in the well-defined regime
    (denominator positive and non-negligible), consistent with
    ``compare_with_baseline``'s existing ``min(..., 1.0)`` behaviour.

    Returns:
        Dict with keys: 'score' (clamped ratio in [0.0, 1.0], or a 0.0
        sentinel when either flag is set), 'raw_ratio' (unclamped, full
        signed signal; still computed for the inverted case so the true
        magnitude stays inspectable), 'degenerate_denominator' (bool),
        'inverted_denominator' (bool).
    """
    denom = clean - corrupt
    if abs(denom) < _DEGENERATE_DENOM_EPS:
        return {
            "score": 0.0,
            "raw_ratio": 0.0,
            "degenerate_denominator": True,
            "inverted_denominator": False,
        }
    raw_ratio = (circuit - corrupt) / denom
    if denom < 0:
        return {
            "score": 0.0,
            "raw_ratio": raw_ratio,
            "degenerate_denominator": False,
            "inverted_denominator": True,
        }
    return {
        "score": max(0.0, min(raw_ratio, 1.0)),
        "raw_ratio": raw_ratio,
        "degenerate_denominator": False,
        "inverted_denominator": False,
    }


class Pillar1_CausalPatching:
    """
    Pillar 1: Causal Patching Faithfulness.

    Measures whether the circuit explains model behavior using interchange
    intervention (patching). The circuit is considered faithful if:
    - Ablating out-of-circuit edges minimally affects model performance
    - Clean activations through circuit edges are sufficient for behavior

    This pillar answers: "Does the circuit contain the mechanisms the model
    actually uses to produce this behavior?"
    """

    @staticmethod
    def run(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        device: str = "auto",
        quiet: bool = False,
    ) -> dict:
        """
        Run causal patching evaluation on a circuit.

        Uses interchange intervention: out-of-circuit edges are replaced with
        corrupted activations, simulating what would happen if those edges
        didn't carry the necessary information.

        The headline ``score`` is the normalized faithfulness ratio
        ``F = (y_circuit - y_corrupt) / (y_clean - y_corrupt)``, NOT the raw
        (possibly unbounded) metric. The raw circuit metric is still reported
        under ``raw_score`` for inspection.

        Args:
            model: HookedTransformer model with use_attn_result=True.
            graph: Circuit graph with in_graph flags set on edges/nodes.
            dataloader: Evaluation dataset yielding (clean, corrupted, label) batches.
            metric_fn: Metric function with signature
                (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
            device: Target device ("cuda" or "cpu"). Defaults to "cuda".
            quiet: Suppress progress bar. Defaults to False.

        Returns:
            dict with keys:
            - 'score' (float): Normalized faithfulness ratio, clamped to [0.0, 1.0].
                1.0 = circuit fully recovers clean behavior;
                0.0 = circuit no better than the corrupt baseline.
                The full signed signal remains available in 'raw_ratio'.
                If the denominator is degenerate, 0.0 with the flag below set.
            - 'raw_score' (float): Mean raw metric on the patched circuit
                (unbounded for loss-style metrics).
            - 'raw_ratio' (float): Unclamped faithfulness ratio.
            - 'clean_score' (float): Mean raw metric on the full (clean) model.
            - 'corrupt_score' (float): Mean raw metric on the corrupt baseline.
            - 'degenerate_denominator' (bool): True if |clean - corrupt| ~ 0,
                making the ratio undefined; 'score' is then a 0.0 sentinel.

        Raises:
            AssertionError: If model.cfg.use_attn_result is False.
            ValueError: If graph is None or invalid.
        """
        if graph is None:
            raise ValueError("Graph cannot be None")

        if not hasattr(model.cfg, "use_attn_result") or not model.cfg.use_attn_result:
            raise AssertionError(
                "Model must be configured with use_attn_result=True. "
                "Configure model with: model.cfg.use_attn_result = True"
            )

        logger.info("Pillar 1: Running causal patching evaluation...")

        # Run evaluation with patching intervention
        try:
            # Clean (full model) and corrupt baselines from a single evaluate_baseline
            # path each — needed to normalize the raw metric into a 0-1 ratio.
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
                intervention="patching",
                skip_clean=True,
            )
            raw_score = _mean_metric(circuit_scores)

            ratio = _faithfulness_ratio(raw_score, clean_score, corrupt_score)

            if ratio["degenerate_denominator"]:
                # |clean - corrupt| ~ 0: the ratio is undefined. This is
                # semantically the same as the inverted case below — an
                # undefined faithfulness, not a genuine 0.0 — so surface it as
                # status='invalid' (score=None) rather than a silent 0.0 sentinel
                # that a caller could mistake for "no faithfulness".
                logger.warning(
                    "Pillar 1: clean and corrupt baselines are near-identical "
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
                # clean < corrupt: the metric runs backwards for this task, so
                # a clamped 0.0 would misreport "no faithfulness" when the real
                # issue is metric direction. Report status='invalid' (matching
                # Pillars 4/5/6's convention) and keep the raw values inspectable.
                logger.warning(
                    f"Pillar 1: clean score ({clean_score:.4f}) is BELOW the corrupt score "
                    f"({corrupt_score:.4f}) — the faithfulness metric is inverted for this "
                    "task/metric combination, so the normalized ratio is undefined. "
                    "Reporting status='invalid'. Check the metric's sign/direction "
                    "(e.g. loss- vs reward-style logit_diff) rather than the circuit."
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
                f"Pillar 1 Score (Causal Patching faithfulness ratio): {result['score']:.4f} "
                f"(raw metric: {raw_score:.4f}, clean: {clean_score:.4f}, "
                f"corrupt: {corrupt_score:.4f})"
            )
            return result

        except Exception as e:
            logger.error(f"Pillar 1 evaluation failed: {e}")
            raise

    @staticmethod
    def compare_with_baseline(
        model: HookedTransformer,
        graph: Graph,
        dataloader: DataLoader,
        metric_fn,
        device: str = "auto",
        quiet: bool = False,
    ) -> dict:
        """
        Compare circuit performance with baseline (full model).

        Useful for understanding the circuit's performance relative to the
        original model's behavior.

        Args:
            model: HookedTransformer model.
            graph: Circuit graph.
            dataloader: Evaluation dataset.
            metric_fn: Metric function.
            device: Target device.
            quiet: Suppress progress bar.

        Returns:
            Dict with keys:
            - 'circuit_score': Raw circuit metric (patched).
            - 'baseline_score': Full model (clean) metric.
            - 'faithfulness': Ratio (circuit_score / baseline_score), clamped
              to [0.0, 1.0]. ``None`` when the ratio is undefined (see below).
            - 'raw_faithfulness': The unclamped quotient, kept inspectable.
            When ``baseline_score <= 0`` (a signed metric where the full model
            itself scores non-positive), the quotient flips or loses its
            meaning, so instead of a fabricated number the dict carries
            ``faithfulness=None``, ``status='invalid'`` and a ``reason`` —
            the same convention as ``run()`` and Pillars 4/5/6.
        """
        logger.info("Computing Pillar 1 with baseline comparison...")

        # Get baseline (full model) performance
        baseline_scores = evaluate_baseline(model, dataloader, metric_fn)
        if isinstance(baseline_scores, list):
            baseline_scores = baseline_scores[0]
        baseline_avg = float(baseline_scores.mean().item())

        # Get raw circuit performance (patched)
        run_result = Pillar1_CausalPatching.run(
            model=model,
            graph=graph,
            dataloader=dataloader,
            metric_fn=metric_fn,
            device=device,
            quiet=quiet,
        )
        circuit_score = run_result["raw_score"]

        # circuit/baseline is only interpretable when the baseline is positive
        # and non-negligible. The old `if baseline_avg > 0 else 0.0` returned a
        # 0.0 SENTINEL for a negative baseline (reads as "completely
        # unfaithful"), and the cap-above-only `min(x, 1.0)` let a negative
        # circuit score pass through as a negative "faithfulness".
        if baseline_avg <= _DEGENERATE_DENOM_EPS:
            logger.warning(
                f"Pillar 1 compare_with_baseline: baseline score ({baseline_avg:.4f}) is "
                "non-positive — the circuit/baseline ratio is undefined for a signed metric. "
                "Reporting status='invalid'."
            )
            return {
                "circuit_score": circuit_score,
                "baseline_score": baseline_avg,
                "faithfulness": None,
                "raw_faithfulness": None,
                "status": "invalid",
                "reason": (
                    "faithfulness ratio undefined: the full-model baseline is non-positive "
                    "(signed metric), so circuit_score / baseline_score flips or loses its "
                    "meaning. Check the metric's sign/direction rather than the circuit."
                ),
            }

        raw_faithfulness = circuit_score / baseline_avg
        return {
            "circuit_score": circuit_score,
            "baseline_score": baseline_avg,
            # Clamped both sides in the valid regime; the raw quotient stays
            # available so a below-zero circuit score is not silently hidden.
            "faithfulness": max(0.0, min(raw_faithfulness, 1.0)),
            "raw_faithfulness": raw_faithfulness,
        }
