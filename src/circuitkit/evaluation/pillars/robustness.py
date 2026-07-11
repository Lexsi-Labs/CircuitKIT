"""
Pillar 4: Robustness Under Corruption Shift

Evaluates whether the discovered circuit is robust to input corruptions and
distribution shifts. Measures circuit faithfulness when evaluated on paraphrased,
entity-swapped, or other corrupted variants of the original task data.

Core Concept:
- Circuit is discovered on original data
- Circuit is then evaluated on corrupted data variants
- Score: How well the circuit maintains performance under corruption
- Comparison: Original performance vs variant performance

High score (near 1.0): Circuit is robust to corruption; minimal performance drop
Low score (near 0.0): Circuit is brittle; large performance drop under corruption

Generate-or-skip contract
-------------------------
Pillar 4 only produces a *measured* robustness number for a variant when a real
corrupted dataloader is available for it. There are two ways that happens:

1. The caller pre-supplies a ``corruption_dataloader`` for the variant.
2. Pillar 4 *generates* one itself by applying the matching corruption strategy
   from :mod:`circuitkit.corruption` to the clean prompts of the original
   dataloader.

If neither path works (e.g. ``entity_swap`` needs the optional ``spacy``
package and it is missing, the strategy errors, or the original dataloader
exposes only token tensors rather than text), the variant is reported as
``{"corruption_variant": ..., "status": "skipped", "reason": ...}`` with NO
``robustness_ratio``/``delta``. Pillar 4 NEVER falls back to evaluating the
circuit against the uncorrupted data twice — a fabricated zero-delta /
``robustness_ratio=1.0`` result that looks like a perfect PASS but tested
nothing. A skipped variant is always visibly distinguishable from a measured
one.

The same generate-or-skip semantics surface on ``FaithfulnessReport.robustness``:
it is a dict mapping each requested corruption variant to either a measured
result dict (with ``original_score``/``variant_score``/``delta``/
``robustness_ratio``) or a skip marker (``status="skipped"`` + ``reason``, no
``robustness_ratio``). Consumers must check for ``status == "skipped"`` before
treating an entry as a real score.
"""

import logging
from circuitkit.utils.device import get_device, empty_cache
from typing import Any, Dict, List, Literal, Optional, Tuple

from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from ...backends.eap.eap_utils import collate_EAP
from ...backends.eap.graph import Graph
from ..evaluate import evaluate_baseline, evaluate_graph

logger = logging.getLogger(__name__)


# Maps the user-facing corruption_variant name onto a factory that builds the
# matching corruption strategy from circuitkit.corruption. Factories are lazy
# (imports happen inside) so that an optional dependency missing for ONE
# strategy (e.g. spaCy for entity_swap) never breaks the others.
def _make_paraphrase():
    from ...corruption.paraphrase import ParaphraseCorruption

    # Surface mode is rule-based (synonym substitution): instant, deterministic
    # and dependency-free — an evaluation pillar should not silently download a
    # multi-hundred-MB LLM. When surface paraphrase cannot change a prompt
    # (e.g. diagnostic IOI prompts contain no substitutable words) the variant
    # is cleanly SKIPPED — never faked. Callers who want LLM-grade semantic
    # paraphrases pre-supply their own corruption_dataloader (see run()).
    return ParaphraseCorruption(mode="surface")


def _make_entity_swap():
    from ...corruption.entity_swap import EntitySwapCorruption

    # Constructing EntitySwapCorruption never raises even when spaCy is
    # missing; it records the failure and raises on first corrupt() call.
    return EntitySwapCorruption(entity_pool="auto")


def _make_distractor():
    from ...corruption.distractor import DistractorInjectionCorruption

    return DistractorInjectionCorruption(position="after", distractor_source="corpus")


def _make_token_swap():
    from ...corruption.token_swap import TokenSwapCorruption
    import spacy

    nlp = spacy.load("en_core_web_sm")

    def _spacy_tagger(text):
        doc = nlp(text)
        tokens = [t.text for t in doc]
        pos_tags = [t.pos_ for t in doc]
        return tokens, pos_tags

    strategy = TokenSwapCorruption(pos_tags=["NOUN", "NUM", "PROPN"])
    strategy._default_tagger = _spacy_tagger
    return strategy


def _make_role_swap():
    from ...corruption.role_swap import RoleSwapCorruption

    return RoleSwapCorruption()


def _make_negation():
    from ...corruption.negation import NegationCorruption

    return NegationCorruption()


def _make_position_shift():
    from ...corruption.position_shift import PositionShiftCorruption

    return PositionShiftCorruption(strategy="shuffle")


_STRATEGY_FACTORIES = {
    "paraphrase": _make_paraphrase,
    "entity_swap": _make_entity_swap,
    "distractor": _make_distractor,
    "format_distractor": _make_distractor,
    "token_swap": _make_token_swap,
    "role_swap": _make_role_swap,
    "logical_negation": _make_negation,
    "position_shift": _make_position_shift,
}


class CorruptionUnavailableError(RuntimeError):
    """Raised when a corrupted dataloader cannot be produced for a variant.

    Carrying this as a dedicated exception lets :meth:`Pillar4_Robustness.run`
    distinguish "corruption could not be generated, skip the variant" from a
    genuine evaluation failure, and never silently fall back to the
    uncorrupted dataloader.
    """


def _extract_text_triples(dataloader: DataLoader) -> List[Tuple[str, str, Any]]:
    """Extract (clean_text, corrupted_text, label) triples from a dataloader.

    Pillar 4 builds corrupted variants by applying a text-level corruption
    strategy to the clean prompts. That requires the original dataloader to
    carry text (the EAP/collate_EAP format yields lists of strings). Token-only
    dataloaders (e.g. ACDC / IBCircuit single-batch loaders) cannot be
    corrupted at the text level here.

    Args:
        dataloader: The original evaluation dataloader.

    Returns:
        List of (clean_text, corrupted_text, label) triples.

    Raises:
        CorruptionUnavailableError: If the dataloader does not yield text
            prompts (so no text-level corruption can be applied).
    """
    triples: List[Tuple[str, str, Any]] = []
    for batch in dataloader:
        if not (isinstance(batch, (tuple, list)) and len(batch) >= 3):
            raise CorruptionUnavailableError(
                "original dataloader does not yield (clean, corrupted, label) "
                "batches; cannot generate a text-level corrupted variant."
            )
        clean, corrupted, labels = batch[0], batch[1], batch[2]
        if not (isinstance(clean, (list, tuple)) and isinstance(corrupted, (list, tuple))):
            raise CorruptionUnavailableError(
                "original dataloader yields token tensors rather than text "
                "prompts; text-level corruption strategies cannot be applied. "
                "Pre-supply a corruption_dataloader for this variant instead."
            )
        for i in range(len(clean)):
            clean_text = clean[i]
            corrupted_text = corrupted[i]
            if not isinstance(clean_text, str) or not isinstance(corrupted_text, str):
                raise CorruptionUnavailableError(
                    "original dataloader yields non-string prompts; "
                    "text-level corruption cannot be applied."
                )
            label = labels[i] if hasattr(labels, "__getitem__") else labels
            if hasattr(label, "tolist"):
                label = label.tolist()
            triples.append((clean_text, corrupted_text, label))

    if not triples:
        raise CorruptionUnavailableError("original dataloader produced no examples to corrupt.")
    return triples


def _build_corrupted_dataloader(
    original_dataloader: DataLoader,
    corruption_variant: str,
    model=None,
) -> DataLoader:
    """Generate a corrupted-variant dataloader from the original dataloader.

    Applies the corruption strategy matching ``corruption_variant`` (from
    :mod:`circuitkit.corruption`) to the *clean* prompts of the original
    dataloader, keeping the corrupted (patch-source) prompt and label
    unchanged. The result is a DataLoader of the exact same shape as the
    original (text triples collated with ``collate_EAP``) so it is a drop-in
    replacement for ``corruption_dataloader`` in :func:`evaluate_graph`.

    Pairs where the corruption changes the token count are silently dropped
    when ``model`` is provided. EAP's positional patching requires clean and
    corrupted sequences to share positional correspondence; mid-sequence
    insertions/deletions (distractor injection, logical negation) shift every
    later position and produce unreliable attribution. Dropping such pairs is
    preferable to propagating misaligned activations.

    Args:
        original_dataloader: The original evaluation dataloader (must yield
            text prompts).
        corruption_variant: Corruption variant name (must be in
            ``_STRATEGY_FACTORIES``).
        model: Optional HookedTransformer used to check token-length parity.
            When provided, any pair whose corrupted clean prompt tokenizes to a
            different length than the original clean prompt is discarded.

    Returns:
        A DataLoader yielding (clean, corrupted, label) batches where the
        clean prompts have been corrupted.

    Raises:
        CorruptionUnavailableError: If the strategy is unknown, its optional
            dependency is missing, it errors, or it produces no usable
            corrupted examples.
    """
    import random

    factory = _STRATEGY_FACTORIES.get(corruption_variant)
    if factory is None:
        raise CorruptionUnavailableError(
            f"no corruption strategy registered for variant '{corruption_variant}'."
        )

    try:
        strategy = factory()
    except Exception as e:  # missing dependency, bad config, etc.
        raise CorruptionUnavailableError(
            f"could not construct '{corruption_variant}' corruption strategy: {e}"
        ) from e

    triples = _extract_text_triples(original_dataloader)

    rng = random.Random(42)
    corrupted_triples: List[Tuple[str, str, Any]] = []
    n_failures = 0
    n_length_mismatch = 0
    last_error: Optional[str] = None

    for clean_text, corrupted_text, label in triples:
        try:
            tagger = getattr(strategy, "_default_tagger", None)
            metadata = {"tagger": tagger} if tagger is not None else None
            result = strategy.corrupt({"prompt": clean_text}, rng=rng, metadata=metadata)
        except Exception as e:
            n_failures += 1
            last_error = str(e)
            continue

        new_clean = result.get("prompt", clean_text)
        # Reject no-op corruptions: an unchanged prompt would re-create exactly
        # the fabricated zero-delta result this pillar exists to prevent.
        if new_clean == clean_text:
            continue

        # Reject pairs where corruption changes the token count. EAP's
        # positional patching requires clean/corrupted to share per-position
        # correspondence; a mid-sequence length change (e.g. distractor
        # injection, logical negation adding "not") shifts every later token
        # and produces unreliable attribution.
        if model is not None:
            try:
                n_orig = model.to_tokens(clean_text, prepend_bos=True).shape[-1]
                n_new = model.to_tokens(new_clean, prepend_bos=True).shape[-1]
                if n_orig != n_new:
                    n_length_mismatch += 1
                    continue
            except Exception:
                pass  # tokenization error: keep the pair and let evaluate_graph surface it

        corrupted_triples.append((new_clean, corrupted_text, label))

    if not corrupted_triples:
        if n_failures and last_error is not None:
            raise CorruptionUnavailableError(
                f"'{corruption_variant}' corruption strategy failed on all "
                f"{n_failures} examples (last error: {last_error})."
            )
        if n_length_mismatch:
            raise CorruptionUnavailableError(
                f"'{corruption_variant}' corruption changed the token count on all "
                f"{n_length_mismatch} examples; EAP requires length-preserving corruptions "
                f"for positional patching. Pre-supply a corruption_dataloader for this variant."
            )
        raise CorruptionUnavailableError(
            f"'{corruption_variant}' corruption strategy produced no modified "
            f"prompts (every corrupted prompt was identical to the original)."
        )

    if n_failures:
        logger.warning(
            f"Pillar 4: '{corruption_variant}' corruption failed on "
            f"{n_failures}/{len(triples)} examples; proceeding with "
            f"{len(corrupted_triples)} corrupted examples."
        )
    if n_length_mismatch:
        logger.warning(
            f"Pillar 4: '{corruption_variant}' dropped {n_length_mismatch}/{len(triples)} "
            f"examples whose corrupted clean prompt changed token count (EAP requires "
            f"length-preserving corruptions); proceeding with {len(corrupted_triples)} examples."
        )

    batch_size = getattr(original_dataloader, "batch_size", None) or len(corrupted_triples)
    new_loader = DataLoader(
        corrupted_triples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_EAP,
    )
    # Preserve padding-side metadata if the original carried it.
    side = getattr(original_dataloader, "pair_padding_side", None)
    if side is not None:
        new_loader.pair_padding_side = side
    return new_loader


class Pillar4_Robustness:
    """
    Pillar 4: Robustness Under Corruption Shift.

    Measures whether a circuit discovered on original data remains faithful
    when evaluated on corrupted data variants (paraphrased text, entity swaps,
    distractors, etc.).

    This pillar answers: "How robust is the circuit to input corruptions?
    Does it degrade gracefully or catastrophically under distribution shift?"

    Generate-or-skip contract
    -------------------------
    For every requested corruption variant, Pillar 4 needs a real corrupted
    dataloader. It obtains one in one of two ways:

    1. The caller pre-supplies a ``corruption_dataloader`` for the variant.
    2. Pillar 4 *builds* one itself by applying the matching corruption
       strategy from :mod:`circuitkit.corruption` to the clean prompts of the
       original dataloader (see :func:`_build_corrupted_dataloader`).

    If neither path succeeds — for example ``entity_swap`` requires the
    optional ``spacy`` package and it is not installed, the strategy raises,
    or the original dataloader exposes only token tensors — the variant is
    reported as ``{"status": "skipped", "reason": ...}`` with NO
    ``robustness_ratio``/``delta`` and a clear ``logger.warning``.

    Pillar 4 **never** evaluates the circuit against the uncorrupted data twice
    as a fallback. Doing so would fabricate a ``delta=0`` /
    ``robustness_ratio=1.0`` result that reads as a perfect PASS while having
    tested nothing. A skipped variant is therefore always visibly
    distinguishable from a measured one.

    Supported corruption variants:
    - 'paraphrase': Semantic-preserving rewrites of prompts
    - 'entity_swap': Different entity, same task structure (needs spaCy)
    - 'distractor': Added irrelevant context
    - 'role_swap': Swapped participant roles
    - 'token_swap': Swapped tokens/words
    - 'logical_negation', 'format_distractor', 'position_shift': rule-based
    """

    @staticmethod
    def run(
        model: HookedTransformer,
        graph: Graph,
        original_dataloader: DataLoader,
        corruption_variant: str = "paraphrase",
        corruption_dataloader: Optional[DataLoader] = None,
        metric_fn=None,
        intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
        device: str = "auto",
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """
        Run robustness evaluation on a circuit under corruption shift.

        Evaluates circuit performance on both original and corrupted data,
        then computes the performance delta and relative drop.

        If ``corruption_dataloader`` is None, Pillar 4 generates a corrupted
        variant itself by applying the matching corruption strategy to the
        clean prompts of ``original_dataloader``. If that generation is not
        possible (missing optional dependency, strategy error, token-only
        dataloader, ...), the variant is **skipped** — a result dict with
        ``status="skipped"`` and a ``reason`` is returned, with NO
        ``robustness_ratio``. Pillar 4 never falls back to the uncorrupted
        dataloader, so it can never fabricate a zero-delta "perfectly robust"
        result.

        Args:
            model: HookedTransformer model with use_attn_result=True.
            graph: Circuit graph with in_graph flags set on edges/nodes.
            original_dataloader: Original evaluation dataset yielding
                (clean, corrupted, label) batches.
            corruption_variant: Type of corruption applied:
                - 'paraphrase': Semantic-preserving rewrites
                - 'entity_swap': Different entity with same structure
                - 'distractor': Added irrelevant context
                - 'role_swap': Swapped participant roles
                - 'token_swap': Swapped tokens/words
                Defaults to 'paraphrase'.
            corruption_dataloader: Corrupted variant dataloader. If None,
                Pillar 4 generates one from ``original_dataloader``; if that
                fails the variant is reported as skipped (never falls back to
                the uncorrupted dataloader).
            metric_fn: Metric function with signature
                (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
            intervention: Ablation method for out-of-circuit edges:
                - 'patching': Replace with corrupted activations (default)
                - 'zero': Replace with zeros
                - 'mean': Replace with dataset mean activations
                - 'mean-positional': Replace with position-specific means
            intervention_dataloader: Dataset for computing mean activations
                (required if intervention is 'mean' or 'mean-positional').
            device: Target device ("cuda" or "cpu"). Defaults to "cuda".
            quiet: Suppress progress bar. Defaults to False.

        Returns:
            On success, a dict with keys:
            - 'original_score': Circuit performance on original data
            - 'variant_score': Circuit performance on corrupted variant
            - 'delta': Absolute difference (original_score - variant_score)
            - 'relative_drop': Relative drop ((original - variant) / original)
            - 'corruption_variant': Type of corruption applied
            - 'robustness_ratio': Ratio (variant_score / original_score)
            - 'corruption_source': 'caller' or 'generated'

            When the corrupted dataloader cannot be produced, a *skip marker*:
            - 'corruption_variant': Type of corruption requested
            - 'status': 'skipped'
            - 'reason': Human-readable explanation
            (NO 'robustness_ratio' / 'delta' — the variant was not measured.)

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

        if intervention:
            intervention = intervention.lower()

        if intervention in ["mean", "mean-positional"] and intervention_dataloader is None:
            raise ValueError(
                f"intervention={intervention!r} requires an "
                "'intervention_dataloader' to compute mean activations, "
                "but none was provided. Pass intervention_dataloader=<DataLoader>, "
                "or use intervention='zero' which needs no extra data."
            )

        corruption_variant = corruption_variant.lower()
        valid_variants = [
            "paraphrase",
            "entity_swap",
            "distractor",
            "role_swap",
            "token_swap",
            "logical_negation",
            "format_distractor",
            "position_shift",
        ]
        if corruption_variant not in valid_variants:
            raise ValueError(
                f"Invalid corruption_variant: {corruption_variant}. "
                f"Must be one of {valid_variants}"
            )

        logger.info(
            f"Pillar 4: Running robustness evaluation (corruption: {corruption_variant})..."
        )

        # ── Obtain the corrupted dataloader: caller-supplied, else generated ──
        # If generation fails, the variant is SKIPPED. We never fall back to
        # the original (uncorrupted) dataloader — that would fabricate a
        # zero-delta / robustness_ratio=1.0 result that tested nothing.
        corruption_source = "caller"
        if corruption_dataloader is None:
            logger.info(
                f"No corruption_dataloader supplied for variant "
                f"'{corruption_variant}'; generating one from the original "
                f"dataloader's clean prompts."
            )
            try:
                corruption_dataloader = _build_corrupted_dataloader(
                    original_dataloader, corruption_variant, model=model
                )
                corruption_source = "generated"
                logger.info(
                    f"Pillar 4: generated corrupted dataloader for " f"'{corruption_variant}'."
                )
            except CorruptionUnavailableError as e:
                reason = f"{corruption_variant} corruption unavailable: {e}"
                logger.warning(
                    f"Pillar 4: SKIPPING variant '{corruption_variant}' — {reason}. "
                    f"No robustness score is produced for this variant (it was "
                    f"NOT evaluated against the uncorrupted data)."
                )
                return {
                    "corruption_variant": corruption_variant,
                    "status": "skipped",
                    "reason": reason,
                }

        # Evaluate circuit on original data
        try:
            logger.info("Evaluating circuit on original data...")
            original_scores = evaluate_graph(
                model=model,
                graph=graph,
                dataloader=original_dataloader,
                metrics=metric_fn,
                quiet=quiet,
                intervention=intervention,
                intervention_dataloader=intervention_dataloader,
                skip_clean=True,
            )

            # Convert to scalar if needed
            if isinstance(original_scores, list):
                original_scores = original_scores[0]

            if hasattr(original_scores, "cpu"):
                original_scores = original_scores.cpu()

            # Compute average score
            if original_scores.ndim == 0:
                score_original = float(original_scores.item())
            else:
                score_original = float(original_scores.mean().item())

            logger.info(f"Pillar 4 Original Score: {score_original:.4f}")

        except Exception as e:
            logger.error(f"Pillar 4 evaluation on original data failed: {e}")
            raise

        # Evaluate circuit on corrupted variant data
        try:
            logger.info(f"Evaluating circuit on {corruption_variant} corrupted data...")
            variant_scores = evaluate_graph(
                model=model,
                graph=graph,
                dataloader=corruption_dataloader,
                metrics=metric_fn,
                quiet=quiet,
                intervention=intervention,
                intervention_dataloader=intervention_dataloader,
                skip_clean=True,
            )

            # Convert to scalar if needed
            if isinstance(variant_scores, list):
                variant_scores = variant_scores[0]

            if hasattr(variant_scores, "cpu"):
                variant_scores = variant_scores.cpu()

            # Compute average score
            if variant_scores.ndim == 0:
                score_variant = float(variant_scores.item())
            else:
                score_variant = float(variant_scores.mean().item())

            logger.info(f"Pillar 4 Variant Score ({corruption_variant}): {score_variant:.4f}")

        except Exception as e:
            logger.error(f"Pillar 4 evaluation on {corruption_variant} data failed: {e}")
            raise

        # Compute robustness metrics
        delta = score_original - score_variant

        # robustness_ratio = variant/original is only interpretable in [0, 1]
        # for a non-negative, bounded metric. On a signed/unbounded metric
        # (e.g. logit_diff) a non-positive original or a negative variant makes
        # the ratio meaningless (negative, or > 1), so report status='invalid'
        # rather than fabricating a worst-case FAIL for an undefined case.
        if score_original <= 0 or score_variant < 0:
            logger.warning(
                f"Pillar 4 ({corruption_variant}): robustness_ratio undefined for a signed/unbounded "
                f"metric when original <= 0 or variant < 0 (original={score_original:.4f}, "
                f"variant={score_variant:.4f}). Reporting status='invalid'."
            )
            return {
                "original_score": score_original,
                "variant_score": score_variant,
                "delta": delta,
                "status": "invalid",
                "reason": (
                    "robustness_ratio undefined: signed/unbounded faithfulness metric (e.g. logit_diff) "
                    "with non-positive original or negative variant score, so variant/original does not "
                    "lie in [0, 1]. Use a bounded metric or a task where corruption degrades gracefully."
                ),
                "corruption_variant": corruption_variant,
                "corruption_source": corruption_source,
            }

        relative_drop = delta / score_original
        robustness_ratio = score_variant / score_original

        logger.info(
            f"Pillar 4 Robustness Summary (corruption: {corruption_variant}, "
            f"source: {corruption_source}): "
            f"original={score_original:.4f}, variant={score_variant:.4f}, "
            f"delta={delta:.4f}, relative_drop={relative_drop:.4f}, "
            f"robustness_ratio={robustness_ratio:.4f}"
        )

        return {
            "original_score": score_original,
            "variant_score": score_variant,
            "delta": delta,
            "relative_drop": relative_drop,
            "robustness_ratio": robustness_ratio,
            "corruption_variant": corruption_variant,
            "corruption_source": corruption_source,
        }

    @staticmethod
    def compare_corruption_variants(
        model: HookedTransformer,
        graph: Graph,
        original_dataloader: DataLoader,
        corruption_dataloaders: Dict[str, DataLoader],
        metric_fn,
        intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
        device: str = "auto",
        quiet: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare circuit robustness across multiple corruption variants.

        Useful for understanding which types of corruption affect the circuit
        most severely and how robust the circuit is to different distribution shifts.

        Args:
            model: HookedTransformer model.
            graph: Circuit graph.
            original_dataloader: Original evaluation dataset.
            corruption_dataloaders: Dict mapping corruption variant names to dataloaders.
                Example: {'paraphrase': dl1, 'entity_swap': dl2, 'distractor': dl3}
            metric_fn: Metric function.
            intervention: Ablation method.
            intervention_dataloader: Dataset for mean ablations.
            device: Target device.
            quiet: Suppress progress bar.

        Returns:
            Dict mapping corruption variant to robustness result dict (a
            measured result, or a ``status="skipped"`` marker — see
            :meth:`run`):
            {
                'paraphrase': {'original_score': ..., 'variant_score': ..., ...},
                'entity_swap': {'status': 'skipped', 'reason': ...},
                ...
            }
        """
        logger.info("Comparing circuit robustness across corruption variants...")

        results = {}

        for variant_name, variant_loader in corruption_dataloaders.items():
            logger.info(f"Evaluating robustness for {variant_name}...")
            try:
                result = Pillar4_Robustness.run(
                    model=model,
                    graph=graph,
                    original_dataloader=original_dataloader,
                    corruption_variant=variant_name,
                    corruption_dataloader=variant_loader,
                    metric_fn=metric_fn,
                    intervention=intervention,
                    intervention_dataloader=intervention_dataloader,
                    device=device,
                    quiet=quiet,
                )
                results[variant_name] = result
            except Exception as e:
                logger.warning(f"Failed to evaluate robustness for {variant_name}: {e}")
                results[variant_name] = {"error": str(e)}

        return results

    @staticmethod
    def compare_with_baseline(
        model: HookedTransformer,
        graph: Graph,
        original_dataloader: DataLoader,
        corruption_variant: str = "paraphrase",
        corruption_dataloader: Optional[DataLoader] = None,
        metric_fn=None,
        intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
        intervention_dataloader: Optional[DataLoader] = None,
        device: str = "auto",
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """
        Compare circuit robustness with baseline (full model).

        Useful for understanding circuit robustness relative to the original
        model's behavior. If the circuit is more robust than the full model,
        it suggests the circuit represents a more robust component of the model.

        If the requested variant has to be skipped (no corrupted dataloader can
        be produced — see :meth:`run`), a skip marker is returned instead of a
        baseline comparison.

        Args:
            model: HookedTransformer model.
            graph: Circuit graph.
            original_dataloader: Original evaluation dataset.
            corruption_variant: Type of corruption.
            corruption_dataloader: Corrupted variant dataloader. If None, Pillar
                4 generates one (or skips the variant — never falls back to the
                uncorrupted dataloader).
            metric_fn: Metric function.
            intervention: Ablation method.
            intervention_dataloader: Dataset for mean ablations.
            device: Target device.
            quiet: Suppress progress bar.

        Returns:
            Dict with keys:
            - 'circuit_robustness': Dict from Pillar4_Robustness.run()
            - 'baseline_original_score': Baseline score on original data
            - 'baseline_variant_score': Baseline score on corrupted data
            - 'baseline_relative_drop': Baseline relative drop
            - 'circuit_vs_baseline_relative_drop': Comparison of relative drops

            Or, if the variant is skipped, the skip marker from :meth:`run`
            (``status="skipped"``).
        """
        logger.info(
            f"Computing Pillar 4 with baseline comparison (corruption: {corruption_variant})..."
        )

        # Get circuit performance first. This also resolves the corrupted
        # dataloader (caller-supplied or generated) and decides whether the
        # variant can be measured at all.
        circuit_result = Pillar4_Robustness.run(
            model=model,
            graph=graph,
            original_dataloader=original_dataloader,
            corruption_variant=corruption_variant,
            corruption_dataloader=corruption_dataloader,
            metric_fn=metric_fn,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            device=device,
            quiet=quiet,
        )

        # If the variant was skipped (no corrupted data) or the ratio is
        # undefined (status='invalid', so no relative_drop/robustness_ratio
        # keys exist), there is nothing to compare against a baseline; surface
        # the marker unchanged rather than fabricating numbers or KeyError-ing
        # on relative_drop below.
        if circuit_result.get("status") in ("skipped", "invalid"):
            return circuit_result

        # Re-derive the corrupted dataloader for the baseline pass, using the
        # SAME generate-or-skip path so the baseline is measured on identical
        # corrupted data (never the uncorrupted dataloader).
        baseline_corruption_dataloader = corruption_dataloader
        if baseline_corruption_dataloader is None:
            baseline_corruption_dataloader = _build_corrupted_dataloader(
                original_dataloader, corruption_variant.lower(), model=model
            )

        # Get baseline (full model) performance on original data
        baseline_original_scores = evaluate_baseline(model, original_dataloader, metric_fn)
        if isinstance(baseline_original_scores, list):
            baseline_original_scores = baseline_original_scores[0]
        baseline_original = float(baseline_original_scores.mean().item())

        # Get baseline performance on corrupted data
        baseline_variant_scores = evaluate_baseline(
            model, baseline_corruption_dataloader, metric_fn
        )
        if isinstance(baseline_variant_scores, list):
            baseline_variant_scores = baseline_variant_scores[0]
        baseline_variant = float(baseline_variant_scores.mean().item())

        # Compute baseline relative drop. Mirror run()'s own guard: on a signed
        # metric, a non-positive baseline original (or negative baseline
        # variant) makes the relative drop undefined. The old
        # `if baseline_original > 0 else 0.0` returned a 0.0 SENTINEL, which
        # then FABRICATED the headline `is_circuit_more_robust` boolean
        # (circuit_drop - 0.0 > 0 -> False) from an undefined comparison.
        baseline_delta = baseline_original - baseline_variant
        if baseline_original <= 0 or baseline_variant < 0:
            logger.warning(
                f"Pillar 4 compare_with_baseline ({corruption_variant}): baseline relative "
                f"drop undefined for a signed metric (original={baseline_original:.4f}, "
                f"variant={baseline_variant:.4f}). Reporting status='invalid' for the "
                "comparison; the circuit-side result is preserved."
            )
            return {
                "circuit_robustness": circuit_result,
                "baseline_original_score": baseline_original,
                "baseline_variant_score": baseline_variant,
                "baseline_relative_drop": None,
                "circuit_vs_baseline_relative_drop_difference": None,
                "status": "invalid",
                "reason": (
                    "baseline relative drop undefined: signed/unbounded faithfulness metric "
                    "with non-positive baseline original or negative baseline variant score, "
                    "so the circuit-vs-baseline robustness comparison cannot be computed. "
                    "The circuit-side robustness result is still reported under "
                    "'circuit_robustness'."
                ),
            }

        baseline_relative_drop = baseline_delta / baseline_original

        # Compare robustness: lower relative drop is better
        circuit_vs_baseline = circuit_result["relative_drop"] - baseline_relative_drop

        return {
            "circuit_robustness": circuit_result,
            "baseline_original_score": baseline_original,
            "baseline_variant_score": baseline_variant,
            "baseline_relative_drop": baseline_relative_drop,
            "circuit_vs_baseline_relative_drop_difference": circuit_vs_baseline,
            "is_circuit_more_robust": circuit_vs_baseline < 0,  # Negative = circuit is more robust
        }
