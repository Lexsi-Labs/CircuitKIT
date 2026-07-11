"""
InvarianceGroupBuilder — wraps CircuitKit's existing corruption transforms
to produce typed, contracted InvarianceGroups from raw task examples.

Usage:
    from circuitkit.data.invariance_groups import InvarianceGroupBuilder, VariantType

    groups = InvarianceGroupBuilder.from_task_examples(
        examples=ioi_examples,        # List[Dict] with 'prompt' and 'answer' keys
        task="ioi",
        variant_types=[VariantType.ENTITY_SWAP, VariantType.PARAPHRASE],
        tokenizer=model.tokenizer,    # for length-delta computation
    )
    # Filter to QC-passed groups
    groups = [g for g in groups if g.passed_qc()]

Each transform comes from src/circuitkit/corruption/ — this module adds
the InvarianceContract wrapper and length-delta annotation on top.
"""

from __future__ import annotations

import logging
import random
from typing import Callable, Dict, List, Optional, Tuple

from .schema import (
    DEFAULT_CONTRACTS,
    InvarianceContract,
    InvarianceGroup,
    InvarianceVariant,
    VariantType,
    new_group_id,
)

logger = logging.getLogger(__name__)


class InvarianceGroupBuilder:
    """
    Builds InvarianceGroups from plain task examples using registered
    corruption transforms.

    The builder is stateless — all methods are classmethods.
    """

    # Registry: VariantType → callable(prompt, answer, **kwargs) → (new_prompt, new_answer)
    # Populated lazily from the corruption module on first use.
    _transforms: Dict[VariantType, Callable] = {}

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    @classmethod
    def from_task_examples(
        cls,
        examples: List[Dict],
        task: str,
        variant_types: Optional[List[VariantType]] = None,
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        tokenizer=None,
        contract_overrides: Optional[Dict[VariantType, InvarianceContract]] = None,
        transform_kwargs: Optional[Dict[VariantType, Dict]] = None,
    ) -> List[InvarianceGroup]:
        """
        Convert a list of raw task examples into InvarianceGroups.

        Args:
            examples: List of dicts with at least prompt_key and answer_key.
            task: Task name string (e.g. 'ioi', 'greater_than').
            variant_types: Which transforms to apply. Defaults to
                [ENTITY_SWAP, PARAPHRASE] if None.
            prompt_key: Key for the input prompt in each example dict.
            answer_key: Key for the expected answer in each example dict.
            tokenizer: Optional tokenizer for length-delta computation.
                If None, length_delta is set to -1 (unknown).
            contract_overrides: Per-VariantType InvarianceContract overrides.
                Merged with DEFAULT_CONTRACTS.
            transform_kwargs: Extra keyword arguments forwarded to each
                transform function, keyed by VariantType.

        Returns:
            List of InvarianceGroups, one per input example. Groups whose
            variants all fail QC are retained but marked qc_passed=False.
        """
        if variant_types is None:
            variant_types = [VariantType.ENTITY_SWAP, VariantType.PARAPHRASE]

        contracts = {**DEFAULT_CONTRACTS, **(contract_overrides or {})}
        kwargs_map = transform_kwargs or {}

        cls._register_default_transforms()
        groups: List[InvarianceGroup] = []

        for idx, example in enumerate(examples):
            base_prompt = example.get(prompt_key, "")
            base_answer = example.get(answer_key, "")

            if not base_prompt:
                logger.debug(f"Skipping example {idx}: empty prompt")
                continue

            group_id = new_group_id(task, idx)
            variants: List[InvarianceVariant] = []

            for vtype in variant_types:
                transform_fn = cls._transforms.get(vtype)
                if transform_fn is None:
                    logger.warning(f"No transform registered for {vtype.value}, skipping")
                    continue

                try:
                    extra = kwargs_map.get(vtype, {})
                    new_prompt, new_answer = transform_fn(base_prompt, base_answer, **extra)
                except Exception as exc:
                    logger.debug(f"Transform {vtype.value} failed for example {idx}: {exc}")
                    # Record failed variant so QC can catch it
                    variants.append(
                        InvarianceVariant(
                            variant_type=vtype,
                            prompt=base_prompt,
                            answer=base_answer,
                            contract=contracts[vtype],
                            qc_passed=False,
                            transformation_params={"error": str(exc)},
                        )
                    )
                    continue

                contract = contracts[vtype]
                length_delta = cls._compute_length_delta(base_prompt, new_prompt, tokenizer)
                # QC: if contract requires length matching but delta != 0, fail
                # length_delta == -1 means unknown (no tokenizer) — skip the check
                qc_passed = True
                if contract.length_matched and length_delta not in (0, -1):
                    qc_passed = False
                # QC: if label_invariant, check answer didn't change
                if contract.label_invariant and new_answer != base_answer:
                    qc_passed = False

                variants.append(
                    InvarianceVariant(
                        variant_type=vtype,
                        prompt=new_prompt,
                        answer=new_answer,
                        contract=contract,
                        qc_passed=qc_passed,
                        length_delta=length_delta,
                        transformation_params=extra,
                    )
                )

            groups.append(
                InvarianceGroup(
                    group_id=group_id,
                    task=task,
                    base_prompt=base_prompt,
                    base_answer=base_answer,
                    variants=variants,
                    metadata={
                        k: v for k, v in example.items() if k not in (prompt_key, answer_key)
                    },
                )
            )

        passed = sum(1 for g in groups if g.passed_qc())
        logger.info(
            f"Built {len(groups)} groups for task={task!r}; "
            f"{passed} pass QC ({100*passed//max(len(groups), 1)}%)"
        )
        return groups

    # ------------------------------------------------------------------
    # Transform registration
    # ------------------------------------------------------------------

    @classmethod
    def register_transform(
        cls,
        vtype: VariantType,
        fn: Callable[[str, str], Tuple[str, str]],
    ) -> None:
        """
        Register a custom transform function for a VariantType.

        fn signature: (prompt: str, answer: str, **kwargs) -> (new_prompt, new_answer)
        """
        cls._transforms[vtype] = fn

    @classmethod
    def _register_default_transforms(cls) -> None:
        """Lazy-register default transforms from the corruption module."""
        if cls._transforms:
            return  # already registered

        # Entity swap — replace named entities with alternatives from a fixed list
        try:
            from ...corruption.entity_swap import EntitySwapCorruption

            cls._transforms[VariantType.ENTITY_SWAP] = cls._make_corruption_adapter(
                EntitySwapCorruption()
            )
        except ImportError:
            logger.debug("entity_swap not available")

        # Role swap
        try:
            from ...corruption.role_swap import RoleSwapCorruption

            cls._transforms[VariantType.ROLE_SWAP] = cls._make_corruption_adapter(
                RoleSwapCorruption()
            )
        except ImportError:
            logger.debug("role_swap not available")

        # Token swap
        try:
            from ...corruption.token_swap import TokenSwapCorruption

            cls._transforms[VariantType.TOKEN_SWAP] = cls._make_corruption_adapter(
                TokenSwapCorruption()
            )
        except ImportError:
            logger.debug("token_swap not available")

        # Distractor insertion
        try:
            from ...corruption.distractor import DistractorInjectionCorruption

            dist = DistractorInjectionCorruption()
            cls._transforms[VariantType.DISTRACTOR] = cls._make_corruption_adapter(dist)
        except ImportError:
            logger.debug("distractor not available")

        # Negation
        try:
            from ...corruption.negation import NegationCorruption

            cls._transforms[VariantType.NEGATION] = cls._make_corruption_adapter(
                NegationCorruption()
            )
        except ImportError:
            logger.debug("negation not available")

        # Voice swap
        try:
            from ...corruption.voice_swap import VoiceSwapCorruption

            cls._transforms[VariantType.VOICE_SWAP] = cls._make_corruption_adapter(
                VoiceSwapCorruption()
            )
        except ImportError:
            logger.debug("voice_swap not available")

        # Paraphrase — no default (requires an LLM or a cached paraphrase set);
        # callers must register their own paraphrase transform.
        logger.debug(
            "Paraphrase transform not auto-registered — use "
            "InvarianceGroupBuilder.register_transform(VariantType.PARAPHRASE, fn)"
        )

    # ------------------------------------------------------------------
    # Adapter closure — maps corruption class API (dict in/out) → (prompt, answer) API
    # ------------------------------------------------------------------

    @staticmethod
    def _make_corruption_adapter(corruption_obj):
        """
        Generic adapter for corruption classes whose .corrupt() signature is:
            corrupt(example: Dict, *, rng: random.Random, metadata=None) -> Dict

        The 'example' dict must have 'prompt' and 'answer' keys; the returned
        dict is expected to have the same keys (possibly modified).
        """

        def _fn(prompt: str, answer: str, **kw) -> Tuple[str, str]:
            seed = kw.pop("seed", None)
            rng = kw.pop("rng", random.Random(seed))
            example = {"prompt": prompt, "answer": answer}
            result = corruption_obj.corrupt(example, rng=rng, **kw)
            return result.get("prompt", prompt), result.get("answer", answer)

        return _fn

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_length_delta(
        base_prompt: str,
        new_prompt: str,
        tokenizer,
    ) -> int:
        """Token count of new_prompt minus token count of base_prompt."""
        if tokenizer is None:
            return -1  # unknown
        try:
            base_len = len(tokenizer.encode(base_prompt))
            new_len = len(tokenizer.encode(new_prompt))
            return new_len - base_len
        except Exception:
            return -1


def register_paraphrase_transform(
    fn: Callable[[str, str], Tuple[str, str]],
) -> None:
    """
    Convenience wrapper to register a paraphrase transform globally.

    fn: (prompt, answer) -> (paraphrased_prompt, answer)
    Typically wraps a cached LLM paraphrase function or a local model.

    Example:
        def my_paraphrase(prompt, answer, **kw):
            return cached_paraphrases[prompt], answer

        register_paraphrase_transform(my_paraphrase)
    """
    InvarianceGroupBuilder.register_transform(VariantType.PARAPHRASE, fn)
