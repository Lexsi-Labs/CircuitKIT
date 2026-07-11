"""Strategy protocol and registry.

A ``CorruptionStrategy`` consumes a ``ContrastiveRecord`` (normalized by
an Adapter) and produces a corrupted partner: ``(corrupt_prompt,
corrupt_answer)``. The strategy may also leave the record untouched if
it is already natively paired (``contrast_source == NATIVE_PAIR``).

Implementations live in ``data/corruption/<strategy>.py`` and self-register
via ``@register_strategy("strategy-name")``.

Length contracts
----------------
Every strategy declares whether its corrupt output preserves token-count
relative to the clean input. This is critical: gradient-based circuit
discovery (EAP / EAP-IG) requires aligned activation shapes between
clean and corrupt forward passes, so a strategy that breaks length
silently produces noisy attribution. The ``length_contract`` attribute
is checked by ``data.worthiness`` before discovery runs.
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from ..normalized import ContrastiveRecord, ContrastSource, NormalizedDataset


class LengthContract(str, Enum):
    """How the corrupt prompt's token count relates to the clean prompt's."""

    PRESERVE = "preserve"  # entity_swap, token_swap, role_swap, position_shift
    EXTEND = "extend"  # distractor (insertion at end-of-context)
    SHRINK = "shrink"  # rare; e.g. compaction strategies
    UNKNOWN = "unknown"  # paraphrase, llm_counterfactual — depends on output
    NATIVE = "native"  # dataset-supplied; no transformation


@dataclass
class CorruptionResult:
    """Outcome of applying a strategy to one record.

    Attributes:
        corrupt_prompt: generated counter-factual prompt
        corrupt_answer: generated counter-factual answer token
        notes:          optional debug info ("dropped due to length mismatch", etc.)
        succeeded:      False if the strategy could not produce a valid pair
    """

    corrupt_prompt: Optional[str]
    corrupt_answer: Optional[str]
    notes: str = ""
    succeeded: bool = True


class CorruptionStrategy(abc.ABC):
    """Generate the corrupt half of a contrastive pair.

    Subclasses MUST override:
      - ``name``                — short string used in YAML / CLI / config
      - ``length_contract``     — LengthContract for this strategy
      - ``corrupt(record)``     — produce CorruptionResult for one record

    Subclasses MAY override:
      - ``description``         — human-readable summary
      - ``fits(record)``        — quick check whether this strategy is
                                  applicable to a given record (used by
                                  auto-detect; default returns True)
    """

    name: str = ""
    description: str = ""
    length_contract: LengthContract = LengthContract.UNKNOWN

    def fits(self, record: ContrastiveRecord) -> bool:
        """Default: strategy applies to anything not natively paired."""
        return record.contrast_source != ContrastSource.NATIVE_PAIR

    @abc.abstractmethod
    def corrupt(self, record: ContrastiveRecord, **kwargs: Any) -> CorruptionResult:
        """Produce the corrupt half of the pair for one record."""

    def corrupt_example(
        self,
        example: Dict[str, Any],
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """Bridge: converts GenericTaskSpec dict ↔ ContrastiveRecord ↔ dict.

        Allows new-style strategies to be used with GenericTaskSpec's
        ``_apply_corruptions()`` pipeline. The returned dict preserves
        all original keys, with ``prompt`` and ``answer`` replaced by
        their corrupted counterparts.
        """
        rng = rng or random.Random()

        answer = example.get("answer") or (
            example.get("answers", [""])[0] if example.get("answers") else ""
        )

        meta: Dict[str, Any] = {}
        if "choices" in example:
            meta["choices"] = example["choices"]
        if "correct_choice_idx" in example:
            meta["correct_idx"] = example["correct_choice_idx"]

        record = ContrastiveRecord(
            record_id=str(hash(str(example))),
            clean_prompt=str(example.get("prompt", "")),
            clean_answer=str(answer),
            corrupt_prompt=None,
            corrupt_answer=None,
            target_field="answer",
            contrast_source=ContrastSource.GENERATED,
            meta=meta,
        )

        result = self.apply(record, rng=rng)

        corrupted = dict(example)
        if result.corrupt_prompt is not None:
            corrupted["prompt"] = result.corrupt_prompt
        corrupted["answer"] = (
            result.corrupt_answer if result.corrupt_answer is not None else str(answer)
        )
        return corrupted

    def apply(
        self,
        record: ContrastiveRecord,
        **kwargs: Any,
    ) -> ContrastiveRecord:
        """Convenience: run ``corrupt`` and return an updated record copy.

        If the record already has a native pair, returns it unchanged.
        If the strategy fails, returns the original record with
        ``contrast_source`` unchanged but ``meta['_strategy_error']`` set.
        """
        if record.is_paired and record.contrast_source == ContrastSource.NATIVE_PAIR:
            return record
        result = self.corrupt(record, **kwargs)
        if not result.succeeded:
            new_meta = {**record.meta, "_strategy_error": result.notes, "_strategy_name": self.name}
            return ContrastiveRecord(
                record_id=record.record_id,
                clean_prompt=record.clean_prompt,
                clean_answer=record.clean_answer,
                corrupt_prompt=record.corrupt_prompt,
                corrupt_answer=record.corrupt_answer,
                target_field=record.target_field,
                contrast_source=record.contrast_source,
                meta=new_meta,
                spans=record.spans,
            )
        return ContrastiveRecord(
            record_id=record.record_id,
            clean_prompt=record.clean_prompt,
            clean_answer=record.clean_answer,
            corrupt_prompt=result.corrupt_prompt,
            corrupt_answer=result.corrupt_answer,
            target_field=record.target_field,
            contrast_source=ContrastSource.GENERATED,
            meta={
                **record.meta,
                "_strategy_name": self.name,
                **({"_strategy_notes": result.notes} if result.notes else {}),
            },
            spans=record.spans,
        )
        
    def apply_to_dataset(
        self,
        ds: "NormalizedDataset",
        **kwargs: Any,
    ) -> "NormalizedDataset":
        """Apply this strategy to every record in a dataset.

        Subclasses that need dataset-level context (e.g. pool access for
        Resample or TemplateStrategy in auto_peer mode) should override this.
        The default applies ``self.apply(record, **kwargs)`` to each record.
        """

        new_records = [self.apply(r, **kwargs) for r in ds.records]
        return NormalizedDataset(
            name=ds.name,
            shape=ds.shape,
            records=new_records,
            source=ds.source,
            schema_version=ds.schema_version,
            meta={**ds.meta, "_corruption": self.name},
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, Type[CorruptionStrategy]] = {}


def register_strategy(name: str):
    """Decorator: register a strategy class under ``name``."""

    def deco(cls: Type[CorruptionStrategy]) -> Type[CorruptionStrategy]:
        if not issubclass(cls, CorruptionStrategy):
            raise TypeError(f"{cls!r} is not a CorruptionStrategy subclass")
        cls.name = name
        STRATEGY_REGISTRY[name] = cls
        return cls

    return deco


def get_strategy(name: str) -> Type[CorruptionStrategy]:
    """Look up a strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"No corruption strategy registered as {name!r}. "
            f"Registered: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]


def list_strategies() -> List[str]:
    return sorted(STRATEGY_REGISTRY)


__all__ = [
    "CorruptionStrategy",
    "CorruptionResult",
    "LengthContract",
    "STRATEGY_REGISTRY",
    "register_strategy",
    "get_strategy",
    "list_strategies",
]
