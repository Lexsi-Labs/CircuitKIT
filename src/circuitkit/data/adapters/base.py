"""Adapter protocol and registry.

A ``DataAdapter`` converts a raw dataset of a known *shape* into a
``NormalizedDataset[ContrastiveRecord]``. The adapter knows nothing
about model tokenisers, attribution algorithms, or corruption.

Implementations live in ``data/adapters/<shape>.py`` and self-register
via ``@register_adapter(DatasetShape.<X>)`` on the class.

Detecting the shape (when not user-specified) is the job of
``data.auto_detect`` which sniffs HF dataset features and picks the
right adapter.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Type

from ..normalized import DatasetShape, NormalizedDataset


class DataAdapter(abc.ABC):
    """Convert a raw dataset of a known shape into ContrastiveRecords.

    Implementations:
      1. Override ``shape`` (class attr) — DatasetShape this adapter handles.
      2. Override ``fits(raw)`` — quick boolean: does this adapter
         understand the given raw dataset's columns / structure? Used by
         the auto-detector.
      3. Override ``adapt(raw, max_records=None, **kwargs)`` — the actual
         normalization. Returns a NormalizedDataset.
    """

    #: DatasetShape this adapter handles. Subclasses MUST override.
    shape: DatasetShape = DatasetShape.UNKNOWN

    #: Human-readable description (used in CLI help).
    description: str = ""

    @classmethod
    @abc.abstractmethod
    def fits(cls, raw: Any) -> bool:
        """Quick fit-check: does this adapter understand ``raw``?

        Used by auto-detect. Implementations should be cheap (look at
        column names / first row) and never raise.
        """

    @abc.abstractmethod
    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs: Any,
    ) -> NormalizedDataset:
        """Normalize ``raw`` into a ``NormalizedDataset``.

        Args:
            raw:         a HuggingFace ``datasets.Dataset``, ``pandas.DataFrame``,
                         list of dicts, or any iterable the subclass accepts.
            max_records: optional cap on record count.
            name:        explicit dataset name (defaults to subclass-detected).
            source:      provenance string (e.g. ``"hf://glue/sst2"``).
            **kwargs:    adapter-specific options (e.g. column overrides).

        Returns:
            NormalizedDataset whose ``records`` are ContrastiveRecords. If
            the dataset has native pairs (e.g. CrowS-Pairs ``sent_more`` /
            ``sent_less``) the corrupt half is filled in; otherwise it is
            ``None`` and a CorruptionStrategy must be applied later.
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: Dict[DatasetShape, Type[DataAdapter]] = {}


def register_adapter(shape: DatasetShape):
    """Decorator: register an adapter class as the handler for ``shape``."""

    def deco(cls: Type[DataAdapter]) -> Type[DataAdapter]:
        if not issubclass(cls, DataAdapter):
            raise TypeError(f"{cls!r} is not a DataAdapter subclass")
        cls.shape = shape
        ADAPTER_REGISTRY[shape] = cls
        return cls

    return deco


def get_adapter(shape: DatasetShape) -> Type[DataAdapter]:
    """Look up the adapter registered for ``shape``.

    Raises:
        KeyError: if no adapter is registered for that shape.
    """
    if shape not in ADAPTER_REGISTRY:
        raise KeyError(
            f"No adapter registered for shape {shape!r}. "
            f"Registered: {sorted(s.value for s in ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[shape]


def list_adapters() -> List[DatasetShape]:
    """Return the list of shapes with registered adapters."""
    return list(ADAPTER_REGISTRY.keys())


__all__ = [
    "DataAdapter",
    "ADAPTER_REGISTRY",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]
