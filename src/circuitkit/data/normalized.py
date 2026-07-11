"""Normalized data types — the lingua franca between adapters and strategies.

The data layer is split into two decoupled stages:

    raw dataset  ──[Adapter]──>  NormalizedDataset[ContrastiveRecord]
                                          │
                                  ──[Worthiness]──>  pass / warn / fail
                                          │
                          ──[CorruptionStrategy]──>  attribution-ready pairs
                                          │
                                          ▼
                                  discover_circuit

Adapters know only the *shape* of a raw dataset (Q&A vs MCQ vs ShareGPT
vs CrowS-Pairs etc). Strategies know only how to *corrupt* a normalized
record (entity-swap, final-answer-swap, logical-negation, ...). Adding
support for a new dataset shape = writing one adapter. Adding a new
corruption family = writing one strategy. They are independent.

Every adapter outputs ``ContrastiveRecord`` instances. Every strategy
consumes one and produces a corrupted partner. This module defines the
shared schema.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Enums describing dataset shape & contrast nature
# ---------------------------------------------------------------------------


class DatasetShape(str, Enum):
    """Native shape of a raw dataset, before adapter normalization.

    Drives which Adapter is selected by ``data.auto_detect``. New shapes
    can be appended without breaking existing code (Enum value is the
    string used in YAML/CLI).
    """

    QA = "qa"  # {question, context?, answer}
    MCQ = "mcq"  # {question, choices[], correct_idx}
    CLASSIFICATION = "classification"  # {input, label}
    PAIRWISE = "pairwise"  # {sent_more, sent_less}  (CrowS, StereoSet pairs)
    TRIPLE = "triple"  # 3-way: stereotype/anti-stereo/unrelated (StereoSet intrasentence)
    CONVERSATIONAL = "conversational"  # ShareGPT / OpenAI: [{role, content}, ...]
    INSTRUCTION = "instruction"  # Alpaca / Dolly: {instruction, input?, output}
    FORGET_RETAIN = "forget_retain"  # TOFU / MUSE / WMDP: split-tagged QA
    MULTI_HOP = "multi_hop"  # MQuAKE: chained edits
    CODE = "code"  # HumanEval / MBPP: {prompt, solution, test}
    MATH = "math"  # GSM8K / MATH: {question, answer_with_work}
    REFUSAL = "refusal"  # AdvBench / HarmBench: harmful instruction
    SENTENCE_COMPLETION = "sentence_completion"  # HellaSwag-style
    TEMPLATE = "template"  # user-defined template-based pairing
    CLEAN_ONLY = "clean_only"  # clean prompts only; no corrupt partner
    UNKNOWN = "unknown"


class ContrastSource(str, Enum):
    """Where the corrupt partner of a record came from."""

    NATIVE_PAIR = "native_pair"  # dataset already supplies (clean, corrupt)
    GENERATED = "generated"  # produced by a CorruptionStrategy
    EXTERNAL = "external"  # user-provided
    NOT_PAIRED_YET = "not_paired_yet"  # only clean known so far


# ---------------------------------------------------------------------------
# Core record — one (clean, corrupt?) example
# ---------------------------------------------------------------------------


@dataclass
class ContrastiveRecord:
    """One example normalized to the shared schema.

    Fields cover both halves of the contrastive pair when known. Adapters
    that emit dataset-native pairs (CrowS-Pairs, StereoSet, TOFU, etc.)
    populate ``corrupt_prompt`` / ``corrupt_answer`` directly; otherwise
    those start as ``None`` and a CorruptionStrategy fills them in.

    Fields:
        record_id:     stable string id, unique within the dataset
        clean_prompt:  the canonical prompt to attribute to
        clean_answer:  the target token (or first token of a span)
        corrupt_prompt: counter-factual prompt (if known)
        corrupt_answer: counter-factual target token (if known)
        target_field:  which field/column the answer maps to (e.g. 'choice_A')
        contrast_source: where the corrupt half came from
        meta:          arbitrary per-row provenance (label, source-set tag, etc.)
        spans:         OPTIONAL / RESERVED. Named character-spans into
                       ``clean_prompt`` for position-aware circuit discovery
                       (PEAP). Maps span name -> (start_char, end_char).

                       This field is reserved for future use: no adapter
                       currently populates it, and it is *not* required by any
                       strategy, validator, or the discovery pipeline. It
                       defaults to an empty dict and every consumer treats an
                       empty ``spans`` as "no span info available". Adapters or
                       callers that *do* have span information may populate it
                       (see :meth:`set_span`); everything downstream tolerates
                       it being empty.
    """

    record_id: str
    clean_prompt: str
    clean_answer: str
    corrupt_prompt: Optional[str] = None
    corrupt_answer: Optional[str] = None
    target_field: Optional[str] = None
    contrast_source: ContrastSource = ContrastSource.NOT_PAIRED_YET
    meta: Dict[str, Any] = field(default_factory=dict)
    spans: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def is_paired(self) -> bool:
        return self.corrupt_prompt is not None and self.corrupt_answer is not None

    @property
    def has_spans(self) -> bool:
        """True if any named span has been recorded (``spans`` is optional)."""
        return bool(self.spans)

    def set_span(self, name: str, start: int, end: int) -> None:
        """Record a named character-span into ``clean_prompt``.

        Optional helper for adapters/callers that have position information.
        Validates the span lies within ``clean_prompt`` so a bad span fails
        loudly here rather than producing misaligned discovery later.

        Args:
            name:  span identifier (e.g. "subject", "answer").
            start: inclusive start char offset into ``clean_prompt``.
            end:   exclusive end char offset into ``clean_prompt``.

        Raises:
            ValueError: if the span is out of range or start >= end.
        """
        if not (0 <= start < end <= len(self.clean_prompt)):
            raise ValueError(
                f"span '{name}' = ({start}, {end}) is out of range for a "
                f"clean_prompt of length {len(self.clean_prompt)}"
            )
        self.spans[name] = (start, end)

    def get_span_text(self, name: str) -> Optional[str]:
        """Return the substring of ``clean_prompt`` for a named span.

        Returns ``None`` if the span was never set, so callers can safely
        probe an optional span without a KeyError.
        """
        span = self.spans.get(name)
        if span is None:
            return None
        start, end = span
        return self.clean_prompt[start:end]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["contrast_source"] = self.contrast_source.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContrastiveRecord":
        d = dict(d)
        if "contrast_source" in d and isinstance(d["contrast_source"], str):
            d["contrast_source"] = ContrastSource(d["contrast_source"])
        if "spans" in d and d["spans"]:
            d["spans"] = {k: tuple(v) for k, v in d["spans"].items()}
        return cls(**d)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


@dataclass
class NormalizedDataset:
    """A homogeneous sequence of ContrastiveRecord plus dataset-level metadata.

    Adapters return one of these. Strategies consume one and may produce
    an updated copy with all records paired.
    """

    name: str
    shape: DatasetShape
    records: List[ContrastiveRecord]
    source: str = "unknown"  # short string: 'hf://glue/sst2', 'csv://...', 'builtin', etc.
    schema_version: str = "1.0"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[ContrastiveRecord]:
        return iter(self.records)

    def __getitem__(self, i):
        return self.records[i]

    @property
    def n_paired(self) -> int:
        return sum(1 for r in self.records if r.is_paired)

    @property
    def fully_paired(self) -> bool:
        return self.n_paired == len(self.records) and len(self.records) > 0

    def take(self, n: int) -> "NormalizedDataset":
        return NormalizedDataset(
            name=self.name,
            shape=self.shape,
            records=self.records[:n],
            source=self.source,
            schema_version=self.schema_version,
            meta={**self.meta, "subset_n": n},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape.value,
            "source": self.source,
            "schema_version": self.schema_version,
            "n_records": len(self.records),
            "n_paired": self.n_paired,
            "meta": self.meta,
            "records": [r.to_dict() for r in self.records],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "NormalizedDataset":
        with open(path) as f:
            d = json.load(f)
        return cls(
            name=d["name"],
            shape=DatasetShape(d["shape"]),
            records=[ContrastiveRecord.from_dict(r) for r in d["records"]],
            source=d.get("source", "unknown"),
            schema_version=d.get("schema_version", "1.0"),
            meta=d.get("meta", {}),
        )


__all__ = [
    "DatasetShape",
    "ContrastSource",
    "ContrastiveRecord",
    "NormalizedDataset",
]
