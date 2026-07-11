"""DatasetSchema — named token-spans across variable-length examples.

PEAP (Position-aware Edge Attribution Patching, ACL 2025) requires
knowing which token positions across different examples are *semantically
equivalent*. For example, in IOI-style "When [SUBJ_A] and [SUBJ_B] went
to the [PLACE], [SUBJ_C] gave a [OBJECT] to" the SUBJ_A position is
"equivalent" across all 256 IOI examples even though their absolute
indices vary by 1-3 tokens.

A DatasetSchema names spans (e.g. "subject_a", "place", "object") and
provides a way to resolve them to concrete token-index ranges per
ContrastiveRecord. This in turn lets PEAP aggregate edge-attribution
scores per-position across examples.

Reference: Haklay, Orgad, Bau, Mueller, Belinkov ACL 2025
([arxiv:2502.04577](https://arxiv.org/abs/2502.04577)).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .normalized import ContrastiveRecord


@dataclass
class SpanDef:
    """One named span definition.

    Attributes:
        name:    canonical span name ("subject_a", "object", ...)
        regex:   optional Python regex (matched against the prompt). The
                 first group captures the span's text. Use ``None`` when
                 the span comes from explicit ``ContrastiveRecord.spans``
                 metadata instead.
        invariance: free-text contract — what *should* stay invariant
                    when this span is corrupted.
    """

    name: str
    regex: Optional[str] = None
    invariance: str = ""

    def find(self, prompt: str) -> Optional[Tuple[int, int]]:
        """Return (char_start, char_end) of the first match in ``prompt``,
        or None if no match. Returns the *first capture group*'s bounds
        when the regex has groups, otherwise the full-match bounds."""
        if not self.regex:
            return None
        m = re.search(self.regex, prompt)
        if not m:
            return None
        if m.groups():
            return m.span(1)
        return m.span()


@dataclass
class DatasetSchema:
    """Collection of named spans for a dataset.

    Attributes:
        spans:        ordered list of SpanDef
        invariances:  free-text mapping span_name -> what should be
                      invariant when this span varies (used by PEAP and
                      by Pillar-4 robustness reports)
    """

    spans: List[SpanDef] = field(default_factory=list)
    invariances: Dict[str, str] = field(default_factory=dict)

    def find_all(self, prompt: str) -> Dict[str, Tuple[int, int]]:
        """Return {span_name: (char_start, char_end)} for spans the
        regex finds in ``prompt``. Spans without a hit are omitted.
        """
        out: Dict[str, Tuple[int, int]] = {}
        for span in self.spans:
            hit = span.find(prompt)
            if hit:
                out[span.name] = hit
        return out

    def annotate(self, record: ContrastiveRecord) -> ContrastiveRecord:
        """Return a new record with spans populated from this schema."""
        spans = self.find_all(record.clean_prompt)
        return ContrastiveRecord(
            record_id=record.record_id,
            clean_prompt=record.clean_prompt,
            clean_answer=record.clean_answer,
            corrupt_prompt=record.corrupt_prompt,
            corrupt_answer=record.corrupt_answer,
            target_field=record.target_field,
            contrast_source=record.contrast_source,
            meta=record.meta,
            spans={**record.spans, **spans},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spans": [
                {"name": s.name, "regex": s.regex, "invariance": s.invariance} for s in self.spans
            ],
            "invariances": self.invariances,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetSchema":
        return cls(
            spans=[SpanDef(**s) for s in d.get("spans", [])],
            invariances=d.get("invariances", {}),
        )


# ---------------------------------------------------------------------------
# Built-in schemas for canonical mech-interp tasks
# ---------------------------------------------------------------------------

IOI_SCHEMA = DatasetSchema(
    spans=[
        SpanDef(
            "subject_a",
            r"When (\w+) and \w+ went",
            "swap with subject_b without changing answer's role",
        ),
        SpanDef("subject_b", r"When \w+ and (\w+) went", "swap with subject_a"),
        SpanDef("place", r"to the (\w+)", "any place noun preserves the IOI structure"),
        SpanDef(
            "subject_c",
            r"\.\s+(\w+) gave",
            "the giver's name; answer is one of subject_a/b that ISN'T this",
        ),
        SpanDef("object", r"a (\w+) to", "any object noun preserves the IOI structure"),
    ],
    invariances={
        "subject_a": "answer position",
        "subject_b": "answer position",
        "subject_c": "subject_c is the giver; the receiver is the OTHER subject",
    },
)

CAPITAL_COUNTRY_SCHEMA = DatasetSchema(
    spans=[
        SpanDef(
            "country",
            r"The capital of (\w+(?:\s+\w+)?) is",
            "swap with another country preserves task structure",
        ),
    ],
    invariances={"country": "the answer is the capital of the country mentioned"},
)


def get_builtin_schema(task_name: str) -> Optional[DatasetSchema]:
    """Return the built-in schema for a canonical task, or None."""
    return {
        "ioi": IOI_SCHEMA,
        "capital_country": CAPITAL_COUNTRY_SCHEMA,
    }.get(task_name)


__all__ = [
    "SpanDef",
    "DatasetSchema",
    "IOI_SCHEMA",
    "CAPITAL_COUNTRY_SCHEMA",
    "get_builtin_schema",
]
