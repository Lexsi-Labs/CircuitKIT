"""QA adapter — plain question/answer tables, the simplest custom-data shape.

Schema variants handled:
  - {question, answer}                 clean-only; a CorruptionStrategy must
    {prompt, answer} / {query, target}  supply the corrupt half before discovery
  - the same plus an explicit counterfactual pair:
    {corrupted_question | corrupted_prompt | corrupt_prompt,
     corrupted_answer   | corrupt_answer}              -> NATIVE_PAIR

Detected after every more specific shape (GSM8K is {question, answer} too, and
is MATH), so this only claims tables nothing else recognises.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from .base import DataAdapter, register_adapter
from .pairwise import _iter_rows, _peek_columns

_QUESTION_KEYS = ("question", "prompt", "query", "input")
_ANSWER_KEYS = ("answer", "target", "label", "output")
_CORRUPT_QUESTION_KEYS = ("corrupted_question", "corrupted_prompt", "corrupt_prompt", "corrupted")
_CORRUPT_ANSWER_KEYS = ("corrupted_answer", "corrupt_answer")


def _first(keys, cols) -> Optional[str]:
    return next((k for k in keys if k in cols), None)


def _text(row: dict, key: Optional[str]) -> str:
    value = row.get(key) if key else None
    return "" if value is None or value != value else str(value).strip()  # value != value: NaN


@register_adapter(DatasetShape.QA)
class QAAdapter(DataAdapter):
    """Adapter for plain question/answer tables, with or without corrupt columns."""

    description = (
        "Plain question/answer tables (CSV / JSONL / HF): {question, answer} -> "
        "ContrastiveRecord. Explicit corrupted_question / corrupted_answer columns "
        "make a native pair; otherwise a corruption strategy must be applied."
    )

    @classmethod
    def fits(cls, raw: Any) -> bool:
        cols = _peek_columns(raw)
        return bool(_first(_QUESTION_KEYS, cols) and _first(_ANSWER_KEYS, cols))

    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        **_unused: Any,
    ) -> NormalizedDataset:
        cols = _peek_columns(raw)
        q_key, a_key = _first(_QUESTION_KEYS, cols), _first(_ANSWER_KEYS, cols)
        if not q_key or not a_key:
            raise ValueError(
                f"QA adapter requires one of {_QUESTION_KEYS} and one of {_ANSWER_KEYS}; "
                f"got cols={cols}"
            )
        cq_key, ca_key = _first(_CORRUPT_QUESTION_KEYS, cols), _first(_CORRUPT_ANSWER_KEYS, cols)

        records: List[ContrastiveRecord] = []
        for i, row in enumerate(_iter_rows(raw)):
            question, answer = _text(row, q_key), _text(row, a_key)
            if not question or not answer:
                continue
            corrupt_q, corrupt_a = _text(row, cq_key), _text(row, ca_key)
            paired = bool(corrupt_q)
            records.append(
                ContrastiveRecord(
                    record_id=f"{i:06d}",
                    clean_prompt=question,
                    clean_answer=" " + answer,
                    corrupt_prompt=corrupt_q if paired else None,
                    corrupt_answer=(" " + corrupt_a) if paired and corrupt_a else None,
                    contrast_source=(
                        ContrastSource.NATIVE_PAIR if paired else ContrastSource.NOT_PAIRED_YET
                    ),
                    target_field="answer",
                )
            )
            if max_records and len(records) >= max_records:
                break

        return NormalizedDataset(
            name=name or "qa",
            shape=DatasetShape.QA,
            records=records,
            source=source or "raw",
            meta={
                "question_key": q_key,
                "answer_key": a_key,
                "corrupt_question_key": cq_key,
                "corrupt_answer_key": ca_key,
                "n_loaded": len(records),
            },
        )
