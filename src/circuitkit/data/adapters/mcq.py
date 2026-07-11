"""MCQ adapter — general multiple-choice question datasets.

Schema variants handled (auto-detected by column names):

  - MMLU (cais/mmlu):
      {question, choices: List[str], answer: int, subject?}

  - ARC (allenai/ai2_arc):
      {question, choices: {text: List[str], label: List[str]}, answerKey: str}

  - HellaSwag (Rowan/hellaswag):
      {ctx, endings: List[str], label: str}     (label is index as str)

  - BBQ (heegyu/bbq):
      {context, question, ans0, ans1, ans2, label: int, ...}

  - CommonSenseQA, OpenBookQA, BoolQ-MC, etc. — same general shape.

Output: clean_prompt is the question with all choices laid out, ending
with "Answer:" so the model can predict the answer letter. clean_answer
is " A" / " B" / ... corresponding to the correct choice.
``meta['choices']`` and ``meta['correct_idx']`` are preserved so a
``CorruptionStrategy`` (e.g. position-swap of two choices) can later
generate the corrupt half.

No native pair (one prompt per record). Apply a strategy before discovery.
"""

from __future__ import annotations

import string
from typing import Any, Dict, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from .base import DataAdapter, register_adapter
from .pairwise import _iter_rows, _peek_columns

_LETTERS = string.ascii_uppercase  # "A", "B", "C", ...


def _normalize_choices(row: Dict[str, Any], cols: List[str]) -> Optional[Dict[str, Any]]:
    """Extract (choices: List[str], correct_idx: int) from heterogeneous schemas."""
    # MMLU / OpenBookQA: choices is a list of strings, answer is int
    if "choices" in cols:
        ch = row.get("choices")
        if isinstance(ch, list):
            ans = row.get("answer", row.get("answerKey", row.get("label")))
            try:
                idx = int(ans)
            except (TypeError, ValueError):
                # ARC sends answerKey as letter "A"-"E"
                idx = _LETTERS.index(str(ans).strip().upper()) if ans else 0
            return {"choices": list(ch), "correct_idx": idx}
        # ARC nested: {"text": [...], "label": [...]}
        if isinstance(ch, dict) and "text" in ch and "label" in ch:
            texts = list(ch["text"])
            labels = list(ch["label"])
            ans_key = row.get("answerKey", row.get("answer"))
            try:
                idx = labels.index(str(ans_key).strip())
            except (ValueError, AttributeError):
                idx = 0
            return {"choices": texts, "correct_idx": idx}

    # HellaSwag: endings + label
    if "endings" in cols:
        endings = row.get("endings", [])
        if isinstance(endings, list) and endings:
            try:
                idx = int(row.get("label", 0))
            except (TypeError, ValueError):
                idx = 0
            return {"choices": list(endings), "correct_idx": idx}

    # BBQ: ans0, ans1, ans2 + label
    if all(f"ans{i}" in cols for i in range(3)):
        choices = [row.get(f"ans{i}", "") for i in range(3)]
        if "ans3" in cols:
            choices.append(row.get("ans3", ""))
        try:
            idx = int(row.get("label", row.get("answer", 0)))
        except (TypeError, ValueError):
            idx = 0
        return {"choices": choices, "correct_idx": idx}

    return None


def _format_question(row: Dict[str, Any], cols: List[str]) -> str:
    """Get the question stem from heterogeneous schemas."""
    if "context" in cols and "question" in cols:
        return f"{row['context']}\n{row['question']}".strip()
    if "question" in cols:
        return str(row["question"]).strip()
    # OpenBookQA / CommonSenseQA use 'question_stem'
    if "question_stem" in cols:
        return str(row["question_stem"]).strip()
    if "ctx" in cols:
        return str(row["ctx"]).strip()
    if "stem" in cols:
        return str(row["stem"]).strip()
    return str(row.get("text", "")).strip()


@register_adapter(DatasetShape.MCQ)
class MCQAdapter(DataAdapter):
    """General adapter for multiple-choice datasets (MMLU / ARC / HellaSwag / BBQ / ...)."""

    description = (
        "General MCQ datasets: builds a single clean prompt listing all "
        "choices and the correct-letter target. Stores choices and "
        "correct_idx in meta for downstream corruption strategies."
    )

    @classmethod
    def fits(cls, raw: Any) -> bool:
        cols = _peek_columns(raw)
        if not cols:
            return False
        # MMLU / OpenBookQA / ARC
        if "choices" in cols and any(k in cols for k in ("answer", "answerKey", "label")):
            return True
        # HellaSwag
        if "endings" in cols and "label" in cols:
            return True
        # BBQ
        if all(f"ans{i}" in cols for i in range(3)) and "label" in cols:
            return True
        return False

    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        prompt_template: str = ("{question}\n" "{choices_block}" "Answer:"),
        choice_letter_format: str = "{letter}. {text}\n",
        **_unused: Any,
    ) -> NormalizedDataset:
        cols = _peek_columns(raw)
        records: List[ContrastiveRecord] = []
        for i, row in enumerate(_iter_rows(raw)):
            extracted = _normalize_choices(row, cols)
            if extracted is None:
                continue
            choices = extracted["choices"]
            correct_idx = extracted["correct_idx"]
            if not choices or not (0 <= correct_idx < len(choices)):
                continue
            question = _format_question(row, cols)
            if not question:
                continue
            choices_block = "".join(
                choice_letter_format.format(letter=_LETTERS[k], text=str(c).strip())
                for k, c in enumerate(choices)
            )
            prompt = prompt_template.format(
                question=question,
                choices_block=choices_block,
            )
            answer = " " + _LETTERS[correct_idx]
            extra_meta = {
                "choices": [str(c) for c in choices],
                "correct_idx": correct_idx,
                "n_choices": len(choices),
            }
            for keep in ("subject", "category", "id", "ind", "source_id", "activity_label"):
                if keep in cols:
                    extra_meta[keep] = row.get(keep)
            records.append(
                ContrastiveRecord(
                    record_id=str(row.get("id", row.get("ind", f"{i:06d}"))),
                    clean_prompt=prompt,
                    clean_answer=answer,
                    target_field="answer_letter",
                    contrast_source=ContrastSource.NOT_PAIRED_YET,
                    meta=extra_meta,
                )
            )
            if max_records and len(records) >= max_records:
                break

        return NormalizedDataset(
            name=name or "mcq",
            shape=DatasetShape.MCQ,
            records=records,
            source=source or "raw",
            meta={
                "n_loaded": len(records),
                "choice_letter_format": choice_letter_format,
            },
        )
