"""Math adapter — GSM8K / MATH / AQuA / MR-GSM8K / GSM-Plus.

Schema variants handled:

  - GSM8K (gsm8k):              {question, answer}  (answer ends with "#### N")
  - MATH (hendrycks/competition_math): {problem, level, type, solution}
  - AQuA-RAT (deepmind/aqua_rat): {question, options, rationale, correct}
  - GSM-Plus (qintongli/GSM-Plus): {question, perturbation_type, answer}

Output: clean_prompt is the formatted question with chain-of-thought
template; clean_answer is the *final numeric answer* extracted from the
solution. ``meta['solution_text']`` holds the full reasoning trace for
strategies that want to corrupt intermediate steps (deferred to v2).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from .base import DataAdapter, register_adapter
from .pairwise import _iter_rows, _peek_columns

# Match GSM8K's "#### 42" final-answer marker, MATH's \boxed{}, plain numbers.
_GSM_FINAL = re.compile(r"####\s*(-?\d[\d,]*)")
_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
_TRAILING_NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*\.?\s*$")


def _extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = _GSM_FINAL.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = _BOXED.search(text)
    if m:
        return m.group(1).strip()
    m = _TRAILING_NUM.search(text)
    if m:
        return m.group(1)
    return None


@register_adapter(DatasetShape.MATH)
class MathAdapter(DataAdapter):
    """Math word-problem datasets with extractable final answers."""

    description = (
        "Math reasoning datasets (GSM8K / MATH / AQuA / GSM-Plus). "
        "Extracts the final numeric answer; preserves the full solution "
        "in meta['solution_text'] for downstream chain-of-thought "
        "corruption strategies (deferred)."
    )

    @classmethod
    def fits(cls, raw: Any) -> bool:
        cols = _peek_columns(raw)
        if not cols:
            return False
        # GSM8K
        if "question" in cols and "answer" in cols:
            # Sample a row to see if the answer looks math-flavoured
            try:
                first = next(iter(_iter_rows(raw)))
                ans = str(first.get("answer", ""))
                if "####" in ans:
                    return True
            except (StopIteration, TypeError):
                pass
        # MATH-style
        if "problem" in cols and "solution" in cols:
            return True
        # AQuA-style
        if "question" in cols and "options" in cols and "correct" in cols:
            return True
        return False

    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        prompt_template: str = "Question: {question}\nAnswer:",
        first_token_only: bool = True,
        **_unused: Any,
    ) -> NormalizedDataset:
        cols = _peek_columns(raw)
        records: List[ContrastiveRecord] = []
        for i, row in enumerate(_iter_rows(raw)):
            if "problem" in cols and "solution" in cols:
                question = str(row["problem"]).strip()
                solution = str(row["solution"]).strip()
                final = _extract_final_answer(solution)
            elif "question" in cols and "answer" in cols:
                question = str(row["question"]).strip()
                solution = str(row["answer"]).strip()
                final = _extract_final_answer(solution)
            elif "question" in cols and "correct" in cols:
                question = str(row["question"]).strip()
                solution = ""
                final = str(row["correct"]).strip()
            else:
                continue
            if not question or not final:
                continue
            answer = " " + (final.split()[0] if first_token_only else final)
            records.append(
                ContrastiveRecord(
                    record_id=str(row.get("id", f"{i:06d}")),
                    clean_prompt=prompt_template.format(question=question),
                    clean_answer=answer,
                    contrast_source=ContrastSource.NOT_PAIRED_YET,
                    target_field="final_answer_token",
                    meta={
                        "solution_text": solution,
                        "final_answer_full": final,
                        "category": row.get("type", row.get("level", row.get("subject"))),
                    },
                )
            )
            if max_records and len(records) >= max_records:
                break
        return NormalizedDataset(
            name=name or "math",
            shape=DatasetShape.MATH,
            records=records,
            source=source or "raw",
            meta={"n_loaded": len(records)},
        )
