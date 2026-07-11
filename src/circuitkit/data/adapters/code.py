"""Code adapter — HumanEval / MBPP / similar code-generation datasets.

Schema variants handled:

  - HumanEval (openai/openai_humaneval / openai_humaneval):
      {prompt, canonical_solution, test, entry_point, task_id}
  - MBPP (mbpp):
      {text, code, test_list, task_id, ...}
  - BigCodeBench: similar to HumanEval but with multi-file tests.

Output: clean_prompt is the function signature + docstring; clean_answer
is the first content token of the canonical solution. Stores the full
solution + tests in meta for syntax-level corruption strategies (which
are deferred to v2 as 'full code adapter beyond final-answer swap').
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from .base import DataAdapter, register_adapter
from .pairwise import _iter_rows, _peek_columns


@register_adapter(DatasetShape.CODE)
class CodeAdapter(DataAdapter):
    """Code-generation datasets (HumanEval / MBPP / BigCodeBench)."""

    description = (
        "Code-generation datasets. Extracts the function signature + "
        "docstring as the clean prompt and the first solution token as "
        "the answer. Full solution + test cases preserved in meta for "
        "syntax-corruption strategies."
    )

    @classmethod
    def fits(cls, raw: Any) -> bool:
        cols = _peek_columns(raw)
        if not cols:
            return False
        # HumanEval
        if "prompt" in cols and "canonical_solution" in cols:
            return True
        # MBPP
        if "text" in cols and "code" in cols and "test_list" in cols:
            return True
        return False

    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        first_token_only: bool = True,
        **_unused: Any,
    ) -> NormalizedDataset:
        cols = _peek_columns(raw)
        records: List[ContrastiveRecord] = []
        for i, row in enumerate(_iter_rows(raw)):
            if "prompt" in cols and "canonical_solution" in cols:
                prompt = str(row["prompt"])
                solution = str(row["canonical_solution"])
                tests = str(row.get("test", ""))
                ep = row.get("entry_point", "")
                tid = str(row.get("task_id", f"{i:06d}"))
            elif "text" in cols and "code" in cols:
                prompt = str(row["text"])  # natural-language description
                solution = str(row["code"])
                tests_raw = row.get("test_list") or []
                tests = "\n".join(tests_raw) if isinstance(tests_raw, list) else str(tests_raw)
                ep = ""
                tid = str(row.get("task_id", f"{i:06d}"))
            else:
                continue
            if not prompt or not solution:
                continue
            stripped = solution.lstrip("\n")
            first_word = stripped.split()[0] if stripped.split() else stripped[:1]
            answer = " " + (first_word if first_token_only else stripped)
            records.append(
                ContrastiveRecord(
                    record_id=tid,
                    clean_prompt=prompt,
                    clean_answer=answer,
                    contrast_source=ContrastSource.NOT_PAIRED_YET,
                    target_field="first_solution_token",
                    meta={
                        "solution_text": solution,
                        "tests": tests,
                        "entry_point": ep,
                        "solution_lines": len(solution.split("\n")),
                    },
                )
            )
            if max_records and len(records) >= max_records:
                break
        return NormalizedDataset(
            name=name or "code",
            shape=DatasetShape.CODE,
            records=records,
            source=source or "raw",
            meta={"n_loaded": len(records)},
        )
