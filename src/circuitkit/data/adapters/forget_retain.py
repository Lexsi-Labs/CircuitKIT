"""Forget/Retain adapter — TOFU / MUSE / WMDP -style unlearning datasets.

Schema variants handled:

  - TOFU (locuslab/TOFU):
      configs forget01 / forget05 / forget10 (the unlearn target) and
      retain99 / retain95 / retain90 (the retain set), all with columns
      ``{question, answer}``. Caller passes both halves (or two adapter
      runs) and the records are tagged with ``meta['split'] = 'forget'``
      or ``'retain'``.

  - MUSE: similar QA-pair structure with explicit forget/retain splits.

  - WMDP (cais/wmdp): the existing built-in `WMDPTaskSpec` already
    handles WMDP at the task level; this adapter is the data-side
    normalizer for the same dataset family when used outside that task.

Output: clean_prompt is the question (with optional system prefix);
clean_answer is the first token of the answer span. The corrupt half is
NOT_PAIRED_YET — for unlearning the natural strategy is to pair a
forget-set question's clean_prompt with a retain-set question's prompt
as the corrupt counterfactual (see ``corruption/forget_swap.py``).

Worthiness ``shape_specific`` check verifies BOTH splits are populated.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from .base import DataAdapter, register_adapter
from .pairwise import _iter_rows, _peek_columns


@register_adapter(DatasetShape.FORGET_RETAIN)
class ForgetRetainAdapter(DataAdapter):
    """Adapter for TOFU / MUSE / WMDP -style forget/retain QA splits."""

    description = (
        "Unlearning benchmarks with explicit forget/retain splits "
        "(TOFU, MUSE, WMDP). Caller supplies both splits via "
        "`raw=({split_name: hf_split, ...})` or by passing a tagged dataset."
    )

    @classmethod
    def fits(cls, raw: Any) -> bool:
        # Match split-tagged dict input: {'forget': raw1, 'retain': raw2}
        if isinstance(raw, dict):
            keys = {str(k).lower() for k in raw.keys()}
            if "forget" in keys and "retain" in keys:
                return True
        # Match a single split with QA cols + a 'split' col
        cols = _peek_columns(raw)
        if not cols:
            return False
        if "question" in cols and "answer" in cols and "split" in cols:
            return True
        return False

    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        question_col: str = "question",
        answer_col: str = "answer",
        prompt_template: str = "Question: {question}\nAnswer:",
        first_token_only: bool = True,
        **_unused: Any,
    ) -> NormalizedDataset:
        # Normalize the input shape into a list of (split_name, rows) pairs.
        if isinstance(raw, dict):
            iter_pairs = []
            for split_name, sub in raw.items():
                iter_pairs.append((str(split_name).lower(), sub))
        else:
            cols = _peek_columns(raw)
            if "split" in cols:
                # Group rows by their 'split' column value.
                forget_rows = []
                retain_rows = []
                for r in _iter_rows(raw):
                    s = str(r.get("split", "")).lower()
                    (forget_rows if "forget" in s else retain_rows).append(r)
                iter_pairs = [("forget", forget_rows), ("retain", retain_rows)]
            else:
                # Single split, treat as forget-only.
                iter_pairs = [("forget", raw)]

        records: List[ContrastiveRecord] = []
        per_split_caps = {"forget": None, "retain": None}
        if max_records is not None:
            # Distribute the max across the available splits evenly.
            half = max_records // max(1, sum(1 for _, s in iter_pairs if _truthy_iter(s)))
            for split_name, sub in iter_pairs:
                per_split_caps[split_name] = half

        for split_name, sub in iter_pairs:
            cap = per_split_caps.get(split_name) or max_records
            count = 0
            for i, row in enumerate(_iter_rows(sub)):
                q = (row.get(question_col) or "").strip()
                a = (row.get(answer_col) or "").strip()
                if not q or not a:
                    continue
                prompt = prompt_template.format(question=q)
                answer = " " + (a.split()[0] if first_token_only and a.split() else a)
                records.append(
                    ContrastiveRecord(
                        record_id=f"{split_name}-{i:06d}",
                        clean_prompt=prompt,
                        clean_answer=answer,
                        contrast_source=ContrastSource.NOT_PAIRED_YET,
                        target_field="first_answer_token",
                        meta={
                            "split": split_name,
                            "question_chars": len(q),
                            "answer_chars": len(a),
                        },
                    )
                )
                count += 1
                if cap and count >= cap:
                    break

        return NormalizedDataset(
            name=name or "forget_retain",
            shape=DatasetShape.FORGET_RETAIN,
            records=records,
            source=source or "raw",
            meta={
                "splits": [s for s, _ in iter_pairs],
                "n_loaded": len(records),
                "n_forget": sum(1 for r in records if r.meta.get("split") == "forget"),
                "n_retain": sum(1 for r in records if r.meta.get("split") == "retain"),
            },
        )


def _truthy_iter(raw: Any) -> bool:
    """Best-effort check: does this raw split have records?"""
    if isinstance(raw, list):
        return bool(raw)
    if hasattr(raw, "__len__"):
        try:
            return len(raw) > 0
        except TypeError:
            return True
    return True
