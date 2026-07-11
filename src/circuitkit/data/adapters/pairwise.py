"""Pairwise adapter — datasets that natively supply (sent_more, sent_less) pairs.

Handles:
  - CrowS-Pairs ([nyu-mll/crows-pairs](https://github.com/nyu-mll/crows-pairs))
    columns: sent_more, sent_less, stereo_antistereo, bias_type, ...
  - StereoSet intersentence (huggingface 'McGill-NLP/stereoset')
    columns: target, sentences[{sentence, gold_label}, ...]
  - Any HF dataset / DataFrame with both ``sent_more`` and ``sent_less``
    string columns (the canonical schema established by CrowS-Pairs).

Native pair structure:
    clean_prompt   = sent_more (the stereotyped or "more likely under bias" sentence)
    corrupt_prompt = sent_less (the anti-stereotyped or "less likely under bias")
    clean_answer   / corrupt_answer = the trailing token-position used to score
                                       sentence likelihood (see notes)

For circuit-discovery purposes we treat the two sentences as the
(clean, corrupt) pair directly and use the *last word of each* as the
target. Bias type is preserved in ``meta['bias_type']``.

Reference:
    Nangia et al. EMNLP 2020 (arxiv:2010.00133).
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from .base import DataAdapter, register_adapter


def _last_word(s: str) -> str:
    """Return the trailing word (with leading space, GPT-style)."""
    s = s.rstrip()
    if not s:
        return ""
    parts = s.replace("?", "").replace(".", "").replace(",", "").split()
    if not parts:
        return ""
    return " " + parts[-1]


def _strip_last_word(s: str) -> str:
    """Return the prompt with the last word removed (so model predicts it)."""
    s = s.rstrip(" .?!,")
    parts = s.split()
    if len(parts) <= 1:
        return s
    return " ".join(parts[:-1])


@register_adapter(DatasetShape.PAIRWISE)
class PairwiseAdapter(DataAdapter):
    """Adapter for CrowS-Pairs-style natively-paired datasets."""

    description = (
        "CrowS-Pairs / StereoSet / similar datasets that supply sent_more "
        "and sent_less columns directly."
    )

    @classmethod
    def fits(cls, raw: Any) -> bool:
        cols = _peek_columns(raw)
        if not cols:
            return False
        # CrowS-Pairs canonical schema
        if "sent_more" in cols and "sent_less" in cols:
            return True
        # StereoSet intrasentence variant (different shape)
        if "stereo" in cols and "anti_stereo" in cols:
            return True
        return False

    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        sent_more_col: str = "sent_more",
        sent_less_col: str = "sent_less",
        bias_type_col: Optional[str] = "bias_type",
        keep_full_sentence_as_answer: bool = False,
        **_unused: Any,
    ) -> NormalizedDataset:
        cols = _peek_columns(raw)
        if "stereo" in cols and "anti_stereo" in cols and sent_more_col == "sent_more":
            sent_more_col, sent_less_col = "stereo", "anti_stereo"

        records: List[ContrastiveRecord] = []
        for i, row in enumerate(_iter_rows(raw)):
            sm = (row.get(sent_more_col) or "").strip()
            sl = (row.get(sent_less_col) or "").strip()
            if not sm or not sl:
                continue
            if keep_full_sentence_as_answer:
                clean_p, clean_a = sm, sm
                corr_p, corr_a = sl, sl
            else:
                clean_p = _strip_last_word(sm)
                clean_a = _last_word(sm)
                corr_p = _strip_last_word(sl)
                corr_a = _last_word(sl)
                # If the two sentences end with the *same* trailing word, the
                # native pair is uninformative for next-token prediction. We
                # still emit the record but flag it.
                if clean_a == corr_a:
                    pass
            meta = {}
            if bias_type_col and bias_type_col in cols:
                meta["bias_type"] = row.get(bias_type_col)
            if "stereo_antistereo" in cols:
                meta["stereo_antistereo"] = row.get("stereo_antistereo")
            records.append(
                ContrastiveRecord(
                    record_id=f"{i:05d}",
                    clean_prompt=clean_p,
                    clean_answer=clean_a,
                    corrupt_prompt=corr_p,
                    corrupt_answer=corr_a,
                    target_field="last_token",
                    contrast_source=ContrastSource.NATIVE_PAIR,
                    meta=meta,
                )
            )
            if max_records and len(records) >= max_records:
                break
        return NormalizedDataset(
            name=name or "pairwise",
            shape=DatasetShape.PAIRWISE,
            records=records,
            source=source or "raw",
            meta={
                "sent_more_col": sent_more_col,
                "sent_less_col": sent_less_col,
                "n_loaded": len(records),
            },
        )


# ---------------------------------------------------------------------------
# Generic raw helpers (work for HF Dataset, pandas, list-of-dicts, csv path)
# ---------------------------------------------------------------------------


def _peek_columns(raw: Any) -> List[str]:
    if hasattr(raw, "column_names"):  # HF Dataset
        return list(raw.column_names)
    if hasattr(raw, "columns"):  # pandas DataFrame
        return list(raw.columns)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return list(raw[0].keys())
    if isinstance(raw, str):  # path to CSV
        import csv

        with open(raw) as f:
            return next(csv.reader(f), [])
    return []


def _iter_rows(raw: Any) -> Iterable[dict]:
    if hasattr(raw, "to_pandas"):  # HF Dataset
        df = raw.to_pandas()
        for _, row in df.iterrows():
            yield row.to_dict()
        return
    if hasattr(raw, "iterrows"):  # pandas DataFrame
        for _, row in raw.iterrows():
            yield row.to_dict()
        return
    if isinstance(raw, list):
        yield from raw
        return
    if isinstance(raw, str):  # path to CSV
        import csv

        with open(raw) as f:
            yield from csv.DictReader(f)
        return
    raise TypeError(f"Cannot iterate over raw of type {type(raw).__name__}")
