"""clean_only normalizer — load clean-only data for IBCircuit / CD-T discovery.

No corruption is needed. Records carry corrupt_prompt=None and
contrast_source=NOT_PAIRED_YET. NormalizedTaskSpec accepts these when the
algorithm is IBCircuit or CD-T (both need only the clean prompt).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Optional

from .normalized import (
    ContrastiveRecord,
    ContrastSource,
    DatasetShape,
    NormalizedDataset,
)


def clean_only_normalize(
    raw: Any,
    *,
    prompt_column: str = "prompt",
    answer_column: Optional[str] = "answer",
    max_records: Optional[int] = None,
    name: Optional[str] = None,
    source: Optional[str] = None,
) -> NormalizedDataset:
    """Load a CSV / list-of-dicts as a clean-only NormalizedDataset.

    No corrupt partner is generated. Use with IBCircuit or CD-T, which
    require only the clean prompt.

    Args:
        raw: Path to a CSV file (str or Path), or a list of dicts.
        prompt_column: Column name for the clean prompt. Default "prompt".
        answer_column: Column name for the answer token. Pass None to skip
            answer loading (valid for CD-T which ignores it). Default "answer".
        max_records: If set, truncate to this many records.
        name: Dataset name for reporting. Defaults to the source path.
        source: Source descriptor stored in NormalizedDataset.source.

    Returns:
        NormalizedDataset with shape=CLEAN_ONLY and all records unpaired.

    Raises:
        ValueError: If prompt_column is not found in the data, or if
            answer_column is not None and not found in the data.
        ValueError: If raw is not a recognised input type.
    """
    rows = _load_rows(raw)
    if not rows:
        raise ValueError("clean_only_normalize received an empty dataset.")

    # Validate columns once on the first row.
    first = rows[0]
    if prompt_column not in first:
        available = sorted(first.keys())
        raise ValueError(
            f"prompt_column={prompt_column!r} not found. "
            f"Available columns: {available}"
        )
    if answer_column is not None and answer_column not in first:
        available = sorted(first.keys())
        raise ValueError(
            f"answer_column={answer_column!r} not found. "
            f"Pass answer_column=None to skip answer loading, "
            f"or pick from: {available}"
        )

    if max_records is not None:
        rows = rows[:max_records]

    records: List[ContrastiveRecord] = []
    for i, row in enumerate(rows):
        clean_answer = row[answer_column].strip() if answer_column is not None else ""
        records.append(
            ContrastiveRecord(
                record_id=str(i),
                clean_prompt=row[prompt_column],
                clean_answer=clean_answer,
                corrupt_prompt=None,
                corrupt_answer=None,
                contrast_source=ContrastSource.NOT_PAIRED_YET,
            )
        )

    _name = name or (str(raw) if isinstance(raw, (str, Path)) else "clean_only")
    _source = source or _name

    return NormalizedDataset(
        name=_name,
        shape=DatasetShape.CLEAN_ONLY,
        records=records,
        source=_source,
    )


def _load_rows(raw: Any) -> List[dict]:
    """Normalise the various accepted input formats into a list of dicts."""
    if isinstance(raw, (str, Path)):
        path = Path(raw)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if hasattr(raw, "to_dict"):
        return raw.to_dict("records")
    if isinstance(raw, list):
        if not raw:
            return []
        # Accept list of dicts directly.
        if isinstance(raw[0], dict):
            return raw
        raise ValueError(
            f"clean_only_normalize: list input must be a list of dicts, "
            f"got list of {type(raw[0]).__name__}"
        )
    raise ValueError(
        f"clean_only_normalize: unsupported input type {type(raw).__name__}. "
        f"Pass a CSV path (str/Path), pandas DataFrame, or a list of dicts."
    )
