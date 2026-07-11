"""Regression tests — data adapter / util bugs (hardening pass).

Covers:
  * Bug 6 — MCQ adapter must handle the ``question_stem`` column
            (OpenBookQA / CommonsenseQA), not silently drop every row.
  * Bug 9 — TransferMatrix._convert_for_json must recurse so a NaN inside an
            array becomes JSON ``null`` (not a non-serialisable float).
"""

from __future__ import annotations

import json
import math

import numpy as np

# ---------------------------------------------------------------------------
# Bug 6 — MCQ adapter handles `question_stem`
# ---------------------------------------------------------------------------


def test_mcq_adapter_handles_question_stem_column():
    """OpenBookQA / CommonsenseQA rows use 'question_stem' instead of 'question'.

    The historical bug dropped every row because ``_format_question`` only
    looked at 'question'/'context', so the stem-only schema produced an empty
    question and the row was skipped.
    """
    from circuitkit.data.adapters.mcq import MCQAdapter

    raw = [
        {
            "question_stem": "The sun is best described as a",
            "choices": {
                "text": ["star", "planet", "moon", "comet"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "question_stem": "Photosynthesis primarily occurs in",
            "choices": {
                "text": ["leaves", "roots", "stems", "flowers"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
    ]

    ds = MCQAdapter().adapt(raw, name="openbookqa")

    assert len(ds.records) == 2, (
        "MCQ adapter dropped question_stem rows — every OpenBookQA/CSQA row "
        "was discarded (question_stem regression)."
    )
    # The stem text must actually appear in the built prompt.
    assert "The sun is best described as a" in ds.records[0].clean_prompt
    assert ds.records[0].clean_prompt.strip() != ""
    # Choices must still be laid out and the answer letter resolved.
    assert "A. star" in ds.records[0].clean_prompt
    assert ds.records[0].clean_answer.strip() == "A"


def test_mcq_format_question_prefers_question_stem_when_present():
    """Unit-level check on the helper that resolves the question text."""
    from circuitkit.data.adapters.mcq import _format_question

    row = {"question_stem": "Which gas do plants absorb?"}
    assert _format_question(row, ["question_stem"]) == "Which gas do plants absorb?"


# ---------------------------------------------------------------------------
# Bug 9 — TransferMatrix._convert_for_json recurses into arrays
# ---------------------------------------------------------------------------


def test_transfer_matrix_json_converts_nan_inside_array(tmp_path):
    """A NaN inside the transfer matrix must serialise to JSON null.

    The historical bug only checked the top-level object: a NaN buried inside
    a numpy array was passed straight to ``json.dump``, which either raised or
    emitted invalid JSON. The fix recurses element-wise.
    """
    from circuitkit.evaluation.transfer import TransferMatrix

    tm = TransferMatrix(task_names=["ioi", "sva"])
    # 2x2 matrix with a NaN entry (e.g. a failed transfer evaluation).
    tm.matrix = np.array([[1.0, float("nan")], [0.5, 0.8]], dtype=float)

    out_path = tmp_path / "transfer.json"
    tm.to_json(out_path)

    # Must be strict, valid JSON (json.load rejects bare NaN).
    with open(out_path) as f:
        data = json.load(f)

    matrix = data["matrix"]
    assert matrix[0][0] == 1.0
    assert matrix[0][1] is None, (
        "NaN inside the matrix array did not become JSON null — "
        "_convert_for_json did not recurse into the array."
    )
    assert matrix[1][1] == 0.8

    # And the raw file text must not contain a bare 'NaN' literal.
    assert "NaN" not in out_path.read_text()


def test_transfer_matrix_json_handles_nested_nan_in_analysis(tmp_path):
    """NaN nested in the analysis dict/lists must also become null."""
    from circuitkit.evaluation.transfer import TransferMatrix

    tm = TransferMatrix(task_names=["ioi", "sva", "greater_than"])
    m = np.full((3, 3), 0.7, dtype=float)
    m[1, 2] = float("nan")
    tm.matrix = m

    out_path = tmp_path / "transfer_nested.json"
    tm.to_json(out_path)
    # Round-trips as strict JSON => every nested NaN was converted.
    with open(out_path) as f:
        data = json.load(f)
    assert "analysis" in data

    # Spot-check: no NaN survived anywhere serialisable.
    def _no_nan(obj):
        if isinstance(obj, float):
            assert not math.isnan(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _no_nan(v)
        elif isinstance(obj, list):
            for v in obj:
                _no_nan(v)

    _no_nan(data)
