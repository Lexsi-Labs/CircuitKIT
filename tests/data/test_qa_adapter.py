"""The QA shape: plain {question, answer} tables, the simplest custom data.
`circuitkit data check` named `qa` in its help but no adapter was registered,
so a two-column CSV crashed with "Could not auto-detect dataset shape"."""

import pandas as pd
import pytest

from circuitkit.data.auto_detect import auto_normalize, detect_shape
from circuitkit.data.normalized import ContrastSource, DatasetShape


def test_clean_only_table_is_qa_and_unpaired():
    raw = pd.DataFrame({"question": ["The capital of France is"], "answer": ["Paris"]})
    assert detect_shape(raw) is DatasetShape.QA
    ds = auto_normalize(raw)
    assert ds.n_paired == 0
    assert ds.records[0].clean_answer == " Paris"
    assert ds.records[0].contrast_source is ContrastSource.NOT_PAIRED_YET


def test_explicit_corrupt_columns_make_a_native_pair():
    raw = pd.DataFrame(
        {
            "question": ["The capital of France is"],
            "answer": ["Paris"],
            "corrupted_question": ["The capital of Italy is"],
            "corrupted_answer": ["Rome"],
        }
    )
    rec = auto_normalize(raw).records[0]
    assert (rec.corrupt_prompt, rec.corrupt_answer) == ("The capital of Italy is", " Rome")
    assert rec.contrast_source is ContrastSource.NATIVE_PAIR


@pytest.mark.parametrize(
    "columns, shape",
    [
        ({"instruction": ["Say hi"], "output": ["hi"]}, DatasetShape.INSTRUCTION),
        ({"problem": ["1+1"], "solution": ["2"]}, DatasetShape.MATH),
        ({"question": ["2+2?"], "answer": ["so 4\n#### 4"]}, DatasetShape.MATH),
    ],
)
def test_more_specific_shapes_still_win(columns, shape):
    assert detect_shape(pd.DataFrame(columns)) is shape
