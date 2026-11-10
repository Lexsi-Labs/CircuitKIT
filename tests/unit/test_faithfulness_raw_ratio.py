"""A faithfulness ratio outside 0 to 1 is reported as such, not shown as a bare 0 or 1.

The clipped ``score`` keeps its meaning. These tests pin what is added next to
it: the ``out_of_range`` flag, the warning, and the raw ratio on the report.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from circuitkit.evaluation.pillars.ablation import Pillar2_Ablation
from circuitkit.evaluation.pillars.causal_patching import (
    Pillar1_CausalPatching,
    _faithfulness_ratio,
    describe_raw_ratio,
)
from circuitkit.evaluation.report import FaithfulnessReport

PILLARS = [
    (Pillar1_CausalPatching, "circuitkit.evaluation.pillars.causal_patching"),
    (Pillar2_Ablation, "circuitkit.evaluation.pillars.ablation"),
]


@pytest.mark.parametrize(
    "circuit,score,raw,flag",
    [
        (0.6, 0.6, 0.6, False),
        (5.842, 1.0, 5.842, True),
        (-0.3, 0.0, -0.3, True),
        (1.0, 1.0, 1.0, False),
    ],
)
def test_ratio_flags_values_outside_zero_to_one(circuit, score, raw, flag):
    r = _faithfulness_ratio(circuit, clean=1.0, corrupt=0.0)
    assert r["score"] == pytest.approx(score)  # clipped value is unchanged
    assert r["raw_ratio"] == pytest.approx(raw)
    assert r["out_of_range"] is flag


def test_undefined_ratio_is_not_called_out_of_range():
    assert _faithfulness_ratio(1.0, clean=0.5, corrupt=0.5)["out_of_range"] is False
    assert _faithfulness_ratio(1.0, clean=0.0, corrupt=1.0)["out_of_range"] is False


@pytest.mark.parametrize("pillar,module", PILLARS)
@pytest.mark.parametrize("circuit,word", [(5.842, "above 1"), (-0.3, "below 0"), (0.6, None)])
def test_pillar_warns_when_the_score_was_clipped(pillar, module, circuit, word, caplog):
    model = MagicMock()
    model.cfg.use_attn_result = True
    with (
        patch(f"{module}.evaluate_graph", return_value=torch.tensor([circuit])),
        patch(
            f"{module}.evaluate_baseline", side_effect=[torch.tensor([1.0]), torch.tensor([0.0])]
        ),
        caplog.at_level(logging.WARNING, logger=module),
    ):
        result = pillar.run(model, MagicMock(), MagicMock(), MagicMock(), quiet=True)
    assert result["raw_ratio"] == pytest.approx(circuit)
    assert result["out_of_range"] is (word is not None)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    if word is None:
        assert not warned
    else:
        assert len(warned) == 1 and word in warned[0] and f"{circuit:.4f}" in warned[0]


def test_good_and_bad_circuit_that_both_clip_stay_apart_in_the_raw_ratio():
    """Two circuits can share a clipped score; the raw ratio still orders them."""
    good = _faithfulness_ratio(-0.15, clean=1.0, corrupt=0.0)
    bad = _faithfulness_ratio(-0.73, clean=1.0, corrupt=0.0)
    assert good["score"] == bad["score"] == 0.0
    assert good["raw_ratio"] > bad["raw_ratio"]


def test_report_shows_the_raw_ratio_and_says_when_it_is_out_of_range():
    report = FaithfulnessReport(
        patching_score=1.0,
        ablation_score=0.0,
        metadata={"patching_raw_ratio": 5.842, "ablation_raw_ratio": -0.3},
    )
    assert report.patching_raw_ratio == 5.842 and report.ablation_raw_ratio == -0.3
    text = repr(report)
    assert "Score: 1.0000  (OUT OF RANGE: raw ratio 5.8420 is above 1" in text
    assert "Score: 0.0000  (OUT OF RANGE: raw ratio -0.3000 is below 0" in text
    s = report.summary()
    assert s["pillar_1_patching"] == 1.0 and s["pillar_1_patching_raw_ratio"] == 5.842
    assert s["pillar_2_ablation"] == 0.0 and s["pillar_2_ablation_raw_ratio"] == -0.3
    assert s["overall_average"] == 0.5  # still the average of the clipped scores


def test_report_in_range_and_without_a_raw_ratio():
    in_range = FaithfulnessReport(0.8, 0.6, metadata={"patching_raw_ratio": 0.8})
    assert "Score: 0.8000  (raw ratio 0.8000)" in repr(in_range)
    assert "OUT OF RANGE" not in repr(in_range)
    bare = FaithfulnessReport(patching_score=0.8, ablation_score=None)
    assert bare.patching_raw_ratio is None and bare.ablation_raw_ratio is None
    assert "  Score: 0.8000\n" in repr(bare) and "  Score: N/A\n" in repr(bare)


def test_raw_ratio_survives_a_json_round_trip(tmp_path):
    report = FaithfulnessReport(1.0, 0.0, metadata={"patching_raw_ratio": 5.842})
    report.to_json(tmp_path / "r.json")
    assert FaithfulnessReport.from_json(tmp_path / "r.json").patching_raw_ratio == 5.842


def test_describe_raw_ratio_is_empty_inside_the_range():
    assert describe_raw_ratio(None) == "" and describe_raw_ratio(0.0) == ""
    assert describe_raw_ratio(1.0) == ""
