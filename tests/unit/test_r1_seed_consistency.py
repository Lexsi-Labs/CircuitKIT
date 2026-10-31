"""Pillar 7 R1: Spearman rho must hold up on tied, constant and NaN node scores.

The old computation gave tied scores distinct ordinal ranks and used the
`1 - 6*sum(d^2) / n(n^2 - 1)` shortcut, which is only exact without ties: two
lists with no relation scored 0.57, and two constant lists scored a perfect 1.0
just because a stable sort kept them in the same order. Graphs do produce ties:
IBCircuit pins out-of-scope nodes to inf, and a graph with no node scores reads
as all zeros.
"""

import math

import pytest

from circuitkit.evaluation.pillars.intervention_reliability import _spearman_rho


def test_no_ties_matches_the_closed_form():
    # d = [1, 1, 1, 1, 1, 1] -> 1 - 6 * 6 / (6 * 35)
    rho = _spearman_rho([1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5])
    assert rho == pytest.approx(1 - 36 / 210)


def test_unrelated_lists_with_ties_score_zero():
    rho = _spearman_rho([0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1])
    assert rho == pytest.approx(0.0, abs=1e-9)


def test_opposite_lists_with_ties_score_minus_one():
    rho = _spearman_rho([0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0])
    assert rho == pytest.approx(-1.0)


def test_nodes_pinned_to_inf_tie_with_each_other():
    inf = float("inf")
    # average ranks [4.5, 4.5, 1, 2, 3] and [4.5, 4.5, 2, 1, 3]: 8.5 / 9.5
    rho = _spearman_rho([inf, inf, 1, 2, 3], [inf, inf, 2, 1, 3])
    assert rho == pytest.approx(8.5 / 9.5)


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
@pytest.mark.parametrize(
    "x, y",
    [
        ([0.0] * 6, [0.0] * 6),
        ([0.0] * 6, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_constant_list_has_no_correlation(x, y):
    """Undefined rho is neutral, not a perfect score."""
    assert _spearman_rho(x, y) == 0.0


def test_nan_scores_are_skipped_pairwise():
    nan = float("nan")
    rho = _spearman_rho([1, nan, 3, 4, 2], [1, 5, nan, 4, 2])
    assert rho == pytest.approx(1.0)


@pytest.mark.parametrize(
    "x, y",
    [
        ([], []),
        ([1.0, 2.0], [2.0, 1.0]),
        ([float("nan")] * 4, [1.0, 2.0, 3.0, 4.0]),
        ([1.0, 2.0, float("nan"), float("nan")], [1.0, 2.0, 3.0, 4.0]),
        ([1.0, float("nan"), float("nan"), float("nan")], [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_fewer_than_three_pairs_is_neutral(x, y):
    assert _spearman_rho(x, y) == 0.0


def test_returns_a_plain_float():
    rho = _spearman_rho([1, 2, 3, 4], [1, 3, 2, 4])
    assert type(rho) is float
    assert not math.isnan(rho)
