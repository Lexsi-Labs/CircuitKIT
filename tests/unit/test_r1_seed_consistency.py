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
import torch

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


# ---------------------------------------------------------------------------
# run_intervention_reliability: which score vectors feed R1
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch  # noqa: E402

from circuitkit.backends.eap.graph import Graph  # noqa: E402
from circuitkit.evaluation.pillars import intervention_reliability as ir  # noqa: E402

_TINY_CFG = {"n_layers": 2, "n_heads": 2, "d_model": 64, "d_mlp": 128}


def _graph(scores=None):
    """A real Graph. ``scores`` is a function of the node count, or None for no node scores."""
    graph = Graph.from_model(_TINY_CFG, node_scores=scores is not None)
    if scores is not None:
        graph.nodes_scores[:] = scores(graph.nodes_scores.numel())
    return graph


def _run(primary, reruns, algorithm="eap", seeds=None):
    """Run Pillar 7 with only the model-dependent calls replaced."""
    seeds = seeds or [1 + 10 * i for i in range(len(reruns) + 1)]
    rebuilt = iter([(g, {}) for g in reruns])
    with (
        patch.object(ir, "_score_circuit_vs_baseline", return_value=(0.9, 1.0)),
        patch.object(ir, "_rebuild_graph_with_seed", side_effect=lambda *a, **k: next(rebuilt)),
    ):
        return ir.run_intervention_reliability(
            model=MagicMock(),
            graph=primary,
            task_spec=MagicMock(),
            discovery_cfg={"algorithm": algorithm},
            pruning_cfg={},
            device="cpu",
            metric_fn=MagicMock(),
            dataloader=MagicMock(),
            seeds=seeds,
        )


def _ascending(n):
    return torch.arange(n, dtype=torch.float32)


def _descending(n):
    return torch.arange(n, 0, -1, dtype=torch.float32)


def test_has_node_scores():
    assert not ir._has_node_scores(_graph(scores=None))  # from_model default: no tensor
    assert not ir._has_node_scores(Graph.from_model(_TINY_CFG, node_scores=True))  # all NaN
    assert ir._has_node_scores(_graph(_ascending))


def test_primary_without_node_scores_is_left_out_of_r1():
    """Edge-level discovery leaves no node scores; its zero vector must not enter R1."""
    result = _run(_graph(scores=None), [_graph(_ascending), _graph(_ascending)])

    assert result["r1_seed_consistency"] == 1.0  # the two re-runs agree exactly


def test_r1_is_none_and_left_out_of_the_index_with_one_comparable_vector():
    result = _run(_graph(scores=None), [_graph(_ascending)], seeds=[1, 2])

    assert result["r1_seed_consistency"] is None
    r2, r3 = result["r2_effect_magnitude"], result["r3_effect_variance"]
    expected = 2 / (1 / (r2 + 1e-8) + 1 / (r3 + 1e-8))
    assert result["reliability_index"] == pytest.approx(expected, abs=1e-4)


def test_algorithm_the_reruns_cannot_reproduce_is_left_out_of_r1():
    """ibcircuit is re-run with EAP, so its primary scores are a different quantity."""
    result = _run(
        _graph(_descending), [_graph(_ascending), _graph(_ascending)], algorithm="ibcircuit"
    )

    assert result["r1_seed_consistency"] == 1.0  # not dragged down by the opposite primary


def test_mapped_algorithm_keeps_the_primary_vector():
    result = _run(_graph(_descending), [_graph(_ascending), _graph(_ascending)], algorithm="eap")

    # pairs: (primary, rerun1) = -1, (primary, rerun2) = -1, (rerun1, rerun2) = 1
    assert result["r1_seed_consistency"] == pytest.approx(-1 / 3, abs=1e-4)
