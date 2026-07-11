"""Unit tests for the P5 multi-draw random baseline in Pillar5_Baselines.run.

The random baseline must be estimated from N draws with distinct seeds (so
"better than chance" has a distribution), not a single seed=42 sample. These
tests patch the (model-dependent) internals and check the aggregation:
n draws, distinct seeds, mean 'score', std, z-score, and determinism.
"""

from unittest.mock import MagicMock, patch

from circuitkit.evaluation.pillars.baselines import Pillar5_Baselines


def _mock_graph(n_nodes=4):
    g = MagicMock()
    g.nodes = {f"n{i}": MagicMock() for i in range(n_nodes)}
    return g


def _fake_random(
    model, graph, dataloader, metric_fn, circuit_sparsity, device="auto", quiet=False, seed=42
):
    # deterministic, seed-dependent: seeds 42..46 -> 0.10..0.14
    return 0.10 + 0.01 * (seed - 42)


def _run(n_draws):
    with (
        patch.object(Pillar5_Baselines, "_is_neuron_level", return_value=False),
        patch.object(Pillar5_Baselines, "_extract_circuit_nodes", return_value=["n0"]),
        patch.object(Pillar5_Baselines, "_evaluate_circuit", return_value=0.9),
        patch.object(Pillar5_Baselines, "_evaluate_random_baseline", side_effect=_fake_random) as m,
    ):
        result = Pillar5_Baselines.run(
            MagicMock(),
            _mock_graph(),
            MagicMock(),
            MagicMock(),
            baseline_types=["random"],
            n_random_draws=n_draws,
            quiet=True,
        )
    seeds = [c.kwargs.get("seed") for c in m.call_args_list]
    return result, seeds


def test_random_baseline_multi_draw_distribution():
    result, seeds = _run(5)
    rand = result["baselines"]["random"]
    assert seeds == [42, 43, 44, 45, 46], seeds  # N distinct seeds
    assert rand["n_draws"] == 5
    assert len(rand["random_scores"]) == 5
    assert abs(rand["score"] - 0.12) < 1e-9  # mean of 0.10..0.14
    assert rand["score_std"] > 0  # a real distribution
    assert rand["z_score"] is not None and rand["z_score"] > 0  # circuit >> chance
    # backward-compat keys still present
    assert "improvement" in rand and "improvement_valid" in rand


def test_random_baseline_deterministic():
    r1, _ = _run(4)
    r2, _ = _run(4)
    assert r1["baselines"]["random"]["random_scores"] == r2["baselines"]["random"]["random_scores"]
    assert r1["baselines"]["random"]["score"] == r2["baselines"]["random"]["score"]


# ---------------------------------------------------------------------------
# Regression: _sync_edges_to_nodes must REBUILD the edge matrix from the node
# mask, not intersect with the stale (discovered-circuit) matrix.
# ---------------------------------------------------------------------------


def _tiny_real_graph():
    from circuitkit.backends.eap.graph import Graph

    return Graph.from_model({"n_layers": 2, "n_heads": 2, "d_model": 64, "d_mlp": 128})


def test_sync_edges_rebuilds_node_induced_subgraph():
    """A baseline over ALL nodes must reproduce the FULL real-edge set.

    The cloned baseline graph carries the discovered circuit's edge matrix.
    The old `graph.in_graph *= edge_mask` could only remove edges, so a
    random/magnitude/wanda baseline collapsed to a subset of the circuit's
    own edges — here, 2 edges instead of every structurally valid edge —
    understating baseline scores and overstating circuit_advantage/z-score.
    """
    g = _tiny_real_graph()
    # Simulate the cloned state: a tiny "discovered circuit" of 2 edges.
    stale = g.real_edge_mask.nonzero()[:2]
    for i, j in stale.tolist():
        g.in_graph[i, j] = True
    # Baseline builder selects ALL nodes...
    g.nodes_in_graph[:] = True

    Pillar5_Baselines._sync_edges_to_nodes(g)

    n_edges = int(g.in_graph.sum())
    n_real = int(g.real_edge_mask.sum())
    assert n_edges == n_real, (
        f"expected the full node-induced subgraph ({n_real} edges), got {n_edges} "
        f"(collapse to the stale circuit edges = the pre-fix bug)"
    )


def test_sync_edges_recovers_from_empty_matrix():
    """Even from an all-empty edge matrix (multiplication's absorbing state),
    the sync must produce the node-induced subgraph."""
    g = _tiny_real_graph()  # in_graph starts all False
    g.nodes_in_graph[:] = True

    Pillar5_Baselines._sync_edges_to_nodes(g)

    assert int(g.in_graph.sum()) == int(g.real_edge_mask.sum())
    # And never invents structurally invalid edges:
    assert bool((g.in_graph & ~g.real_edge_mask).sum() == 0)
