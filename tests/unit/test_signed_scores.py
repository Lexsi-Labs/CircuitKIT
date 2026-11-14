"""``signed: true`` keeps EAP node scores' signs next to the absolute scores (#47)."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from circuitkit.api import discover_circuit
from circuitkit.backends.eap.graph import Graph
from circuitkit.circuit import Circuit

_TINY = {"n_layers": 2, "n_heads": 2, "d_model": 8, "d_mlp": 16}


def _run(tmp_path, name, **discovery):
    real_from_model = Graph.from_model

    def fake_attribute(model, graph, *args, **kwargs):
        # Alternating signs with distinct magnitudes, so the ranking is unambiguous.
        n = graph.nodes_scores.shape[0]
        graph.nodes_scores[:] = torch.tensor(
            [(i + 1) * (-1) ** i for i in range(n)], dtype=graph.nodes_scores.dtype
        )

    out = tmp_path / f"{name}.pt"
    config = {
        "model": {"name": "gpt2", "device": "cpu"},
        "discovery": {"algorithm": "eap", "task": "ioi", "level": "node", **discovery},
        "pruning": {"target_sparsity": 0.3, "scope": "both"},
        "output_path": str(out),
    }
    with (
        patch("circuitkit.api._get_task"),
        patch.object(Graph, "from_model", lambda model, **kw: real_from_model(_TINY, **kw)),
        patch("circuitkit.api.attribute_node", side_effect=fake_attribute),
    ):
        pruned = discover_circuit(config, _model=MagicMock())
    return pruned, Circuit.from_artifact(out), out


def test_signed_scores_are_extra_and_leave_the_circuit_unchanged(tmp_path):
    pruned_abs, plain, _ = _run(tmp_path, "plain")
    pruned_signed, signed, _ = _run(tmp_path, "signed", signed=True)

    assert pruned_signed == pruned_abs
    assert signed.scores == plain.scores
    assert plain.signed_scores == {}
    assert set(signed.signed_scores) == set(signed.scores)
    assert any(v < 0 for v in signed.signed_scores.values())
    for node, value in signed.signed_scores.items():
        assert abs(value) == pytest.approx(signed.scores[node])

    # Both side-cars carry it, and Circuit.save keeps it through a reload.
    assert torch.load(tmp_path / "signed_scores.pt", weights_only=True)["signed_node_scores"]
    resaved = Circuit.from_artifact(signed.save(tmp_path / "again.pt"))
    assert resaved.signed_scores == pytest.approx(signed.signed_scores)


def test_default_scores_json_has_no_signed_key(tmp_path):
    import json

    _run(tmp_path, "plain")
    assert "signed_node_scores" not in json.loads((tmp_path / "plain_scores.json").read_text())


@pytest.mark.parametrize(
    "discovery, match",
    [({"algorithm": "acdc"}, "EAP family"), ({"level": "neuron"}, "node-level")],
)
def test_signed_rejected_where_it_does_not_apply(tmp_path, discovery, match):
    config = {
        "model": {"name": "gpt2", "device": "cpu"},
        "discovery": {"algorithm": "eap", "task": "ioi", "level": "node", "signed": True},
        "pruning": {"target_sparsity": 0.3},
        "output_path": str(tmp_path / "x.pt"),
    }
    config["discovery"].update(discovery)
    with patch("circuitkit.api._get_task"), pytest.raises(Exception, match=match):
        discover_circuit(config, _model=MagicMock())
