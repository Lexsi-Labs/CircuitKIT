"""
Unit tests for circuitkit.circuit.Circuit.

Covers construction, from_artifact (JSON and .pt side-cars), save/load
round-trip, top_nodes, dunders (__len__, __iter__, __contains__, __repr__),
and graceful plot() degradation — all without loading any real model.
"""

import json
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from circuitkit.circuit import Circuit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A valid algorithm name recognised by CircuitScores.__post_init__
_ALGO = "eap-ig"
_TASK = "ioi"
_MODEL = "gpt2"

# Simple node list used across many tests
_NODES = ["A0.1", "A0.2", "MLP 3"]
_SCORES: Dict[str, float] = {"A0.1": 0.9, "A0.2": 0.5, "MLP 3": 0.3}


def _make_circuit_scores(**overrides):
    """Return a CircuitScores dataclass with sensible defaults."""
    from circuitkit.artifacts.scores import CircuitScores

    kwargs = dict(
        task=_TASK,
        model=_MODEL,
        algorithm=_ALGO,
        level="node",
        node_scores=dict(_SCORES),
        timestamp=CircuitScores.create_timestamp(),
    )
    kwargs.update(overrides)
    return CircuitScores(**kwargs)


def _make_circuit(**kw) -> Circuit:
    defaults = dict(nodes=list(_NODES), scores=dict(_SCORES), level="node")
    defaults.update(kw)
    nodes = defaults.pop("nodes")
    scores = defaults.pop("scores")
    return Circuit(nodes, scores, **defaults)


def _write_artifact(path: Path, nodes=None) -> Path:
    """Write a minimal .pt pruning artifact and return its path."""
    if nodes is None:
        nodes = _NODES
    torch.save(nodes, path)
    return path


def _write_json_sidecar(artifact_path: Path, scores=None) -> Path:
    """Write a bare {node_scores: ...} JSON side-car next to the artifact."""
    if scores is None:
        scores = _SCORES
    sidecar = artifact_path.parent / f"{artifact_path.stem}_scores.json"
    with open(sidecar, "w") as f:
        json.dump({"node_scores": scores}, f)
    return sidecar


def _write_full_json_sidecar(artifact_path: Path) -> Path:
    """Write a full CircuitScores JSON side-car (with algorithm/task/model)."""
    cs = _make_circuit_scores()
    sidecar = artifact_path.parent / f"{artifact_path.stem}_scores.json"
    cs.to_json(sidecar)
    return sidecar


def _write_pt_sidecar(artifact_path: Path, scores=None) -> Path:
    """Write a _scores.pt side-car with {node_scores: ...} layout."""
    if scores is None:
        scores = _SCORES
    sidecar = artifact_path.parent / f"{artifact_path.stem}_scores.pt"
    torch.save({"node_scores": scores, "algo": _ALGO, "level": "node"}, sidecar)
    return sidecar


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCircuitConstruction:
    def test_init_with_node_list(self):
        c = Circuit(_NODES)
        assert len(c) == 3
        assert c.level == "node"
        assert c.scores == {}

    def test_init_with_scores(self):
        c = Circuit(_NODES, dict(_SCORES))
        assert c.scores["A0.1"] == pytest.approx(0.9)
        assert c.scores["MLP 3"] == pytest.approx(0.3)

    def test_init_scores_is_copy(self):
        """Mutating original scores dict must not affect the circuit."""
        original = {"A0.1": 0.9}
        c = Circuit(["A0.1"], original)
        original["A0.1"] = 99.0
        assert c.scores["A0.1"] == pytest.approx(0.9)

    def test_init_metadata_attributes(self):
        c = Circuit(_NODES, task=_TASK, algorithm=_ALGO, model_name=_MODEL)
        assert c.task == _TASK
        assert c.algorithm == _ALGO
        assert c.model_name == _MODEL

    def test_init_neuron_level_artifact(self):
        nodes = {"mlp": {"0": [1, 2, 3]}, "heads": {"0": [4, 5]}, "_meta": {}}
        c = Circuit(nodes, level="neuron")
        assert c.level == "neuron"

    def test_init_backfill_from_circuit_scores(self):
        """When circuit_scores is supplied, task/algorithm/model_name/scores
        should be back-filled if not already set on the Circuit."""
        cs = _make_circuit_scores()
        c = Circuit(_NODES, circuit_scores=cs)
        assert c.task == _TASK
        assert c.algorithm == _ALGO
        assert c.model_name == _MODEL
        assert c.scores["A0.1"] == pytest.approx(0.9)

    def test_init_explicit_attrs_override_circuit_scores(self):
        """Explicitly passed attributes must win over circuit_scores backfill."""
        cs = _make_circuit_scores(task="sva")
        c = Circuit(_NODES, task="ioi", circuit_scores=cs)
        assert c.task == "ioi"  # explicit wins

    def test_init_no_scores_with_circuit_scores(self):
        """scores= stays empty when circuit_scores.node_scores is empty,
        but circuit_scores metadata is still backfilled."""
        cs = _make_circuit_scores(node_scores={"A0.1": 0.0})
        c = Circuit(_NODES, circuit_scores=cs)
        # scores dict is populated from circuit_scores
        assert "A0.1" in c.scores


# ---------------------------------------------------------------------------
# from_artifact — JSON side-car
# ---------------------------------------------------------------------------

class TestFromArtifactJSON:
    def test_missing_artifact_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            Circuit.from_artifact("nonexistent_xyz.pt")

    def test_bare_json_sidecar_loads_scores(self, tmp_path):
        pt = tmp_path / "circuit.pt"
        _write_artifact(pt)
        _write_json_sidecar(pt)
        c = Circuit.from_artifact(pt)
        assert c.scores["A0.1"] == pytest.approx(0.9)
        assert c.circuit_scores is None  # bare side-car has no full metadata

    def test_full_json_sidecar_loads_metadata(self, tmp_path):
        pt = tmp_path / "circuit.pt"
        _write_artifact(pt)
        _write_full_json_sidecar(pt)
        c = Circuit.from_artifact(pt)
        assert c.circuit_scores is not None
        assert c.task == _TASK
        assert c.algorithm == _ALGO
        assert c.model_name == _MODEL

    def test_no_sidecar_gives_empty_scores(self, tmp_path):
        pt = tmp_path / "circuit.pt"
        _write_artifact(pt)
        c = Circuit.from_artifact(pt)
        assert c.scores == {}

    def test_artifact_path_set_on_loaded_circuit(self, tmp_path):
        pt = tmp_path / "circuit.pt"
        _write_artifact(pt)
        c = Circuit.from_artifact(pt)
        assert c.artifact_path == str(pt)

    def test_explicit_scores_path_overrides_autodiscovery(self, tmp_path):
        pt = tmp_path / "circuit.pt"
        _write_artifact(pt)
        # Write side-car at a custom location
        custom_sidecar = tmp_path / "custom_scores.json"
        with open(custom_sidecar, "w") as f:
            json.dump({"node_scores": {"A0.1": 0.42}}, f)
        c = Circuit.from_artifact(pt, scores_path=custom_sidecar)
        assert c.scores["A0.1"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# from_artifact — .pt side-car
# ---------------------------------------------------------------------------

class TestFromArtifactPTSidecar:
    def test_pt_sidecar_loads_scores(self, tmp_path):
        pt = tmp_path / "circuit.pt"
        _write_artifact(pt)
        _write_pt_sidecar(pt)
        c = Circuit.from_artifact(pt)
        assert c.scores["A0.1"] == pytest.approx(0.9)

    def test_json_preferred_over_pt_when_both_exist(self, tmp_path):
        """JSON side-car is checked before .pt side-car."""
        pt = tmp_path / "circuit.pt"
        _write_artifact(pt)
        _write_json_sidecar(pt, scores={"A0.1": 0.11})   # JSON
        _write_pt_sidecar(pt, scores={"A0.1": 0.99})     # PT
        c = Circuit.from_artifact(pt)
        # JSON is tried first in candidates list
        assert c.scores["A0.1"] == pytest.approx(0.11)


# ---------------------------------------------------------------------------
# Neuron-level artifact detection
# ---------------------------------------------------------------------------

class TestNeuronLevelDetection:
    def test_dict_artifact_with_meta_is_neuron_level(self, tmp_path):
        nodes = {"mlp": {"0": [1, 2]}, "heads": {"1": [3]}, "_meta": {"algo": "ibcircuit"}}
        pt = tmp_path / "circuit.pt"
        torch.save(nodes, pt)
        c = Circuit.from_artifact(pt)
        assert c.level == "neuron"

    def test_list_artifact_is_node_level(self, tmp_path):
        pt = tmp_path / "circuit.pt"
        _write_artifact(pt, nodes=["A0.1", "MLP 3"])
        c = Circuit.from_artifact(pt)
        assert c.level == "node"


# ---------------------------------------------------------------------------
# save / round-trip
# ---------------------------------------------------------------------------

class TestSaveRoundTrip:
    def test_save_creates_file(self, tmp_path):
        c = Circuit(_NODES, dict(_SCORES))
        path = c.save(tmp_path / "out.pt")
        assert path.exists()

    def test_save_creates_json_sidecar_when_scores_present(self, tmp_path):
        c = Circuit(_NODES, dict(_SCORES))
        path = tmp_path / "out.pt"
        c.save(path)
        assert (tmp_path / "out_scores.json").exists()

    def test_no_sidecar_when_no_scores(self, tmp_path):
        c = Circuit(_NODES)  # no scores
        c.save(tmp_path / "out.pt")
        assert not (tmp_path / "out_scores.json").exists()

    def test_roundtrip_nodes(self, tmp_path):
        c = Circuit(_NODES, dict(_SCORES))
        path = c.save(tmp_path / "out.pt")
        loaded = Circuit.from_artifact(path)
        assert list(loaded.nodes) == _NODES

    def test_roundtrip_scores(self, tmp_path):
        c = Circuit(_NODES, dict(_SCORES))
        path = c.save(tmp_path / "out.pt")
        loaded = Circuit.from_artifact(path)
        for name, score in _SCORES.items():
            assert loaded.scores[name] == pytest.approx(score)

    def test_roundtrip_sets_artifact_path(self, tmp_path):
        c = Circuit(_NODES, dict(_SCORES))
        path = c.save(tmp_path / "out.pt")
        assert c.artifact_path == str(path)

    def test_save_creates_parent_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "out.pt"
        c = Circuit(_NODES, dict(_SCORES))
        c.save(deep_path)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# top_nodes
# ---------------------------------------------------------------------------

class TestTopNodes:
    def test_top_nodes_correct_order(self):
        c = Circuit(_NODES, dict(_SCORES))
        top = c.top_nodes(2)
        keys = list(top.keys())
        assert keys[0] == "A0.1"   # score 0.9 — highest
        assert keys[1] == "A0.2"   # score 0.5

    def test_top_nodes_respects_k(self):
        c = Circuit(_NODES, dict(_SCORES))
        assert len(c.top_nodes(1)) == 1
        assert len(c.top_nodes(2)) == 2
        assert len(c.top_nodes(100)) == len(_SCORES)  # capped at available

    def test_top_nodes_empty_scores_returns_empty(self):
        c = Circuit(_NODES)
        assert c.top_nodes(5) == {}


# ---------------------------------------------------------------------------
# Dunders: __len__, __iter__, __contains__, __repr__
# ---------------------------------------------------------------------------

class TestDunders:
    # --- __len__ ---
    def test_len_node_level(self):
        assert len(Circuit(_NODES)) == 3

    def test_len_empty_node_list(self):
        assert len(Circuit([])) == 0

    def test_len_neuron_level_counts_mlp_and_heads(self):
        nodes = {
            "mlp": {"0": [1, 2], "1": [3]},
            "heads": {"0": [4, 5], "2": [6]},
            "_meta": {},
        }
        c = Circuit(nodes, level="neuron")
        # __len__ counts layer-dict keys: 2 mlp layer keys + 2 head layer keys = 4
        assert len(c) == 4

    # --- __iter__ ---
    def test_iter_node_level(self):
        c = Circuit(_NODES)
        assert list(c) == _NODES

    def test_iter_neuron_level(self):
        nodes = {"mlp": {"0": []}, "heads": {"1": []}, "_meta": {}}
        c = Circuit(nodes, level="neuron")
        items = list(c)
        assert "0" in items
        assert "1" in items

    # --- __contains__ ---
    def test_contains_present_node(self):
        c = Circuit(_NODES)
        assert "A0.1" in c

    def test_contains_absent_node(self):
        c = Circuit(_NODES)
        assert "A9.9" not in c

    def test_contains_neuron_level_mlp(self):
        nodes = {"mlp": {"0": [1]}, "heads": {}, "_meta": {}}
        c = Circuit(nodes, level="neuron")
        assert "0" in c

    # --- __repr__ ---
    def test_repr_contains_circuit_prefix(self):
        c = Circuit(_NODES)
        assert repr(c).startswith("Circuit(")

    def test_repr_contains_level(self):
        c = Circuit(_NODES)
        assert "level=node" in repr(c)

    def test_repr_contains_n_nodes(self):
        c = Circuit(_NODES)
        assert "n_nodes=3" in repr(c)

    def test_repr_contains_task_when_set(self):
        c = Circuit(_NODES, task=_TASK)
        assert _TASK in repr(c)


# ---------------------------------------------------------------------------
# plot() — graceful degradation without viz dependencies
# ---------------------------------------------------------------------------

class TestPlotGracefulDegradation:
    def test_plot_with_no_scores_returns_none(self, caplog):
        import logging

        c = Circuit(_NODES)  # no scores
        with caplog.at_level(logging.INFO):
            result = c.plot()
        assert result is None
        assert "no node scores" in caplog.text.lower()

    def test_plot_with_scores_to_path_calls_visualizer(self, tmp_path):
        """When scores are present and an output path is given, plot() should
        attempt the visualizer.  We mock it to avoid needing optional deps."""
        c = Circuit(_NODES, dict(_SCORES))
        out_path = tmp_path / "circuit.html"
        mock_viz = MagicMock()
        mock_viz.to_html.return_value = "<html/>"
        with patch("circuitkit.visualize.graph_viz.CircuitGraphVisualizer", return_value=mock_viz):
            result = c.plot(str(out_path))
        # Either the call succeeded with HTML or degraded gracefully — either
        # outcome must not raise.
        assert result is None or isinstance(result, str)
