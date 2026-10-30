"""
Unit tests for circuitkit.circuit.Circuit.

Covers construction, from_artifact (JSON and .pt side-cars), save/load
round-trip, top_nodes, dunders (__len__, __iter__, __contains__, __repr__),
and graceful plot() degradation — all without loading any real model.
"""

import json
import sys
import warnings
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

    def test_pt_sidecar_metadata_gives_the_graph_its_title(self, tmp_path):
        """A neuron-level run writes no scores JSON, only task, model and algo in
        the .pt side-car; a side-car without them (written before CK-26) is
        labelled unknown, never as eap-ig. Non-string values are ignored."""
        new_pt = tmp_path / "new.pt"
        _write_artifact(new_pt)
        torch.save(
            {"node_scores": _SCORES, "algo": "ibcircuit", "task": "ioi", "model": "gpt2"},
            tmp_path / "new_scores.pt",
        )
        c = Circuit.from_artifact(new_pt)
        assert (c.algorithm, c.task, c.model_name) == ("ibcircuit", "ioi", "gpt2")
        assert "ioi / ibcircuit" in c.plot(str(tmp_path / "new.html"))

        old_pt = tmp_path / "old.pt"
        _write_artifact(old_pt)
        torch.save(
            {"node_scores": _SCORES, "algo": 3, "task": None, "level": "neuron"},
            tmp_path / "old_scores.pt",
        )
        c = Circuit.from_artifact(old_pt)
        assert (c.algorithm, c.task, c.model_name) == (None, None, None)
        html = c.plot(str(tmp_path / "old.html"))
        assert "unknown / unknown" in html
        assert "eap-ig" not in html

    def test_json_preferred_over_pt_when_both_exist(self, tmp_path):
        """JSON side-car is checked before .pt side-car."""
        pt = tmp_path / "circuit.pt"
        _write_artifact(pt)
        _write_json_sidecar(pt, scores={"A0.1": 0.11})  # JSON
        _write_pt_sidecar(pt, scores={"A0.1": 0.99})  # PT
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
        assert keys[0] == "A0.1"  # score 0.9 — highest
        assert keys[1] == "A0.2"  # score 0.5

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
    def test_plot_with_no_scores_warns_and_returns_none(self, tmp_path):
        """A circuit without node scores is a RuntimeWarning the caller sees, even
        with the UserWarning filter importing circuitkit.api installs (CK-25), and
        plot() still returns None without writing a file."""
        c = Circuit(_NODES)  # no scores
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=UserWarning)  # as circuitkit.api does
            result = c.plot(str(tmp_path / "circuit.html"))
        assert result is None
        assert not (tmp_path / "circuit.html").exists()
        no_scores = [w for w in caught if "no node scores" in str(w.message).lower()]
        assert len(no_scores) == 1
        assert no_scores[0].category is RuntimeWarning
        assert "output_path" in str(no_scores[0].message)

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

    def test_plot_json_path_dispatches_to_json_with_bounds(self, tmp_path):
        """A .json output_path must call CircuitGraphVisualizer.to_json (not
        to_html), forwarding max_nodes/max_edges/edge_threshold (PR-L2)."""
        c = Circuit(_NODES, dict(_SCORES))
        out_path = tmp_path / "circuit.json"
        mock_viz = MagicMock()
        mock_viz.to_json.return_value = {"nodes": [], "edges": []}
        with patch("circuitkit.visualize.graph_viz.CircuitGraphVisualizer", return_value=mock_viz):
            result = c.plot(str(out_path), max_nodes=300, max_edges=3000, edge_threshold=0.1)

        mock_viz.to_json.assert_called_once_with(
            str(out_path), max_nodes=300, max_edges=3000, edge_threshold=0.1
        )
        mock_viz.to_html.assert_not_called()
        assert result == {"nodes": [], "edges": []}

    def test_plot_without_viz_stack_warns_and_returns_none(self, tmp_path, monkeypatch):
        """A missing visualisation stack is a RuntimeWarning the caller sees, even
        with the UserWarning filter importing circuitkit.api installs (CK-25), and
        plot() still returns None without raising."""
        c = Circuit(_NODES, dict(_SCORES))
        # A None entry makes `from .visualize.graph_viz import ...` raise ImportError.
        monkeypatch.setitem(sys.modules, "circuitkit.visualize.graph_viz", None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=UserWarning)  # as circuitkit.api does
            result = c.plot(str(tmp_path / "circuit.html"))

        assert result is None
        assert not (tmp_path / "circuit.html").exists()
        plot_warnings = [w for w in caught if "Plotting unavailable" in str(w.message)]
        assert len(plot_warnings) == 1
        assert plot_warnings[0].category is RuntimeWarning
        assert "ModuleNotFoundError" in str(plot_warnings[0].message)


class TestPlotNeuronLevelPrunedComponents:
    def test_only_fully_pruned_components_are_pruned(self, tmp_path):
        """A neuron-level component is pruned only when all of its neurons are
        (CK-26), in both naming styles; an artifact written before the neuron
        counts were recorded marks none and still plots."""
        cases = [
            # EAP / EAP-IG names. m0 and a0.h0 are fully pruned; m1 and a0.h1 are not.
            (
                {"m0": 0.5, "m1": 0.4, "a0.h0": 0.3, "a0.h1": 0.2},
                {"mlp": {0: [0, 1, 2, 3], 1: [0]}, "heads": {(0, 0): [0, 1], (0, 1): [0]}},
                {"m0", "a0.h0"},
            ),
            # IBCircuit names, same pruning.
            (
                {"MLP 0": 0.5, "MLP 1": 0.4, "A0.0": 0.3, "A0.1": 0.2},
                {"mlp": {0: [0, 1, 2, 3], 1: [0]}, "heads": {(0, 0): [0, 1], (0, 1): [0]}},
                {"MLP 0", "A0.0"},
            ),
        ]
        for i, (scores, pruned, expected) in enumerate(cases):
            meta = {"mlp_hook": "mlp_out", "neurons_per_mlp": 4, "neurons_per_head": 2}
            nodes = {**pruned, "_meta": meta}
            # A real save/load round trip: the head keys are (layer, head) tuples.
            Circuit(nodes, scores, level="neuron").save(tmp_path / f"c{i}.pt")
            circuit = Circuit.from_artifact(tmp_path / f"c{i}.pt")
            out = circuit.plot(str(tmp_path / f"c{i}.json"))
            assert {n["id"] for n in out["nodes"] if not n["in_circuit"]} == expected

            # Same artifact without the neuron counts: nothing is pruned.
            legacy = {**pruned, "_meta": {"mlp_hook": "mlp_out"}}
            out = Circuit(legacy, scores, level="neuron").plot(str(tmp_path / f"l{i}.json"))
            assert all(n["in_circuit"] for n in out["nodes"])
