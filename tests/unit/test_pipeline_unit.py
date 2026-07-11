"""
Unit tests for circuitkit.pipeline.Pipeline.

All tests operate without loading a real model.  Heavy operations
(api.discover_circuit, api.evaluate_circuit, quick.*) are patched at the
point of import inside pipeline.py — i.e. at their canonical paths so that
Pipeline's local `from .api import ...` picks up the mock.

Covers: __init__, alternative constructors, _require_circuit, discover
(validation + CDT warning + history), evaluate guards, evaluate_advanced,
prune/quantize/export error paths, selective_finetune, method chaining,
_resolve_pillars, _build_eval_config, summary, __repr__.
"""

import json
import warnings
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from circuitkit.circuit import Circuit
from circuitkit.pipeline import Pipeline, _resolve_pillars

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_NODES = ["A0.1", "A0.2", "MLP 3"]
_SCORES: Dict[str, float] = {"A0.1": 0.9, "A0.2": 0.5, "MLP 3": 0.3}


def _make_circuit(**kw) -> Circuit:
    defaults = dict(nodes=list(_NODES), scores=dict(_SCORES), level="node")
    defaults.update(kw)
    nodes = defaults.pop("nodes")
    scores = defaults.pop("scores")
    return Circuit(nodes, scores, **defaults)


def _write_artifact(tmp_path: Path) -> Path:
    """Write a minimal .pt + _scores.json and return the artifact path."""
    pt = tmp_path / "circuit.pt"
    torch.save(list(_NODES), pt)
    sidecar = tmp_path / "circuit_scores.json"
    with open(sidecar, "w") as f:
        json.dump({"node_scores": _SCORES}, f)
    return pt


@pytest.fixture()
def pipe():
    return Pipeline("gpt2", task="ioi", precision="float32")


@pytest.fixture()
def pipe_with_circuit(pipe, tmp_path):
    """Pipeline with a pre-loaded circuit (no real discovery)."""
    pipe._circuit = _make_circuit(artifact_path=str(tmp_path / "circuit.pt"))
    pipe._artifact_path = str(tmp_path / "circuit.pt")
    return pipe


# ---------------------------------------------------------------------------
# __init__: constructor arguments
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_stores_model_name(self):
        p = Pipeline("gpt2")
        assert p.model_name == "gpt2"

    def test_stores_task(self):
        p = Pipeline("gpt2", task="ioi")
        assert p.task == "ioi"

    def test_stores_precision(self):
        p = Pipeline("gpt2", precision="float32")
        assert p.precision == "float32"

    def test_default_precision_is_bfloat16(self):
        p = Pipeline("gpt2")
        assert p.precision == "bfloat16"

    def test_default_output_dir(self):
        p = Pipeline("gpt2")
        assert p.output_dir == "./pipeline_output"

    def test_custom_output_dir(self):
        p = Pipeline("gpt2", output_dir="/tmp/mydir")
        assert p.output_dir == "/tmp/mydir"

    def test_initial_circuit_is_none(self):
        assert Pipeline("gpt2")._circuit is None

    def test_initial_history_is_empty(self):
        assert Pipeline("gpt2")._history == []

    def test_initial_pruned_model_is_none(self):
        assert Pipeline("gpt2")._pruned_model is None

    def test_task_defaults_to_none(self):
        assert Pipeline("gpt2").task is None


# ---------------------------------------------------------------------------
# from_artifact constructor
# ---------------------------------------------------------------------------

class TestFromArtifact:
    def test_populates_circuit(self, tmp_path):
        pt = _write_artifact(tmp_path)
        p = Pipeline.from_artifact(pt, "gpt2")
        assert p._circuit is not None
        assert isinstance(p._circuit, Circuit)

    def test_sets_artifact_path(self, tmp_path):
        pt = _write_artifact(tmp_path)
        p = Pipeline.from_artifact(pt, "gpt2")
        assert p._artifact_path == str(pt)

    def test_history_contains_from_artifact(self, tmp_path):
        pt = _write_artifact(tmp_path)
        p = Pipeline.from_artifact(pt, "gpt2")
        assert "from_artifact" in p._history

    def test_task_resolved_from_circuit_metadata(self, tmp_path):
        # Write a full CircuitScores JSON so task is baked into the artifact
        from circuitkit.artifacts.scores import CircuitScores
        pt = tmp_path / "circuit.pt"
        torch.save(list(_NODES), pt)
        cs = CircuitScores(
            task="sva", model="gpt2", algorithm="eap-ig", level="node",
            node_scores={k: v for k, v in _SCORES.items()},
            timestamp=CircuitScores.create_timestamp(),
        )
        cs.to_json(tmp_path / "circuit_scores.json")
        p = Pipeline.from_artifact(pt, "gpt2")
        assert p.task == "sva"

    def test_explicit_task_overrides_circuit_metadata(self, tmp_path):
        pt = _write_artifact(tmp_path)
        p = Pipeline.from_artifact(pt, "gpt2", task="ioi")
        assert p.task == "ioi"


# ---------------------------------------------------------------------------
# from_custom_data constructor
# ---------------------------------------------------------------------------

class TestFromCustomData:
    def test_stores_custom_data_cfg(self):
        p = Pipeline.from_custom_data(
            "gpt2", "data.csv",
            clean_prompt="{question}", clean_answer="{answer}",
        )
        assert p._custom_data_cfg is not None
        assert p._custom_data_cfg["template"]["clean_prompt"] == "{question}"

    def test_task_name_auto_derived_from_stem(self):
        p = Pipeline.from_custom_data(
            "gpt2", "/some/path/my_dataset.csv",
            clean_prompt="{q}", clean_answer="{a}",
        )
        assert p.task == "custom:my_dataset"

    def test_explicit_task_name_used(self):
        p = Pipeline.from_custom_data(
            "gpt2", "data.csv",
            clean_prompt="{q}", clean_answer="{a}",
            task_name="custom:myexperiment",
        )
        assert p.task == "custom:myexperiment"
        assert p._task_name_override == "custom:myexperiment"

    def test_corrupt_pair_stored_when_provided(self):
        p = Pipeline.from_custom_data(
            "gpt2", "data.csv",
            clean_prompt="{q}", clean_answer="{a}",
            corrupt_prompt="{bad_q}", corrupt_answer="{bad_a}",
        )
        tmpl = p._custom_data_cfg["template"]
        assert "corrupt_prompt" in tmpl
        assert "corrupt_answer" in tmpl

    def test_no_corrupt_pair_when_omitted(self):
        p = Pipeline.from_custom_data(
            "gpt2", "data.csv",
            clean_prompt="{q}", clean_answer="{a}",
        )
        tmpl = p._custom_data_cfg["template"]
        assert "corrupt_prompt" not in tmpl

    def test_history_contains_from_custom_data(self):
        p = Pipeline.from_custom_data(
            "gpt2", "data.csv", clean_prompt="{q}", clean_answer="{a}",
        )
        assert "from_custom_data" in p._history


# ---------------------------------------------------------------------------
# _require_circuit
# ---------------------------------------------------------------------------

class TestRequireCircuit:
    def test_raises_before_discover(self, pipe):
        with pytest.raises(RuntimeError, match="discover"):
            pipe._require_circuit("prune")

    def test_error_message_includes_caller_name(self, pipe):
        with pytest.raises(RuntimeError, match="my_method"):
            pipe._require_circuit("my_method")

    def test_no_raise_when_circuit_present(self, pipe):
        pipe._circuit = _make_circuit()
        pipe._require_circuit("prune")  # must not raise


# ---------------------------------------------------------------------------
# _resolve_pillars (module-level helper)
# ---------------------------------------------------------------------------

class TestResolvePillars:
    def test_none_returns_none(self):
        assert _resolve_pillars(None) is None

    def test_all_string_returns_none(self):
        assert _resolve_pillars("all") is None

    def test_int_list_maps_to_names(self):
        # Integers 1-6 resolve to their canonical pillar names.
        assert _resolve_pillars([1, 3, 5]) == ["patching", "stability", "baselines"]

    def test_name_list_passthrough(self):
        # String names pass through unchanged (validated downstream).
        assert _resolve_pillars(["patching", "ablation"]) == ["patching", "ablation"]

    def test_all_pillar_ids(self):
        expected = ["patching", "ablation", "stability", "robustness", "baselines", "generalization"]
        assert _resolve_pillars([1, 2, 3, 4, 5, 6]) == expected

    def test_mixed_int_and_name(self):
        assert _resolve_pillars([1, "ablation", 3]) == ["patching", "ablation", "stability"]

    def test_unnumbered_name_passes_through(self):
        # Pillar 7 has no number; it must survive normalisation for the
        # evaluator to validate it.
        assert _resolve_pillars(["intervention_reliability"]) == ["intervention_reliability"]

    def test_unknown_int_raises(self):
        with pytest.raises(ValueError, match="Unknown pillar id"):
            _resolve_pillars([7])

    def test_bool_raises(self):
        with pytest.raises(TypeError):
            _resolve_pillars([True])

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            _resolve_pillars([3.14])


# ---------------------------------------------------------------------------
# discover() — validation guards (mocked api.discover_circuit)
# ---------------------------------------------------------------------------

class TestDiscoverValidation:
    _MOCK_RETURN = list(_NODES)

    def test_no_task_raises(self):
        p = Pipeline("gpt2")  # no task
        with pytest.raises(ValueError, match="task must be set"):
            with patch("circuitkit.api.discover_circuit"):
                p.discover()

    def test_cdt_with_neuron_level_emits_warning(self, tmp_path):
        p = Pipeline("gpt2", task="ioi", output_dir=str(tmp_path))
        with patch("circuitkit.api.discover_circuit", return_value=self._MOCK_RETURN), \
             patch("circuitkit.api.prepare_custom_task"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                p.discover(algorithm="cdt", level="neuron")
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1
        assert "node" in str(user_warnings[0].message).lower()

    def test_cdt_neuron_level_forces_node(self, tmp_path):
        p = Pipeline("gpt2", task="ioi", output_dir=str(tmp_path))
        with patch("circuitkit.api.discover_circuit", return_value=self._MOCK_RETURN), \
             patch("circuitkit.api.prepare_custom_task"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                p.discover(algorithm="cdt", level="neuron")
        assert p._circuit.level == "node"

    def test_discover_appends_to_history(self, tmp_path):
        p = Pipeline("gpt2", task="ioi", output_dir=str(tmp_path))
        with patch("circuitkit.api.discover_circuit", return_value=self._MOCK_RETURN):
            p.discover()
        assert "discover" in p._history

    def test_discover_stores_circuit(self, tmp_path):
        p = Pipeline("gpt2", task="ioi", output_dir=str(tmp_path))
        with patch("circuitkit.api.discover_circuit", return_value=self._MOCK_RETURN):
            p.discover()
        assert p._circuit is not None

    def test_seed_propagates_into_data_params(self, tmp_path):
        """discover(seed=s) must reach discovery.data_params.seed.

        Regression: the seed used to land only at discovery.seed (top level),
        while data generation (e.g. IOI) reads discovery.data_params.seed and
        silently defaulted to 42 — so multi-seed runs produced identical
        circuits and zero-variance error bars.
        """
        p = Pipeline("gpt2", task="ioi", output_dir=str(tmp_path))
        with patch(
            "circuitkit.api.discover_circuit", return_value=self._MOCK_RETURN
        ) as mock_dc:
            p.discover(seed=7)
        cfg = mock_dc.call_args[0][0]
        assert cfg["discovery"]["data_params"]["seed"] == 7
        assert cfg["discovery"]["seed"] == 7

    def test_no_seed_leaves_data_params_seed_unset(self, tmp_path):
        """Omitting seed must not inject one — data generation keeps its default."""
        p = Pipeline("gpt2", task="ioi", output_dir=str(tmp_path))
        with patch(
            "circuitkit.api.discover_circuit", return_value=self._MOCK_RETURN
        ) as mock_dc:
            p.discover()
        cfg = mock_dc.call_args[0][0]
        assert "seed" not in cfg["discovery"]["data_params"]
        assert "seed" not in cfg["discovery"]

    def test_discover_returns_self_for_chaining(self, tmp_path):
        p = Pipeline("gpt2", task="ioi", output_dir=str(tmp_path))
        with patch("circuitkit.api.discover_circuit", return_value=self._MOCK_RETURN):
            result = p.discover()
        assert result is p


# ---------------------------------------------------------------------------
# evaluate() guards
# ---------------------------------------------------------------------------

class TestEvaluateGuards:
    def test_requires_circuit(self, pipe):
        with pytest.raises(RuntimeError, match="discover"):
            pipe.evaluate()

    def test_requires_existing_artifact_file(self, pipe):
        pipe._circuit = _make_circuit()
        pipe._artifact_path = "/nonexistent/path/circuit.pt"
        with pytest.raises(RuntimeError, match="artifact"):
            pipe.evaluate()


# ---------------------------------------------------------------------------
# evaluate_advanced() mode validation
# ---------------------------------------------------------------------------

class TestEvaluateAdvanced:
    def test_unknown_mode_raises(self, pipe):
        with pytest.raises(ValueError, match="Unknown evaluate_advanced mode"):
            pipe.evaluate_advanced(mode="bogus_mode_xyz")


# ---------------------------------------------------------------------------
# Applications — error paths (no model needed)
# ---------------------------------------------------------------------------

class TestApplicationErrorPaths:
    def test_prune_without_circuit_raises(self, pipe):
        with pytest.raises(RuntimeError, match="discover"):
            pipe.prune()

    def test_quantize_without_circuit_raises(self, pipe):
        with pytest.raises(RuntimeError, match="discover"):
            pipe.quantize()

    def test_selective_finetune_without_circuit_raises(self, pipe):
        with pytest.raises(RuntimeError, match="discover"):
            pipe.selective_finetune()

    def test_export_without_pruned_model_raises(self, pipe_with_circuit):
        """export() must raise if prune()/quantize() has not been called yet."""
        with pytest.raises(RuntimeError, match="prune"):
            pipe_with_circuit.export("some/path")

    def test_visualize_without_circuit_raises(self, pipe):
        with pytest.raises(RuntimeError, match="discover"):
            pipe.visualize()


# ---------------------------------------------------------------------------
# _build_eval_config
# ---------------------------------------------------------------------------

class TestBuildEvalConfig:
    def test_uses_discovery_cfg_when_present(self, pipe):
        pipe._discovery_cfg = {
            "model": {"name": "gpt2", "precision": "float32"},
            "discovery": {"algorithm": "eap-ig", "task": "ioi", "level": "node",
                          "batch_size": 4, "data_params": {"num_examples": 128}},
            "pruning": {"target_sparsity": 0.3, "scope": "both"},
            "output_path": "out.pt",
        }
        result = pipe._build_eval_config({"pillars": [1, 2]})
        assert result["eval"] == {"pillars": [1, 2]}
        assert result["model"]["name"] == "gpt2"

    def test_deep_copies_discovery_cfg(self, pipe):
        pipe._discovery_cfg = {
            "model": {"name": "gpt2", "precision": "float32"},
            "discovery": {},
            "pruning": {},
            "output_path": "",
        }
        result = pipe._build_eval_config({})
        result["model"]["name"] = "MUTATED"
        # Original must be unchanged
        assert pipe._discovery_cfg["model"]["name"] == "gpt2"

    def test_fallback_when_no_discovery_cfg(self, pipe):
        pipe._circuit = _make_circuit()
        result = pipe._build_eval_config({})
        assert "model" in result
        assert result["model"]["name"] == "gpt2"

    def test_eval_key_always_present(self, pipe):
        pipe._circuit = _make_circuit()
        result = pipe._build_eval_config({"pillars": [3]})
        assert "eval" in result


# ---------------------------------------------------------------------------
# prune() method chaining and history
# ---------------------------------------------------------------------------

class TestPruneMethod:
    def test_prune_appends_to_history(self, pipe_with_circuit):
        mock_pruned = MagicMock()
        with patch("circuitkit.quick.prune", return_value=mock_pruned):
            with patch.object(pipe_with_circuit, "_ensure_model", return_value=MagicMock()):
                pipe_with_circuit.prune()
        assert "prune" in pipe_with_circuit._history

    def test_prune_stores_pruned_model(self, pipe_with_circuit):
        mock_pruned = MagicMock()
        with patch("circuitkit.quick.prune", return_value=mock_pruned):
            with patch.object(pipe_with_circuit, "_ensure_model", return_value=MagicMock()):
                pipe_with_circuit.prune()
        assert pipe_with_circuit._pruned_model is mock_pruned

    def test_prune_returns_self(self, pipe_with_circuit):
        with patch("circuitkit.quick.prune", return_value=MagicMock()):
            with patch.object(pipe_with_circuit, "_ensure_model", return_value=MagicMock()):
                result = pipe_with_circuit.prune()
        assert result is pipe_with_circuit

    def test_prune_does_not_overwrite_original_model(self, pipe_with_circuit):
        """_model and _pruned_model must remain separate."""
        original_model = MagicMock(name="original")
        pipe_with_circuit._model = original_model
        mock_pruned = MagicMock(name="pruned")
        with patch("circuitkit.quick.prune", return_value=mock_pruned):
            with patch.object(pipe_with_circuit, "_ensure_model", return_value=original_model):
                pipe_with_circuit.prune()
        assert pipe_with_circuit._model is original_model
        assert pipe_with_circuit._pruned_model is mock_pruned


# ---------------------------------------------------------------------------
# device property
# ---------------------------------------------------------------------------

class TestDeviceProperty:
    def test_device_returns_string(self):
        p = Pipeline("gpt2")
        device = p.device
        assert isinstance(device, str)

    def test_device_is_cpu_or_cuda(self):
        p = Pipeline("gpt2")
        assert p.device in ("cpu", "cuda")

    def test_explicit_device_returned(self):
        p = Pipeline("gpt2", device="cpu")
        assert p.device == "cpu"


# ---------------------------------------------------------------------------
# summary() — must not raise in any state
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_empty_state(self, pipe):
        pipe.summary()  # must not raise

    def test_summary_with_circuit(self, pipe):
        pipe._circuit = _make_circuit()
        pipe.summary()  # must not raise

    def test_summary_with_eval_report(self, pipe):
        from circuitkit.evaluation.report import FaithfulnessReport

        pipe._circuit = _make_circuit()
        # evaluate_circuit returns a FaithfulnessReport for every path as of 1.0.
        pipe._eval_report = FaithfulnessReport(patching_score=0.8, ablation_score=0.75)
        pipe.summary()  # must not raise


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_starts_with_pipeline(self, pipe):
        assert repr(pipe).startswith("Pipeline(")

    def test_repr_contains_model_name(self, pipe):
        assert "gpt2" in repr(pipe)

    def test_repr_contains_task(self, pipe):
        assert "ioi" in repr(pipe)

    def test_repr_contains_steps(self, pipe):
        assert "steps" in repr(pipe)
