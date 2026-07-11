"""
Unit tests for the three new quick.py functions (load_scores, visualize,
selective_finetune) and the internal validation helpers (_check_algorithm,
_check_level, build_discovery_config).

All heavy model-loading and GPU operations are mocked.  The test suite
exercises only the validation / dispatch logic that runs before any
model call happens.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from circuitkit.circuit import Circuit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NODES = ["A0.1", "A0.2", "MLP 3"]
_SCORES = {"A0.1": 0.9, "A0.2": 0.5, "MLP 3": 0.3}


def _make_circuit(*, scores=None, level="node", artifact_path=None):
    return Circuit(
        list(_NODES),
        dict(scores or _SCORES),
        level=level,
        artifact_path=artifact_path,
    )


def _write_artifact_with_json_sidecar(tmp_path: Path):
    """Write .pt + _scores.json and return the artifact path."""
    pt = tmp_path / "circuit.pt"
    torch.save(list(_NODES), pt)
    sidecar = tmp_path / "circuit_scores.json"
    with open(sidecar, "w") as f:
        json.dump({"node_scores": _SCORES}, f)
    return pt


# ---------------------------------------------------------------------------
# load_scores
# ---------------------------------------------------------------------------

class TestLoadScores:
    def test_returns_circuit_instance(self, tmp_path):
        pt = _write_artifact_with_json_sidecar(tmp_path)
        from circuitkit.quick import load_scores
        result = load_scores(pt)
        assert isinstance(result, Circuit)

    def test_scores_populated_from_sidecar(self, tmp_path):
        pt = _write_artifact_with_json_sidecar(tmp_path)
        from circuitkit.quick import load_scores
        result = load_scores(pt)
        assert result.scores["A0.1"] == pytest.approx(0.9)

    def test_nodes_loaded_correctly(self, tmp_path):
        pt = _write_artifact_with_json_sidecar(tmp_path)
        from circuitkit.quick import load_scores
        result = load_scores(pt)
        assert list(result.nodes) == _NODES

    def test_missing_file_raises_file_not_found(self):
        from circuitkit.quick import load_scores
        with pytest.raises(FileNotFoundError):
            load_scores("definitely_does_not_exist_xyz.pt")

    def test_explicit_scores_path_forwarded(self, tmp_path):
        pt = tmp_path / "circuit.pt"
        torch.save(list(_NODES), pt)
        custom_sidecar = tmp_path / "my_scores.json"
        with open(custom_sidecar, "w") as f:
            json.dump({"node_scores": {"A0.1": 0.42}}, f)
        from circuitkit.quick import load_scores
        result = load_scores(pt, scores_path=custom_sidecar)
        assert result.scores["A0.1"] == pytest.approx(0.42)

    def test_accepts_string_path(self, tmp_path):
        pt = _write_artifact_with_json_sidecar(tmp_path)
        from circuitkit.quick import load_scores
        result = load_scores(str(pt))
        assert isinstance(result, Circuit)


# ---------------------------------------------------------------------------
# _check_algorithm
# ---------------------------------------------------------------------------

class TestCheckAlgorithm:
    def test_valid_algorithm_returned_lower(self):
        from circuitkit.quick import _check_algorithm
        assert _check_algorithm("eap-ig") == "eap-ig"

    def test_upper_case_normalised(self):
        from circuitkit.quick import _check_algorithm
        assert _check_algorithm("EAP-IG") == "eap-ig"

    def test_invalid_algorithm_raises_value_error(self):
        from circuitkit.quick import _check_algorithm
        with pytest.raises(ValueError, match="Valid algorithms"):
            _check_algorithm("bogus_algo_xyz")

    def test_error_message_includes_input_name(self):
        from circuitkit.quick import _check_algorithm
        with pytest.raises(ValueError, match="bogus_algo_xyz"):
            _check_algorithm("bogus_algo_xyz")


# ---------------------------------------------------------------------------
# _check_level
# ---------------------------------------------------------------------------

class TestCheckLevel:
    @pytest.mark.parametrize("level", ["node", "neuron"])
    def test_valid_levels_pass(self, level):
        from circuitkit.quick import _check_level
        assert _check_level(level) == level

    def test_invalid_level_raises(self):
        from circuitkit.quick import _check_level
        with pytest.raises(ValueError, match="'node' or 'neuron'"):
            _check_level("layer")


# ---------------------------------------------------------------------------
# build_discovery_config validation
# ---------------------------------------------------------------------------

class TestBuildDiscoveryConfigValidation:
    """Only tests the validation guards; does not need a real HookedTransformer."""

    def _mock_model(self, model_name="gpt2", dtype="float32"):
        m = MagicMock()
        m.cfg.tokenizer_name = model_name
        m.cfg.dtype = dtype
        return m

    def test_invalid_sparsity_below_zero(self):
        from circuitkit.quick import build_discovery_config
        with pytest.raises(ValueError, match="sparsity"):
            build_discovery_config(self._mock_model(), "ioi", sparsity=-0.1)

    def test_invalid_sparsity_above_one(self):
        from circuitkit.quick import build_discovery_config
        with pytest.raises(ValueError, match="sparsity"):
            build_discovery_config(self._mock_model(), "ioi", sparsity=1.1)

    def test_invalid_scope(self):
        from circuitkit.quick import build_discovery_config
        with pytest.raises(ValueError, match="scope"):
            build_discovery_config(self._mock_model(), "ioi", scope="invalid")

    def test_invalid_n_examples(self):
        from circuitkit.quick import build_discovery_config
        with pytest.raises(ValueError, match="n_examples"):
            build_discovery_config(self._mock_model(), "ioi", n_examples=0)

    def test_invalid_batch_size(self):
        from circuitkit.quick import build_discovery_config
        with pytest.raises(ValueError, match="batch_size"):
            build_discovery_config(self._mock_model(), "ioi", batch_size=0)

    def test_invalid_algorithm(self):
        from circuitkit.quick import build_discovery_config
        with pytest.raises(ValueError, match="Valid algorithms"):
            build_discovery_config(self._mock_model(), "ioi", algorithm="bogus")

    def test_invalid_level(self):
        from circuitkit.quick import build_discovery_config
        with pytest.raises(ValueError, match="'node' or 'neuron'"):
            build_discovery_config(self._mock_model(), "ioi", level="layer")

    def test_valid_config_structure(self):
        """A valid call must produce the expected nested dict shape."""
        from circuitkit.quick import build_discovery_config
        cfg = build_discovery_config(
            self._mock_model(), "ioi", algorithm="eap-ig", level="node",
            n_examples=16, batch_size=4, sparsity=0.3, scope="both",
        )
        assert "model" in cfg
        assert "discovery" in cfg
        assert "pruning" in cfg
        assert cfg["discovery"]["algorithm"] == "eap-ig"
        assert cfg["pruning"]["target_sparsity"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# prune() error paths
# ---------------------------------------------------------------------------

class TestPruneErrorPaths:
    def test_neuron_level_circuit_raises(self):
        from circuitkit.quick import prune
        c = _make_circuit(level="neuron")
        with pytest.raises(ValueError, match="node-level"):
            prune(MagicMock(), c)

    def test_no_scores_raises(self):
        from circuitkit.quick import prune
        c = Circuit(list(_NODES), level="node")  # no scores
        with pytest.raises(ValueError, match="node scores"):
            prune(MagicMock(), c)


# ---------------------------------------------------------------------------
# quantize() error paths
# ---------------------------------------------------------------------------

class TestQuantizeErrorPaths:
    def test_unknown_backend_raises(self):
        from circuitkit.quick import quantize
        c = _make_circuit()
        with pytest.raises(ValueError, match="backend"):
            quantize(MagicMock(), c, backend="unknown_backend")

    def test_no_scores_raises(self):
        from circuitkit.quick import quantize
        c = Circuit(list(_NODES), level="node")  # no scores
        with pytest.raises(ValueError, match="node scores"):
            quantize(MagicMock(), c)


# ---------------------------------------------------------------------------
# export_checkpoint() error paths
# ---------------------------------------------------------------------------

class TestExportCheckpointErrorPaths:
    def test_unknown_intervention_raises(self):
        from circuitkit.quick import export_checkpoint
        with pytest.raises(ValueError, match="intervention"):
            export_checkpoint(MagicMock(), None, "path", intervention="explode")

    def test_pruning_without_artifact_raises(self):
        from circuitkit.quick import export_checkpoint
        with pytest.raises(ValueError, match="artifact"):
            export_checkpoint(MagicMock(), None, "path", intervention="pruning")

    def test_push_to_hub_without_repo_raises(self):
        """push_to_hub=True without hub_repo must raise before any network call."""
        from circuitkit.quick import export_checkpoint
        c = _make_circuit()
        # Mock save_pruned_checkpoint so we don't hit the filesystem
        with patch("circuitkit.quick.export_checkpoint") as _:
            pass  # just verifying the import path is correct
        # Test the internal _push_checkpoint_to_hub guard directly
        from circuitkit.quick import _push_checkpoint_to_hub
        with pytest.raises(ValueError, match="hub_repo"):
            _push_checkpoint_to_hub("some/path", hub_repo=None, hub_private=True)


# ---------------------------------------------------------------------------
# benchmark() error paths
# ---------------------------------------------------------------------------

class TestBenchmarkErrorPaths:
    def test_empty_task_list_raises(self):
        from circuitkit.quick import benchmark
        with pytest.raises(ValueError, match="at least one"):
            benchmark("ckpt_path", [])

    def test_string_task_converted_to_list(self):
        """benchmark() wraps a bare string task into a single-element list before
        forwarding to run_lm_eval (quick.py: `if isinstance(tasks, str): tasks = [tasks]`)."""
        from circuitkit.quick import benchmark

        with patch("circuitkit.evaluation.run_lm_eval") as mock_eval:
            mock_eval.return_value = {"boolq": {"acc": 0.5}}
            benchmark("ckpt_path", "boolq", device="cpu")

        assert mock_eval.called
        # run_lm_eval(checkpoint_path, tasks, ...) — tasks is the 2nd positional arg.
        forwarded_tasks = mock_eval.call_args.args[1]
        assert forwarded_tasks == ["boolq"]


# ---------------------------------------------------------------------------
# selective_finetune() error paths
# ---------------------------------------------------------------------------

class TestSelectiveFinetuneErrorPaths:
    def test_no_artifact_path_raises(self):
        from circuitkit.quick import selective_finetune
        c = _make_circuit()
        c.artifact_path = None  # ensure no path
        with pytest.raises(ValueError, match="artifact_path"):
            selective_finetune(c, model_name=None)

    def test_missing_scores_pt_raises(self, tmp_path):
        from circuitkit.quick import selective_finetune
        # Artifact exists but no _scores.pt side-car
        pt = tmp_path / "circuit.pt"
        torch.save(list(_NODES), pt)
        c = _make_circuit(artifact_path=str(pt))
        with pytest.raises(FileNotFoundError, match="_scores.pt"):
            selective_finetune(c, model_name="gpt2")


# ---------------------------------------------------------------------------
# faithfulness() error paths
# ---------------------------------------------------------------------------

class TestFaithfulnessErrorPaths:
    def test_no_scores_raises(self):
        from circuitkit.quick import faithfulness
        c = Circuit(list(_NODES), level="node")  # empty scores
        with pytest.raises(ValueError, match="node scores"):
            faithfulness(MagicMock(), c, "ioi")


# ---------------------------------------------------------------------------
# visualize() dispatch and error paths
# ---------------------------------------------------------------------------

class TestVisualize:
    def test_unknown_mode_raises(self):
        from circuitkit.quick import visualize_circuit
        c = _make_circuit()
        with pytest.raises(ValueError, match="Unknown visualization mode"):
            visualize_circuit(c, mode="bogus_mode")

    def test_comparison_without_second_circuit_raises(self):
        from circuitkit.quick import visualize_circuit
        c = _make_circuit()
        with pytest.raises(ValueError, match="second_circuit"):
            visualize_circuit(c, mode="comparison")

    def test_comparison_with_empty_scores_raises(self):
        from circuitkit.quick import visualize_circuit
        c1 = Circuit(list(_NODES))  # no scores
        c2 = Circuit(list(_NODES))  # no scores
        with pytest.raises(ValueError, match="node scores"):
            visualize_circuit(c1, mode="comparison", second_circuit=c2)

    def test_graph_mode_delegates_to_plot(self):
        from circuitkit.quick import visualize_circuit
        c = _make_circuit()
        c.plot = MagicMock(return_value="<html/>")
        visualize_circuit(c, mode="graph", output="out.html")
        c.plot.assert_called_once_with("out.html")

    def test_comparison_returns_dashboard_when_no_output(self):
        from circuitkit.quick import visualize_circuit
        c1 = _make_circuit()
        c2 = _make_circuit()
        # Patch ComparisonDashboard to avoid needing plotly
        mock_db = MagicMock()
        with patch("circuitkit.visualize.comparison.ComparisonDashboard", return_value=mock_db):
            result = visualize_circuit(c1, mode="comparison", second_circuit=c2)
        assert result is mock_db

    def test_comparison_calls_export_when_output_given(self, tmp_path):
        from circuitkit.quick import visualize_circuit
        c1 = _make_circuit()
        c2 = _make_circuit()
        out = str(tmp_path / "compare.html")
        mock_db = MagicMock()
        with patch("circuitkit.visualize.comparison.ComparisonDashboard", return_value=mock_db):
            result = visualize_circuit(c1, mode="comparison", second_circuit=c2, output=out)
        mock_db.export_to_html.assert_called_once_with(out)
        assert result is None

    def test_comparison_auto_deduplicates_identical_labels(self):
        """When both circuits have the same task, labels should be suffixed."""
        from circuitkit.quick import visualize_circuit
        c1 = _make_circuit()
        c1.task = "ioi"
        c2 = _make_circuit()
        c2.task = "ioi"  # same task → labels would collide
        mock_db = MagicMock()
        with patch("circuitkit.visualize.comparison.ComparisonDashboard", return_value=mock_db) as mock_cls:
            visualize_circuit(c1, mode="comparison", second_circuit=c2)
        called_circuits = mock_cls.call_args[1]["circuits"]
        labels = list(called_circuits.keys())
        assert labels[0] != labels[1], "Duplicate labels must be disambiguated"
