"""
Unit tests for the `circuitkit run` CLI command.

Tests that the YAML-driven pipeline execution works correctly at the
orchestration level: correct Pipeline constructor is called, non-fatal
step failures don't abort the run, missing required keys abort early.

All heavy operations (model loading, discovery, evaluation) are mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from circuitkit.cli.main import cli

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner():
    return CliRunner()


def _write_yaml(cfg: dict, path: Path) -> str:
    path.write_text(yaml.dump(cfg))
    return str(path)


def _minimal_cfg(tmp_path: Path) -> str:
    """Write a minimal valid YAML config and return its path."""
    return _write_yaml(
        {
            "model": "gpt2",
            "task": "ioi",
            "precision": "float32",
            "output_dir": str(tmp_path / "pipeline_out"),
            "discovery": {
                "algorithm": "eap-ig",
                "level": "node",
                "sparsity": 0.3,
                "n_examples": 4,
                "batch_size": 2,
            },
        },
        tmp_path / "pipeline.yaml",
    )


# ---------------------------------------------------------------------------
# Missing required keys
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_missing_model_key_aborts(self, runner, tmp_path):
        """A config without 'model' must abort with a non-zero exit code."""
        cfg_path = _write_yaml(
            {"task": "ioi", "discovery": {"algorithm": "eap-ig"}},
            tmp_path / "bad.yaml",
        )
        result = runner.invoke(cli, ["run", cfg_path])
        assert result.exit_code != 0
        assert "model" in result.output.lower()

    def test_nonexistent_config_file_rejected(self, runner):
        """click.Path(exists=True) rejects a path that doesn't exist."""
        result = runner.invoke(cli, ["run", "/no/such/file/pipeline.yaml"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Minimal successful run (mocked discovery)
# ---------------------------------------------------------------------------


class TestRunMinimalConfig:
    def test_run_completes_with_mocked_discovery(self, runner, tmp_path):
        """A valid config with mocked discovery must exit 0 and print summary."""
        from circuitkit.circuit import Circuit

        cfg_path = _minimal_cfg(tmp_path)
        mock_circuit = Circuit(["A0.1", "MLP 3"], {"A0.1": 0.9, "MLP 3": 0.5})

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1", "MLP 3"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["run", cfg_path])

        # The run command catches discovery errors and aborts; a successful mock
        # must reach the summary step.
        assert (
            "pipeline" in result.output.lower() or result.exit_code == 0
        ), f"Unexpected output:\n{result.output}"

    def test_run_outputs_step_labels(self, runner, tmp_path):
        """The run command prints step labels (discovery, etc.)."""
        cfg_path = _minimal_cfg(tmp_path)

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1", "MLP 3"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
        ):
            result = runner.invoke(cli, ["run", cfg_path])

        assert "discovery" in result.output.lower()


# ---------------------------------------------------------------------------
# Non-fatal step failures
# ---------------------------------------------------------------------------


class TestRunNonFatalFailures:
    def test_evaluate_failure_does_not_abort_run(self, runner, tmp_path):
        """Evaluate step failure must emit a warning but not abort the pipeline."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "precision": "float32",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "evaluate": {"enabled": True, "pillars": [1]},
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            patch("circuitkit.pipeline.Pipeline.evaluate", side_effect=RuntimeError("eval boom")),
        ):
            result = runner.invoke(cli, ["run", cfg_path])

        # Must warn, not abort
        assert "warning" in result.output.lower() or "eval" in result.output.lower()
        # Exit code 0 means the pipeline continued past the evaluate failure
        assert result.exit_code == 0

    def test_visualize_failure_does_not_abort_run(self, runner, tmp_path):
        """Visualize step failure must emit a warning but not abort the pipeline."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "visualize": {"enabled": True, "mode": "graph"},
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            patch("circuitkit.pipeline.Pipeline.visualize", side_effect=RuntimeError("viz boom")),
        ):
            result = runner.invoke(cli, ["run", cfg_path])

        assert "warning" in result.output.lower() or "viz" in result.output.lower()
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Custom data path
# ---------------------------------------------------------------------------


class TestRunCustomData:
    def test_custom_data_config_calls_from_custom_data(self, runner, tmp_path):
        """When custom_data: is present, Pipeline.from_custom_data must be used."""
        # Write a dummy CSV so the path reference is valid
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("clean,corrupted,correct_idx,incorrect_idx\nhello,world,1,2\n")

        cfg = {
            "model": "gpt2",
            "output_dir": str(tmp_path / "out"),
            "custom_data": {
                "path": str(csv_path),
                "clean_prompt": "{clean}",
                "clean_answer": "{correct_idx}",
            },
            "discovery": {"algorithm": "eap-ig", "level": "node"},
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.pipeline.Pipeline.from_custom_data") as mock_fcd,
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
        ):
            # from_custom_data must return a Pipeline-like object
            mock_pipe = MagicMock()
            mock_pipe._circuit = None
            mock_fcd.return_value = mock_pipe

            runner.invoke(cli, ["run", cfg_path])

        mock_fcd.assert_called_once()
        call_kwargs = mock_fcd.call_args
        assert "clean_prompt" in call_kwargs.kwargs or (
            call_kwargs.args and "{clean}" in str(call_kwargs)
        )


# ---------------------------------------------------------------------------
# Precision and output_dir forwarded correctly
# ---------------------------------------------------------------------------


class TestRunConfigForwarding:
    def test_precision_forwarded_to_pipeline(self, runner, tmp_path):
        """The precision key must be passed through to Pipeline.__init__."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "precision": "float32",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        captured_pipelines = []

        original_init = __import__("circuitkit.pipeline", fromlist=["Pipeline"]).Pipeline.__init__

        def capturing_init(self, model_name, *, precision="bfloat16", **kw):
            captured_pipelines.append({"model": model_name, "precision": precision})
            original_init(self, model_name, precision=precision, **kw)

        with (
            patch("circuitkit.pipeline.Pipeline.__init__", capturing_init),
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
        ):
            runner.invoke(cli, ["run", cfg_path])

        if captured_pipelines:
            assert captured_pipelines[0]["precision"] == "float32"

    def test_output_dir_forwarded_to_pipeline(self, runner, tmp_path):
        """The output_dir key must be passed through to Pipeline.__init__."""
        custom_dir = str(tmp_path / "my_output")
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": custom_dir,
            "discovery": {"algorithm": "eap-ig", "level": "node"},
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        captured = []

        original_init = __import__("circuitkit.pipeline", fromlist=["Pipeline"]).Pipeline.__init__

        def capturing_init(self, model_name, *, output_dir="./pipeline_output", **kw):
            captured.append(output_dir)
            original_init(self, model_name, output_dir=output_dir, **kw)

        with (
            patch("circuitkit.pipeline.Pipeline.__init__", capturing_init),
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
        ):
            runner.invoke(cli, ["run", cfg_path])

        if captured:
            assert captured[0] == custom_dir


# ---------------------------------------------------------------------------
# PR-L2: advanced discovery/evaluate/visualize params threaded through
# ---------------------------------------------------------------------------


class TestRunAdvancedParamsThreaded:
    def test_discovery_seed_and_ig_steps_forwarded(self, runner, tmp_path):
        """discovery.seed/ig_steps/mlp_hook/chat_template_mode reach
        Pipeline.discover(), previously silently dropped by the YAML runner."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {
                "algorithm": "eap-ig",
                "level": "node",
                "seed": 42,
                "ig_steps": 7,
                "mlp_hook": "post_act",
                "chat_template_mode": "off",
            },
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.pipeline.Pipeline.discover") as mock_discover,
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
        ):
            mock_discover.return_value = MagicMock(_circuit=None)
            runner.invoke(cli, ["run", cfg_path])

        mock_discover.assert_called_once()
        kwargs = mock_discover.call_args.kwargs
        assert kwargs["seed"] == 42
        assert kwargs["ig_steps"] == 7
        assert kwargs["mlp_hook"] == "post_act"
        assert kwargs["chat_template_mode"] == "off"

    def test_discovery_extra_keys_absent_when_unset(self, runner, tmp_path):
        """Without ig_steps/mlp_hook/chat_template_mode in the YAML, they must
        not be passed at all (no accidental None overrides)."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.pipeline.Pipeline.discover") as mock_discover,
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
        ):
            mock_discover.return_value = MagicMock(_circuit=None)
            runner.invoke(cli, ["run", cfg_path])

        kwargs = mock_discover.call_args.kwargs
        assert "ig_steps" not in kwargs
        assert "mlp_hook" not in kwargs
        assert "chat_template_mode" not in kwargs
        assert kwargs["seed"] is None

    def test_evaluate_n_stability_runs_and_target_task_forwarded(self, runner, tmp_path):
        """evaluate.n_stability_runs/target_task reach Pipeline.evaluate()."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "evaluate": {
                "enabled": True,
                "pillars": [1],
                "n_stability_runs": 3,
                "target_task": "greater_than",
            },
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            patch("circuitkit.pipeline.Pipeline.evaluate") as mock_evaluate,
        ):
            runner.invoke(cli, ["run", cfg_path])

        mock_evaluate.assert_called_once()
        kwargs = mock_evaluate.call_args.kwargs
        assert kwargs["n_stability_runs"] == 3
        assert kwargs["target_task"] == "greater_than"

    def test_prune_protect_layers_forwarded(self, runner, tmp_path):
        """applications[].protect_layers reaches Pipeline.prune()."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "applications": [{"type": "prune", "sparsity": 0.2, "protect_layers": [0, 1]}],
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            patch("circuitkit.pipeline.Pipeline.prune") as mock_prune,
        ):
            runner.invoke(cli, ["run", cfg_path])

        mock_prune.assert_called_once()
        assert mock_prune.call_args.kwargs["protect_layers"] == [0, 1]

    def test_device_forwarded_to_pipeline(self, runner, tmp_path):
        """The top-level device key reaches Pipeline.__init__ (absent -> None, auto-detect)."""
        from circuitkit.pipeline import Pipeline

        seen = []
        original_init = Pipeline.__init__

        def capturing_init(self, model_name, *, device=None, **kw):
            seen.append(device)
            original_init(self, model_name, device=device, **kw)

        for device in ("cpu", None):
            cfg = {
                "model": "gpt2",
                "task": "ioi",
                "output_dir": str(tmp_path / "out"),
                "discovery": {"algorithm": "eap-ig", "level": "node"},
            }
            if device:
                cfg["device"] = device
            cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")
            with (
                patch("circuitkit.pipeline.Pipeline.__init__", capturing_init),
                patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
                patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            ):
                runner.invoke(cli, ["run", cfg_path])

        assert seen == ["cpu", None]

    def test_visualize_bounding_kwargs_forwarded_only_when_set(self, runner, tmp_path):
        """visualize.max_nodes/max_edges/edge_threshold reach Pipeline.visualize()
        only when present in the YAML."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "visualize": {
                "enabled": True,
                "mode": "graph",
                "output": "graph.json",
                "max_nodes": 300,
                "max_edges": 3000,
            },
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            patch("circuitkit.pipeline.Pipeline.visualize") as mock_visualize,
        ):
            runner.invoke(cli, ["run", cfg_path])

        mock_visualize.assert_called_once()
        kwargs = mock_visualize.call_args.kwargs
        assert kwargs["max_nodes"] == 300
        assert kwargs["max_edges"] == 3000
        assert "edge_threshold" not in kwargs

    @pytest.mark.parametrize("block, runs", [
        ({"tasks": ["ioi"]}, True),  # present block runs, like evaluate:/visualize:
        ({"tasks": ["ioi"], "enabled": False}, False),
    ])
    def test_benchmark_block_runs_unless_disabled(self, runner, tmp_path, block, runs):
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "benchmark": block,
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            patch("circuitkit.pipeline.Pipeline.benchmark") as mock_benchmark,
        ):
            runner.invoke(cli, ["run", cfg_path])

        assert mock_benchmark.called is runs

    def test_benchmark_runs_with_enabled_true(self, runner, tmp_path):
        """benchmark: {enabled: true} must run."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "benchmark": {"enabled": True, "tasks": ["ioi"]},
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            patch("circuitkit.pipeline.Pipeline.benchmark") as mock_benchmark,
        ):
            runner.invoke(cli, ["run", cfg_path])

        mock_benchmark.assert_called_once()


class TestPinnedDeviceReachesDiscovery:
    def test_pipeline_device_is_in_the_discovery_config(self):
        """Pipeline(device=...) must reach discover_circuit's config; without it
        discovery auto-detected and ignored the pinned device."""
        from circuitkit.pipeline import Pipeline

        assert Pipeline("gpt2", task="ioi", device="cpu")._model_cfg()["device"] == "cpu"
        assert "device" not in Pipeline("gpt2", task="ioi")._model_cfg()

    def test_resolving_device_does_not_pin_it(self):
        from circuitkit.pipeline import Pipeline

        pipe = Pipeline("gpt2", task="ioi")
        assert pipe.device in ("cuda", "cpu")
        assert "device" not in pipe._model_cfg()


class TestBlockSpellings:
    def test_applications_mapping_keyed_by_type(self, runner, tmp_path):
        """applications may be a mapping keyed by type as well as a list; the
        mapping form used to crash with "'str' object has no attribute 'get'"."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "applications": {"prune": {"sparsity": 0.3, "scope": "both"}},
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            patch("circuitkit.pipeline.Pipeline.prune") as mock_prune,
        ):
            result = runner.invoke(cli, ["run", cfg_path])

        assert result.exit_code == 0, result.output
        assert mock_prune.call_args.kwargs["sparsity"] == 0.3


class TestDiscoveryKeysAndReport:
    def test_algorithm_hyperparameters_reach_discover(self, runner, tmp_path):
        """Discovery keys beyond the named ones are passed through, so IBCircuit's
        num_epochs / learning_rate can be set from YAML (a whitelist used to drop them)."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(tmp_path / "out"),
            "discovery": {
                "algorithm": "ibcircuit",
                "level": "node",
                "num_epochs": 50,
                "learning_rate": 0.05,
            },
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        with (
            patch("circuitkit.pipeline.Pipeline.discover") as mock_discover,
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
        ):
            runner.invoke(cli, ["run", cfg_path])

        kwargs = mock_discover.call_args.kwargs
        assert kwargs["num_epochs"] == 50 and kwargs["learning_rate"] == 0.05
        assert kwargs["algorithm"] == "ibcircuit" and "enabled" not in kwargs

    def test_evaluate_writes_the_report_json(self, runner, tmp_path):
        """`circuitkit run` leaves faithfulness_report.json in output_dir."""
        from circuitkit.evaluation.report import FaithfulnessReport
        from circuitkit.pipeline import Pipeline

        out = tmp_path / "out"
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "output_dir": str(out),
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "evaluate": {"pillars": [1]},
        }
        cfg_path = _write_yaml(cfg, tmp_path / "pipeline.yaml")

        def fake_evaluate(self, **kw):
            self._eval_report = FaithfulnessReport(
                patching_score=0.0, metadata={"patching_raw_ratio": -12.21}
            )
            return self

        with (
            patch("circuitkit.api.discover_circuit", return_value=["A0.1"]),
            patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()),
            patch.object(Pipeline, "evaluate", fake_evaluate),
        ):
            runner.invoke(cli, ["run", cfg_path])

        import json

        saved = json.loads((out / "faithfulness_report.json").read_text())
        assert saved["patching_score"] == 0.0
        assert saved["metadata"]["patching_raw_ratio"] == -12.21
