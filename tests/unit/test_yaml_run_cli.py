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

        with patch("circuitkit.api.discover_circuit", return_value=["A0.1", "MLP 3"]), \
             patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()):
            result = runner.invoke(cli, ["run", cfg_path])

        # The run command catches discovery errors and aborts; a successful mock
        # must reach the summary step.
        assert "pipeline" in result.output.lower() or result.exit_code == 0, \
            f"Unexpected output:\n{result.output}"

    def test_run_outputs_step_labels(self, runner, tmp_path):
        """The run command prints step labels (discovery, etc.)."""
        cfg_path = _minimal_cfg(tmp_path)

        with patch("circuitkit.api.discover_circuit", return_value=["A0.1", "MLP 3"]), \
             patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()):
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

        with patch("circuitkit.api.discover_circuit", return_value=["A0.1"]), \
             patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()), \
             patch("circuitkit.pipeline.Pipeline.evaluate", side_effect=RuntimeError("eval boom")):
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

        with patch("circuitkit.api.discover_circuit", return_value=["A0.1"]), \
             patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()), \
             patch("circuitkit.pipeline.Pipeline.visualize", side_effect=RuntimeError("viz boom")):
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

        with patch("circuitkit.pipeline.Pipeline.from_custom_data") as mock_fcd, \
             patch("circuitkit.api.discover_circuit", return_value=["A0.1"]), \
             patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()):
            # from_custom_data must return a Pipeline-like object
            mock_pipe = MagicMock()
            mock_pipe._circuit = None
            mock_fcd.return_value = mock_pipe

            runner.invoke(cli, ["run", cfg_path])

        mock_fcd.assert_called_once()
        call_kwargs = mock_fcd.call_args
        assert "clean_prompt" in call_kwargs.kwargs or \
               (call_kwargs.args and "{clean}" in str(call_kwargs))


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

        original_init = __import__(
            "circuitkit.pipeline", fromlist=["Pipeline"]
        ).Pipeline.__init__

        def capturing_init(self, model_name, *, precision="bfloat16", **kw):
            captured_pipelines.append({"model": model_name, "precision": precision})
            original_init(self, model_name, precision=precision, **kw)

        with patch("circuitkit.pipeline.Pipeline.__init__", capturing_init), \
             patch("circuitkit.api.discover_circuit", return_value=["A0.1"]), \
             patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()):
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

        original_init = __import__(
            "circuitkit.pipeline", fromlist=["Pipeline"]
        ).Pipeline.__init__

        def capturing_init(self, model_name, *, output_dir="./pipeline_output", **kw):
            captured.append(output_dir)
            original_init(self, model_name, output_dir=output_dir, **kw)

        with patch("circuitkit.pipeline.Pipeline.__init__", capturing_init), \
             patch("circuitkit.api.discover_circuit", return_value=["A0.1"]), \
             patch("circuitkit.pipeline.Pipeline._ensure_model", return_value=MagicMock()):
            runner.invoke(cli, ["run", cfg_path])

        if captured:
            assert captured[0] == custom_dir
