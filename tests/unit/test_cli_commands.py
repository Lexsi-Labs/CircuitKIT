"""
Unit tests for the five new CLI commands added in Phase 5:
inspect, prune, quantize, export, run.

Uses click.testing.CliRunner — no model loading, no GPU.
All tests verify command registration and --help output only;
end-to-end execution is covered by the integration suite.
"""

import re

import pytest
from click.testing import CliRunner

from circuitkit.cli.main import cli

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text):
    """CLI output without colour codes. Rich colours it when the shell exports FORCE_COLOR, which
    splits a plain substring such as "Hyperparameter warnings (2)" into styled fragments."""
    return _ANSI.sub("", text)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

class TestCommandRegistration:
    NEW_COMMANDS = {"inspect", "prune", "quantize", "export", "run"}

    def test_all_new_commands_registered(self):
        registered = set(cli.commands.keys())
        missing = self.NEW_COMMANDS - registered
        assert not missing, f"Missing CLI commands: {sorted(missing)}"

    def test_existing_discover_command_still_present(self):
        """Regression: the original discover command must not have been removed."""
        assert "discover" in cli.commands

    def test_existing_discover_yaml_command_still_present(self):
        assert "discover-yaml" in cli.commands


# ---------------------------------------------------------------------------
# inspect --help
# ---------------------------------------------------------------------------

class TestInspectHelp:
    def test_exits_zero(self, runner):
        result = runner.invoke(cli, ["inspect", "--help"])
        assert result.exit_code == 0, result.output

    def test_output_mentions_artifact(self, runner):
        result = runner.invoke(cli, ["inspect", "--help"])
        assert "artifact" in result.output.lower()

    def test_output_is_non_empty(self, runner):
        result = runner.invoke(cli, ["inspect", "--help"])
        assert len(result.output.strip()) > 0


# ---------------------------------------------------------------------------
# prune --help
# ---------------------------------------------------------------------------

class TestPruneHelp:
    def test_exits_zero(self, runner):
        result = runner.invoke(cli, ["prune", "--help"])
        assert result.exit_code == 0, result.output

    def test_output_mentions_model(self, runner):
        result = runner.invoke(cli, ["prune", "--help"])
        assert "--model" in result.output

    def test_output_mentions_artifact(self, runner):
        result = runner.invoke(cli, ["prune", "--help"])
        assert "--artifact" in result.output

    def test_output_mentions_sparsity(self, runner):
        result = runner.invoke(cli, ["prune", "--help"])
        assert "sparsity" in result.output.lower()


# ---------------------------------------------------------------------------
# quantize --help
# ---------------------------------------------------------------------------

class TestQuantizeHelp:
    def test_exits_zero(self, runner):
        result = runner.invoke(cli, ["quantize", "--help"])
        assert result.exit_code == 0, result.output

    def test_output_mentions_model(self, runner):
        result = runner.invoke(cli, ["quantize", "--help"])
        assert "--model" in result.output

    def test_output_mentions_artifact(self, runner):
        result = runner.invoke(cli, ["quantize", "--help"])
        assert "--artifact" in result.output


# ---------------------------------------------------------------------------
# export --help
# ---------------------------------------------------------------------------

class TestExportHelp:
    def test_exits_zero(self, runner):
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0, result.output

    def test_output_mentions_intervention(self, runner):
        result = runner.invoke(cli, ["export", "--help"])
        assert "intervention" in result.output.lower()

    def test_output_mentions_output_flag(self, runner):
        result = runner.invoke(cli, ["export", "--help"])
        assert "--output" in result.output

    def test_output_mentions_model(self, runner):
        result = runner.invoke(cli, ["export", "--help"])
        assert "--model" in result.output


# ---------------------------------------------------------------------------
# run --help
# ---------------------------------------------------------------------------

class TestRunHelp:
    def test_exits_zero(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0, result.output

    def test_output_mentions_yaml_or_config(self, runner):
        result = runner.invoke(cli, ["run", "--help"])
        lower = result.output.lower()
        assert "yaml" in lower or "config" in lower

    def test_output_mentions_config_path(self, runner):
        """run takes a positional CONFIG_PATH argument."""
        result = runner.invoke(cli, ["run", "--help"])
        assert "config" in result.output.lower()


# ---------------------------------------------------------------------------
# run command — missing model key aborts gracefully
# ---------------------------------------------------------------------------

class TestRunCommandValidation:
    def test_run_yaml_missing_model_aborts(self, runner, tmp_path):
        """A YAML file without a 'model' key must cause an abort with a clear error."""
        import yaml
        cfg = {"task": "ioi", "discovery": {"algorithm": "eap-ig"}}
        config_file = tmp_path / "bad_pipeline.yaml"
        config_file.write_text(yaml.dump(cfg))

        result = runner.invoke(cli, ["run", str(config_file)])
        assert result.exit_code != 0
        assert "model" in result.output.lower()

    def test_run_nonexistent_config_file_fails(self, runner):
        """Passing a path that does not exist must fail (click.Path(exists=True) guard)."""
        result = runner.invoke(cli, ["run", "/nonexistent/path/pipeline.yaml"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Top-level CLI --help
# ---------------------------------------------------------------------------

class TestTopLevelHelp:
    def test_main_help_exits_zero(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0, result.output

    def test_main_help_lists_new_commands(self, runner):
        result = runner.invoke(cli, ["--help"])
        for cmd in ("inspect", "prune", "quantize", "export", "run"):
            assert cmd in result.output, f"'{cmd}' not shown in top-level --help"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestCliDefaults:
    def test_discover_ig_steps_default_is_three(self):
        option = next(p for p in cli.commands["discover"].params if p.name == "ig_steps")
        assert option.default == 3


class TestValidateConfigCommand:
    @staticmethod
    def _write(tmp_path, text):
        path = tmp_path / "cfg.yaml"
        path.write_text(text)
        return str(path)

    VALID = (
        "model: {name: gpt2}\n"
        "discovery: {algorithm: eap-ig, task: ioi, level: node, data_params: {num_examples: 128}}\n"
        "pruning: {target_sparsity: 0.3, scope: both}\n"
    )

    def test_valid_config_passes(self, runner, tmp_path):
        result = runner.invoke(cli, ["validate-config", "--config", self._write(tmp_path, self.VALID)])
        assert result.exit_code == 0, _plain(result.output)
        assert "Configuration is valid" in _plain(result.output)
        assert "warnings" not in _plain(result.output).lower()

    def test_invalid_range_fails_and_names_the_parameter(self, runner, tmp_path):
        text = self.VALID.replace("level: node,", "level: node, ig_steps: 0,")
        result = runner.invoke(cli, ["validate-config", "--config", self._write(tmp_path, text)])
        assert result.exit_code != 0
        assert "discovery.ig_steps" in _plain(result.output)

    def test_irrelevant_parameter_is_ignored(self, runner, tmp_path):
        text = self.VALID.replace("algorithm: eap-ig,", "algorithm: eap,").replace(
            "level: node,", "level: node, ig_steps: 0,"
        )
        result = runner.invoke(cli, ["validate-config", "--config", self._write(tmp_path, text)])
        assert result.exit_code == 0, _plain(result.output)

    WARN = (
        "model: {name: gpt2}\n"
        "discovery: {algorithm: eap-ig, task: ioi, data_params: {num_examples: 8}}\n"
        "pruning: {target_sparsity: 0.6, scope: both}\n"
    )

    def test_sensible_range_warns_but_passes(self, runner, tmp_path):
        result = runner.invoke(cli, ["validate-config", "--config", self._write(tmp_path, self.WARN)])
        assert result.exit_code == 0, _plain(result.output)
        assert "Hyperparameter warnings (2)" in _plain(result.output)

    def test_strict_fails_on_warnings(self, runner, tmp_path):
        result = runner.invoke(
            cli, ["validate-config", "--config", self._write(tmp_path, self.WARN), "--strict"]
        )
        assert result.exit_code != 0
        assert "num_examples" in _plain(result.output)

    def test_pipeline_yaml_gets_a_clear_message(self, runner, tmp_path):
        text = "model: gpt2\ntask: ioi\ndiscovery: {algorithm: eap-ig}\n"
        result = runner.invoke(cli, ["validate-config", "--config", self._write(tmp_path, text)])
        assert result.exit_code != 0
        assert "circuitkit run" in _plain(result.output)

    def test_missing_file_aborts(self, runner, tmp_path):
        result = runner.invoke(cli, ["validate-config", "--config", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0
        assert "not found" in _plain(result.output)


class TestHparamsCommand:
    def test_table_lists_the_parameters(self, runner):
        result = runner.invoke(cli, ["hparams"])
        assert result.exit_code == 0, _plain(result.output)
        for name in ("discovery.ig_steps", "eval.n_stability_runs", "pruning.target_sparsity"):
            assert name in _plain(result.output)

    def test_markdown_matches_the_docs_generator(self, runner):
        from circuitkit.utils.hparams import render_markdown

        result = runner.invoke(cli, ["hparams", "--markdown"])
        assert result.exit_code == 0, _plain(result.output)
        assert _plain(result.output) == render_markdown()
