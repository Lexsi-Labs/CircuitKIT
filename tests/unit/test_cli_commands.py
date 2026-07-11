"""
Unit tests for the five new CLI commands added in Phase 5:
inspect, prune, quantize, export, run.

Uses click.testing.CliRunner — no model loading, no GPU.
All tests verify command registration and --help output only;
end-to-end execution is covered by the integration suite.
"""

import pytest
from click.testing import CliRunner

from circuitkit.cli.main import cli

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
