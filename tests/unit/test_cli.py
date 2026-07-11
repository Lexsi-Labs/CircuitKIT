"""
Unit tests for CircuitKit CLI.
"""

from click.testing import CliRunner

from circuitkit.cli.main import cli
from circuitkit.cli.utils import get_supported_models, validate_model_name


class TestCLI:
    """Test CLI functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CircuitKit" in result.output

    def test_discover_help(self):
        """Test discover command help."""
        result = self.runner.invoke(cli, ["discover", "--help"])
        assert result.exit_code == 0
        assert "Run circuit discovery" in result.output

    def test_evaluate_help(self):
        """Test evaluate command help."""
        result = self.runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "Evaluate circuit faithfulness" in result.output

    def test_list_models(self):
        """Test list models command."""
        result = self.runner.invoke(cli, ["list-models"])
        assert result.exit_code == 0
        assert "Supported Models" in result.output


class TestCLIUtils:
    """Test CLI utility functions."""

    def test_validate_model_name_valid(self):
        """Test model name validation with valid names."""
        valid_names = ["gpt2", "gpt2-medium", "meta-llama/Meta-Llama-3-8B"]

        for name in valid_names:
            assert validate_model_name(name) is True

    def test_validate_model_name_invalid(self):
        """Test model name validation with invalid names."""
        invalid_names = ["", "a", "invalid@name", "name with spaces"]

        for name in invalid_names:
            assert validate_model_name(name) is False

    def test_get_supported_models(self):
        """Test getting supported models list."""
        models = get_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt2" in models
