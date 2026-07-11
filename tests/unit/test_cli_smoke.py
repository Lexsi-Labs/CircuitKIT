"""Fast CLI plumbing tests (CliRunner + mocked discovery).

These guard the CLI's output-path handling without loading a model. In
particular they cover the regression where a bare output filename (no directory
component) crashed `discover` with ``FileNotFoundError`` — because
``os.makedirs(os.path.dirname("out.pt"))`` calls ``makedirs("")``. Discovery
itself is mocked, so these run in milliseconds and belong in the fast CI subset.
"""
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from circuitkit.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize(
    "out",
    [
        "out.pt",            # bare filename — dirname("") regression
        "sub/out.pt",        # one nested dir
        "a/b/c/out.pt",      # several nested dirs
    ],
)
def test_discover_output_path_variants(runner, out):
    """`discover -o <path>` must handle bare and nested output paths.

    Pre-fix, a bare filename raised FileNotFoundError from os.makedirs("").
    """
    with patch("circuitkit.api.discover_circuit", return_value=["a0.h0", "m1"]) as mock:
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli,
                ["discover", "-m", "gpt2", "-t", "ioi", "-a", "eap-ig",
                 "-o", out, "--num-examples", "4", "-b", "2"],
            )
    assert result.exit_code == 0, result.output
    mock.assert_called_once()


def test_discover_default_output_path(runner):
    """With no -o, the default ``results/...`` path is created, not crashed."""
    with patch("circuitkit.api.discover_circuit", return_value=["a0.h0"]):
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["discover", "-m", "gpt2", "-t", "ioi"])
    assert result.exit_code == 0, result.output


def test_discover_help_lists_commands(runner):
    """`circuitkit --help` exposes the core interface commands."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("discover", "discover-yaml", "run", "prune", "evaluate"):
        assert cmd in result.output
