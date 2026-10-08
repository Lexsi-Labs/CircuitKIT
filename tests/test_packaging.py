"""Packaging / import smoke tests.

Fast, offline, dependency-light checks that the package is importable and the
CLI entry point is wired. These run in the default fast tier and catch
packaging regressions (missing modules, broken entry points, version drift)
that source-tree tests cannot.
"""

from __future__ import annotations

import importlib
import importlib.metadata

import pytest


def test_top_level_import():
    import circuitkit

    assert circuitkit is not None


def test_version_matches_pyproject():
    # pyproject.toml reads circuitkit.__version__, so the installed metadata must match it.
    import circuitkit

    assert circuitkit.__version__ == importlib.metadata.version("circuitkit")


@pytest.mark.parametrize(
    "module",
    [
        "circuitkit.api",
        "circuitkit.cli.main",
        "circuitkit.backends",
        "circuitkit.applications",
        "circuitkit.corruption",
        "circuitkit.pipeline",
    ],
)
def test_public_subpackages_import(module):
    assert importlib.import_module(module) is not None


def test_cli_entry_point_target_exists():
    """The object named by the ``circuitkit`` console-script entry point exists."""
    from circuitkit.cli.main import main

    assert callable(main)


def test_cli_help_runs():
    """``circuitkit --help`` parses and exits cleanly (in-process, no network)."""
    from click.testing import CliRunner

    from circuitkit.cli.main import cli

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert result.output.strip()
