"""
Importing CircuitKIT must not write to the working directory.

The loggers used to create ``logs/`` in the current directory at import time, which fails
with PermissionError when that directory is read-only. Each test runs a fresh interpreter
in an empty directory so the module-level logger is created from scratch.
"""

import os
import subprocess
import sys
from pathlib import Path

import circuitkit

SRC = str(Path(circuitkit.__file__).resolve().parents[1])


def _run(code: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": SRC},
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_import_leaves_cwd_empty(tmp_path):
    result = _run(
        "from circuitkit.utils.logging import get_logger\n"
        "get_logger().info('global')\n"
        "get_logger('task.ioi').info('named')\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_setup_logging_log_file_still_writes(tmp_path):
    result = _run(
        "from circuitkit.utils.logging import setup_logging\n"
        "lg = setup_logging(verbose=True, log_file='run.log')\n"
        "lg.debug('detail', k=1)\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.log"]
    assert 'detail | Context: {"k": 1}' in (tmp_path / "run.log").read_text()
