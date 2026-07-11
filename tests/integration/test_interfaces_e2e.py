"""End-to-end interface tests on a real model (gpt2).

Exercises all four entry points — Python ``Pipeline``, the flat ``ck.discover``
API, the ``circuitkit discover`` CLI, and ``circuitkit run <yaml>`` — and does so
across several built-in tasks, not only IOI. These load gpt2 and run real
discovery on CPU/MPS, so they are marked ``slow``/``integration`` and excluded
from the fast CI subset. Run them with ``pytest -m slow``.
"""
import os

import pytest
import yaml
from click.testing import CliRunner

pytestmark = [pytest.mark.slow, pytest.mark.integration]

# Built-in gpt2 tasks (deliberately not only "ioi").
TASKS = ["ioi", "sva", "greater_than"]
_DISCO = dict(algorithm="eap-ig", sparsity=0.3, n_examples=4, batch_size=2)


@pytest.mark.parametrize("task", TASKS)
def test_pipeline_discovers(task, tmp_path):
    """Python Pipeline API discovers a non-empty circuit for each task."""
    from circuitkit import Pipeline

    pipe = Pipeline("gpt2", task=task, output_dir=str(tmp_path))
    pipe.discover(**_DISCO)
    assert pipe._circuit is not None and len(pipe._circuit) > 0


@pytest.mark.parametrize("task", ["ioi", "greater_than"])
def test_flat_api_discovers(task, tmp_path):
    """Flat ``ck.discover(model, task, ...)`` returns a non-empty Circuit."""
    import circuitkit as ck

    model = ck.load_model("gpt2")
    circuit = ck.discover(model, task, output_path=str(tmp_path / "flat.pt"), **_DISCO)
    assert len(circuit) > 0


def test_cli_discover_bare_filename(tmp_path):
    """`circuitkit discover -o out.pt` (bare filename) exits 0 and writes the file.

    Real end-to-end regression for the os.makedirs("") crash.
    """
    from circuitkit.cli.main import cli

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["discover", "-m", "gpt2", "-t", "ioi", "-a", "eap-ig",
             "-o", "cli_out.pt", "--num-examples", "4", "-b", "2"],
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists("cli_out.pt")


def test_yaml_run(tmp_path):
    """`circuitkit run <yaml>` executes a full pipeline for a non-ioi task."""
    from circuitkit.cli.main import cli

    cfg = {
        "model": "gpt2",
        "task": "greater_than",
        "precision": "float32",
        "output_dir": str(tmp_path / "out"),
        "discovery": {
            "algorithm": "eap-ig", "level": "node", "sparsity": 0.3,
            "n_examples": 4, "batch_size": 2, "scope": "both",
        },
    }
    yaml_path = tmp_path / "pipe.yaml"
    yaml_path.write_text(yaml.dump(cfg))

    result = CliRunner().invoke(cli, ["run", str(yaml_path)])
    assert result.exit_code == 0, result.output
