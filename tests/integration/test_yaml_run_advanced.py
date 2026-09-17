"""Real gpt2/CPU acceptance tests for PR-L2 (YAML-runner completeness).

Exercises the parameters `circuitkit run <yaml>` previously dropped
(discovery.seed/ig_steps, evaluate.n_stability_runs/target_task) and the new
bounded graph.json export (visualize.max_nodes/max_edges/edge_threshold),
plus the existing prune+export path. All on gpt2/CPU with tiny n_examples,
so these are marked slow/integration and excluded from the fast CI subset.
"""

import json

import pytest
import yaml
from click.testing import CliRunner

pytestmark = [pytest.mark.slow, pytest.mark.integration]

_DISCO_BASE = dict(algorithm="eap-ig", level="node", sparsity=0.3, n_examples=4, batch_size=2)


def _run(cfg: dict, tmp_path, name: str = "pipeline.yaml"):
    from circuitkit.cli.main import cli

    yaml_path = tmp_path / name
    yaml_path.write_text(yaml.dump(cfg))
    return CliRunner().invoke(cli, ["run", str(yaml_path)])


def test_run_yaml_discovery_advanced(tmp_path):
    """discovery.seed + ig_steps thread through and produce circuit.pt + graph.json."""
    out_dir = tmp_path / "out"
    cfg = {
        "model": "gpt2",
        "task": "ioi",
        "precision": "float32",
        "output_dir": str(out_dir),
        "discovery": {**_DISCO_BASE, "seed": 7, "ig_steps": 3},
        "visualize": {"mode": "graph", "output": str(out_dir / "graph.json")},
    }
    result = _run(cfg, tmp_path)
    assert result.exit_code == 0, result.output

    pt_files = list(out_dir.glob("*.pt"))
    assert pt_files, f"no circuit .pt artifact written to {out_dir}"

    graph_path = out_dir / "graph.json"
    assert graph_path.exists()
    with open(graph_path) as f:
        graph = json.load(f)
    assert "nodes" in graph and "edges" in graph


def test_run_yaml_eval_generalization(tmp_path):
    """evaluate.n_stability_runs + target_task run without error."""
    out_dir = tmp_path / "out"
    cfg = {
        "model": "gpt2",
        "task": "ioi",
        "precision": "float32",
        "output_dir": str(out_dir),
        "discovery": dict(_DISCO_BASE),
        "evaluate": {
            "enabled": True,
            "pillars": [1, 2],
            "n_examples": 8,
            "n_stability_runs": 1,
            "target_task": "greater_than",
        },
    }
    result = _run(cfg, tmp_path)
    assert result.exit_code == 0, result.output
    assert "warning" not in result.output.lower()

    pt_files = list(out_dir.glob("*.pt"))
    assert pt_files


def test_run_yaml_prune_and_export(tmp_path):
    """applications:[{type: prune}] + export produces an exported checkpoint dir."""
    out_dir = tmp_path / "out"
    export_dir = tmp_path / "exported"
    cfg = {
        "model": "gpt2",
        "task": "ioi",
        "precision": "float32",
        "output_dir": str(out_dir),
        "discovery": dict(_DISCO_BASE),
        "applications": [{"type": "prune", "sparsity": 0.3, "scope": "both"}],
        "export": {"path": str(export_dir), "intervention": "pruning"},
    }
    result = _run(cfg, tmp_path)
    assert result.exit_code == 0, result.output
    assert export_dir.exists() and any(export_dir.iterdir())


def test_graph_json_emitted(tmp_path):
    """visualize step with a .json output + max_nodes/max_edges writes a
    bounded, parseable graph.json (the L1 bounded-export contract)."""
    out_dir = tmp_path / "out"
    graph_path = out_dir / "graph.json"
    cfg = {
        "model": "gpt2",
        "task": "ioi",
        "precision": "float32",
        "output_dir": str(out_dir),
        "discovery": dict(_DISCO_BASE),
        "visualize": {
            "mode": "graph",
            "output": str(graph_path),
            "max_nodes": 50,
            "max_edges": 200,
            "edge_threshold": 0.0,
        },
    }
    result = _run(cfg, tmp_path)
    assert result.exit_code == 0, result.output

    assert graph_path.exists()
    with open(graph_path) as f:
        graph = json.load(f)
    assert len(graph["nodes"]) <= 50
    assert len(graph["edges"]) <= 200
    assert "n_nodes_hidden" in graph["metadata"]
