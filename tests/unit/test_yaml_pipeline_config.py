"""
Unit tests for YAML pipeline config parsing.

Tests the structure and semantics of YAML configs that the `circuitkit run`
command and Pipeline consume — without invoking any real model loading.
All tests operate on in-memory dicts and temp files only.
"""

import os
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(cfg: dict, path: Path) -> Path:
    """Serialise cfg to YAML at path and return path."""
    path.write_text(yaml.dump(cfg))
    return path


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Minimal valid config
# ---------------------------------------------------------------------------

class TestMinimalConfig:
    def test_minimal_config_roundtrips(self, tmp_path):
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "discovery": {"algorithm": "eap-ig", "level": "node"},
        }
        path = _write_yaml(cfg, tmp_path / "minimal.yaml")
        loaded = _load_yaml(path)
        assert loaded["model"] == "gpt2"
        assert loaded["task"] == "ioi"
        assert loaded["discovery"]["algorithm"] == "eap-ig"

    def test_minimal_config_is_valid_yaml(self, tmp_path):
        """yaml.safe_load must not raise on a well-formed minimal config."""
        cfg = {"model": "gpt2", "task": "ioi"}
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        assert isinstance(loaded, dict)


# ---------------------------------------------------------------------------
# evaluate key semantics
# ---------------------------------------------------------------------------

class TestEvaluateKeySemantics:
    def test_evaluate_enabled_defaults_true_when_key_present(self, tmp_path):
        """When evaluate: is present but has no enabled field, it defaults True."""
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "discovery": {"algorithm": "eap-ig"},
            "evaluate": {"pillars": [1, 2]},
        }
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        eval_cfg = loaded.get("evaluate", {})
        # Mirrors the run command logic: eval_cfg.get("enabled", True)
        assert eval_cfg.get("enabled", True) is True

    def test_evaluate_key_absent_means_no_eval(self, tmp_path):
        """Absence of evaluate: key means no evaluation step is triggered."""
        cfg = {"model": "gpt2", "task": "ioi", "discovery": {}}
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        # run command: `if eval_cfg and eval_cfg.get("enabled", True)`
        # cfg.get("evaluate", {}) returns {} which is falsy
        eval_cfg = loaded.get("evaluate", {})
        assert not eval_cfg

    def test_evaluate_disabled_explicitly(self, tmp_path):
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "evaluate": {"enabled": False},
        }
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        eval_cfg = loaded.get("evaluate", {})
        assert eval_cfg.get("enabled", True) is False


# ---------------------------------------------------------------------------
# applications list format
# ---------------------------------------------------------------------------

class TestApplicationsList:
    def test_applications_is_a_list(self, tmp_path):
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "applications": [
                {"type": "prune", "sparsity": 0.3},
            ],
        }
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        assert isinstance(loaded["applications"], list)

    def test_application_entry_has_type_key(self, tmp_path):
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "applications": [{"type": "prune", "sparsity": 0.3}],
        }
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        entry = loaded["applications"][0]
        assert "type" in entry
        assert entry["type"] == "prune"

    def test_multiple_applications_preserved(self, tmp_path):
        cfg = {
            "model": "gpt2",
            "task": "ioi",
            "applications": [
                {"type": "prune", "sparsity": 0.3},
                {"type": "selective_finetune", "top_fraction": 0.2},
            ],
        }
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        assert len(loaded["applications"]) == 2
        assert loaded["applications"][1]["type"] == "selective_finetune"


# ---------------------------------------------------------------------------
# output_dir default
# ---------------------------------------------------------------------------

class TestOutputDirDefault:
    def test_output_dir_absent_means_default(self, tmp_path):
        """When output_dir is missing, the run command falls back to ./pipeline_output."""
        cfg = {"model": "gpt2", "task": "ioi"}
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        output_dir = loaded.get("output_dir", "./pipeline_output")
        assert output_dir == "./pipeline_output"

    def test_output_dir_explicit_value_preserved(self, tmp_path):
        cfg = {"model": "gpt2", "task": "ioi", "output_dir": "/custom/dir"}
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        assert loaded["output_dir"] == "/custom/dir"


# ---------------------------------------------------------------------------
# custom_data block
# ---------------------------------------------------------------------------

class TestCustomDataBlock:
    def test_custom_data_roundtrips(self, tmp_path):
        cfg = {
            "model": "gpt2",
            "custom_data": {
                "path": "data.csv",
                "clean_prompt": "{question}",
                "clean_answer": "{answer}",
            },
            "discovery": {"algorithm": "ibcircuit"},
        }
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        cd = loaded["custom_data"]
        assert cd["path"] == "data.csv"
        assert cd["clean_prompt"] == "{question}"

    def test_custom_data_corrupt_pair_optional(self, tmp_path):
        """corrupt_prompt/answer must be optional (IBCircuit doesn't need them)."""
        cfg = {
            "model": "gpt2",
            "custom_data": {
                "path": "data.csv",
                "clean_prompt": "{q}",
                "clean_answer": "{a}",
            },
        }
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        cd = loaded["custom_data"]
        assert "corrupt_prompt" not in cd
        assert "corrupt_answer" not in cd

    def test_custom_data_with_corrupt_pair(self, tmp_path):
        cfg = {
            "model": "gpt2",
            "custom_data": {
                "path": "data.csv",
                "clean_prompt": "{q}",
                "clean_answer": "{a}",
                "corrupt_prompt": "{bad_q}",
                "corrupt_answer": "{bad_a}",
            },
        }
        path = _write_yaml(cfg, tmp_path / "cfg.yaml")
        loaded = _load_yaml(path)
        cd = loaded["custom_data"]
        assert cd["corrupt_prompt"] == "{bad_q}"


# ---------------------------------------------------------------------------
# Full pipeline config — all sections
# ---------------------------------------------------------------------------

class TestFullPipelineConfig:
    @pytest.fixture()
    def full_cfg(self):
        return {
            "model": "gpt2",
            "task": "ioi",
            "precision": "bfloat16",
            "output_dir": "./results",
            "discovery": {
                "algorithm": "eap-ig",
                "level": "node",
                "sparsity": 0.3,
                "n_examples": 128,
                "batch_size": 4,
                "scope": "both",
            },
            "evaluate": {
                "enabled": True,
                "pillars": [1, 2],
                "n_examples": 256,
            },
            "applications": [
                {"type": "prune", "sparsity": 0.3, "scope": "both"},
            ],
            "export": {
                "path": "./results/checkpoint",
                "intervention": "pruning",
            },
            "benchmark": {
                "enabled": False,
                "tasks": ["boolq"],
                "limit": 50,
            },
            "visualize": {
                "enabled": True,
                "mode": "graph",
                "output": "./results/circuit.html",
            },
        }

    def test_all_top_level_keys_present(self, full_cfg, tmp_path):
        path = _write_yaml(full_cfg, tmp_path / "full.yaml")
        loaded = _load_yaml(path)
        for key in ("model", "task", "precision", "output_dir", "discovery",
                    "evaluate", "applications", "export", "benchmark", "visualize"):
            assert key in loaded, f"Top-level key {key!r} missing after roundtrip"

    def test_discovery_block_structure(self, full_cfg, tmp_path):
        path = _write_yaml(full_cfg, tmp_path / "full.yaml")
        loaded = _load_yaml(path)
        disc = loaded["discovery"]
        assert disc["algorithm"] == "eap-ig"
        assert disc["sparsity"] == pytest.approx(0.3)
        assert disc["level"] == "node"

    def test_export_path_preserved(self, full_cfg, tmp_path):
        path = _write_yaml(full_cfg, tmp_path / "full.yaml")
        loaded = _load_yaml(path)
        assert loaded["export"]["path"] == "./results/checkpoint"

    def test_benchmark_disabled_by_default_in_full_cfg(self, full_cfg, tmp_path):
        path = _write_yaml(full_cfg, tmp_path / "full.yaml")
        loaded = _load_yaml(path)
        bench = loaded.get("benchmark", {})
        assert bench.get("enabled", False) is False


# ---------------------------------------------------------------------------
# Sample pipeline fixture file (written to disk for use by test_yaml_run_cli)
# ---------------------------------------------------------------------------

class TestSampleFixtureFile:
    """Verify the sample fixture YAML that test_yaml_run_cli.py uses is valid
    and readable.  Path is relative to the tests/ root."""

    FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample_pipeline.yaml"

    def test_fixture_file_exists(self):
        if not self.FIXTURE_PATH.exists():
            pytest.skip("fixtures/sample_pipeline.yaml not yet created")
        assert self.FIXTURE_PATH.exists()

    def test_fixture_file_is_valid_yaml(self):
        if not self.FIXTURE_PATH.exists():
            pytest.skip("fixtures/sample_pipeline.yaml not yet created")
        with open(self.FIXTURE_PATH) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "model" in data
