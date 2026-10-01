"""`trust_remote_code` reaches the loader from config and CLI, and stays opt-in.

Sarvam-MoE ships its modeling code in the repository rather than in
``transformers``, so it cannot load without this flag. ``load_model`` always
accepted it, but the YAML/dict config path and the CLI could not set it, which
left Sarvam unreachable outside the quick API. Forwarding it runs code from the
model repository, so it must never be enabled implicitly.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from circuitkit.api import _trust_remote_code_kwargs
from circuitkit.cli.main import discover
from circuitkit.utils.config import load_and_validate_config


class TestForwarding:
    def test_absent_or_false_forwards_nothing(self):
        assert _trust_remote_code_kwargs({}) == {}
        assert _trust_remote_code_kwargs({"trust_remote_code": False}) == {}

    def test_true_forwards_the_flag(self):
        assert _trust_remote_code_kwargs({"trust_remote_code": True}) == {
            "trust_remote_code": True
        }

    def test_warns_when_forwarded(self, caplog):
        with caplog.at_level("WARNING"):
            _trust_remote_code_kwargs({"trust_remote_code": True})
        assert "trust_remote_code" in caplog.text

    def test_default_config_does_not_enable_it(self):
        config = load_and_validate_config(
            {
                "model": {"name": "gpt2"},
                "discovery": {"algorithm": "eap", "task": "ioi"},
                "pruning": {"target_sparsity": 0.5, "scope": "heads"},
            }
        )
        assert config["model"]["trust_remote_code"] is False

    def test_config_rejects_non_boolean(self):
        with pytest.raises(ValueError, match="trust_remote_code"):
            load_and_validate_config(
                {
                    "model": {"name": "gpt2", "trust_remote_code": "yes"},
                    "discovery": {"algorithm": "eap", "task": "ioi"},
                    "pruning": {"target_sparsity": 0.5, "scope": "heads"},
                }
            )


class TestCliFlag:
    def test_flag_is_off_by_default(self):
        result = CliRunner().invoke(discover, ["--help"])
        assert result.exit_code == 0
        assert "--trust-remote-code" in result.output

    def test_flag_reaches_the_config_dict(self):
        with patch("circuitkit.api.discover_circuit", return_value=["a0.h0"]) as mock_discover:
            result = CliRunner().invoke(
                discover,
                [
                    "--model",
                    "sarvamai/sarvam-30b",
                    "--algorithm",
                    "eap",
                    "--task",
                    "ioi",
                    "--trust-remote-code",
                ],
            )
        assert result.exit_code == 0, result.output
        assert mock_discover.call_args.args[0]["model"]["trust_remote_code"] is True

    def test_omitting_the_flag_keeps_it_false(self):
        with patch("circuitkit.api.discover_circuit", return_value=["a0.h0"]) as mock_discover:
            result = CliRunner().invoke(
                discover,
                ["--model", "gpt2", "--algorithm", "eap", "--task", "ioi"],
            )
        assert result.exit_code == 0, result.output
        assert mock_discover.call_args.args[0]["model"]["trust_remote_code"] is False
