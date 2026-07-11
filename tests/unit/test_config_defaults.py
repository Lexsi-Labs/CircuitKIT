#!/usr/bin/env python3
"""
Test cases for configuration defaults consistency.
Ensures that default values come from a single source of truth (DEFAULT_CONFIG).
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from circuitkit.utils.config import (  # noqa: E402 - import after intentional pre-import setup
    DEFAULT_CONFIG,
    load_and_validate_config,
)


class TestConfigDefaults:
    """Test cases for configuration defaults."""

    def test_default_config_structure(self):
        """Test that DEFAULT_CONFIG has the required top-level sections."""
        assert "model" in DEFAULT_CONFIG
        assert "discovery" in DEFAULT_CONFIG
        assert "pruning" in DEFAULT_CONFIG
        assert "output_path" in DEFAULT_CONFIG

    def test_model_defaults(self):
        """Test that model section has required defaults."""
        model_defaults = DEFAULT_CONFIG["model"]
        assert "precision" in model_defaults
        assert model_defaults["precision"] == "bfloat16"

    def test_discovery_defaults(self):
        """Test that discovery section has required defaults."""
        discovery_defaults = DEFAULT_CONFIG["discovery"]

        # Required fields
        assert "algorithm" in discovery_defaults
        assert "level" in discovery_defaults

        # "task" is intentionally NOT defaulted — it must be supplied by user
        # config or an inline data section (see DEFAULT_CONFIG in config.py,
        # changed in 145f935). There is no sensible universal default task.
        assert "task" not in discovery_defaults

        # IBCircuit-specific defaults
        assert "num_epochs" in discovery_defaults
        assert "learning_rate" in discovery_defaults
        assert "alpha" in discovery_defaults
        assert "beta" in discovery_defaults
        assert "alpha_loss" in discovery_defaults
        assert "log_interval" in discovery_defaults
        assert "mask_type" in discovery_defaults

        # Additional discovery parameters
        assert "scope" in discovery_defaults
        assert "mlp_hook" in discovery_defaults
        assert "intervention" in discovery_defaults
        assert "method" in discovery_defaults
        assert "ig_steps" in discovery_defaults

    def test_ibcircuit_defaults_consistency(self):
        """Test that IBCircuit-specific defaults are reasonable."""
        discovery = DEFAULT_CONFIG["discovery"]

        # Check beta default (should be 0.001, not 1.0)
        assert (
            discovery["beta"] == 0.001
        ), f"IBCircuit beta should be 0.001, got {discovery['beta']}"

        # Check learning_rate default (should be 0.01)
        assert (
            discovery["learning_rate"] == 0.01
        ), f"IBCircuit learning_rate should be 0.01, got {discovery['learning_rate']}"

        # Check alpha default
        assert discovery["alpha"] == 1.0

        # Check num_epochs default
        assert discovery["num_epochs"] == 1000

        # Check alpha_loss default
        assert discovery["alpha_loss"] == "kl"

        # Check log_interval default
        assert discovery["log_interval"] == 100

        # Check mask_type default
        assert discovery["mask_type"] == "sigmoid"

    def test_pruning_defaults(self):
        """Test that pruning section has required defaults."""
        pruning_defaults = DEFAULT_CONFIG["pruning"]

        assert "target_sparsity" in pruning_defaults
        assert "scope" in pruning_defaults
        assert pruning_defaults["scope"] == "both"
        assert pruning_defaults["target_sparsity"] == 0.3

    def test_load_and_validate_with_empty_discovery_config(self):
        """Test that load_and_validate_config fills in defaults."""
        # Create a minimal config that only requires essential fields
        minimal_config = {
            "model": {"name": "gpt2"},
            "discovery": {"algorithm": "ibcircuit", "task": "ioi"},
        }

        # This should not raise and should have defaults filled in
        config = load_and_validate_config(minimal_config)

        # Verify that defaults were applied
        discovery = config["discovery"]
        assert (
            discovery["beta"] == 0.001
        ), f"After loading, beta should be 0.001, got {discovery['beta']}"
        assert discovery["learning_rate"] == 0.01
        assert discovery["alpha"] == 1.0
        assert discovery["num_epochs"] == 1000
        assert discovery["alpha_loss"] == "kl"

    def test_user_config_overrides_defaults(self):
        """Test that user config properly overrides defaults."""
        user_config = {
            "model": {"name": "gpt2", "precision": "float32"},
            "discovery": {
                "algorithm": "ibcircuit",
                "task": "ioi",
                "beta": 0.5,  # Override default
                "learning_rate": 0.1,  # Override default
            },
        }

        config = load_and_validate_config(user_config)

        # Verify user overrides are respected
        assert config["discovery"]["beta"] == 0.5
        assert config["discovery"]["learning_rate"] == 0.1

        # Verify other defaults still present
        assert config["discovery"]["alpha"] == 1.0
        assert config["model"]["precision"] == "float32"

    def test_beta_default_mismatch_fixed(self):
        """
        Test that the beta default mismatch (A8) has been fixed.
        Previously: config.py had beta=0.001, but api.py defaulted to beta=1.0
        """
        discovery_defaults = DEFAULT_CONFIG["discovery"]

        # The main issue: beta should be 0.001
        assert (
            discovery_defaults["beta"] == 0.001
        ), "CRITICAL: IBCircuit beta default should be 0.001 (A8 fix)"

        # Verify this is used consistently
        config = load_and_validate_config(
            {"model": {"name": "gpt2"}, "discovery": {"algorithm": "ibcircuit", "task": "ioi"}}
        )

        assert (
            config["discovery"]["beta"] == 0.001
        ), "Beta mismatch not fixed: config doesn't use 0.001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
