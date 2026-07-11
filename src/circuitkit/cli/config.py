"""
Configuration management for CircuitKit CLI.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """Manages configuration loading, validation, and generation."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Generate default configuration."""
        return {
            "model": {"name": "gpt2", "precision": "bfloat16"},
            "discovery": {
                "algorithm": "eap-ig",
                "task": "ioi",  # Built-in task - data is auto-generated
                "level": "node",
                "batch_size": 4,
                "ig_steps": 5,
                "data_params": {"num_examples": 128},
            },
            "pruning": {"target_sparsity": 0.3, "scope": "both"},
            "output_path": "./results/circuit_discovery.pt",
        }

    def save_config(self, output_path: str) -> None:
        """Save current configuration to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def validate_config(self) -> bool:
        """Validate the current configuration."""
        required_keys = ["model", "discovery", "pruning"]

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required section: {key}")

        # Validate model
        if "name" not in self.config["model"]:
            raise ValueError("Missing required key: model.name")

        # Validate discovery
        algo = self.config["discovery"].get("algorithm")
        if not algo:
            raise ValueError("Missing required key: discovery.algorithm")

        valid_algos = ["acdc", "eap", "eap-ig"]
        if algo not in valid_algos:
            raise ValueError(f"Invalid algorithm: {algo}. Must be one of {valid_algos}")

        # Validate pruning
        sparsity = self.config["pruning"].get("target_sparsity")
        if not isinstance(sparsity, (int, float)) or not (0.0 <= sparsity <= 1.0):
            raise ValueError(f"Invalid sparsity: {sparsity}. Must be between 0.0 and 1.0")

        return True
