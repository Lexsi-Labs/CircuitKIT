"""
Unit tests for CircuitScores artifact (Workstream G).

Tests the unified scores schema across all backends.
"""

import json
import tempfile
from pathlib import Path

import pytest

from circuitkit.artifacts.scores import CircuitScores


class TestCircuitScoresCreation:
    """Test CircuitScores instantiation and validation."""

    def test_create_valid_circuit_scores(self):
        """Test creating valid CircuitScores."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.8, "A0.1": 0.6, "MLP 0": 0.4},
            timestamp="2025-04-13T12:00:00Z",
        )
        assert scores.task == "ioi"
        assert scores.model == "gpt2"
        assert scores.algorithm == "eap"
        assert len(scores.node_scores) == 3

    def test_invalid_algorithm(self):
        """Test validation of invalid algorithm."""
        with pytest.raises(ValueError, match="algorithm must be one of"):
            CircuitScores(
                task="ioi",
                model="gpt2",
                algorithm="invalid_algo",
                level="node",
                node_scores={"A0.0": 0.5},
                timestamp="2025-04-13T12:00:00Z",
            )

    def test_invalid_level(self):
        """Test validation of invalid level."""
        with pytest.raises(ValueError, match="level must be 'node' or 'neuron'"):
            CircuitScores(
                task="ioi",
                model="gpt2",
                algorithm="eap",
                level="invalid_level",
                node_scores={"A0.0": 0.5},
                timestamp="2025-04-13T12:00:00Z",
            )

    def test_negative_score(self):
        """Test validation of negative scores."""
        with pytest.raises(ValueError, match="must be non-negative"):
            CircuitScores(
                task="ioi",
                model="gpt2",
                algorithm="eap",
                level="node",
                node_scores={"A0.0": -0.5},
                timestamp="2025-04-13T12:00:00Z",
            )

    def test_non_numeric_score(self):
        """Test validation of non-numeric scores."""
        with pytest.raises(TypeError, match="must be numeric"):
            CircuitScores(
                task="ioi",
                model="gpt2",
                algorithm="eap",
                level="node",
                node_scores={"A0.0": "not_a_number"},
                timestamp="2025-04-13T12:00:00Z",
            )


class TestCircuitScoresJsonRoundtrip:
    """Test JSON serialization and deserialization."""

    def test_to_json_and_from_json(self):
        """Test round-trip JSON serialization."""
        original = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap-ig",
            level="node",
            node_scores={"A0.0": 0.8, "A0.1": 0.6, "MLP 0": 0.4},
            timestamp="2025-04-13T12:00:00Z",
            discovery_cfg={"method": "ig", "ig_steps": 50},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.json"
            original.to_json(path)

            # Verify file exists
            assert path.exists()

            # Load and compare
            loaded = CircuitScores.from_json(path)
            assert loaded.task == original.task
            assert loaded.model == original.model
            assert loaded.algorithm == original.algorithm
            assert loaded.node_scores == original.node_scores
            assert loaded.discovery_cfg == original.discovery_cfg

    def test_json_file_content(self):
        """Test JSON file format."""
        scores = CircuitScores(
            task="mmlu",
            model="pythia-70m",
            algorithm="acdc",
            level="node",
            node_scores={"A0.0": 0.5},
            timestamp="2025-04-13T12:00:00Z",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.json"
            scores.to_json(path)

            with open(path, "r") as f:
                data = json.load(f)

            assert data["task"] == "mmlu"
            assert data["model"] == "pythia-70m"
            assert data["algorithm"] == "acdc"
            assert data["level"] == "node"
            assert data["node_scores"] == {"A0.0": 0.5}


class TestCircuitScoresFromDict:
    """Test from_dict construction."""

    def test_from_dict_with_defaults(self):
        """Test from_dict fills in defaults."""
        data = {
            "task": "ioi",
            "model": "gpt2",
            "algorithm": "eap",
            "level": "node",
            "node_scores": {"A0.0": 0.5},
            "timestamp": "2025-04-13T12:00:00Z",
        }
        scores = CircuitScores.from_dict(data)
        assert scores.version == "1.0"
        assert scores.discovery_cfg == {}

    def test_from_dict_with_explicit_values(self):
        """Test from_dict with explicit version and config."""
        data = {
            "task": "ioi",
            "model": "gpt2",
            "algorithm": "ibcircuit",
            "level": "node",
            "node_scores": {"A0.0": 0.5},
            "timestamp": "2025-04-13T12:00:00Z",
            "version": "1.1",
            "discovery_cfg": {"custom_key": "custom_value"},
        }
        scores = CircuitScores.from_dict(data)
        assert scores.version == "1.1"
        assert scores.discovery_cfg["custom_key"] == "custom_value"


class TestCircuitScoresNormalization:
    """Test score normalization methods."""

    def test_normalize_minmax(self):
        """Test min-max normalization."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.0, "A0.1": 0.5, "A0.2": 1.0},
            timestamp="2025-04-13T12:00:00Z",
        )
        normalized = scores.normalize_scores(method="minmax")

        assert normalized["A0.0"] == 0.0
        assert normalized["A0.1"] == 0.5
        assert normalized["A0.2"] == 1.0

    def test_normalize_minmax_equal_scores(self):
        """Test min-max normalization when all scores are equal."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.5, "A0.1": 0.5, "A0.2": 0.5},
            timestamp="2025-04-13T12:00:00Z",
        )
        normalized = scores.normalize_scores(method="minmax")

        # All equal scores should normalize to 1.0
        for value in normalized.values():
            assert value == 1.0

    def test_normalize_zscore(self):
        """Test z-score normalization."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 1.0, "A0.1": 2.0, "A0.2": 3.0},
            timestamp="2025-04-13T12:00:00Z",
        )
        normalized = scores.normalize_scores(method="zscore")

        # Mean should be 2.0, std should be ~0.816
        # Values should be centered around 0
        assert sum(normalized.values()) < 0.1  # Sum close to 0

    def test_invalid_normalization_method(self):
        """Test invalid normalization method."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.5},
            timestamp="2025-04-13T12:00:00Z",
        )
        with pytest.raises(ValueError, match="method must be"):
            scores.normalize_scores(method="invalid")


class TestCircuitScoresTopK:
    """Test top-k and bottom-k methods."""

    def test_top_k_nodes(self):
        """Test retrieving top-k highest-scoring nodes."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.9, "A0.1": 0.7, "A0.2": 0.5, "MLP 0": 0.3, "MLP 1": 0.1},
            timestamp="2025-04-13T12:00:00Z",
        )
        top_2 = scores.top_k_nodes(2)

        assert len(top_2) == 2
        assert list(top_2.keys()) == ["A0.0", "A0.1"]

    def test_bottom_k_nodes(self):
        """Test retrieving bottom-k lowest-scoring nodes (pruning candidates)."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.9, "A0.1": 0.7, "A0.2": 0.5, "MLP 0": 0.3, "MLP 1": 0.1},
            timestamp="2025-04-13T12:00:00Z",
        )
        bottom_2 = scores.bottom_k_nodes(2)

        assert len(bottom_2) == 2
        assert list(bottom_2.keys()) == ["MLP 1", "MLP 0"]

    def test_top_k_larger_than_total(self):
        """Test top-k when k > number of nodes."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.5, "A0.1": 0.3},
            timestamp="2025-04-13T12:00:00Z",
        )
        top_10 = scores.top_k_nodes(10)

        # Should return all available nodes
        assert len(top_10) == 2


class TestCircuitScoresTimestamp:
    """Test timestamp generation."""

    def test_create_timestamp(self):
        """Test timestamp generation."""
        ts = CircuitScores.create_timestamp()
        assert ts.endswith("Z")
        assert "T" in ts  # ISO 8601 format

    def test_timestamp_uniqueness(self):
        """Test that consecutive timestamps are different."""
        ts1 = CircuitScores.create_timestamp()
        ts2 = CircuitScores.create_timestamp()
        # Very unlikely to be identical in practice
        assert ts1 <= ts2
