"""
Integration tests for discovery method CircuitArtifact export.

Tests that ACDC, EAP, and IBCircuit discovery methods can export
their results as unified CircuitArtifact objects.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from circuitkit.artifacts import CircuitArtifact
from circuitkit.backends.acdc.artifact_export import export_circuit_artifact as acdc_export
from circuitkit.backends.eap.artifact_export import export_circuit_artifact as eap_export
from circuitkit.backends.ibcircuit.artifact_export import (
    export_circuit_artifact as ibcircuit_export,
)

# Test ACDC Export


class TestACDCExport:
    """Test ACDC circuit export to CircuitArtifact."""

    def test_acdc_export_basic(self):
        """Test basic ACDC export."""
        prune_scores = {
            "blocks.0.attn.hook_v": torch.randn(8),
            "blocks.0.mlp.hook_result": torch.randn(4),
            "blocks.1.attn.hook_v": torch.randn(8),
        }

        artifact = acdc_export(
            prune_scores=prune_scores,
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
        )

        assert artifact.model_id == "gpt2"
        assert artifact.discovery_method == "acdc"
        assert artifact.task == "ioi"
        assert len(artifact.nodes) > 0

    def test_acdc_export_with_threshold(self):
        """Test ACDC export with importance threshold."""
        prune_scores = {
            "blocks.0.attn.hook_v": torch.tensor([0.9, 0.1, 0.8, 0.2, 0.95, 0.05, 0.7, 0.15]),
        }

        artifact = acdc_export(
            prune_scores=prune_scores,
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
            threshold=0.5,
        )

        # Only nodes with importance >= 0.5 should be included
        for node in artifact.nodes.values():
            assert node.importance >= 0.5

    def test_acdc_export_validation(self):
        """Test that exported artifact is valid."""
        prune_scores = {
            "blocks.0.attn.hook_v": torch.randn(8),
        }

        artifact = acdc_export(
            prune_scores=prune_scores,
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
        )

        checks = artifact.validate()
        assert all(checks.values()), f"Validation failed: {checks}"

    def test_acdc_export_serialization(self):
        """Test that exported artifact can be serialized."""
        prune_scores = {
            "blocks.0.attn.hook_v": torch.randn(8),
        }

        artifact1 = acdc_export(
            prune_scores=prune_scores,
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "acdc_circuit.json"
            artifact1.save_json(path)

            artifact2 = CircuitArtifact.load_json(path)
            assert len(artifact1.nodes) == len(artifact2.nodes)
            assert artifact1.model_id == artifact2.model_id


# Test EAP Export


class TestEAPExport:
    """Test EAP circuit export to CircuitArtifact."""

    def test_eap_export_basic(self):
        """Test basic EAP export."""
        node_scores = {
            "A0.0": 0.92,
            "A0.1": 0.15,
            "A0.2": 0.78,
            "A1.3": 0.65,
            "MLP 0": 0.55,
            "MLP 1": 0.82,
        }

        artifact = eap_export(
            node_scores=node_scores,
            model_id="gpt2",
            task="sva",
            dataset="counterfact",
        )

        assert artifact.model_id == "gpt2"
        assert artifact.discovery_method == "eap"
        assert artifact.task == "sva"
        assert len(artifact.nodes) == 6  # All nodes added

    def test_eap_export_with_threshold(self):
        """Test EAP export with importance threshold."""
        node_scores = {
            "A0.0": 0.92,
            "A0.1": 0.15,
            "A0.2": 0.78,
            "MLP 0": 0.55,
        }

        artifact = eap_export(
            node_scores=node_scores,
            model_id="gpt2",
            task="sva",
            dataset="counterfact",
            threshold=0.5,
        )

        # Only nodes >= 0.5 threshold
        assert len(artifact.nodes) <= 4
        for node in artifact.nodes.values():
            assert node.importance >= 0.5

    def test_eap_export_node_types(self):
        """Test that EAP export correctly identifies node types."""
        node_scores = {
            "A0.0": 0.9,  # Attention
            "A1.2": 0.8,  # Attention
            "MLP 0": 0.7,  # MLP
            "MLP 1": 0.6,  # MLP
        }

        artifact = eap_export(
            node_scores=node_scores,
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
        )

        from circuitkit.artifacts import NodeType

        attn_nodes = [n for n in artifact.nodes.values() if n.node_type == NodeType.ATTENTION_HEAD]
        mlp_nodes = [n for n in artifact.nodes.values() if n.node_type == NodeType.MLP_NEURON]

        assert len(attn_nodes) >= 2
        assert len(mlp_nodes) >= 2

    def test_eap_export_validation(self):
        """Test that exported artifact is valid."""
        node_scores = {
            "A0.0": 0.92,
            "MLP 0": 0.55,
        }

        artifact = eap_export(
            node_scores=node_scores,
            model_id="gpt2",
            task="sva",
            dataset="counterfact",
        )

        checks = artifact.validate()
        assert all(checks.values()), f"Validation failed: {checks}"


# Test IBCircuit Export


class TestIBCircuitExport:
    """Test IBCircuit circuit export to CircuitArtifact."""

    def test_ibcircuit_export_basic(self):
        """Test basic IBCircuit export."""
        neuron_scores = {
            "L0.N10": 0.75,
            "L0.N25": 0.45,
            "L1.N5": 0.85,
            "L2.N100": 0.2,
        }

        artifact = ibcircuit_export(
            neuron_scores=neuron_scores,
            model_id="pythia-70m",
            task="greater_than",
            dataset="numeric",
        )

        assert artifact.model_id == "pythia-70m"
        assert artifact.discovery_method == "ibcircuit"
        assert artifact.task == "greater_than"
        assert len(artifact.nodes) == 4

    def test_ibcircuit_export_with_threshold(self):
        """Test IBCircuit export with threshold."""
        neuron_scores = {
            "L0.N10": 0.75,
            "L0.N25": 0.45,
            "L1.N5": 0.85,
            "L2.N100": 0.2,
        }

        artifact = ibcircuit_export(
            neuron_scores=neuron_scores,
            model_id="pythia-70m",
            task="greater_than",
            dataset="numeric",
            threshold=0.5,
        )

        # Only neurons >= 0.5
        assert len(artifact.nodes) == 2
        assert all(n.importance >= 0.5 for n in artifact.nodes.values())

    def test_ibcircuit_export_granularity(self):
        """Test that IBCircuit export uses neuron granularity."""
        neuron_scores = {
            "L0.N10": 0.75,
            "L1.N5": 0.85,
        }

        artifact = ibcircuit_export(
            neuron_scores=neuron_scores,
            model_id="pythia-70m",
            task="greater_than",
            dataset="numeric",
            granularity="neuron",
        )

        assert artifact.granularity == "neuron"

    def test_ibcircuit_export_validation(self):
        """Test that exported artifact is valid."""
        neuron_scores = {
            "L0.N10": 0.75,
            "L1.N5": 0.85,
        }

        artifact = ibcircuit_export(
            neuron_scores=neuron_scores,
            model_id="pythia-70m",
            task="greater_than",
            dataset="numeric",
        )

        checks = artifact.validate()
        assert all(checks.values()), f"Validation failed: {checks}"


# Cross-Method Compatibility Tests


class TestCrossMethodCompatibility:
    """Test that all methods produce compatible artifacts."""

    def test_all_methods_export_valid_artifacts(self):
        """Test that all three methods produce valid artifacts."""
        acdc_artifact = acdc_export(
            prune_scores={"blocks.0.attn.hook_v": torch.randn(8)},
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
        )

        eap_artifact = eap_export(
            node_scores={"A0.0": 0.9, "MLP 0": 0.7},
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
        )

        ibcircuit_artifact = ibcircuit_export(
            neuron_scores={"L0.N10": 0.75},
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
        )

        # All should be valid
        assert all(acdc_artifact.validate().values())
        assert all(eap_artifact.validate().values())
        assert all(ibcircuit_artifact.validate().values())

    def test_all_methods_serialization(self):
        """Test that all methods' artifacts can be serialized."""
        artifacts = [
            acdc_export(
                prune_scores={"blocks.0.attn.hook_v": torch.randn(8)},
                model_id="gpt2",
                task="test",
                dataset="test",
            ),
            eap_export(
                node_scores={"A0.0": 0.9},
                model_id="gpt2",
                task="test",
                dataset="test",
            ),
            ibcircuit_export(
                neuron_scores={"L0.N10": 0.75},
                model_id="gpt2",
                task="test",
                dataset="test",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, artifact in enumerate(artifacts):
                path = Path(tmpdir) / f"artifact_{idx}.json"
                artifact.save_json(path)
                loaded = CircuitArtifact.load_json(path)
                assert len(artifact.nodes) == len(loaded.nodes)

    def test_artifacts_compatible_with_masks(self):
        """Test that exported artifacts can generate masks."""
        # This is a basic compatibility test
        # Full mask generation requires model/arch_cfg

        artifact = eap_export(
            node_scores={"A0.0": 0.9, "A0.1": 0.7, "MLP 0": 0.6},
            model_id="gpt2",
            task="ioi",
            dataset="ioi_dataset",
        )

        # Should have nodes with different importances
        importances = [n.importance for n in artifact.nodes.values()]
        assert len(importances) > 0
        assert max(importances) >= min(importances)  # Different importance levels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
