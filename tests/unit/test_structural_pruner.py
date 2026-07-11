"""
Unit tests for StructuralPruner (Workstream H).

Tests real structural pruning using weight matrix manipulation.
"""

from unittest.mock import Mock, patch

import pytest
import torch as t
from transformer_lens import HookedTransformer

from circuitkit.applications.pruning.pruner import StructuralPruner
from circuitkit.artifacts.scores import CircuitScores


@pytest.fixture
def sample_circuit_scores():
    """Create a sample CircuitScores artifact for testing."""
    return CircuitScores(
        task="ioi",
        model="gpt2",
        algorithm="eap",
        level="node",
        node_scores={
            "A0.0": 0.9,
            "A0.1": 0.7,
            "A0.2": 0.5,
            "A0.3": 0.3,
            "A0.4": 0.1,
            "MLP 0": 0.4,
            "MLP 1": 0.2,
        },
        timestamp="2025-04-13T12:00:00Z",
    )


@pytest.fixture
def pruner():
    """Create a StructuralPruner instance."""
    return StructuralPruner()


class TestStructuralPrunerValidation:
    """Test input validation."""

    def test_invalid_sparsity_too_low(self, pruner):
        """Test that sparsity < 0 is rejected."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.5},
            timestamp="2025-04-13T12:00:00Z",
        )
        model = Mock(spec=HookedTransformer)

        with pytest.raises(ValueError, match="sparsity must be in"):
            pruner.prune(model, scores, sparsity=-0.1)

    def test_invalid_sparsity_too_high(self, pruner):
        """Test that sparsity > 1 is rejected."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.5},
            timestamp="2025-04-13T12:00:00Z",
        )
        model = Mock(spec=HookedTransformer)

        with pytest.raises(ValueError, match="sparsity must be in"):
            pruner.prune(model, scores, sparsity=1.5)

    def test_neuron_level_not_supported(self, pruner):
        """Test that neuron-level scores are rejected."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="neuron",
            node_scores={},
            timestamp="2025-04-13T12:00:00Z",
        )
        # Override the __post_init__ check to allow neuron level for testing
        scores.level = "neuron"
        model = Mock(spec=HookedTransformer)

        with pytest.raises(ValueError, match="only supports node-level"):
            pruner.prune(model, scores, sparsity=0.3)


class TestStructuralPrunerSelection:
    """Test node selection for pruning."""

    def test_select_nodes_to_prune(self, pruner, sample_circuit_scores):
        """Test that lowest-scoring nodes are selected."""
        nodes = pruner._select_nodes_to_prune(sample_circuit_scores, sparsity=0.3)

        # 30% of 7 nodes = 2.1, so 2 nodes should be selected
        assert len(nodes) == 2

        # Lowest-scoring nodes should be selected
        node_names = set(nodes.keys())
        assert "MLP 1" in node_names  # Lowest: 0.2
        assert "A0.4" in node_names  # Second lowest: 0.1

    def test_select_zero_sparsity(self, pruner, sample_circuit_scores):
        """Test that 0% sparsity selects no nodes."""
        nodes = pruner._select_nodes_to_prune(sample_circuit_scores, sparsity=0.0)
        assert len(nodes) == 0

    def test_select_full_sparsity(self, pruner, sample_circuit_scores):
        """Test that 100% sparsity selects all nodes."""
        nodes = pruner._select_nodes_to_prune(sample_circuit_scores, sparsity=1.0)
        assert len(nodes) == len(sample_circuit_scores.node_scores)

    def test_select_fractional_sparsity(self, pruner, sample_circuit_scores):
        """Test that sparsity rounds down correctly."""
        # 0.5 sparsity on 7 nodes = 3.5, should select 3
        nodes = pruner._select_nodes_to_prune(sample_circuit_scores, sparsity=0.5)
        assert len(nodes) == 3

    def test_select_scope_mlp_only(self, pruner, sample_circuit_scores):
        """scope='mlp' restricts the candidate set to MLP nodes only."""
        # 2 MLP nodes; 1.0 sparsity within them = both MLPs, no heads.
        nodes = pruner._select_nodes_to_prune(
            sample_circuit_scores, sparsity=1.0, scope="mlp"
        )
        assert set(nodes) == {"MLP 0", "MLP 1"}

    def test_select_scope_heads_only(self, pruner, sample_circuit_scores):
        """scope='heads' restricts the candidate set to attention heads only."""
        # 5 head nodes; 1.0 sparsity = all heads, no MLPs.
        nodes = pruner._select_nodes_to_prune(
            sample_circuit_scores, sparsity=1.0, scope="heads"
        )
        assert all(n.startswith("A") for n in nodes)
        assert len(nodes) == 5

    def test_select_scope_budget_within_component(self, pruner, sample_circuit_scores):
        """Sparsity budget is taken within the chosen component type."""
        # scope='mlp', 0.5 sparsity on 2 MLPs = 1 node (the lowest: MLP 1).
        nodes = pruner._select_nodes_to_prune(
            sample_circuit_scores, sparsity=0.5, scope="mlp"
        )
        assert set(nodes) == {"MLP 1"}

    def test_select_protect_layers(self, pruner, sample_circuit_scores):
        """protect_layers excludes both heads and MLPs in those layers."""
        # All nodes are in layer 0 or 1; protecting both leaves nothing.
        nodes = pruner._select_nodes_to_prune(
            sample_circuit_scores, sparsity=1.0, protect_layers=[0, 1]
        )
        assert nodes == {}

    def test_select_protect_layer_excludes_its_heads(self, pruner):
        """Heads/MLP of a protected layer are dropped before bottom-k."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.1, "A1.0": 0.2, "MLP 0": 0.05, "MLP 1": 0.3},
            timestamp="2025-04-13T12:00:00Z",
        )
        nodes = pruner._select_nodes_to_prune(scores, sparsity=1.0, protect_layers=[0])
        assert set(nodes) == {"A1.0", "MLP 1"}

    def test_select_invalid_scope_rejected(self, pruner):
        """An invalid scope is rejected by prune() with a ValueError."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.5},
            timestamp="2025-04-13T12:00:00Z",
        )
        model = Mock(spec=HookedTransformer)
        with pytest.raises(ValueError, match="scope must be"):
            pruner.prune(model, scores, sparsity=0.3, scope="everything")


class TestStructuralPrunerLogic:
    """Test pruning logic (without real model modifications)."""

    def test_format_nodes_short_list(self, pruner):
        """Test formatting short node lists."""
        nodes = {"A0.0": 0.5, "A0.1": 0.3}
        formatted = pruner._format_nodes(nodes)
        assert "A0.0" in formatted
        assert "A0.1" in formatted

    def test_format_nodes_long_list(self, pruner):
        """Test formatting long node lists (abbreviated)."""
        nodes = {f"A0.{i}": 0.5 for i in range(10)}
        formatted = pruner._format_nodes(nodes)
        # Should include "X more" notation
        assert "more" in formatted or len(nodes) <= 5

    def test_measure_sparsity_zero(self, pruner):
        """Test sparsity measurement when no weights are zeroed."""
        model_before = Mock(spec=HookedTransformer)
        model_before.parameters.return_value = [t.ones(10), t.ones(20)]

        model_after = Mock(spec=HookedTransformer)
        model_after.parameters.return_value = [t.ones(10), t.ones(20)]

        sparsity = pruner.measure_sparsity(model_after, model_before=model_before)
        assert sparsity == 0.0

    def test_measure_sparsity_partial(self, pruner):
        """Test sparsity with a partial weight zeroing.

        Masking keeps tensor shapes constant and only zeroes a subset of
        weights, so model_after has the same numel but more zeros.
        """
        model_before = Mock(spec=HookedTransformer)
        model_before.parameters.return_value = [t.ones(100)]

        model_after = Mock(spec=HookedTransformer)
        model_after.parameters.return_value = [t.cat([t.ones(70), t.zeros(30)])]

        sparsity = pruner.measure_sparsity(model_after, model_before=model_before)
        assert sparsity == 0.3

    def test_measure_sparsity_from_snapshot(self, pruner):
        """measure_sparsity works from a captured pre-masking count.

        This is the in-place path: there is no distinct unmasked model, only a
        snapshot of the nonzero count taken before masking.
        """
        model_after = Mock(spec=HookedTransformer)
        model_after.parameters.return_value = [t.cat([t.ones(70), t.zeros(30)])]

        sparsity = pruner.measure_sparsity(
            model_after, nonzero_before=100, total_params=100
        )
        assert sparsity == 0.3

    def test_measure_sparsity_full(self, pruner):
        """Test sparsity when all weights are zeroed."""
        model_before = Mock(spec=HookedTransformer)
        model_before.parameters.return_value = [t.ones(100)]

        model_after = Mock(spec=HookedTransformer)
        model_after.parameters.return_value = [t.zeros(100)]

        sparsity = pruner.measure_sparsity(model_after, model_before=model_before)
        assert sparsity == 1.0


class TestStructuralPrunerIntegration:
    """Integration tests with minimal real models."""

    def test_prune_dry_run(self, pruner, tiny_model):
        """dry_run=True masks a copy and leaves the original model untouched.

        Uses the real 2-layer HookedTransformer ``tiny_model`` fixture (2 heads,
        d_model=64) so node names match the model's actual components.
        """
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            # Two layers, two heads each, plus the two MLPs.
            node_scores={
                "A0.0": 0.9,
                "A0.1": 0.1,
                "A1.0": 0.8,
                "A1.1": 0.2,
                "MLP 0": 0.5,
                "MLP 1": 0.05,
            },
            timestamp="2025-04-13T12:00:00Z",
        )

        # Snapshot every original parameter before pruning.
        original_snapshot = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}

        # dry_run is the deprecated alias for copy mode — it must warn.
        with pytest.warns(DeprecationWarning):
            pruned = pruner.prune(tiny_model, scores, sparsity=0.5, dry_run=True)

        # 1. The returned model is a distinct object (a copy), not the original.
        assert pruned is not tiny_model

        # 2. The original model is completely unmodified.
        for name, param in tiny_model.named_parameters():
            assert t.allclose(param, original_snapshot[name]), (
                f"dry_run=True mutated original parameter {name!r}"
            )

        # 3. Masking actually happened on the copy: at least one parameter differs.
        pruned_params = dict(pruned.named_parameters())
        changed = any(
            not t.allclose(pruned_params[n], original_snapshot[n])
            for n in original_snapshot
        )
        assert changed, "dry_run=True produced a copy identical to the original (no masking)"

    def test_prune_returns_copy_on_dry_run(self, pruner, sample_circuit_scores):
        """Test that dry_run=True returns a copy."""
        model = Mock(spec=HookedTransformer)
        model.parameters.return_value = []

        # Mock the _select_nodes_to_prune to return empty dict (no pruning)
        with patch.object(pruner, "_select_nodes_to_prune", return_value={}):
            result = pruner.prune(model, sample_circuit_scores, dry_run=True)

        # Should return a model (or mock) for dry_run
        assert result is not None


class TestStructuralPrunerEdgeCases:
    """Test edge cases and error conditions."""

    def test_prune_with_no_prunable_nodes(self, pruner):
        """Test pruning when selected sparsity results in no nodes."""
        scores = CircuitScores(
            task="ioi",
            model="gpt2",
            algorithm="eap",
            level="node",
            node_scores={"A0.0": 0.5},
            timestamp="2025-04-13T12:00:00Z",
        )
        model = Mock(spec=HookedTransformer)
        model.parameters.return_value = []

        # 0.1% sparsity on 1 node = 0 nodes, should log warning
        result = pruner.prune(model, scores, sparsity=0.001, dry_run=True)

        # Should return model unchanged
        assert result is not None

    def test_all_algorithms_supported(self, pruner):
        """Test that all algorithms are accepted in CircuitScores."""
        for algo in ["eap", "eap-ig", "acdc", "ibcircuit"]:
            scores = CircuitScores(
                task="ioi",
                model="gpt2",
                algorithm=algo,
                level="node",
                node_scores={"A0.0": 0.5},
                timestamp="2025-04-13T12:00:00Z",
            )
            # Should not raise
            assert scores.algorithm == algo
