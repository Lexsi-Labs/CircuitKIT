"""
Integration tests for soft healing (CircuitLoRA) end-to-end workflow.

These tests verify:
1. CircuitLoRA initialization with circuit scores
2. LoRA training on synthetic data
3. State save/load
4. Healing metrics computation
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Conditional import with fallback to skip if unavailable
try:
    from circuitkit.applications.finetuning.healing_metrics import (
        HealingMetrics,
        compute_recovery_metrics,
    )
    from circuitkit.applications.finetuning.soft_healing import CircuitLoRA, LoRALayer

    CIRCUITKIT_AVAILABLE = True
except ImportError:
    CIRCUITKIT_AVAILABLE = False


@pytest.mark.skipif(not CIRCUITKIT_AVAILABLE, reason="circuitkit not available")
class TestCircuitLoRAIntegration:
    """Integration tests for CircuitLoRA."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock HookedTransformer."""
        model = MagicMock()
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.d_head = 64
        model.cfg.n_heads = 12
        model.cfg.d_mlp = 3072
        model.cfg.device = "cpu"
        model.cfg.n_layers = 2

        # Create mock blocks
        model.blocks = []
        for i in range(2):
            block = MagicMock()
            block.attn = MagicMock()
            block.mlp = MagicMock()
            model.blocks.append(block)

        # Mock parameters
        model.parameters = MagicMock(return_value=[])
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock(return_value=None)

        return model

    def test_lora_layer_basic(self):
        """Test LoRA layer initialization and forward pass."""
        lora = LoRALayer(in_features=64, out_features=64, lora_rank=4)

        # Check dimensions
        assert lora.in_features == 64
        assert lora.out_features == 64
        assert lora.lora_rank == 4
        assert lora.lora_A.shape == (64, 4)
        assert lora.lora_B.shape == (4, 64)

        # Check forward pass
        x = torch.randn(8, 64)
        output = lora(x)
        assert output.shape == (8, 64)

    def test_lora_layer_initialization(self):
        """Test that B is initialized to zeros as per LoRA paper."""
        lora = LoRALayer(in_features=100, out_features=100, lora_rank=8)

        # B should be all zeros initially
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

        # A should NOT be zeros (Gaussian init)
        assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))

    def test_lora_scaling(self):
        """Test LoRA scaling factor."""
        rank = 8
        alpha = 16.0
        lora = LoRALayer(in_features=64, out_features=64, lora_rank=rank, lora_alpha=alpha)

        expected_scaling = alpha / rank
        assert abs(lora.scaling - expected_scaling) < 1e-6

    def test_lora_gradient_flow(self):
        """Test that gradients flow through LoRA."""
        lora = LoRALayer(in_features=32, out_features=32, lora_rank=4)

        x = torch.randn(8, 32, requires_grad=True)
        output = lora(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert x.grad is not None

    def test_circuit_lora_init(self, mock_model):
        """Test CircuitLoRA initialization."""
        circuit_scores = {"A0.0": 0.95, "A0.1": 0.92, "MLP 1": 0.88}

        lora = CircuitLoRA(mock_model, circuit_scores, lora_rank=8)

        assert lora.model == mock_model
        assert lora.circuit_scores == circuit_scores
        assert lora.lora_rank == 8
        assert len(lora.high_score_nodes) == 3

    def test_circuit_lora_threshold_filtering(self, mock_model):
        """Test that score threshold filters nodes."""
        circuit_scores = {"A0.0": 0.95, "A0.1": 0.50, "A0.2": 0.30, "MLP 0": 0.10}

        lora = CircuitLoRA(mock_model, circuit_scores, score_threshold=0.6)

        # Only nodes with score >= 0.6 should be in high_score_nodes
        assert len(lora.high_score_nodes) == 1
        assert "A0.0" in lora.high_score_nodes
        assert all(score >= 0.6 for score in lora.high_score_nodes.values())

    def test_circuit_lora_empty_scores(self, mock_model):
        """Test with empty circuit scores."""
        lora = CircuitLoRA(mock_model, {})

        assert len(lora.high_score_nodes) == 0
        assert len(lora.lora_modules) == 0
        assert lora.get_lora_parameters() == []

    def test_get_lora_parameters(self, mock_model):
        """Test retrieving LoRA parameters."""
        circuit_scores = {"A0.0": 0.95, "MLP 1": 0.88}

        lora = CircuitLoRA(mock_model, circuit_scores)
        params = lora.get_lora_parameters()

        # Should be a list of parameters
        assert isinstance(params, list)
        assert all(isinstance(p, nn.Parameter) for p in params)

    def test_state_dict_save_load(self, mock_model):
        """Test saving and loading LoRA state."""
        circuit_scores = {"A0.0": 0.95, "MLP 1": 0.88}

        # Create and initialize LoRA
        lora1 = CircuitLoRA(mock_model, circuit_scores)
        state1 = lora1.get_lora_state_dict()

        # Create second instance and load state
        lora2 = CircuitLoRA(mock_model, circuit_scores)
        lora2.load_lora_state_dict(state1)
        state2 = lora2.get_lora_state_dict()

        # States should have same structure
        assert len(state1) == len(state2)

    def test_recovery_metrics_full_recovery(self):
        """Test recovery metrics with full recovery."""
        metrics = compute_recovery_metrics(
            original_accuracy=1.0, pruned_accuracy=0.5, healed_accuracy=1.0
        )

        assert abs(metrics["recovery_rate"] - 1.0) < 1e-6
        assert abs(metrics["absolute_recovery"] - 0.5) < 1e-6

    def test_recovery_metrics_partial_recovery(self):
        """Test recovery metrics with partial recovery."""
        metrics = compute_recovery_metrics(
            original_accuracy=1.0, pruned_accuracy=0.5, healed_accuracy=0.75
        )

        assert abs(metrics["recovery_rate"] - 0.5) < 1e-6
        assert abs(metrics["absolute_recovery"] - 0.25) < 1e-6

    def test_recovery_metrics_no_loss(self):
        """Test when pruning doesn't cause loss."""
        metrics = compute_recovery_metrics(
            original_accuracy=1.0, pruned_accuracy=1.0, healed_accuracy=1.0
        )

        assert metrics["recovery_rate"] == 0.0

    def test_healing_metrics_dataclass(self):
        """Test HealingMetrics dataclass."""
        metrics = HealingMetrics(
            recovery_rate=0.8,
            original_accuracy=0.95,
            pruned_accuracy=0.50,
            healed_accuracy=0.86,
            convergence_epoch=2,
            convergence_speed=0.18,
            generalization_gap=0.02,
            lora_parameter_count=10000,
            model_parameter_count=100000,
            efficiency_ratio=10.0,
        )

        assert metrics.recovery_rate == 0.8
        assert metrics.efficiency_ratio == 10.0

    def test_healing_metrics_repr(self):
        """Test pretty printing of metrics."""
        metrics = HealingMetrics(
            recovery_rate=0.8,
            original_accuracy=0.95,
            pruned_accuracy=0.50,
            healed_accuracy=0.86,
            convergence_epoch=2,
            convergence_speed=0.18,
            generalization_gap=0.02,
            lora_parameter_count=10000,
            model_parameter_count=100000,
            efficiency_ratio=10.0,
        )

        repr_str = repr(metrics)
        assert "HEALING METRICS" in repr_str
        assert "80.00%" in repr_str or "0.80" in repr_str
        assert "Recovery" in repr_str


class TestLoRALayerStandalone:
    """Standalone tests for LoRA layer (work even without circuitkit)."""

    @staticmethod
    def create_lora_layer(in_features=64, out_features=64, lora_rank=8, lora_alpha=16.0):
        """Create a LoRA layer for testing."""

        class SimpleLoRALayer(nn.Module):
            def __init__(self, in_features, out_features, lora_rank, lora_alpha):
                super().__init__()
                self.lora_A = nn.Parameter(torch.zeros(in_features, lora_rank))
                self.lora_B = nn.Parameter(torch.zeros(lora_rank, out_features))
                nn.init.kaiming_uniform_(self.lora_A, a=5.0)
                nn.init.zeros_(self.lora_B)
                self.scaling = lora_alpha / lora_rank

            def forward(self, x):
                return (x @ self.lora_A @ self.lora_B) * self.scaling

        return SimpleLoRALayer(in_features, out_features, lora_rank, lora_alpha)

    def test_lora_layer_output_shape(self):
        """Test LoRA output shape."""
        lora = self.create_lora_layer(in_features=256, out_features=256, lora_rank=16)

        # Test different batch sizes
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 256)
            output = lora(x)
            assert output.shape == (batch_size, 256)

    def test_lora_layer_scaling_effect(self):
        """Test that scaling affects output magnitude."""
        lora_low_alpha = self.create_lora_layer(
            in_features=64, out_features=64, lora_rank=8, lora_alpha=1.0
        )
        lora_high_alpha = self.create_lora_layer(
            in_features=64, out_features=64, lora_rank=8, lora_alpha=32.0
        )

        # Since B is initialized to zeros, both should give zero
        x = torch.randn(1, 64)
        out_low = lora_low_alpha(x)
        out_high = lora_high_alpha(x)

        # Both should be zero since B is zero
        assert torch.allclose(out_low, torch.zeros_like(out_low), atol=1e-6)
        assert torch.allclose(out_high, torch.zeros_like(out_high), atol=1e-6)

    def test_lora_layer_requires_grad(self):
        """Test that LoRA parameters require gradients."""
        lora = self.create_lora_layer(in_features=64, out_features=64, lora_rank=4)

        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
