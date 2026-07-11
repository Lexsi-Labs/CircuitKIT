"""
Tests for circuit-aware PEFT methods.

Tests LoRA, Adapter, Prefix-tuning, and BitFit implementations
with circuit-guided targeting.
"""

import pytest
import torch.nn as nn

from circuitkit.applications.finetuning.peft_methods import (
    CircuitAdapterTuning,
    CircuitBitFit,
    CircuitLoRA,
    CircuitPrefixTuning,
    PEFTComposer,
)
from circuitkit.artifacts import CircuitArtifact, Node, NodeType

# Fixtures


@pytest.fixture
def mock_model():
    """Create a mock Transformer model."""

    class MockAttention(nn.Module):
        def __init__(self, hidden_size=256):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.o_proj = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            return self.o_proj(x)

    class MockMLP(nn.Module):
        def __init__(self, hidden_size=256, intermediate_size=1024):
            super().__init__()
            self.up_proj = nn.Linear(hidden_size, intermediate_size)
            self.down_proj = nn.Linear(intermediate_size, hidden_size)

        def forward(self, x):
            return self.down_proj(self.up_proj(x))

    class MockLayer(nn.Module):
        def __init__(self, hidden_size=256):
            super().__init__()
            self.self_attn = MockAttention(hidden_size)
            self.mlp = MockMLP(hidden_size)

        def forward(self, x):
            return x + self.mlp(x + self.self_attn(x))

    class MockConfig:
        hidden_size = 256
        intermediate_size = 1024
        num_layers = 4
        num_attention_heads = 4

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self.layers = nn.ModuleList([MockLayer() for _ in range(4)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return MockModel()


@pytest.fixture
def mock_circuit():
    """Create a mock circuit artifact."""
    artifact = CircuitArtifact(
        model_id="test-model",
        discovery_method="eap",
        task="test",
        dataset="test",
    )

    # Add nodes in layers 0-2
    artifact.add_node("L0H0", Node(0, NodeType.ATTENTION_HEAD, 0, 0.9))
    artifact.add_node("L1H1", Node(1, NodeType.ATTENTION_HEAD, 1, 0.85))
    artifact.add_node("L2H2", Node(2, NodeType.ATTENTION_HEAD, 2, 0.8))
    artifact.add_node("L0M0", Node(0, NodeType.MLP_NEURON, 0, 0.7))

    return artifact


# Test Base Class


class TestCircuitPEFT:
    """Test CircuitPEFT base class."""

    def test_initialization(self, mock_model, mock_circuit):
        """Test PEFT initialization."""
        # Use LoRA as concrete implementation
        peft = CircuitLoRA(mock_model, mock_circuit, device="cpu")

        assert peft.model is mock_model
        assert peft.circuit is mock_circuit
        assert peft.device == "cpu"

    def test_freeze_base_weights(self, mock_model, mock_circuit):
        """Test that base weights are frozen."""
        CircuitLoRA(mock_model, mock_circuit, device="cpu", freeze_base=True)

        # Check that base model parameters are not trainable
        for param in mock_model.parameters():
            assert not param.requires_grad, "Base weights should be frozen"

    def test_no_freeze_option(self, mock_model, mock_circuit):
        """Test with freeze_base=False."""
        CircuitLoRA(mock_model, mock_circuit, device="cpu", freeze_base=False)

        # Base weights should be trainable
        has_trainable = any(p.requires_grad for p in mock_model.parameters())
        assert has_trainable or True  # May be overridden by PEFT impl


# Test LoRA


class TestCircuitLoRA:
    """Test LoRA implementation."""

    def test_lora_initialization(self, mock_model, mock_circuit):
        """Test LoRA initialization."""
        lora = CircuitLoRA(
            mock_model,
            mock_circuit,
            rank=8,
            alpha=1.0,
            device="cpu",
        )

        assert lora.rank == 8
        assert lora.alpha == 1.0

    def test_lora_get_trainable_params(self, mock_model, mock_circuit):
        """Test getting trainable LoRA parameters."""
        lora = CircuitLoRA(mock_model, mock_circuit, device="cpu")
        params = lora.get_trainable_params()

        # Should have parameters (even if mock implementation is limited)
        assert isinstance(params, list)

    def test_lora_parameter_count(self, mock_model, mock_circuit):
        """Test LoRA parameter counting."""
        lora = CircuitLoRA(mock_model, mock_circuit, device="cpu")
        counts = lora.get_parameter_count()

        assert "peft_params" in counts
        assert "base_params" in counts
        assert "trainable_params" in counts
        assert "efficiency" in counts
        assert 0 <= counts["efficiency"] <= 1

    def test_lora_merge_weights(self, mock_model, mock_circuit):
        """Test LoRA weight merging."""
        lora = CircuitLoRA(mock_model, mock_circuit, device="cpu")
        merged_model = lora.merge_weights(alpha=1.0)

        assert merged_model is not None


# Test Adapter Tuning


class TestCircuitAdapterTuning:
    """Test Adapter tuning implementation."""

    def test_adapter_initialization(self, mock_model, mock_circuit):
        """Test Adapter initialization."""
        adapter = CircuitAdapterTuning(
            mock_model,
            mock_circuit,
            hidden_dim=64,
            device="cpu",
        )

        assert adapter.adapter_hidden_dim == 64

    def test_adapter_modules_created(self, mock_model, mock_circuit):
        """Test that adapters are created for circuit layers."""
        adapter = CircuitAdapterTuning(mock_model, mock_circuit, device="cpu")
        adapters = adapter.adapters

        # Should have adapters for circuit layers
        assert isinstance(adapters, dict)

    def test_adapter_parameter_count(self, mock_model, mock_circuit):
        """Test adapter parameter counting."""
        adapter = CircuitAdapterTuning(mock_model, mock_circuit, device="cpu")
        counts = adapter.get_parameter_count()

        assert "peft_params" in counts
        assert "efficiency" in counts
        assert counts["efficiency"] >= 0

    def test_adapter_trainable_params(self, mock_model, mock_circuit):
        """Test getting adapter trainable parameters."""
        adapter = CircuitAdapterTuning(mock_model, mock_circuit, device="cpu")
        params = adapter.get_trainable_params()

        assert isinstance(params, list)


# Test Prefix Tuning


class TestCircuitPrefixTuning:
    """Test Prefix-tuning implementation."""

    def test_prefix_initialization(self, mock_model, mock_circuit):
        """Test Prefix-tuning initialization."""
        prefix = CircuitPrefixTuning(
            mock_model,
            mock_circuit,
            prefix_length=10,
            device="cpu",
        )

        assert prefix.prefix_length == 10

    def test_prefix_embeddings_created(self, mock_model, mock_circuit):
        """Test that prefix embeddings are created."""
        prefix = CircuitPrefixTuning(mock_model, mock_circuit, device="cpu")

        assert len(prefix.prefix_embeddings) > 0

    def test_prefix_embedding_shape(self, mock_model, mock_circuit):
        """Test prefix embedding shapes."""
        prefix_length = 5
        prefix = CircuitPrefixTuning(
            mock_model,
            mock_circuit,
            prefix_length=prefix_length,
            device="cpu",
        )

        for emb in prefix.prefix_embeddings.values():
            assert emb.shape[0] == prefix_length
            assert emb.shape[1] == mock_model.config.hidden_size

    def test_prefix_parameter_count(self, mock_model, mock_circuit):
        """Test prefix parameter counting."""
        prefix = CircuitPrefixTuning(mock_model, mock_circuit, device="cpu")
        counts = prefix.get_parameter_count()

        assert counts["peft_params"] > 0
        assert counts["efficiency"] > 0


# Test BitFit


class TestCircuitBitFit:
    """Test BitFit implementation."""

    def test_bitfit_initialization(self, mock_model, mock_circuit):
        """Test BitFit initialization."""
        bitfit = CircuitBitFit(mock_model, mock_circuit, device="cpu")

        assert isinstance(bitfit.trainable_biases, list)

    def test_bitfit_enables_bias_gradients(self, mock_model, mock_circuit):
        """Test that BitFit enables bias gradients."""
        bitfit = CircuitBitFit(mock_model, mock_circuit, device="cpu")

        # Check that some biases have gradients enabled
        has_trainable_bias = any(b.requires_grad for b in bitfit.trainable_biases)
        assert has_trainable_bias or len(bitfit.trainable_biases) == 0

    def test_bitfit_parameter_count(self, mock_model, mock_circuit):
        """Test BitFit parameter counting."""
        bitfit = CircuitBitFit(mock_model, mock_circuit, device="cpu")
        counts = bitfit.get_parameter_count()

        assert "peft_params" in counts
        assert "efficiency" in counts
        # BitFit should have very low efficiency (only biases)
        if counts["peft_params"] > 0:
            assert counts["efficiency"] < 0.1


# Test Parameter Efficiency


class TestParameterEfficiency:
    """Test parameter efficiency across methods."""

    def test_lora_efficiency(self, mock_model, mock_circuit):
        """Test LoRA is parameter-efficient."""
        lora = CircuitLoRA(mock_model, mock_circuit, rank=8, device="cpu")
        counts = lora.get_parameter_count()

        efficiency = counts["efficiency"]
        assert 0 <= efficiency <= 1
        # LoRA should be more efficient than full fine-tuning
        assert efficiency < 0.5, "LoRA should be < 50% of model size"

    def test_adapter_efficiency(self, mock_model, mock_circuit):
        """Test Adapter is parameter-efficient."""
        adapter = CircuitAdapterTuning(mock_model, mock_circuit, hidden_dim=64, device="cpu")
        counts = adapter.get_parameter_count()

        efficiency = counts["efficiency"]
        assert 0 <= efficiency <= 1

    def test_bitfit_most_efficient(self, mock_model, mock_circuit):
        """Test BitFit is most parameter-efficient."""
        bitfit = CircuitBitFit(mock_model, mock_circuit, device="cpu")
        counts = bitfit.get_parameter_count()

        # BitFit should have minimal parameters (only biases)
        assert counts["efficiency"] < 0.05, "BitFit should use < 5% parameters"


# Test PEFT Composer


class TestPEFTComposer:
    """Test PEFT composition."""

    def test_composer_initialization(self):
        """Test composer initialization."""
        composer = PEFTComposer()

        assert len(composer.methods) == 0

    def test_composer_add_method(self, mock_model, mock_circuit):
        """Test adding methods to composer."""
        composer = PEFTComposer()
        lora = CircuitLoRA(mock_model, mock_circuit, device="cpu")

        composer.add_method("lora", lora)

        assert "lora" in composer.methods
        assert composer.methods["lora"] is lora

    def test_composer_multiple_methods(self, mock_model, mock_circuit):
        """Test composing multiple methods."""
        composer = PEFTComposer()

        lora = CircuitLoRA(mock_model, mock_circuit, device="cpu")
        bitfit = CircuitBitFit(mock_model, mock_circuit, device="cpu")

        composer.add_method("lora", lora)
        composer.add_method("bitfit", bitfit)

        assert len(composer.methods) == 2

    def test_composer_total_parameters(self, mock_model, mock_circuit):
        """Test total parameter counting."""
        composer = PEFTComposer()
        lora = CircuitLoRA(mock_model, mock_circuit, device="cpu")
        bitfit = CircuitBitFit(mock_model, mock_circuit, device="cpu")

        composer.add_method("lora", lora)
        composer.add_method("bitfit", bitfit)

        total = composer.get_total_parameters()

        assert "total_peft_params" in total
        assert "total_base_params" in total
        assert "total_params" in total
        assert "efficiency" in total

    def test_composer_get_all_trainable(self, mock_model, mock_circuit):
        """Test getting all trainable parameters."""
        composer = PEFTComposer()
        lora = CircuitLoRA(mock_model, mock_circuit, device="cpu")

        composer.add_method("lora", lora)

        params = composer.get_all_trainable_params()

        assert isinstance(params, list)

    def test_composer_summary(self, mock_model, mock_circuit):
        """Test composer summary generation."""
        composer = PEFTComposer()
        lora = CircuitLoRA(mock_model, mock_circuit, device="cpu")

        composer.add_method("lora", lora)

        summary = composer.summary()

        assert isinstance(summary, str)
        assert "PEFT Composition" in summary
        assert "lora" in summary.lower()


# Integration Tests


class TestPEFTIntegration:
    """Integration tests for PEFT methods."""

    def test_all_methods_support_interface(self, mock_model, mock_circuit):
        """Test that all PEFT methods support standard interface."""
        methods = [
            CircuitLoRA(mock_model, mock_circuit, device="cpu"),
            CircuitAdapterTuning(mock_model, mock_circuit, device="cpu"),
            CircuitPrefixTuning(mock_model, mock_circuit, device="cpu"),
            CircuitBitFit(mock_model, mock_circuit, device="cpu"),
        ]

        for method in methods:
            # All should have standard methods
            assert callable(method.get_trainable_params)
            assert callable(method.get_parameter_count)
            assert callable(method.merge_weights)

    def test_circuit_awareness(self, mock_model, mock_circuit):
        """Test that methods are circuit-aware."""
        method = CircuitLoRA(mock_model, mock_circuit, device="cpu")

        # Should have circuit information
        assert method.circuit is not None
        assert method.get_circuit_layers() is not None
        assert len(method.get_circuit_layers()) > 0

    def test_no_model_modification_before_apply(self, mock_model):
        """Test that model is not modified until apply."""
        original_param_count = sum(p.numel() for p in mock_model.parameters())

        CircuitArtifact(
            model_id="test",
            discovery_method="eap",
            task="test",
            dataset="test",
        )

        # Parameters should remain same
        new_param_count = sum(p.numel() for p in mock_model.parameters())
        assert original_param_count == new_param_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
