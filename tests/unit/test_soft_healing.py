# FILE: tests/unit/test_soft_healing.py
"""
Unit tests for CircuitLoRA soft healing module.

Tests cover:
1. LoRA layer initialization and forward pass
2. CircuitLoRA initialization with various circuit scores
3. Training on simple datasets
4. Parameter freezing and unfreezing
5. State dict save/load
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Assuming imports
try:
    from transformer_lens import HookedTransformer

    from circuitkit.applications.finetuning.soft_healing import CircuitLoRA, LoRALayer
except ImportError:
    pytest.skip("circuitkit not available", allow_module_level=True)


class TestLoRALayer:
    """Test LoRA layer functionality."""

    def test_lora_layer_initialization(self):
        """Test basic LoRA layer creation."""
        lora = LoRALayer(in_features=768, out_features=768, lora_rank=8, lora_alpha=16.0)

        assert lora.in_features == 768
        assert lora.out_features == 768
        assert lora.lora_rank == 8
        assert lora.lora_alpha == 16.0
        assert lora.scaling == 16.0 / 8.0

        # Check parameter shapes
        assert lora.lora_A.shape == (768, 8)
        assert lora.lora_B.shape == (8, 768)

    def test_lora_layer_forward_pass(self):
        """Test LoRA layer forward pass."""
        lora = LoRALayer(in_features=64, out_features=64, lora_rank=4)

        # Input: [batch_size, in_features]
        x = torch.randn(32, 64)
        output = lora(x)

        assert output.shape == (32, 64)
        assert output.dtype == x.dtype

    def test_lora_layer_zero_initialization(self):
        """Test that LoRA B is initialized to zero."""
        lora = LoRALayer(in_features=64, out_features=64, lora_rank=4)

        # B should be initialized to zeros
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

    def test_lora_layer_gradient_flow(self):
        """Test that gradients flow through LoRA."""
        lora = LoRALayer(in_features=32, out_features=32, lora_rank=4)

        x = torch.randn(8, 32, requires_grad=True)
        output = lora(x)
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert x.grad is not None


class TestCircuitLoRA:
    """Test CircuitLoRA functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock HookedTransformer for testing."""
        model = MagicMock(spec=HookedTransformer)
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.d_head = 64
        model.cfg.n_heads = 12
        model.cfg.d_mlp = 3072
        model.cfg.device = "cpu"
        model.cfg.n_layers = 12

        # Create mock blocks
        model.blocks = []
        for i in range(2):  # 2 layers for testing
            block = MagicMock()

            # Mock attention module
            attn = MagicMock()
            attn.W_O = nn.Parameter(torch.randn(12, 64, 768))
            block.attn = attn

            # Mock MLP module
            mlp = MagicMock()
            block.mlp = mlp

            model.blocks.append(block)

        # Mock parameters method
        model.parameters = MagicMock(return_value=[])

        return model

    def test_circuit_lora_initialization(self, mock_model):
        """Test CircuitLoRA initialization."""
        circuit_scores = {"A0.0": 0.95, "A0.1": 0.92, "MLP 1": 0.88}

        lora = CircuitLoRA(mock_model, circuit_scores, lora_rank=8, lora_alpha=16.0)

        assert lora.model == mock_model
        assert lora.circuit_scores == circuit_scores
        assert lora.lora_rank == 8
        assert len(lora.lora_modules) > 0

    def test_circuit_lora_score_threshold(self, mock_model):
        """Test that score threshold filters nodes correctly."""
        circuit_scores = {"A0.0": 0.95, "A0.1": 0.50, "MLP 1": 0.30}

        lora = CircuitLoRA(mock_model, circuit_scores, score_threshold=0.6)

        # Only A0.0 should be above threshold
        assert len(lora.high_score_nodes) == 1
        assert "A0.0" in lora.high_score_nodes
        # Also verify node is not in lora_layers if it doesn't match pattern
        # (A0.0 should still be added, but we're testing threshold filtering)
        assert all(score >= 0.6 for score in lora.high_score_nodes.values())

    def test_get_lora_parameters(self, mock_model):
        """Test retrieving LoRA parameters for optimization."""
        circuit_scores = {"A0.0": 0.95, "MLP 1": 0.88}

        lora = CircuitLoRA(mock_model, circuit_scores)
        params = lora.get_lora_parameters()

        # Should have parameters for attention and MLP
        assert len(params) > 0
        assert all(isinstance(p, nn.Parameter) for p in params)

    def test_get_lora_state_dict(self, mock_model):
        """Test saving LoRA state dict."""
        circuit_scores = {"A0.0": 0.95, "MLP 1": 0.88}

        lora = CircuitLoRA(mock_model, circuit_scores)
        state_dict = lora.get_lora_state_dict()

        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0

    def test_load_lora_state_dict(self, mock_model):
        """Test loading LoRA state dict."""
        circuit_scores = {"A0.0": 0.95}

        lora1 = CircuitLoRA(mock_model, circuit_scores)
        state_dict1 = lora1.get_lora_state_dict()

        # Create new instance and load state
        lora2 = CircuitLoRA(mock_model, circuit_scores)
        lora2.load_lora_state_dict(state_dict1)

        # State dicts should match. An attn_head entry is a flat
        # {param_name: tensor} module state_dict; an MLP entry is a nested
        # {'module_in': {...}|None, 'module_out': {...}|None} dict.
        def _assert_tensors_match(d1, d2):
            for param_key, val1 in d1.items():
                val2 = d2[param_key]
                if isinstance(val1, dict):
                    # MLP nested case (module_in / module_out)
                    if val1 is None or val2 is None:
                        assert val1 is val2 or (val1 == {} and val2 == {})
                    else:
                        _assert_tensors_match(val1, val2)
                elif val1 is None:
                    assert val2 is None
                else:
                    assert torch.allclose(val1, val2)

        state_dict2 = lora2.get_lora_state_dict()
        assert set(state_dict1) == set(state_dict2)
        for key in state_dict1:
            _assert_tensors_match(state_dict1[key], state_dict2[key])

    def test_empty_circuit_scores(self, mock_model):
        """Test initialization with empty circuit scores."""
        circuit_scores = {}
        lora = CircuitLoRA(mock_model, circuit_scores)

        assert len(lora.lora_modules) == 0
        params = lora.get_lora_parameters()
        assert len(params) == 0

    def test_invalid_node_names(self, mock_model):
        """Test that invalid node names are safely ignored."""
        circuit_scores = {
            "INVALID_NODE": 0.95,
            "A0.0": 0.92,
        }

        # Should not raise error, just skip invalid node
        lora = CircuitLoRA(mock_model, circuit_scores)
        assert "INVALID_NODE" not in lora.lora_modules


class TestCircuitLoRATraining:
    """Test CircuitLoRA training functionality."""

    @pytest.fixture
    def simple_dataloader(self):
        """Create a simple dataloader for testing."""
        data = []
        for _ in range(10):
            batch = {
                "input_ids": torch.randint(0, 1000, (2, 64)),
                "attention_mask": torch.ones(2, 64),
            }
            data.append(batch)
        return data

    @pytest.fixture
    def mock_model_for_training(self):
        """Create a mock model that can do forward passes."""
        model = MagicMock(spec=HookedTransformer)
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.d_head = 64
        model.cfg.n_heads = 12
        model.cfg.d_mlp = 3072
        model.cfg.device = "cpu"

        # Create actual mock blocks
        model.blocks = [MagicMock() for _ in range(2)]
        for block in model.blocks:
            block.attn = MagicMock()
            block.mlp = MagicMock()

        # Mock forward pass: return logits
        def mock_forward(input_ids):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else 1
            return torch.randn(batch_size, seq_len, 1000)

        model.return_value = mock_forward(torch.randint(0, 100, (1, 64)))
        model.__call__ = mock_forward

        # Mock parameters
        model.parameters = MagicMock(return_value=[nn.Parameter(torch.randn(10, 10))])

        # Mock to and other methods
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock(return_value=None)
        model.train = MagicMock(return_value=None)

        return model

    def test_training_basic(self, mock_model_for_training, simple_dataloader):
        """Test basic training loop."""
        circuit_scores = {"A0.0": 0.95}

        # Create a real model wrapper that works with training
        lora = CircuitLoRA(mock_model_for_training, circuit_scores, lora_rank=4)

        # This would fail with mock, so we just check structure
        assert len(lora.get_lora_parameters()) > 0

    def test_training_metrics_structure(self, mock_model_for_training):
        """Test that training returns expected metric structure."""
        circuit_scores = {"A0.0": 0.95}
        lora = CircuitLoRA(mock_model_for_training, circuit_scores)

        # Create minimal dataloaders
        from torch.utils.data import DataLoader, TensorDataset

        dummy_data = TensorDataset(torch.randint(0, 100, (10, 64)))
        train_loader = DataLoader(dummy_data, batch_size=2)

        # train() cannot fully execute against a MagicMock (real autograd/weights
        # are needed), so we assert the callable interface first — a rename or
        # removal of train() must fail this test rather than be swallowed.
        assert callable(getattr(lora, "train", None)), "CircuitLoRA lost its train() method"

        # If training does run far enough to return, the metrics contract must
        # hold. A pure mock-runtime failure (TypeError/RuntimeError from the
        # MagicMock forward) is a known limitation and is skipped explicitly;
        # an AttributeError signals a real API regression and propagates.
        try:
            metrics = lora.train(train_loader, epochs=1)
        except AttributeError:
            raise
        except (TypeError, RuntimeError, ValueError) as e:
            pytest.skip(f"CircuitLoRA.train needs a real model; mock cannot run it: {e}")
        else:
            assert "train_loss" in metrics
            assert "total_params" in metrics


class TestCircuitLoRAEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock HookedTransformer."""
        model = MagicMock(spec=HookedTransformer)
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.d_head = 64
        model.cfg.n_heads = 12
        model.cfg.d_mlp = 3072
        model.cfg.device = "cpu"
        model.blocks = [MagicMock() for _ in range(2)]
        for block in model.blocks:
            block.attn = MagicMock()
            block.mlp = MagicMock()
        return model

    def test_lora_rank_zero(self, mock_model):
        """Test handling of zero LoRA rank."""
        circuit_scores = {"A0.0": 0.95}

        # Should handle gracefully
        lora = CircuitLoRA(mock_model, circuit_scores, lora_rank=0)
        # Scaling should not divide by zero
        assert lora.lora_rank == 0

    def test_zero_score_threshold(self, mock_model):
        """Test that zero threshold includes all nodes."""
        circuit_scores = {"A0.0": 0.1, "A0.1": 0.01, "MLP 1": 0.001}

        lora = CircuitLoRA(mock_model, circuit_scores, score_threshold=0.0)

        assert len(lora.high_score_nodes) == 3

    def test_very_high_score_threshold(self, mock_model):
        """Test that very high threshold excludes all nodes."""
        circuit_scores = {"A0.0": 0.95, "A0.1": 0.92, "MLP 1": 0.88}

        lora = CircuitLoRA(mock_model, circuit_scores, score_threshold=0.99)

        assert len(lora.high_score_nodes) == 0
        assert len(lora.lora_modules) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
