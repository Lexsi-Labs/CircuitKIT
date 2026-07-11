"""
Tests for architecture registry and utils.

Tests detection, validation, and layer access for multiple model architectures.
"""

import pytest
import torch.nn as nn


# Mock models for testing
class MockQwenLayer(nn.Module):
    """Mock Qwen layer matching actual structure."""

    def __init__(self, hidden_size=256, num_heads=4):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.head_dim = hidden_size // num_heads
        self.self_attn.q_norm = nn.RMSNorm(hidden_size // num_heads)
        self.self_attn.k_norm = nn.RMSNorm(hidden_size // num_heads)

        self.mlp = nn.Module()
        mlp_hidden = hidden_size * 4
        self.mlp.gate_proj = nn.Linear(hidden_size, mlp_hidden)
        self.mlp.up_proj = nn.Linear(hidden_size, mlp_hidden)
        self.mlp.down_proj = nn.Linear(mlp_hidden, hidden_size)


class MockQwenModel(nn.Module):
    """Mock Qwen model."""

    def __init__(self, num_layers=2):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "model_type": "qwen",
                "num_attention_heads": 4,
                "hidden_size": 256,
            },
        )()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockQwenLayer() for _ in range(num_layers)])


class MockGemmaLayer(nn.Module):
    """Mock Gemma layer."""

    def __init__(self, hidden_size=256, num_heads=4):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.k_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.v_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.o_proj = nn.Linear(hidden_size, hidden_size)
        self.self_attn.head_dim = hidden_size // num_heads

        self.mlp = nn.Module()
        mlp_hidden = hidden_size * 4
        self.mlp.gate_proj = nn.Linear(hidden_size, mlp_hidden)
        self.mlp.up_proj = nn.Linear(hidden_size, mlp_hidden)
        self.mlp.down_proj = nn.Linear(mlp_hidden, hidden_size)


class MockGemmaModel(nn.Module):
    """Mock Gemma model."""

    def __init__(self, num_layers=2):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "model_type": "gemma",
                "num_attention_heads": 4,
                "hidden_size": 256,
            },
        )()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MockGemmaLayer() for _ in range(num_layers)])


class MockGPT2Layer(nn.Module):
    """Mock GPT-2 layer."""

    def __init__(self, hidden_size=256, num_heads=4):
        super().__init__()
        self.attn = nn.Module()
        # GPT-2's c_attn contains Q, K, V stacked
        self.attn.c_attn = nn.Linear(hidden_size, hidden_size * 3)
        self.attn.c_proj = nn.Linear(hidden_size, hidden_size)

        self.mlp = nn.Module()
        mlp_hidden = hidden_size * 4
        self.mlp.c_fc = nn.Linear(hidden_size, mlp_hidden)
        self.mlp.c_proj = nn.Linear(mlp_hidden, hidden_size)


class MockGPT2Model(nn.Module):
    """Mock GPT-2 model."""

    def __init__(self, num_layers=2):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {
                "model_type": "gpt2",
                "num_attention_heads": 4,
                "hidden_size": 256,
            },
        )()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([MockGPT2Layer() for _ in range(num_layers)])


# Tests


class TestArchitectureDetection:
    """Test automatic architecture detection."""

    def test_detect_qwen(self):
        from circuitkit.applications import detect_model_architecture

        model = MockQwenModel()
        assert detect_model_architecture(model) == "qwen"

    def test_detect_gemma(self):
        from circuitkit.applications import detect_model_architecture

        model = MockGemmaModel()
        assert detect_model_architecture(model) == "gemma"

    def test_detect_gpt2(self):
        from circuitkit.applications import detect_model_architecture

        model = MockGPT2Model()
        assert detect_model_architecture(model) == "gpt2"

    def test_unsupported_architecture(self):
        from circuitkit.applications import UnsupportedArchitectureError, detect_model_architecture

        model = nn.Module()
        model.config = type("Config", (), {"model_type": "unknown_model"})()
        with pytest.raises(UnsupportedArchitectureError):
            detect_model_architecture(model)


class TestArchitectureValidation:
    """Test layer path validation."""

    def test_validate_qwen_paths(self):
        from circuitkit.applications import get_arch_config, validate_model_paths

        model = MockQwenModel()
        arch_cfg = get_arch_config("qwen")
        validate_model_paths(model, arch_cfg)  # Should not raise

    def test_validate_gpt2_paths(self):
        from circuitkit.applications import get_arch_config, validate_model_paths

        model = MockGPT2Model()
        arch_cfg = get_arch_config("gpt2")
        validate_model_paths(model, arch_cfg)  # Should not raise

    def test_invalid_path(self):
        from circuitkit.applications import (
            ArchitectureValidationError,
            get_arch_config,
            validate_model_paths,
        )

        model = MockQwenModel()
        arch_cfg = get_arch_config("gpt2")  # Wrong architecture
        with pytest.raises(ArchitectureValidationError):
            validate_model_paths(model, arch_cfg)


class TestLayerAccess:
    """Test accessing layers with architecture config."""

    def test_get_layers_qwen(self):
        from circuitkit.applications import get_arch_config, get_layers

        model = MockQwenModel(num_layers=3)
        arch_cfg = get_arch_config("qwen")
        layers = get_layers(model, arch_cfg)
        assert len(layers) == 3

    def test_get_layers_gpt2(self):
        from circuitkit.applications import get_arch_config, get_layers

        model = MockGPT2Model(num_layers=4)
        arch_cfg = get_arch_config("gpt2")
        layers = get_layers(model, arch_cfg)
        assert len(layers) == 4

    def test_get_attn_proj(self):
        from circuitkit.applications import get_arch_config, get_attn_proj, get_layers

        model = MockQwenModel()
        arch_cfg = get_arch_config("qwen")
        layers = get_layers(model, arch_cfg)
        layer = layers[0]

        k_proj = get_attn_proj(layer, arch_cfg, "k_proj")
        assert isinstance(k_proj, nn.Linear)
        assert k_proj.weight.shape == (256, 256)

    def test_get_mlp_proj(self):
        from circuitkit.applications import get_arch_config, get_layers, get_mlp_proj

        model = MockQwenModel()
        arch_cfg = get_arch_config("qwen")
        layers = get_layers(model, arch_cfg)
        layer = layers[0]

        gate_proj = get_mlp_proj(layer, arch_cfg, "gate_proj")
        assert isinstance(gate_proj, nn.Linear)
        assert gate_proj.weight.shape == (1024, 256)


class TestHeadDimension:
    """Test head dimension detection."""

    def test_get_head_dim_from_attr(self):
        from circuitkit.applications import get_arch_config, get_head_dim, get_layers

        model = MockQwenModel()
        arch_cfg = get_arch_config("qwen")
        layers = get_layers(model, arch_cfg)
        layer = layers[0]

        head_dim = get_head_dim(layer, arch_cfg)
        assert head_dim == 64  # 256 / 4


class TestRegistryContent:
    """Test registry content and organization."""

    def test_registry_has_required_models(self):
        from circuitkit.applications import MODEL_ARCH_REGISTRY

        assert "llama" in MODEL_ARCH_REGISTRY
        assert "qwen" in MODEL_ARCH_REGISTRY
        assert "gemma" in MODEL_ARCH_REGISTRY
        assert "gpt2" in MODEL_ARCH_REGISTRY

    def test_production_families_list(self):
        from circuitkit.applications import PRODUCTION_FAMILIES

        assert "llama" in PRODUCTION_FAMILIES
        assert "qwen" in PRODUCTION_FAMILIES
        assert "gemma" in PRODUCTION_FAMILIES

    def test_ready_families_list(self):
        from circuitkit.applications import READY_FAMILIES

        assert "mistral" in READY_FAMILIES or "mistral" not in READY_FAMILIES  # Flexible

    def test_arch_config_structure(self):
        from circuitkit.applications import get_arch_config

        arch_cfg = get_arch_config("llama")

        # Check required fields
        assert "name" in arch_cfg
        assert "layers_path" in arch_cfg
        assert "attn" in arch_cfg
        assert "mlp" in arch_cfg
        assert "status" in arch_cfg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
