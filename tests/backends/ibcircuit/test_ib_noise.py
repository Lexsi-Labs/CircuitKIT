"""
Tests for ib_noise.py changes:
- apply_ib_noise is shape-agnostic (works for attn AND mlp activations)
- initialize_mlp_ib_weights produces correct shapes and gradients
- Noise behaviour is correct: high weight → low noise, low weight → high noise
"""

import pytest
import torch
import torch.nn as nn

from circuitkit.backends.ibcircuit.ib_noise import (
    apply_ib_noise,
    initialize_attn_ib_weights,
    initialize_mlp_ib_weights,
)

DEVICE = "cpu"
BATCH = 8
N_HEADS = 4
SEQ = 12
D_HEAD = 16
D_MODEL = 64
N_LAYERS = 3


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def attn_activation():
    """Shape: [batch, n_heads, seq, d_head] — as produced by hook_z after permute."""
    return torch.randn(BATCH, N_HEADS, SEQ, D_HEAD)


@pytest.fixture
def attn_weight():
    """Attn IB weight: [batch, n_heads, 1, 1]."""
    return nn.Parameter(torch.full((BATCH, N_HEADS, 1, 1), 5.0))


@pytest.fixture
def mlp_activation():
    """Shape: [batch, seq, d_model] — as produced by hook_mlp_out directly."""
    return torch.randn(BATCH, SEQ, D_MODEL)


@pytest.fixture
def mlp_weight():
    """MLP IB weight: [batch, 1, 1]."""
    return nn.Parameter(torch.full((BATCH, 1, 1), 5.0))


# ── apply_ib_noise: shape tests ───────────────────────────────────────────────


def test_apply_ib_noise_attn_output_shape(attn_activation, attn_weight):
    out, kl = apply_ib_noise(attn_activation, attn_weight)
    assert out.shape == attn_activation.shape, f"Expected {attn_activation.shape}, got {out.shape}"


def test_apply_ib_noise_mlp_output_shape(mlp_activation, mlp_weight):
    out, kl = apply_ib_noise(mlp_activation, mlp_weight)
    assert out.shape == mlp_activation.shape, f"Expected {mlp_activation.shape}, got {out.shape}"


# ── apply_ib_noise: KL loss sanity ───────────────────────────────────────────


def test_kl_loss_is_scalar_and_finite_attn(attn_activation, attn_weight):
    _, kl = apply_ib_noise(attn_activation, attn_weight)
    assert kl.shape == torch.Size([]), "KL loss must be scalar"
    assert torch.isfinite(kl), f"KL loss is not finite: {kl.item()}"
    assert kl.item() >= 0, f"KL loss must be non-negative, got {kl.item()}"


def test_kl_loss_is_scalar_and_finite_mlp(mlp_activation, mlp_weight):
    _, kl = apply_ib_noise(mlp_activation, mlp_weight)
    assert kl.shape == torch.Size([]), "KL loss must be scalar"
    assert torch.isfinite(kl), f"KL loss is not finite: {kl.item()}"
    assert kl.item() >= 0, f"KL loss must be non-negative, got {kl.item()}"


# ── apply_ib_noise: noise behaviour ──────────────────────────────────────────


def test_high_weight_low_noise_attn(attn_activation):
    """sigmoid(15) ≈ 1.0 → output should be very close to input."""
    high_weight = nn.Parameter(torch.full((BATCH, N_HEADS, 1, 1), 15.0))
    out, kl = apply_ib_noise(attn_activation, high_weight)
    mae = (out - attn_activation).abs().mean().item()
    assert mae < 0.1, f"High weight should produce low noise, MAE={mae:.4f}"


def test_low_weight_high_noise_attn(attn_activation):
    """sigmoid(-15) ≈ 0.0 → output should deviate significantly from input."""
    low_weight = nn.Parameter(torch.full((BATCH, N_HEADS, 1, 1), -15.0))
    out, kl = apply_ib_noise(attn_activation, low_weight)
    mae = (out - attn_activation).abs().mean().item()
    assert mae > 0.01, f"Low weight should produce high noise, MAE={mae:.4f}"


def test_high_weight_low_noise_mlp(mlp_activation):
    high_weight = nn.Parameter(torch.full((BATCH, 1, 1), 15.0))
    out, kl = apply_ib_noise(mlp_activation, high_weight)
    mae = (out - mlp_activation).abs().mean().item()
    assert mae < 0.1, f"High MLP weight should produce low noise, MAE={mae:.4f}"


def test_low_weight_high_noise_mlp(mlp_activation):
    low_weight = nn.Parameter(torch.full((BATCH, 1, 1), -15.0))
    out, kl = apply_ib_noise(mlp_activation, low_weight)
    mae = (out - mlp_activation).abs().mean().item()
    assert mae > 0.01, f"Low MLP weight should produce high noise, MAE={mae:.4f}"


# ── apply_ib_noise: gradient flow ────────────────────────────────────────────


def test_gradients_flow_through_attn(attn_activation, attn_weight):
    out, kl = apply_ib_noise(attn_activation, attn_weight)
    kl.backward()
    assert attn_weight.grad is not None, "No gradient on attn IB weight"
    assert torch.isfinite(attn_weight.grad).all(), "Attn weight grad has NaN/Inf"


def test_gradients_flow_through_mlp(mlp_activation, mlp_weight):
    out, kl = apply_ib_noise(mlp_activation, mlp_weight)
    kl.backward()
    assert mlp_weight.grad is not None, "No gradient on MLP IB weight"
    assert torch.isfinite(mlp_weight.grad).all(), "MLP weight grad has NaN/Inf"


# ── initialize_mlp_ib_weights ─────────────────────────────────────────────────


def test_mlp_weights_length():
    weights = initialize_mlp_ib_weights(BATCH, N_LAYERS, DEVICE)
    assert len(weights) == N_LAYERS, f"Expected {N_LAYERS} weight tensors, got {len(weights)}"


def test_mlp_weights_shape():
    weights = initialize_mlp_ib_weights(BATCH, N_LAYERS, DEVICE)
    for i, w in enumerate(weights):
        assert w.shape == (BATCH, 1, 1), f"Layer {i}: expected ({BATCH}, 1, 1), got {w.shape}"


def test_mlp_weights_are_parameters():
    weights = initialize_mlp_ib_weights(BATCH, N_LAYERS, DEVICE)
    for i, w in enumerate(weights):
        assert isinstance(w, nn.Parameter), f"Layer {i} weight is not nn.Parameter"


def test_mlp_weights_init_values():
    """Weights should be tightly clustered around init_mean=5.0."""
    weights = initialize_mlp_ib_weights(BATCH, N_LAYERS, DEVICE, init_mean=5.0, init_std=0.01)
    for i, w in enumerate(weights):
        assert (
            abs(w.mean().item() - 5.0) < 0.1
        ), f"Layer {i} mean {w.mean().item():.3f} far from 5.0"


def test_mlp_weights_trainable():
    """Gradients must flow through mlp_ib_weights."""
    weights = initialize_mlp_ib_weights(BATCH, N_LAYERS, DEVICE)
    activation = torch.randn(BATCH, SEQ, D_MODEL)
    loss = torch.tensor(0.0)
    for w in weights:
        _, kl = apply_ib_noise(activation, w)
        loss = loss + kl
    loss.backward()
    for i, w in enumerate(weights):
        assert w.grad is not None, f"No gradient on MLP weight layer {i}"
        assert torch.isfinite(w.grad).all(), f"NaN/Inf grad on MLP weight layer {i}"


# ── Parity: attn and mlp weights initialise the same way ─────────────────────


def test_attn_and_mlp_init_parity():
    """Both initialisers should use same mean/std defaults."""
    attn_w = initialize_attn_ib_weights(BATCH, 1, N_HEADS, DEVICE, init_mean=5.0, init_std=0.01)
    mlp_w = initialize_mlp_ib_weights(BATCH, 1, DEVICE, init_mean=5.0, init_std=0.01)
    assert (
        abs(attn_w[0].mean().item() - mlp_w[0].mean().item()) < 0.5
    ), "Attn and MLP weights should have similar initial values"
