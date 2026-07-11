"""Unit smoke tests for CircuitWeightSteering (C-ΔΘ).

Validates the public API surface without requiring a full end-to-end
discovery + fine-tune cycle. The detailed paper-faithful verification
lives in validation/applications/20_ctheta_steering_gpt2.py.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformer_lens")


@pytest.fixture(scope="module")
def gpt2_model():
    """Tiny GPT-2 loaded once for all tests in this module."""
    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HookedTransformer.from_pretrained("gpt2", device=device, dtype=torch.float32)


def _fake_circuit_scores(model, k=4):
    """Return a small mock node-scores dict spanning a few heads."""
    n_layers = model.cfg.n_layers
    return {
        "A0.0": 0.9,
        "A0.1": 0.8,
        "A1.0": 0.7,
        f"A{n_layers - 1}.0": 0.6,
        "MLP 0": 0.5,
    }


def test_get_head_weight_info_returns_qkvo(gpt2_model):
    from circuitkit.applications.steering.weight_steering import get_head_weight_info

    info = get_head_weight_info(gpt2_model, head_name="A3.2")
    for key in ("W_Q", "W_K", "W_V", "W_O"):
        assert key in info, f"missing {key} in head-weight info"
        param, head_idx = info[key]
        assert isinstance(param, torch.nn.Parameter)
        assert isinstance(head_idx, int)
        assert head_idx == 2, f"{key}: head_idx should be 2, got {head_idx}"


def test_get_head_weight_slice_shape(gpt2_model):
    from circuitkit.applications.steering.weight_steering import get_head_weight_slice

    slices = get_head_weight_slice(gpt2_model, head_name="A2.1")
    # Returns dict of per-head slices
    assert "W_Q" in slices
    q_slice = slices["W_Q"]
    # Sliced per-head, should be 2D: [d_model, d_head] or [d_head, d_model]
    assert q_slice.dim() == 2, f"expected 2D slice, got {q_slice.shape}"
    assert q_slice.requires_grad is False  # eval mode by default


def test_circuit_weight_steering_init(gpt2_model):
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering

    scores = _fake_circuit_scores(gpt2_model)
    cs = CircuitWeightSteering(gpt2_model, circuit_scores=scores, score_threshold=0.5)
    # Constructor should pick up the attention-head entries from scores.
    assert hasattr(cs, "target_model")
    assert hasattr(cs, "circuit_scores")
    # Should have identified the attention heads (A0.0, A0.1, A1.0, A11.0)
    assert len(cs.head_names) == 4
    assert "A0.0" in cs.head_names
    assert "A0.1" in cs.head_names
    assert "A1.0" in cs.head_names


def test_steering_vector_zero_before_finetune(gpt2_model):
    """compute_steering_vector with no fine-tune should raise."""
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering

    scores = _fake_circuit_scores(gpt2_model)
    cs = CircuitWeightSteering(gpt2_model, circuit_scores=scores, score_threshold=0.5)
    # Without any fine-tuning the +/- deltas haven't been computed yet;
    # compute_steering_vector should raise.
    with pytest.raises(RuntimeError):
        cs.compute_steering_vector()


def test_apply_steering_with_k_zero_is_noop(gpt2_model):
    """θ + 0 · w_b == θ. Applies a zero-coefficient steering and checks
    the model output didn't move."""
    from circuitkit.applications.steering.weight_steering import (
        CircuitWeightSteering,
        get_head_weight_slice,
    )

    scores = _fake_circuit_scores(gpt2_model)
    cs = CircuitWeightSteering(gpt2_model, circuit_scores=scores, score_threshold=0.5)

    # Manually inject a zero steering vector so we don't need fine-tuning.
    cs._steering_vector = {
        head: {k: torch.zeros_like(slc) for k, slc in heads.items()}
        for head, heads in {h: get_head_weight_slice(gpt2_model, h) for h in cs.head_names}.items()
    }

    prompt = "The capital of France is"
    in_ids = gpt2_model.tokenizer(prompt, return_tensors="pt").input_ids.to(
        next(gpt2_model.parameters()).device,
    )
    with torch.inference_mode():
        logits_before = gpt2_model(in_ids)
    cs.apply_steering(k=0.0)
    with torch.inference_mode():
        logits_after = gpt2_model(in_ids)
    diff = (logits_after - logits_before).abs().max().item()
    assert diff < 1e-4, f"k=0 should be a no-op, got max-diff {diff}"
