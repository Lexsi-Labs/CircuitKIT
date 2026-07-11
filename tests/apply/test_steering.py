"""Unit smoke tests for ActivationSteering (apply.steering)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformer_lens")


@pytest.fixture(scope="module")
def gpt2_model():
    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = HookedTransformer.from_pretrained("gpt2", device=device, dtype=torch.float32)
    # Same flags as apps cells set.
    m.cfg.use_attn_result = True
    m.cfg.use_split_qkv_input = True
    m.cfg.use_hook_mlp_in = True
    return m


def test_steering_imports():
    from circuitkit.applications import steering as st

    assert hasattr(st, "ActivationSteering")


def test_activation_steering_construct(gpt2_model):
    from circuitkit.applications.steering.steering import ActivationSteering

    scores = {"A0.0": 0.9, "A1.0": 0.8, "MLP 0": 0.7}
    s = ActivationSteering(gpt2_model, scores, score_threshold=0.5)
    assert s is not None


def test_steer_with_coefficient_zero_is_noop(gpt2_model):
    """coefficient=0 should leave logits unchanged."""
    from circuitkit.applications.steering.steering import ActivationSteering

    scores = {"A0.0": 0.9, "A1.0": 0.8}
    s = ActivationSteering(gpt2_model, scores, score_threshold=0.5)

    same_prompts = [{"text": "The cat sat on the"}, {"text": "The dog ran across the"}]
    s.compute_steering_vector(same_prompts, same_prompts)

    prompt = "The cat sat on the"
    in_ids = gpt2_model.tokenizer(prompt, return_tensors="pt").input_ids.to(
        next(gpt2_model.parameters()).device,
    )
    with torch.inference_mode():
        logits_before = gpt2_model(in_ids)

    out = s.steer(prompt, coefficient=0.0)
    diff = (out["output"] - logits_before).abs().max().item()
    assert diff < 1e-3, f"coefficient=0 should be a no-op, got max-diff {diff}"
