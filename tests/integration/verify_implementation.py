#!/usr/bin/env python3
"""
Verification script for M7.0.2: Perplexity Metric Implementation
"""

import sys

import torch

# Test 1: Import and basic functionality
print("=" * 70)
print("M7.0.2 PERPLEXITY METRIC - IMPLEMENTATION VERIFICATION")
print("=" * 70)
print()

print("Test 1: Import metrics module")
try:
    sys.path.insert(0, "src")
    from circuitkit.backends.eap.metrics import (
        perplexity_legacy,
        perplexity_loss_legacy,
        perplexity_span,
    )

    print("[PASS] Successfully imported perplexity metrics")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

print()
print("Test 2: Single-token perplexity")
try:
    logits = torch.randn(4, 10, 100)
    input_lengths = torch.tensor([10, 10, 10, 10])
    labels = torch.tensor([[42], [15], [73], [28]])

    ppl = perplexity_legacy(logits, None, input_lengths, labels, mean=True)
    assert ppl.shape == torch.Size([])
    assert ppl > 0
    print(f"[PASS] Single-token perplexity: {ppl:.2f}")
except Exception as e:
    print(f"[FAIL] Single-token test failed: {e}")
    sys.exit(1)

print()
print("Test 3: Multi-token answer spans")
try:
    logits = torch.randn(2, 20, 100)
    input_lengths = torch.tensor([20, 20])
    labels = torch.tensor([[42, 55], [10, 20]])
    answer_spans = [(18, 20), (15, 17)]

    ppl = perplexity_span(logits, None, input_lengths, labels, answer_spans=answer_spans, mean=True)
    assert ppl > 0
    print(f"[PASS] Multi-token perplexity: {ppl:.2f}")
except Exception as e:
    print(f"[FAIL] Multi-token test failed: {e}")
    sys.exit(1)

print()
print("Test 4: Perplexity loss variant")
try:
    logits = torch.randn(4, 10, 100)
    input_lengths = torch.tensor([10, 10, 10, 10])
    labels = torch.tensor([[42], [15], [73], [28]])

    loss = perplexity_loss_legacy(logits, None, input_lengths, labels, mean=True)
    ppl = perplexity_legacy(logits, None, input_lengths, labels, mean=True)

    # Verify relationship: log(ppl) = loss
    assert torch.allclose(torch.log(ppl), loss, rtol=1e-4)
    print(f"[PASS] Loss variant: {loss:.4f} (verified relationship)")
except Exception as e:
    print(f"[FAIL] Loss variant test failed: {e}")
    sys.exit(1)

print()
print("Test 5: Gradient support (needed for EAP)")
try:
    logits = torch.randn(4, 10, 100, requires_grad=True)
    input_lengths = torch.tensor([10, 10, 10, 10])
    labels = torch.tensor([[42], [15], [73], [28]])

    loss = perplexity_loss_legacy(logits, None, input_lengths, labels, mean=True)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    print("[PASS] Gradient support confirmed")
except Exception as e:
    print(f"[FAIL] Gradient test failed: {e}")
    sys.exit(1)

print()
print("Test 6: Per-example vs batch mean")
try:
    logits = torch.randn(4, 10, 100)
    input_lengths = torch.tensor([10, 10, 10, 10])
    labels = torch.tensor([[42], [15], [73], [28]])

    ppl_per = perplexity_legacy(logits, None, input_lengths, labels, mean=False)
    ppl_mean = perplexity_legacy(logits, None, input_lengths, labels, mean=True)

    assert ppl_per.shape == torch.Size([4])
    assert ppl_mean.shape == torch.Size([])
    assert torch.allclose(ppl_per.mean(), ppl_mean)
    print("[PASS] Mean modes working correctly")
except Exception as e:
    print(f"[FAIL] Mean mode test failed: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print()
print("Implementation Summary:")
print("  - Core: perplexity_span() + perplexity_loss_span()")
print("  - Legacy: perplexity_legacy() + perplexity_loss_legacy()")
print("  - Features: Single/multi-token, gradient support, per-example output")
print("  - Ready: For use in language modeling and ranking circuits")
print()
