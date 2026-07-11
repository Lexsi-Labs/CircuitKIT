#!/usr/bin/env python3
"""
Standalone verification for M7.0.2: Perplexity Metric Implementation
"""

import torch
import torch.nn.functional as F


# Inline implementation for verification
def perplexity_span(
    logits, clean_logits, input_lengths, labels, answer_spans=None, mean=True, loss=False
):
    batch_size = logits.shape[0]
    results = []
    for i in range(batch_size):
        if answer_spans and answer_spans[i] is not None:
            start, end = answer_spans[i]
            start = max(0, min(start, logits.shape[1] - 1))
            end = max(start + 1, min(end, logits.shape[1]))
        else:
            pos = input_lengths[i].item() - 1
            pos = max(0, min(pos, logits.shape[1] - 1))
            start, end = pos, pos + 1
        span_logits = logits[i, start:end, :]
        span_len = end - start
        if labels.shape[1] == 1:
            target_tokens = labels[i, :1]
        else:
            target_tokens = labels[i, :span_len]
        log_probs = F.log_softmax(span_logits, dim=-1)
        token_log_probs = []
        for j in range(min(span_len, target_tokens.shape[0])):
            target_idx = target_tokens[j].item()
            target_idx = max(0, min(target_idx, log_probs.shape[1] - 1))
            token_log_probs.append(log_probs[j, target_idx])
        if token_log_probs:
            mean_log_prob = torch.stack(token_log_probs).mean()
        else:
            mean_log_prob = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if loss:
            results.append(-mean_log_prob)
        else:
            perplexity = torch.exp(-mean_log_prob)
            results.append(perplexity)
    result_tensor = torch.stack(results)
    if mean:
        return result_tensor.mean()
    else:
        return result_tensor


# Verification Tests
print("=" * 70)
print("M7.0.2 PERPLEXITY METRIC - STANDALONE VERIFICATION")
print("=" * 70)
print()

print("Test 1: Single-token perplexity")
logits = torch.randn(4, 10, 100)
input_lengths = torch.tensor([10, 10, 10, 10])
labels = torch.tensor([[42], [15], [73], [28]])
ppl = perplexity_span(logits, None, input_lengths, labels, mean=True)
assert ppl.shape == torch.Size([])
assert ppl > 0
print(f"[PASS] Single-token perplexity: {ppl:.2f}")

print()
print("Test 2: Multi-token answer spans")
logits = torch.randn(2, 20, 100)
input_lengths = torch.tensor([20, 20])
labels = torch.tensor([[42, 55], [10, 20]])
answer_spans = [(18, 20), (15, 17)]
ppl = perplexity_span(logits, None, input_lengths, labels, answer_spans=answer_spans, mean=True)
assert ppl > 0
print(f"[PASS] Multi-token perplexity: {ppl:.2f}")

print()
print("Test 3: Perplexity loss variant (per-example)")
logits = torch.randn(4, 10, 100)
input_lengths = torch.tensor([10, 10, 10, 10])
labels = torch.tensor([[42], [15], [73], [28]])
loss_per = perplexity_span(logits, None, input_lengths, labels, mean=False, loss=True)
ppl_per = perplexity_span(logits, None, input_lengths, labels, mean=False, loss=False)
assert torch.allclose(torch.log(ppl_per), loss_per, rtol=1e-4)
print("[PASS] Loss variant verified: log(PPL) = loss")

print()
print("Test 4: Gradient support")
logits = torch.randn(4, 10, 100, requires_grad=True)
input_lengths = torch.tensor([10, 10, 10, 10])
labels = torch.tensor([[42], [15], [73], [28]])
loss = perplexity_span(logits, None, input_lengths, labels, mean=True, loss=True)
loss.backward()
assert logits.grad is not None
print("[PASS] Gradient support confirmed")

print()
print("Test 5: Per-example vs batch mean")
logits = torch.randn(4, 10, 100)
input_lengths = torch.tensor([10, 10, 10, 10])
labels = torch.tensor([[42], [15], [73], [28]])
ppl_per = perplexity_span(logits, None, input_lengths, labels, mean=False)
ppl_mean = perplexity_span(logits, None, input_lengths, labels, mean=True)
assert ppl_per.shape == torch.Size([4])
assert ppl_mean.shape == torch.Size([])
assert torch.allclose(ppl_per.mean(), ppl_mean)
print("[PASS] Mean modes working correctly")

print()
print("Test 6: Numerical correctness")
logits = torch.zeros(1, 1, 10)
labels = torch.tensor([[0]])
ppl = perplexity_span(logits, None, torch.tensor([1]), labels, mean=True)
expected = torch.tensor(10.0)
assert torch.allclose(ppl, expected, rtol=1e-5)
print(f"[PASS] Numerical correctness: PPL={ppl:.2f}, Expected=10.00")

print()
print("=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print()
print("Implementation Summary:")
print("  - perplexity_span(): Core implementation")
print("  - Single & multi-token support")
print("  - Loss variant for optimization")
print("  - Full gradient support")
print("  - Per-example and batch mean outputs")
print()
print("Status: Ready for use in language modeling circuits")
print()
