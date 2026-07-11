"""Regression test — CircuitTuner dead-training bug.

Historical bug
--------------
``applications/finetuning/circuit_tuning.py`` originally applied the LoRA
adapter by patching ``W_in.data = W_in.data + delta`` *inside* the forward
loop. A ``.data =`` assignment writes a brand-new leaf tensor and severs the
autograd graph: the adapter parameters never received gradients, the optimiser
updated nothing, and ``fit()`` silently "trained" while producing zero weight
change. No exception was raised — the loss curve just sat flat.

The fix applies the LoRA contribution via a forward *hook* on ``hook_pre`` so
the adapter parameters stay in the autograd graph.

This test fails if the ``.data``-patch (or any other autograd-severing
implementation) is reintroduced:

* the LoRA adapter parameters must actually change after ``fit()``;
* the adapter parameters must receive a non-zero gradient;
* the base-model parameters must stay frozen (unchanged + ``requires_grad`` False);
* the training loss must decrease.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformer_lens")


@pytest.fixture(scope="module")
def gpt2_cpu():
    from transformer_lens import HookedTransformer

    # CPU on purpose — an H200 is occupied by another job; gpt2 is tiny.
    m = HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.float32)
    return m


def test_circuit_tuner_actually_trains(gpt2_cpu):
    """fit() must change LoRA params, leave base params frozen, lower the loss."""
    from circuitkit.applications.finetuning.circuit_tuning import CircuitTuner, CircuitTunerConfig

    model = gpt2_cpu

    # Tiny config — a handful of steps on two MLP layers is enough to expose a
    # severed autograd graph (any working optimiser moves the params).
    cfg = CircuitTunerConfig(
        lora_rank=2,
        lora_alpha=4.0,
        lr=5e-3,
        n_steps=12,
        kl_retain_weight=0.0,  # CE-only keeps the test fast and deterministic.
        top_k_layers=2,
    )
    scores = {"MLP 5": 1.0, "MLP 6": 0.9}
    tuner = CircuitTuner(model, node_scores=scores, config=cfg)

    # Snapshot base-model weights for the to-be-adapted layers BEFORE fit().
    base_before = {lyr: model.blocks[lyr].mlp.W_in.detach().clone() for lyr in (5, 6)}

    result = tuner.fit(prompts=["The capital of France is"], targets=[" Berlin"])

    assert result.error is None, f"fit() raised: {result.error}"
    assert result.n_lora_params > 0, "no LoRA parameters were created"

    # --- 1. LoRA adapter parameters must have actually changed ----------------
    # lora_B starts at all-zeros; after real training it must be non-zero.
    moved = False
    grad_seen = False
    for adapters in tuner._adapters.values():
        for adapter in adapters.values():
            if adapter.lora_B.detach().abs().sum().item() > 0:
                moved = True
            if adapter.lora_A.grad is not None and adapter.lora_A.grad.abs().sum() > 0:
                grad_seen = True
            if adapter.lora_B.grad is not None and adapter.lora_B.grad.abs().sum() > 0:
                grad_seen = True
    assert moved, (
        "LoRA adapter weights did not change after fit() — autograd graph was "
        "severed (dead-training bug regressed)."
    )
    assert grad_seen, (
        "LoRA adapter parameters received no gradient — backward() did not "
        "reach them (dead-training bug regressed)."
    )

    # --- 2. Base-model weights must stay frozen and unchanged -----------------
    for lyr, before in base_before.items():
        after = model.blocks[lyr].mlp.W_in.detach()
        assert torch.equal(before, after), (
            f"base W_in for layer {lyr} changed — hook-based adapter must NOT "
            f"mutate base weights (only bake() may)."
        )
    for name, p in model.named_parameters():
        assert (
            not p.requires_grad
        ), f"base parameter {name} left with requires_grad=True after fit()"

    # --- 3. Loss must decrease ------------------------------------------------
    assert len(result.loss_history) == cfg.n_steps
    assert result.loss_history[-1] < result.loss_history[0], (
        f"loss did not decrease: start={result.loss_history[0]:.4f} "
        f"end={result.loss_history[-1]:.4f} — training had no effect."
    )
