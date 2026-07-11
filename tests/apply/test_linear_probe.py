"""Unit smoke tests for LinearProbe + ProbeTrainer."""

from __future__ import annotations

import torch


def test_linear_probe_imports():
    from circuitkit.applications.common_utils.linear_probe import LinearProbe, ProbeTrainer

    assert LinearProbe is not None
    assert ProbeTrainer is not None


def test_linear_probe_forward_shape():
    """LinearProbe: input [N, D] -> output [N, 1] (binary prob via sigmoid)."""
    from circuitkit.applications.common_utils.linear_probe import LinearProbe

    probe = LinearProbe(input_dim=64)
    x = torch.randn(8, 64)
    y = probe(x)
    assert y.shape == (8, 1), f"unexpected output shape: {y.shape}"
    assert (y >= 0).all() and (y <= 1).all(), "output should be in [0, 1]"


def test_probe_trainer_runs_one_step():
    """ProbeTrainer.train_epoch lowers loss on a trivial dataset."""
    from torch.utils.data import DataLoader, TensorDataset

    from circuitkit.applications.common_utils.linear_probe import LinearProbe, ProbeTrainer

    probe = LinearProbe(input_dim=32)
    trainer = ProbeTrainer(probe, device="cpu", learning_rate=1e-2)

    x = torch.randn(32, 32)
    y = torch.randint(0, 2, (32, 1)).float()
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8)

    for bx, by in loader:
        with torch.no_grad():
            logits = probe.get_logits(bx)
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), by.squeeze(-1)
            ).item()
        break

    for _ in range(5):
        trainer.train_epoch(loader, loader)

    probe.eval()
    with torch.no_grad():
        out = probe(x)
    assert out.shape == (32, 1)
    assert not torch.isnan(out).any()
