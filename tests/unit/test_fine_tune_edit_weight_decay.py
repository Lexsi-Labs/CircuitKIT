"""FineTuneEditHandler hands its weight_decay to AdamW (AdamW's own default is 0.01)."""

import types

import pytest
import torch
import torch.nn as nn

from circuitkit.applications.editing import fine_tune_editing as fte


class _Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.cfg = types.SimpleNamespace(device="cpu")

    def forward(self, x):
        return self.lin(x)


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_fine_tune_edit_handler_passes_weight_decay_to_adamw(monkeypatch, weight_decay):
    seen = []
    real_adamw = torch.optim.AdamW

    class Spy(real_adamw):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self.param_groups[0]["weight_decay"])

    monkeypatch.setattr(torch.optim, "AdamW", Spy)
    net = _Linear()
    handler = fte.FineTuneEditHandler(net, steps=1, lr=1e-3, weight_decay=weight_decay)
    handler._fact_loss = lambda prompt, subject, target: net(torch.ones(1, 8)).pow(2).sum()

    handler._fine_tune([("p", "s", "t")], steps=1, lr=1e-3, batch_size=1, max_grad_norm=1.0)

    assert seen == [weight_decay]
