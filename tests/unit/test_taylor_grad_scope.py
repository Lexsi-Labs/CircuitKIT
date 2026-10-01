"""The Taylor selector must only ask autograd for the weights it reads."""

import pytest
import torch
from torch import nn

from circuitkit.applications.pruning.selectors.taylor_selector import (
    _only_score_params_require_grad,
)


class _Model(nn.Module):
    def __init__(self, with_scored=True):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(2))
        attn, mlp = nn.Module(), nn.Module()
        attn.W_Q = nn.Parameter(torch.zeros(2))
        if with_scored:
            attn.W_O = nn.Parameter(torch.zeros(2))
            mlp.W_out = nn.Parameter(torch.zeros(2))
        mlp.W_in = nn.Parameter(torch.zeros(2))
        self.blocks = nn.ModuleList()
        block = nn.Module()
        block.attn, block.mlp = attn, mlp
        self.blocks.append(block)


def _flags(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


def test_only_scored_weights_require_grad_inside_and_flags_restored():
    model = _Model()
    model.embed.requires_grad_(False)  # a caller-frozen param must stay frozen afterwards
    before = _flags(model)
    with _only_score_params_require_grad(model):
        inside = _flags(model)
        assert {n for n, f in inside.items() if f} == {"blocks.0.attn.W_O", "blocks.0.mlp.W_out"}
    assert _flags(model) == before


def test_flags_restored_when_the_body_raises():
    model = _Model()
    before = _flags(model)
    with pytest.raises(RuntimeError):
        with _only_score_params_require_grad(model):
            raise RuntimeError("boom")
    assert _flags(model) == before


def test_untouched_when_model_has_no_scored_weights():
    model = _Model(with_scored=False)
    before = _flags(model)
    with _only_score_params_require_grad(model):
        assert _flags(model) == before
