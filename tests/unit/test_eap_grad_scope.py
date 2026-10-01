"""EAP scorers keep only the embeddings differentiable while attributing."""

import pytest
import torch
from torch import nn

from circuitkit.backends.eap.eap_utils import (
    _grads_only_through_embeddings,
    grads_through_embeddings_only,
)


class _Model(nn.Module):
    def __init__(self, embed_requires_grad=True):
        super().__init__()
        self.embed = nn.Module()
        self.embed.W_E = nn.Parameter(torch.zeros(2), requires_grad=embed_requires_grad)
        self.pos_embed = nn.Module()
        self.pos_embed.W_pos = nn.Parameter(torch.zeros(2))
        self.blocks = nn.ModuleList([nn.Module()])
        self.blocks[0].W_Q = nn.Parameter(torch.zeros(2))
        self.unembed = nn.Module()
        self.unembed.W_U = nn.Parameter(torch.zeros(2))


def _flags(model):
    return {n: p.requires_grad for n, p in model.named_parameters()}


def test_only_embeddings_require_grad_inside_and_flags_restored():
    model = _Model()
    before = _flags(model)
    with _grads_only_through_embeddings(model):
        assert {n for n, f in _flags(model).items() if f} == {"embed.W_E", "pos_embed.W_pos"}
    assert _flags(model) == before


def test_flags_restored_when_the_body_raises():
    model = _Model()
    before = _flags(model)
    with pytest.raises(RuntimeError):
        with _grads_only_through_embeddings(model):
            raise RuntimeError("boom")
    assert _flags(model) == before


def test_not_frozen_when_no_embedding_can_anchor_the_graph():
    model = _Model()
    model.embed.W_E.requires_grad_(False)
    model.pos_embed.W_pos.requires_grad_(False)
    before = _flags(model)
    with _grads_only_through_embeddings(model):
        assert _flags(model) == before


def test_decorator_forwards_arguments_and_return_value():
    seen = {}

    @grads_through_embeddings_only
    def scorer(model, x, *, y):
        seen["blocks_frozen"] = not model.blocks[0].W_Q.requires_grad
        return x + y

    model = _Model()
    assert scorer(model, 1, y=2) == 3
    assert seen["blocks_frozen"] is True
    assert model.blocks[0].W_Q.requires_grad is True
