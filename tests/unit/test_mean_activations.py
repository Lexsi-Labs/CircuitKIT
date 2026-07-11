"""Unit tests for ``compute_mean_activations`` mean-ablation intervention.

Pin down the *semantics* of the non-positional ("mean") ablation intervention in
``circuitkit.backends.eap.eap_utils.compute_mean_activations``: the per-node mean
must be a TRUE mean over real (non-padding) positions, NOT the old buggy
``last_token_activation / seq_len``. Two weightings are supported and tested:

  * mean_weighting="token"  (default): grand mean over all real positions AND
    examples -> divide the pooled sum by the total real-token count.
  * mean_weighting="example": mean of per-example position-means -> each example
    weighted equally (matches the /n_examples accumulation structure).

For uniform-length data the two are identical; they differ only with padding.
CPU-only; drives the real function with a tiny fake model/graph.
"""

import types
from contextlib import contextmanager

import pytest
import torch

from circuitkit.backends.eap import eap_utils


class _FakeCfg:
    def __init__(self, d_model):
        self.d_model = d_model
        self.dtype = torch.float32
        self.device = "cpu"
        self.n_ctx = 32


class _FakeModel:
    def __init__(self, d_model, replay):
        self.cfg = _FakeCfg(d_model)
        self._replay = replay
        self._fwd_hooks = []

    @contextmanager
    def hooks(self, fwd_hooks=None, bwd_hooks=None):
        self._fwd_hooks = list(fwd_hooks or [])
        try:
            yield self
        finally:
            self._fwd_hooks = []

    def __call__(self, tokens, attention_mask=None):
        for hook_name, fn in self._fwd_hooks:
            fn(self._replay[hook_name], hook=types.SimpleNamespace(name=hook_name))
        return None


class _FakeNode:
    def __init__(self, name, layer, out_hook):
        self.name = name
        self.layer = layer
        self.out_hook = out_hook


class _FakeGraph:
    def __init__(self, n_forward):
        self.nodes = {}
        self.n_forward = n_forward
        self._indices = {}

    def add_node(self, node, index):
        self.nodes[node.name] = node
        self._indices[node.name] = index

    def forward_index(self, node, attn_slice=True):
        return self._indices[node.name]


def _no_op_tokenize(attention_mask, input_lengths, n_pos):
    def _tokenize_plus(model, inputs, max_length=None, **kwargs):
        tokens = torch.zeros((attention_mask.size(0), n_pos), dtype=torch.long)
        return tokens, attention_mask, input_lengths, n_pos

    return _tokenize_plus


def _run(monkeypatch, acts, attention_mask, n_forward, index, weighting):
    out_hook = "blocks.0.hook_out"
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    graph = _FakeGraph(n_forward=n_forward)
    graph.add_node(_FakeNode("n0", layer=0, out_hook=out_hook), index=index)
    model = _FakeModel(d_model=acts.shape[-1], replay={out_hook: acts})
    monkeypatch.setattr(eap_utils, "tokenize_plus", _no_op_tokenize(attention_mask, input_lengths, n_pos))
    return eap_utils.compute_mean_activations(
        model, graph, [["a", "b"][: acts.size(0)]], per_position=False, mean_weighting=weighting
    )


def test_mean_variable_length_both_weightings(monkeypatch):
    """ex0=[1,2,3] (len3), ex1=[4,6,pad] (len2). Buggy last/seq_len would give 2.0.

    token   : pooled sum 16 / total 5 tokens        = 3.2
    example : (6/3 + 10/2)/2 = (2.0 + 5.0)/2         = 3.5
    """
    acts = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [6.0], [99.0]]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
    tok = _run(monkeypatch, acts, mask, n_forward=1, index=0, weighting="token")
    ex = _run(monkeypatch, acts, mask, n_forward=1, index=0, weighting="example")
    assert torch.allclose(tok, torch.tensor(3.2)), f"token expected 3.2, got {tok}"
    assert torch.allclose(ex, torch.tensor(3.5)), f"example expected 3.5, got {ex}"
    assert not torch.allclose(tok, torch.tensor(2.0)) and not torch.allclose(ex, torch.tensor(2.0))


def test_mean_no_padding_identical(monkeypatch):
    """No padding -> both weightings equal the plain average ([2,4,6] -> 4.0)."""
    acts = torch.tensor([[[2.0], [4.0], [6.0]]])
    mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    tok = _run(monkeypatch, acts, mask, n_forward=1, index=0, weighting="token")
    ex = _run(monkeypatch, acts, mask, n_forward=1, index=0, weighting="example")
    assert torch.allclose(tok, torch.tensor(4.0)) and torch.allclose(ex, torch.tensor(4.0))


def test_mean_head_dim_both_weightings(monkeypatch):
    """Optional head dim; final mean(0) over 2 heads.

    token   : head0 14/5=2.8, head1 42/5=8.4 -> (2.8+8.4)/2 = 5.6
    example : head0 3.0,     head1 9.5      -> (3.0+9.5)/2 = 6.25
    """
    acts = torch.tensor(
        [
            [[[1.0], [2.0]], [[2.0], [4.0]], [[3.0], [6.0]]],
            [[[3.0], [10.0]], [[5.0], [20.0]], [[99.0], [99.0]]],
        ]
    )
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
    tok = _run(monkeypatch, acts, mask, n_forward=2, index=slice(0, 2), weighting="token")
    ex = _run(monkeypatch, acts, mask, n_forward=2, index=slice(0, 2), weighting="example")
    assert torch.allclose(tok, torch.tensor(5.6)), f"token expected 5.6, got {tok}"
    assert torch.allclose(ex, torch.tensor(6.25)), f"example expected 6.25, got {ex}"


def test_mean_weighting_validation(monkeypatch):
    acts = torch.tensor([[[2.0], [4.0]]])
    mask = torch.tensor([[1, 1]], dtype=torch.long)
    with pytest.raises(ValueError):
        _run(monkeypatch, acts, mask, n_forward=1, index=0, weighting="bogus")
