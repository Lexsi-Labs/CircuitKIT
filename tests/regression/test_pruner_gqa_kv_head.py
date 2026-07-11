"""Regression test — StructuralPruner GQA KV-head no-op bug.

Historical bug
--------------
``applications/pruning/pruner.py`` pruned attention heads by writing through
``attn.W_K.data[head_idx].zero_()``. On a Grouped-Query-Attention model
(``n_key_value_heads < n_heads``) TransformerLens exposes ``W_K`` / ``W_V`` as
**read-only properties** that *repeat* the underlying ``_W_K`` / ``_W_V``
parameters (one row per query head). Writing through the repeated view mutates
a transient tensor that is immediately discarded — the KV weights were never
actually zeroed, so KV-head pruning was a silent no-op.

The fix zeroes the backing ``_W_K`` / ``_W_V`` parameters directly, and only
when *every* query head sharing a KV group has been pruned (so a KV head still
used by a surviving query head is left intact).

This test builds a tiny synthetic GQA ``HookedTransformer`` and asserts:

* pruning a whole query group zeroes the corresponding ``_W_K`` / ``_W_V`` row;
* a KV head whose group is only partially pruned is left untouched;
* per-query-head ``W_Q`` / ``W_O`` are always zeroed for pruned heads.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformer_lens")


@pytest.fixture
def gqa_model():
    """Tiny synthetic GQA model: 4 query heads, 2 KV heads (group size 2)."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        n_ctx=16,
        d_head=8,
        n_heads=4,
        n_key_value_heads=2,  # GQA: 2 query heads share each KV head.
        d_mlp=64,
        d_vocab=50,
        act_fn="gelu",
    )
    m = HookedTransformer(cfg)
    # Make every backing KV weight non-zero so "is it zeroed?" is meaningful.
    with torch.no_grad():
        for blk in m.blocks:
            blk.attn._W_K.fill_(1.0)
            blk.attn._W_V.fill_(1.0)
            blk.attn.W_Q.fill_(1.0)
            blk.attn.W_O.fill_(1.0)
    return m


def test_gqa_model_uses_repeated_kv_view(gqa_model):
    """Sanity check: W_K is a non-parameter repeated view of _W_K.

    This is the precondition that made the original ``.data`` write a no-op.
    """
    attn = gqa_model.blocks[0].attn
    assert not isinstance(
        attn.W_K, torch.nn.Parameter
    ), "W_K is a Parameter — model is not GQA, bug condition not reproduced"
    assert attn.W_K.shape[0] == 4, "expected W_K repeated to n_heads=4"
    assert attn._W_K.shape[0] == 2, "expected backing _W_K with n_kv=2"


def test_pruning_full_query_group_zeroes_backing_kv(gqa_model):
    """Pruning every query head in a KV group must zero its _W_K / _W_V row."""
    from circuitkit.applications.pruning.pruner import StructuralPruner

    pruner = StructuralPruner()
    attn = gqa_model.blocks[0].attn

    # KV group 0 owns query heads {0, 1}. Prune both → KV head 0 unused.
    pruner._remove_attention_heads_from_layer(gqa_model, layer_idx=0, heads_to_remove=[0, 1])

    # The backing KV parameter row 0 must now be all zero.
    assert torch.all(attn._W_K.data[0] == 0), (
        "_W_K row for the fully-pruned KV group was not zeroed — GQA KV-head "
        "pruning regressed to a silent no-op."
    )
    assert torch.all(
        attn._W_V.data[0] == 0
    ), "_W_V row for the fully-pruned KV group was not zeroed."
    # KV head 1 (query group {2, 3} untouched) must stay intact.
    assert torch.all(
        attn._W_K.data[1] != 0
    ), "_W_K row for an un-pruned KV group was wrongly zeroed."
    assert torch.all(
        attn._W_V.data[1] != 0
    ), "_W_V row for an un-pruned KV group was wrongly zeroed."
    # Per-query-head weights for the pruned heads must be zeroed.
    assert torch.all(attn.W_Q.data[0] == 0) and torch.all(attn.W_Q.data[1] == 0)
    assert torch.all(attn.W_O.data[0] == 0) and torch.all(attn.W_O.data[1] == 0)


def test_partial_query_group_leaves_kv_head_intact(gqa_model):
    """A KV head still used by a surviving query head must NOT be zeroed."""
    from circuitkit.applications.pruning.pruner import StructuralPruner

    pruner = StructuralPruner()
    attn = gqa_model.blocks[0].attn

    # Prune only query head 0; head 1 (same KV group 0) survives.
    pruner._remove_attention_heads_from_layer(gqa_model, layer_idx=0, heads_to_remove=[0])

    assert torch.all(
        attn._W_K.data[0] != 0
    ), "_W_K for KV group 0 was zeroed even though query head 1 still uses it"
    assert torch.all(
        attn._W_V.data[0] != 0
    ), "_W_V for KV group 0 was zeroed even though query head 1 still uses it"
    # The pruned query head's own weights are still zeroed.
    assert torch.all(attn.W_Q.data[0] == 0)
    assert torch.all(attn.W_O.data[0] == 0)
