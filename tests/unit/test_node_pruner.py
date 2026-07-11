"""
Unit tests for node_pruner.py

Tests cover get_nodes_to_prune, zero_head_hook, and zero_mlp_hook.
No model or CUDA needed.
"""

import re
from unittest.mock import MagicMock

import pytest
import torch

from circuitkit.applications.pruning.node_pruner import (  # noqa: adjust path as needed
    get_nodes_to_prune,
    zero_head_hook,
    zero_mlp_hook,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _make_scores(n_layers: int = 3, n_heads: int = 4) -> dict:
    """Generate synthetic node scores covering n_layers * n_heads heads + n_layers MLPs."""
    scores = {}
    val = 1.0
    for layer in range(n_layers):
        for head in range(n_heads):
            scores[f"A{layer}.{head}"] = val
            val += 1.0
        scores[f"MLP {layer}"] = val
        val += 1.0
    return scores


# ===========================================================================
# 1. get_nodes_to_prune — basic behaviour
# ===========================================================================
class TestGetNodesToPrune:
    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="pruning_scope"):
            get_nodes_to_prune({}, 0.5, pruning_scope="invalid")

    def test_zero_sparsity_prunes_nothing(self):
        scores = _make_scores()
        result = get_nodes_to_prune(scores, target_sparsity=0.0)
        assert result == []

    def test_full_sparsity_prunes_all_eligible(self):
        """sparsity=1.0 should prune all unprotected nodes (subject to per-layer cap)."""
        scores = _make_scores(n_layers=2, n_heads=4)
        result = get_nodes_to_prune(scores, target_sparsity=1.0)
        # At most 50% per layer for heads → 2 heads per layer pruned; all MLPs pruned
        assert len(result) > 0

    def test_protected_nodes_never_pruned(self):
        scores = _make_scores()
        scores["Resid Start"] = 0.0  # lowest score → would be first candidate
        result = get_nodes_to_prune(scores, target_sparsity=0.8, protected_nodes=["Resid Start"])
        assert "Resid Start" not in result

    def test_scope_heads_only_prunes_heads(self):
        scores = _make_scores()
        result = get_nodes_to_prune(scores, target_sparsity=0.5, pruning_scope="heads")
        for name in result:
            assert re.match(r"A\d+\.\d+", name), f"Non-head node pruned: {name}"

    def test_scope_mlp_only_prunes_mlp(self):
        scores = _make_scores()
        result = get_nodes_to_prune(scores, target_sparsity=0.5, pruning_scope="mlp")
        for name in result:
            assert re.match(r"MLP \d+", name), f"Non-MLP node pruned: {name}"

    def test_scope_both_prunes_heads_and_mlp(self):
        scores = _make_scores(n_layers=3, n_heads=4)
        result = get_nodes_to_prune(scores, target_sparsity=0.5, pruning_scope="both")
        has_head = any(re.match(r"A\d+\.\d+", n) for n in result)
        has_mlp = any(re.match(r"MLP \d+", n) for n in result)
        assert has_head
        assert has_mlp

    def test_lowest_scores_pruned_first(self):
        """Nodes with lowest scores should be chosen before higher-scored ones."""
        scores = {"A0.0": 100.0, "A0.1": 1.0, "MLP 0": 50.0}
        result = get_nodes_to_prune(scores, target_sparsity=0.5, pruning_scope="heads")
        # Only one head eligible at 50%, A0.1 has lowest score
        if result:
            assert "A0.1" in result
            assert "A0.0" not in result

    def test_per_layer_head_cap_respected(self):
        """No layer should have more than 50% of its heads pruned."""
        n_heads = 4
        scores = _make_scores(n_layers=2, n_heads=n_heads)
        result = get_nodes_to_prune(scores, target_sparsity=1.0, pruning_scope="heads")
        pruned_by_layer: dict = {}
        for name in result:
            m = re.match(r"A(\d+)\.\d+", name)
            if m:
                layer = int(m.group(1))
                pruned_by_layer[layer] = pruned_by_layer.get(layer, 0) + 1
        for layer, count in pruned_by_layer.items():
            assert count <= n_heads * 0.5

    def test_empty_scores_returns_empty(self):
        result = get_nodes_to_prune({}, target_sparsity=0.5)
        assert result == []

    def test_output_contains_no_duplicates(self):
        scores = _make_scores()
        result = get_nodes_to_prune(scores, target_sparsity=0.5)
        assert len(result) == len(set(result))

    def test_only_mlp_scores_with_head_scope_returns_empty(self):
        """If only MLP scores exist and scope is 'heads', nothing is pruned."""
        scores = {"MLP 0": 1.0, "MLP 1": 2.0}
        result = get_nodes_to_prune(scores, target_sparsity=0.9, pruning_scope="heads")
        assert result == []

    def test_only_head_scores_with_mlp_scope_returns_empty(self):
        scores = {"A0.0": 1.0, "A0.1": 2.0}
        result = get_nodes_to_prune(scores, target_sparsity=0.9, pruning_scope="mlp")
        assert result == []


# ===========================================================================
# 2. zero_head_hook
# ===========================================================================
class TestZeroHeadHook:
    def test_zeros_correct_head(self):
        # activation: [batch=2, pos=3, n_heads=4, d_model=8]
        activation = torch.ones(2, 3, 4, 8)
        hook = MagicMock()
        zero_head_hook(activation, hook, head_index=1)
        # Head 1 should be zeroed, others intact
        assert activation[:, :, 1, :].sum() == 0.0
        assert activation[:, :, 0, :].sum() > 0.0
        assert activation[:, :, 2, :].sum() > 0.0

    def test_zeros_first_head(self):
        activation = torch.ones(1, 5, 3, 16)
        hook = MagicMock()
        zero_head_hook(activation, hook, head_index=0)
        assert activation[:, :, 0, :].sum() == 0.0

    def test_zeros_last_head(self):
        activation = torch.ones(1, 5, 3, 16)
        hook = MagicMock()
        zero_head_hook(activation, hook, head_index=2)
        assert activation[:, :, 2, :].sum() == 0.0

    def test_other_heads_unaffected(self):
        activation = torch.ones(2, 4, 3, 8)
        hook = MagicMock()
        zero_head_hook(activation, hook, head_index=1)
        for h in [0, 2]:
            assert activation[:, :, h, :].all()


# ===========================================================================
# 3. zero_mlp_hook
# ===========================================================================
class TestZeroMlpHook:
    def test_returns_zeros(self):
        activation = torch.ones(2, 5, 64)
        hook = MagicMock()
        result = zero_mlp_hook(activation, hook)
        assert result.sum() == 0.0

    def test_shape_preserved(self):
        activation = torch.ones(3, 7, 128)
        hook = MagicMock()
        result = zero_mlp_hook(activation, hook)
        assert result.shape == activation.shape

    def test_returns_new_tensor(self):
        """zero_mlp_hook should return zeros_like, not mutate in-place."""
        activation = torch.ones(2, 4, 32)
        hook = MagicMock()
        result = zero_mlp_hook(activation, hook)
        # Original should be untouched
        assert activation.sum() > 0
        assert result.sum() == 0.0
