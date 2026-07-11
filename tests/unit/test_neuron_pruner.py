"""
Unit tests for neuron_pruner.py

Tests cover zero_neuron_hook and zero_attn_neuron_hook.
No model or CUDA needed.
"""

from unittest.mock import MagicMock

import pytest
import torch

from circuitkit.applications.pruning.neuron_pruner import (  # noqa: adjust path as needed
    zero_attn_neuron_hook,
    zero_neuron_hook,
)


# ===========================================================================
# 1. zero_neuron_hook  (expects 3-D tensor: batch × pos × d_model)
# ===========================================================================
class TestZeroNeuronHook:
    def test_zeros_specified_neurons(self):
        activation = torch.ones(2, 5, 64)
        hook = MagicMock()
        zero_neuron_hook(activation, hook, neuron_indices=[0, 3, 63])
        assert activation[:, :, 0].sum() == 0.0
        assert activation[:, :, 3].sum() == 0.0
        assert activation[:, :, 63].sum() == 0.0

    def test_other_neurons_unaffected(self):
        activation = torch.ones(2, 5, 64)
        hook = MagicMock()
        zero_neuron_hook(activation, hook, neuron_indices=[0])
        # All neurons except 0 should still be 1
        assert activation[:, :, 1:].sum() == 2 * 5 * 63

    def test_empty_indices_no_change(self):
        activation = torch.ones(2, 5, 64)
        original_sum = activation.sum().item()
        hook = MagicMock()
        zero_neuron_hook(activation, hook, neuron_indices=[])
        assert activation.sum().item() == pytest.approx(original_sum)

    def test_wrong_ndim_raises(self):
        activation = torch.ones(2, 5, 4, 16)  # 4-D — wrong for this hook
        hook = MagicMock()
        with pytest.raises(ValueError):
            zero_neuron_hook(activation, hook, neuron_indices=[0])

    def test_2d_tensor_raises(self):
        activation = torch.ones(5, 64)  # 2-D — wrong
        hook = MagicMock()
        with pytest.raises(ValueError):
            zero_neuron_hook(activation, hook, neuron_indices=[0])

    def test_single_neuron_index(self):
        activation = torch.ones(1, 3, 8)
        hook = MagicMock()
        zero_neuron_hook(activation, hook, neuron_indices=[4])
        assert activation[0, :, 4].sum() == 0.0

    def test_all_neurons_zeroed(self):
        activation = torch.ones(2, 4, 16)
        hook = MagicMock()
        zero_neuron_hook(activation, hook, neuron_indices=list(range(16)))
        assert activation.sum() == 0.0


# ===========================================================================
# 2. zero_attn_neuron_hook  (expects 4-D tensor: batch × pos × n_heads × d_head)
# ===========================================================================
class TestZeroAttnNeuronHook:
    def test_zeros_specified_neurons_in_head(self):
        activation = torch.ones(2, 5, 4, 16)  # [batch, pos, heads, d_head]
        hook = MagicMock()
        zero_attn_neuron_hook(activation, hook, head_index=2, neuron_indices=[0, 5])
        assert activation[:, :, 2, 0].sum() == 0.0
        assert activation[:, :, 2, 5].sum() == 0.0

    def test_other_heads_unaffected(self):
        activation = torch.ones(2, 5, 4, 16)
        hook = MagicMock()
        zero_attn_neuron_hook(activation, hook, head_index=1, neuron_indices=[0])
        for h in [0, 2, 3]:
            assert activation[:, :, h, :].sum() == 2 * 5 * 16

    def test_other_neurons_in_same_head_unaffected(self):
        activation = torch.ones(1, 3, 2, 8)
        hook = MagicMock()
        zero_attn_neuron_hook(activation, hook, head_index=0, neuron_indices=[2])
        # Neuron 2 of head 0 is zero; others in head 0 are still 1
        assert activation[:, :, 0, 2].sum() == 0.0
        assert activation[:, :, 0, 0].sum() > 0.0

    def test_wrong_ndim_raises(self):
        activation = torch.ones(2, 5, 64)  # 3-D — wrong for attn hook
        hook = MagicMock()
        with pytest.raises(ValueError):
            zero_attn_neuron_hook(activation, hook, head_index=0, neuron_indices=[0])

    def test_empty_neuron_indices_no_change(self):
        activation = torch.ones(2, 4, 3, 8)
        original_sum = activation.sum().item()
        hook = MagicMock()
        zero_attn_neuron_hook(activation, hook, head_index=0, neuron_indices=[])
        assert activation.sum().item() == pytest.approx(original_sum)

    def test_all_neurons_in_head_zeroed(self):
        d_head = 8
        activation = torch.ones(2, 3, 2, d_head)
        hook = MagicMock()
        zero_attn_neuron_hook(activation, hook, head_index=1, neuron_indices=list(range(d_head)))
        assert activation[:, :, 1, :].sum() == 0.0
        assert activation[:, :, 0, :].sum() > 0.0
