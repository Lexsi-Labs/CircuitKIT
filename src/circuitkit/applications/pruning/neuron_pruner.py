# FILE: circuitkit/applications/pruning/neuron_pruner.py
from typing import List

import torch as t


def zero_neuron_hook(activation: t.Tensor, hook, neuron_indices: List[int]):
    """
    Zeros out specific neurons in the MLP output activation tensor.
    This is designed for activations of shape (batch, pos, d_model).
    """
    if activation.ndim == 3:
        activation[:, :, neuron_indices] = 0.0
    else:
        raise ValueError(f"Hook not supported for tensor of shape {activation.shape}")

    return activation


# --- START OF NEW CODE ---
def zero_attn_neuron_hook(activation: t.Tensor, hook, head_index: int, neuron_indices: List[int]):
    """
    Zeros out specific neuron dimensions in a specific attention head's output.
    This is for activations from `hook_result` of shape [batch, pos, n_heads, d_model].
    """
    if activation.ndim == 4:
        # Zero out the specified neuron indices for the specified head
        activation[:, :, head_index, neuron_indices] = 0.0
    else:
        raise ValueError(
            f"Attention neuron hook expects a 4D tensor, but got shape {activation.shape}"
        )
    return activation


# --- END OF NEW CODE ---
