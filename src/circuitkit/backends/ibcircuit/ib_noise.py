"""
IB noise injection and weight initialisation for IBCircuit.

Implements the core Information Bottleneck formula from IBCircuit
(Niu et al.): applies learnable Gaussian noise to attention and MLP
activations, where noise magnitude is inversely proportional to learned
importance weights. Provides weight initialisers for both node-level and
neuron-level granularity.

Reference: https://github.com/ivanniu/IBCircuit
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def apply_ib_noise(
    activation: torch.Tensor,
    ib_weight: torch.Tensor,
    mask_type: str = "sigmoid",
    epsilon: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Information Bottleneck noise injection to a layer activation.

    Implements the core IB-Circuit formula: components with high importance
    (weight → 1) retain their signal; components with low importance (weight → 0)
    are pulled toward the batch mean and corrupted with noise, effectively masking
    them out. The accompanying KL loss penalises information transmission.

    Steps:
        1. Compute batch mean and std of the activation.
        2. Mix activation toward batch mean, scaled by (1 - weight).
        3. Scale noise std by (1 - weight).
        4. Sample: noisy = noisy_mean + noisy_std * N(0,1).
        5. Compute KL(original || noisy) as the bottleneck regularisation term.

    Args:
        activation (torch.Tensor): Layer activation to corrupt. For attention,
            expected shape is [batch_size, n_heads, pos, d_head] (after permute
            in the hook). For MLP, [batch_size, pos, d_model].
        ib_weight (torch.Tensor): Learnable importance weight, same leading
            dimensions as activation. Raw logits when mask_type='sigmoid',
            values in [0, 1] when mask_type='raw'.
        mask_type (str): How to convert ib_weight to a [0, 1] mask.
            'sigmoid' applies sigmoid (default); 'raw' uses the weight as-is.
        epsilon (float): Small constant for numerical stability in std
            and log computations. Default: 1e-7.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - noisy_activation: Corrupted activation, same shape as input,
            cast back to the original dtype.
            - kl_loss: Scalar KL divergence KL(original || noisy), averaged
            across all dimensions. Used as the IB regularisation term.
    """
    # Cast to float32 for all internal computation.
    orig_dtype = activation.dtype
    activation_f32 = activation.detach().float()

    # Step 1: Apply sigmoid to convert raw weights to [0, 1] range
    if mask_type == "sigmoid":
        node_IB_weight = torch.sigmoid(ib_weight).float()
    else:
        node_IB_weight = ib_weight.float()

    # Step 2: Compute statistics across batch dimension (dim=0).
    # correction=0 (population std) avoids NaN when batch_size=1 — sample std
    # (correction=1, PyTorch default) is undefined for a single sample.
    act_mean = torch.mean(activation_f32, dim=0)
    act_std = torch.std(activation_f32, dim=0, correction=0).clamp(min=epsilon)

    # Step 3: Create noisy mean by mixing original signal with batch mean
    # High weight (→1): noise_mean ≈ activation (keep original)
    # Low weight (→0): noise_mean ≈ act_mean (pull toward average)
    noise_act_mean = node_IB_weight * activation_f32 + (1 - node_IB_weight) * act_mean

    # Step 4: Scale down the standard deviation based on weight
    # High weight (→1): noise_std ≈ 0 (minimal noise)
    # Low weight (→0): noise_std ≈ act_std (full noise)
    noise_act_std = (1 - node_IB_weight) * act_std

    # Step 5: Sample from Gaussian with the noisy statistics
    # Standard formula: Z = mu + sigma * epsilon
    noisy_activation = (noise_act_mean + noise_act_std * torch.randn_like(act_mean)).to(orig_dtype)

    # Step 6: Compute KL divergence (P_noisy || P_original) for Gaussian distributions
    # Measures the information transmitted through the bottleneck
    KL_tensor = 0.5 * (
        # Variance ratio term
        (noise_act_std / (act_std + epsilon)) ** 2
        +
        # Mean difference term
        ((noise_act_mean - act_mean) / (act_std + epsilon)) ** 2
        -
        # Constant term from the Gaussian KL closed form
        1.0
        +
        # Log determinant term (for Gaussian KL)
        torch.log((act_std / (noise_act_std + epsilon) + epsilon) ** 2)
    )

    # Average KL loss across all dimensions
    KL_loss = torch.mean(KL_tensor)

    return noisy_activation, KL_loss


def initialize_attn_ib_weights(
    batch_size: int,
    num_layers: int,
    num_heads: int,
    device: torch.device,
    init_mean: float = 5.0,
    init_std: float = 0.01,
    dtype: torch.dtype = torch.float32,
    level: str = "node",
    d_head: Optional[int] = None,
) -> nn.ParameterList:
    """
    Initialise learnable IB importance weights for all attention layers.

    Each weight tensor is initialised near `init_mean` so that sigmoid(init_mean) ≈ 1,
    meaning all heads start as fully important and sparsity is learned during training.

    Args:
        batch_size (int): Training batch size. IB weights are example-specific
            and must match the fixed training batch.
        num_layers (int): Number of transformer layers.
        num_heads (int): Number of attention heads per layer.
        device (torch.device): Device for weight tensors.
        init_mean (float): Initial weight value before sigmoid. Default 5.0
            gives sigmoid(5.0) ≈ 0.993, i.e. all heads start important.
        init_std (float): Std of random initialisation noise. Default: 0.01.
        dtype (torch.dtype): Weight dtype. Default: torch.float32.
        level (str): Granularity — 'node' (one scalar per head) or 'neuron'
            (one value per head dimension).
        d_head (int | None): Head dimension. Required when level='neuron'.

    Returns:
        nn.ParameterList: One Parameter per layer.
            - node   shape: [batch_size, num_heads, 1, 1]
            - neuron shape: [batch_size, num_heads, 1, d_head]

    Raises:
        ValueError: If level is invalid or d_head is None when level='neuron'.
    """
    if level not in ("node", "neuron"):
        raise ValueError(f"level must be 'node' or 'neuron', got '{level}'")
    if level == "neuron" and d_head is None:
        raise ValueError("d_head must be provided when level = 'neuron'")

    ib_weights = nn.ParameterList()

    for _ in range(num_layers):
        last_dim = 1 if level == "node" else d_head
        weight = torch.empty(batch_size, num_heads, 1, last_dim, device=device, dtype=dtype)
        weight.normal_(mean=init_mean, std=init_std)

        # Convert to learnable parameter
        ib_weights.append(nn.Parameter(weight))

    return ib_weights


def initialize_mlp_ib_weights(
    batch_size: int,
    num_layers: int,
    device: torch.device,
    init_mean: float = 5.0,
    init_std: float = 0.01,
    dtype: torch.dtype = torch.float32,
    level: str = "node",
    d_model: Optional[int] = None,
) -> nn.ParameterList:
    """
    Initialise learnable IB importance weights for all MLP layers.

    Mirrors initialize_attn_ib_weights for MLP outputs. The weight broadcasts
    over [batch_size, pos, d_model] (or d_mlp for post-activation hooks).

    Args:
        batch_size (int): Training batch size. Must match the fixed training batch.
        num_layers (int): Number of transformer layers.
        device (torch.device): Device for weight tensors.
        init_mean (float): Initial weight value before sigmoid. Default 5.0
            gives sigmoid(5.0) ≈ 0.993, i.e. all MLPs start important.
        init_std (float): Std of random initialisation noise. Default: 0.01.
        dtype (torch.dtype): Weight dtype. Default: torch.float32.
        level (str): Granularity — 'node' (one scalar per MLP layer) or
            'neuron' (one value per output dimension).
        d_model (int | None): MLP output dimension (d_model for hook_mlp_out,
            d_mlp for post_act hook). Required when level='neuron'.

    Returns:
        nn.ParameterList: One Parameter per layer.
            - node   shape: [batch_size, 1, 1]
            - neuron shape: [batch_size, 1, d_model]

    Raises:
        ValueError: If level is invalid or d_model is None when level='neuron'.
    """
    if level not in ("node", "neuron"):
        raise ValueError(f"level must be 'node' or 'neuron', got '{level}'")
    if level == "neuron" and d_model is None:
        raise ValueError("d_model must be provided when level = 'neuron'")

    mlp_ib_weights = nn.ParameterList()
    for _ in range(num_layers):
        last_dim = 1 if level == "node" else d_model
        weight = torch.empty(batch_size, 1, last_dim, device=device, dtype=dtype)
        weight.normal_(mean=init_mean, std=init_std)
        mlp_ib_weights.append(nn.Parameter(weight))
    return mlp_ib_weights
