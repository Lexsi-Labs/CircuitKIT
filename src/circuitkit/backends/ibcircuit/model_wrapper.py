"""
IBHookedTransformer: Universal IB wrapper for HookedTransformer models.

Provides a lightweight wrapper that adds Information Bottleneck (IB) noise
injection to any HookedTransformer model via its hook system. Supports both
attention heads and MLP layers at node or neuron granularity.

Architecture:
    IBHookedTransformer
    ├── model:           HookedTransformer (frozen)
    ├── attn_ib_weights: ParameterList — per-layer attn importance weights
    │                    (empty if scope excludes 'heads')
    ├── mlp_ib_weights:  ParameterList — per-layer MLP importance weights
    │                    (empty if scope excludes 'mlp')
    └── forward_with_ib() — injects IB noise via hooks during the forward pass
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from .ib_noise import apply_ib_noise, initialize_attn_ib_weights, initialize_mlp_ib_weights


class IBHookedTransformer:
    """
    Wraps a frozen HookedTransformer with learnable IB noise injection.

    During forward passes, hooks intercept attention outputs (hook_z) and/or MLP
    outputs (hook_mlp_out or mlp.hook_post) and apply IB noise scaled by learned
    importance weights. Components with high importance retain their signal;
    low-importance components are pushed toward the batch mean.

    Attributes:
        model (HookedTransformer): Frozen base model.
        attn_ib_weights (nn.ParameterList): Per-layer attention IB weights.
            node   shape: [batch_size, n_heads, 1, 1] per layer.
            neuron shape: [batch_size, n_heads, 1, d_head] per layer.
            Empty ParameterList when scope excludes 'heads'.
        mlp_ib_weights (nn.ParameterList): Per-layer MLP IB weights.
            node   shape: [batch_size, 1, 1] per layer.
            neuron shape: [batch_size, 1, d_model_or_d_mlp] per layer.
            Empty ParameterList when scope excludes 'mlp'.
        scope (str): Components scored — 'heads', 'mlp', or 'both'.
        level (str): Scoring granularity — 'node' or 'neuron'.
        mlp_hook (str): MLP hook point — 'mlp_out' or 'post_act'.
        n_layers (int): Number of transformer layers.
        n_heads (int): Attention heads per layer.
        d_head (int): Dimension of each attention head.
        d_model (int): Residual stream dimension.
        d_mlp (int): MLP hidden dimension.
        batch_size (int): Fixed training batch size.
        mask_type (str): IB weight activation — 'sigmoid' or 'raw'.
        device (torch.device): Device of the base model.

    Example:
        >>> model = HookedTransformer.from_pretrained('gpt2')
        >>> ib_model = IBHookedTransformer(model, batch_size=100, device='cuda')
        >>> output, ib_lambdas, kl_loss = ib_model.forward_with_ib(input_ids)
        >>> optimizer = torch.optim.Adam(ib_model.get_trainable_parameters(), lr=0.05)
        >>> total_loss = task_loss + beta * kl_loss
        >>> total_loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        model: HookedTransformer,
        batch_size: int,
        device: str,
        mask_type: str = "sigmoid",
        scope: str = "heads",
        init_mean: float = 5.0,
        init_std: float = 0.01,
        level: str = "node",
        mlp_hook: str = "mlp_out",
    ):
        """
        Initialise the IB wrapper around a frozen HookedTransformer.

        Args:
            model (HookedTransformer): Pre-loaded model. Weights are frozen on init.
            batch_size (int): Training batch size. IB weights are example-specific
                and must match the fixed batch used throughout training.
            device (str): Device for IB weight tensors ('cuda' or 'cpu').
            mask_type (str): How to convert raw IB weights to [0, 1] importance values.
                'sigmoid' applies sigmoid (recommended); 'raw' uses weights as-is.
            scope (str): Which components to score.
                'heads' — attention heads only.
                'mlp'   — MLP layers only.
                'both'  — attention heads and MLP layers.
            init_mean (float): Initial raw weight value. sigmoid(5.0) ≈ 0.993, so
                all components start as important and sparsity is learned. Default: 5.0.
            init_std (float): Std of weight initialisation noise. Default: 0.01.
            level (str): Scoring granularity — 'node' (one score per component)
                or 'neuron' (one score per activation dimension). Default: 'node'.
            mlp_hook (str): MLP hook point — 'mlp_out' (hook_mlp_out, scores d_model
                dimensions) or 'post_act' (mlp.hook_post, scores d_mlp dimensions).
                Default: 'mlp_out'.

        Raises:
            ValueError: If batch_size <= 0.
            ValueError: If model has no n_heads attribute (non-attention architecture).
        """
        # Validate inputs
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if not hasattr(model.cfg, "n_heads"):
            raise ValueError(
                f"Model {model.cfg.model_name} doesn't have n_heads attribute. "
                f"IB-Circuit requires attention-based models."
            )

        # Store model reference
        self.model = model
        # Freeze base model — only IB weights are trainable
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.device = next(model.parameters()).device
        self.mask_type = mask_type
        self.batch_size = batch_size

        # Extract architecture info from model config
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.d_head = model.cfg.d_head
        self.d_model = model.cfg.d_model
        self.mlp_hook = mlp_hook
        self.d_mlp = model.cfg.d_mlp

        # Enable hook_z which gives us attention output before W_O projection
        # This is critical for IB noise injection
        self.scope = scope
        self.level = level

        # Attention IB weights — one [batch, n_heads, 1, 1] tensor per layer.
        # Always a ParameterList; empty when scope excludes attention so that
        # state_dict / save paths remain consistent regardless of scope.
        if scope in ("heads", "both"):
            self.attn_ib_weights = initialize_attn_ib_weights(
                batch_size=batch_size,
                num_layers=self.n_layers,
                num_heads=self.n_heads,
                device=self.device,
                init_mean=init_mean,
                init_std=init_std,
                level=self.level,
                d_head=self.d_head,
            )
        else:
            self.attn_ib_weights = nn.ParameterList()

        # MLP IB weights — one [batch, 1, 1] tensor per layer.
        if scope in ("mlp", "both"):
            self.mlp_ib_weights = initialize_mlp_ib_weights(
                batch_size=batch_size,
                num_layers=self.n_layers,
                device=self.device,
                init_mean=init_mean,
                init_std=init_std,
                level=self.level,
                d_model=self.d_mlp if mlp_hook == "post_act" else self.d_model,
            )
        else:
            self.mlp_ib_weights = nn.ParameterList()

        # Temporary storage for KL losses and IB lambdas during forward pass
        # These are cleared at the start of each forward_with_ib() call
        self._kl_losses: List[torch.Tensor] = []
        self._ib_lambdas: List[torch.Tensor] = []

    def forward_with_ib(
        self, input_ids: torch.LongTensor, **model_kwargs
    ) -> Tuple[Any, List[torch.Tensor], torch.Tensor]:
        """
        Run a forward pass with IB noise injection at all in-scope layers.

        Registers hooks at attn.hook_z (for heads) and/or the MLP output hook
        (for mlp), depending on scope. Each hook applies IB noise, accumulates
        the per-layer KL loss, and stores the sigmoid-activated importance weights.

        Args:
            input_ids (torch.LongTensor): Input token IDs [batch_size, seq_len].
                batch_size must match the value passed at initialisation.
            **model_kwargs: Additional arguments forwarded to model.forward()
                (e.g., attention_mask, position_ids).

        Returns:
            tuple:
                - output: Model output object with a .logits attribute
                [batch_size, seq_len, vocab_size].
                - ib_lambdas (list[torch.Tensor]): Sigmoid-activated importance
                weights accumulated across all hooked layers in forward order.
                Shape per entry depends on scope and level.
                - avg_kl_loss (torch.Tensor): Mean KL loss across all hooked layers
                (scalar). Used as the IB regularisation term in the training loss.

        Raises:
            ValueError: If input batch_size does not match self.batch_size.
        """
        # Validate batch size
        if input_ids.shape[0] != self.batch_size:
            raise ValueError(
                f"Input batch size {input_ids.shape[0]} doesn't match "
                f"initialized batch size {self.batch_size}. "
                f"IB weights are batch-specific and cannot be resized."
            )

        # Clear accumulators from any previous forward pass
        self._kl_losses = []
        # the accumulation of IB lamda values in an empty list was deemed unnecessary and removed from here,
        # and returning an empty list instead

        hook_name = "mlp.hook_post" if self.mlp_hook == "post_act" else "hook_mlp_out"
        fwd_hooks = []
        if self.scope in ("heads", "both"):
            fwd_hooks += [
                (f"blocks.{lyr}.attn.hook_z", self._make_attn_ib_hook(lyr))
                for lyr in range(self.n_layers)
            ]
        if self.scope in ("mlp", "both"):
            fwd_hooks += [
                (f"blocks.{lyr}.{hook_name}", self._make_mlp_ib_hook(lyr))
                for lyr in range(self.n_layers)
            ]

        # Run forward pass with hooks active; each hooked layer fires automatically.
        # Wrap raw tensor output so callers can always access .logits consistently.
        with self.model.hooks(fwd_hooks):
            output = self.model(input_ids, **model_kwargs)

        # HookedTransformer returns a raw Tensor; wrap it for consistent .logits access
        if isinstance(output, torch.Tensor):
            output = type("ModelOutput", (), {"logits": output})()

        # Average KL losses across all hooked layers (attention + MLP combined)
        # Original implementation averages: final_KL_loss = total_KL_loss / n_layers
        avg_kl_loss = torch.stack(self._kl_losses).mean()

        # Return output wrapped for .logits access, per-layer lambdas, and aggregated KL loss
        return output, [], avg_kl_loss

    def _make_attn_ib_hook(self, layer_idx: int):
        """
        Return an IB noise hook for attention layer `layer_idx`.

        Intercepts hook_z — attention output before W_O projection — with shape
        [batch, pos, n_heads, d_head]. Permutes to [batch, n_heads, pos, d_head]
        for apply_ib_noise, then permutes back. KL loss and sigmoid weights are
        appended to self._kl_losses and self._ib_lambdas respectively.

        Args:
            layer_idx (int): Layer index (0 to n_layers - 1).

        Returns:
            Callable: Hook function with signature (z, hook) -> noisy_z.
        """

        def ib_hook_fn(z: torch.Tensor, hook) -> torch.Tensor:

            # Permute to match IB noise function's expected shape
            # From: [batch, pos, head_index, d_head]
            # To:   [batch, head_index, pos, d_head]
            z_heads = z.permute(0, 2, 1, 3)

            # Apply IB noise using the extracted formula
            # Returns: noisy attention output + KL divergence loss
            noisy_z_heads, kl_loss = apply_ib_noise(
                activation=z_heads,
                ib_weight=self.attn_ib_weights[layer_idx],
                mask_type=self.mask_type,
                epsilon=1e-7,
            )

            # Store KL loss for aggregation
            self._kl_losses.append(kl_loss)
            
            # the storing of IB lamda was deemed unnecessary and removed from here

            # Permute back to HookedTransformer's expected shape
            # From: [batch, head_index, pos, d_head]
            # To:   [batch, pos, head_index, d_head]
            noisy_z = noisy_z_heads.permute(0, 2, 1, 3)

            # Return noisy activation - this continues through W_O and rest of model
            return noisy_z

        return ib_hook_fn

    def _make_mlp_ib_hook(self, layer_idx: int):
        """
        Return an IB noise hook for MLP layer `layer_idx`.

        Intercepts hook_mlp_out or mlp.hook_post (shape [batch, pos, d_model or d_mlp]).
        No permutation needed — apply_ib_noise operates on dim=0 and the weight
        [batch, 1, 1] or [batch, 1, d_model] broadcasts cleanly. KL loss and sigmoid
        weights are appended to self._kl_losses and self._ib_lambdas.

        Args:
            layer_idx (int): Layer index (0 to n_layers - 1).

        Returns:
            Callable: Hook function with signature (mlp_out, hook) -> noisy_mlp_out.
        """

        def mlp_hook_fn(mlp_out: torch.Tensor, hook) -> torch.Tensor:
            noisy_mlp_out, kl_loss = apply_ib_noise(
                activation=mlp_out,
                ib_weight=self.mlp_ib_weights[layer_idx],
                mask_type=self.mask_type,
                epsilon=1e-7,
            )
            self._kl_losses.append(kl_loss)
            # the appending of IB lamda values was deemed unnecessary and removed from here
            return noisy_mlp_out

        return mlp_hook_fn

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Return all trainable IB weight parameters.

        The base model is frozen; only attn_ib_weights and mlp_ib_weights are
        trainable. Which lists are non-empty depends on scope.

        Returns:
            list[nn.Parameter]: Combined parameters from attn_ib_weights and
                mlp_ib_weights (both ParameterLists). Pass directly to an optimiser.

        Example:
            >>> optimizer = torch.optim.Adam(ib_model.get_trainable_parameters(), lr=0.05)
        """
        return list(self.attn_ib_weights.parameters()) + list(self.mlp_ib_weights.parameters())

    def extract_node_scores(self, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Extract node-level importance scores after training.

        Applies sigmoid to raw IB weights, averages across the batch dimension,
        and returns scores in CircuitKit's naming convention.

        Args:
            threshold (float | None): If provided, binarises scores — values above
                threshold become 1.0, others 0.0. If None, returns continuous
                scores in (0, 1). Default: None.

        Returns:
            dict[str, float]: Node name → importance score.
                Attention heads: "A{layer}.{head}" (e.g. "A0.3").
                MLP layers:      "MLP {layer}"     (e.g. "MLP 2").
                Keys present depend on scope.

        Raises:
            RuntimeError: If called on a neuron-level model (use extract_neuron_scores).

        Example:
            >>> scores = ib_model.extract_node_scores(threshold=0.5)
            >>> circuit = [k for k, v in scores.items() if v == 1.0]
        """
        if self.level == "neuron":
            raise RuntimeError(
                "This model was initialised with level='neuron'. "
                "Call extract_neuron_scores() instead."
            )
        node_scores = {}

        if self.scope in ("heads", "both"):
            for layer_idx in range(self.n_layers):
                # [batch, n_heads, 1, 1] → average batch → [n_heads]
                layer_scores = torch.sigmoid(self.attn_ib_weights[layer_idx]).mean(dim=0).squeeze()
                for head_idx in range(self.n_heads):
                    score = layer_scores[head_idx].item()
                    if threshold is not None:
                        score = 1.0 if score > threshold else 0.0
                    node_scores[f"A{layer_idx}.{head_idx}"] = score

        if self.scope in ("mlp", "both"):
            for layer_idx in range(self.n_layers):
                # [batch, 1, 1] → scalar
                score = torch.sigmoid(self.mlp_ib_weights[layer_idx]).mean().item()
                if threshold is not None:
                    score = 1.0 if score > threshold else 0.0
                node_scores[f"MLP {layer_idx}"] = score

        return node_scores

    def extract_neuron_scores(self) -> Dict[str, torch.Tensor]:
        """
        Extract per-dimension importance scores after neuron-level training.

        Applies sigmoid to raw IB weights and averages across the batch dimension.

        Returns:
            dict[str, torch.Tensor]: Component name → 1-D score tensor (detached).
                Attention heads: "A{layer}.{head}" → Tensor[d_head].
                MLP layers:      "MLP {layer}"     → Tensor[d_model] or Tensor[d_mlp]
                    (depends on mlp_hook: 'mlp_out' → d_model, 'post_act' → d_mlp).
                Keys present depend on scope.

        Raises:
            RuntimeError: If called on a node-level model (use extract_node_scores).
        """
        if self.level != "neuron":
            raise RuntimeError(
                "This model was initialised with level='node'. "
                "Call extract_node_scores() instead."
            )

        neuron_scores: Dict[str, torch.Tensor] = {}

        if self.scope in ("heads", "both"):
            for layer_idx in range(self.n_layers):
                # [batch, n_heads, 1, d_head] -> sigmoid -> mean over batch -> [n_heads, d_head]
                layer_scores = torch.sigmoid(self.attn_ib_weights[layer_idx]).mean(dim=0).squeeze(1)
                for head_idx in range(self.n_heads):
                    neuron_scores[f"A{layer_idx}.{head_idx}"] = layer_scores[
                        head_idx
                    ].detach()  # [d_head]

        if self.scope in ("mlp", "both"):
            for layer_idx in range(self.n_layers):
                # [batch, 1, d_model] -> sigmoid -> mean over batch -> [d_model]
                neuron_scores[f"MLP {layer_idx}"] = (
                    torch.sigmoid(self.mlp_ib_weights[layer_idx]).mean(dim=0).squeeze(0).detach()
                )

        return neuron_scores

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return current importance statistics for logging and debugging.

        All lambda values are sigmoid-activated IB weights averaged over the batch.
        Keys present in the returned dict depend on scope.

        Returns:
            dict: Subset of the following keys depending on scope:
                avg_attn_lambda_per_layer (list[float]): Mean lambda per attn layer.
                overall_avg_attn_lambda (float): Mean lambda across all attn layers.
                n_attn_heads_important (int): Count of heads/neurons with lambda > 0.5.
                total_attn_heads (int): Total attention heads or neurons scored.
                avg_mlp_lambda_per_layer (list[float]): Mean lambda per MLP layer.
                overall_avg_mlp_lambda (float): Mean lambda across all MLP layers.
                n_mlps_important (int): Count of MLP layers/neurons with lambda > 0.5.
                total_mlps (int): Total MLP layers or neurons scored.

        Example:
            >>> stats = ib_model.get_statistics()
            >>> print(f"Avg attn λ: {stats['overall_avg_attn_lambda']:.3f}, "
            ...       f"important: {stats['n_attn_heads_important']}/{stats['total_attn_heads']}")
        """
        stats = {}

        if self.scope in ("heads", "both"):
            attn_lambdas = [
                torch.sigmoid(self.attn_ib_weights[lyr]).mean().item()
                for lyr in range(self.n_layers)
            ]
            if self.level == "neuron":
                # Count individual neurons (each weight is [batch, heads, 1, d_head])
                n_important = sum(
                    (torch.sigmoid(self.attn_ib_weights[lyr]).mean(dim=0).squeeze(1) > 0.5)
                    .sum()
                    .item()
                    for lyr in range(self.n_layers)
                )
            else:
                # Count whole heads (each weight is [batch, heads, 1, 1])
                n_important = sum(
                    (torch.sigmoid(self.attn_ib_weights[lyr]).mean(dim=0).squeeze() > 0.5)
                    .sum()
                    .item()
                    for lyr in range(self.n_layers)
                )
            stats.update(
                {
                    "avg_attn_lambda_per_layer": attn_lambdas,
                    "overall_avg_attn_lambda": sum(attn_lambdas) / len(attn_lambdas),
                    "n_attn_heads_important": n_important,
                    "total_attn_heads": (
                        self.n_layers * self.n_heads * self.d_head
                        if self.level == "neuron"
                        else self.n_layers * self.n_heads
                    ),
                }
            )

        if self.scope in ("mlp", "both"):
            mlp_lambdas = [
                torch.sigmoid(self.mlp_ib_weights[lyr]).mean().item()
                for lyr in range(self.n_layers)
            ]
            stats.update(
                {
                    "avg_mlp_lambda_per_layer": mlp_lambdas,
                    "overall_avg_mlp_lambda": sum(mlp_lambdas) / len(mlp_lambdas),
                    "n_mlps_important": (
                        sum(
                            (torch.sigmoid(self.mlp_ib_weights[lyr]).mean(dim=0).squeeze(0) > 0.5)
                            .sum()
                            .item()
                            for lyr in range(self.n_layers)
                        )
                        if self.level == "neuron"
                        else sum(1 for s in mlp_lambdas if s > 0.5)
                    ),
                    "total_mlps": (
                        self.n_layers
                        * (self.d_mlp if self.mlp_hook == "post_act" else self.d_model)
                        if self.level == "neuron"
                        else self.n_layers
                    ),
                }
            )

        return stats

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"IBHookedTransformer(\n"
            f"  model={self.model.cfg.model_name},\n"
            f"  mlp_hook={self.mlp_hook},\n"
            f"  scope={self.scope},\n"
            f"  level={self.level},\n"
            f"  n_layers={self.n_layers},\n"
            f"  n_heads={self.n_heads},\n"
            f"  batch_size={self.batch_size},\n"
            f"  device={self.device},\n"
            f"  mask_type={self.mask_type},\n"
            f"  trainable_params={sum(p.numel() for p in self.get_trainable_parameters())}\n"
            f")"
        )
