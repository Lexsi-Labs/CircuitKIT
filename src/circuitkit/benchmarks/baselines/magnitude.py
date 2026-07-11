"""
Magnitude baseline: Select parameters by weight magnitude.

Simple baseline that prunes weights with smallest absolute values.
This is a common heuristic for model compression.
"""

import logging
from typing import Dict

import numpy as np
import torch
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


class MagnitudeBaseline:
    """
    Magnitude-based pruning baseline.

    Selects the largest-magnitude weights to keep during pruning.
    This is one of the simplest and most widely-used baselines.

    Reference:
        Song Han et al. "Learning both Weights and Connections for
        Efficient Neural Networks" (2015)
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize MagnitudeBaseline.

        Args:
            verbose: Print debug info
        """
        self.verbose = verbose
        self.scores: Dict[str, float] = {}

    def score_parameters(
        self,
        model: HookedTransformer,
        target_sparsity: float = 0.1,
    ) -> Dict[str, float]:
        """
        Score all parameters by absolute value.

        Args:
            model: Model to score
            target_sparsity: Target sparsity level (unused for scoring,
                used only for calibration)

        Returns:
            Dict mapping parameter names to importance scores
        """
        self.scores = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                # Score = absolute value of parameter
                score = torch.abs(param).mean().item()
                self.scores[name] = float(score)

        if self.verbose:
            logger.info(f"Scored {len(self.scores)} parameters")

        return self.scores

    def select_parameters(
        self,
        model: HookedTransformer,
        target_sparsity: float = 0.1,
    ) -> Dict[str, bool]:
        """
        Select parameters to keep (inverted sparsity).

        Args:
            model: Model to prune
            target_sparsity: Fraction of parameters to remove

        Returns:
            Dict mapping parameter names to keep/prune boolean
        """
        if not self.scores:
            self.score_parameters(model, target_sparsity)

        # Collect all scores and sort
        scores_list = list(self.scores.items())
        scores_list.sort(key=lambda x: x[1], reverse=True)

        # Select top (1 - sparsity) fraction
        num_to_keep = int(len(scores_list) * (1 - target_sparsity))
        keep_params = set([name for name, _ in scores_list[:num_to_keep]])

        selection = {}
        for name in self.scores.keys():
            selection[name] = name in keep_params

        if self.verbose:
            num_kept = sum(selection.values())
            num_total = len(selection)
            actual_sparsity = 1.0 - (num_kept / num_total)
            logger.info(
                f"Selected {num_kept}/{num_total} parameters "
                f"(actual sparsity: {actual_sparsity:.2%})"
            )

        return selection

    def prune_model(
        self,
        model: HookedTransformer,
        target_sparsity: float = 0.1,
        inplace: bool = False,
    ) -> HookedTransformer:
        """
        Prune model by zeroing small-magnitude weights.

        Args:
            model: Model to prune
            target_sparsity: Target sparsity level
            inplace: Modify model in-place

        Returns:
            Pruned model (same object if inplace=True, else new copy)
        """
        import copy

        pruned_model = model if inplace else copy.deepcopy(model)
        selection = self.select_parameters(pruned_model, target_sparsity)

        num_pruned = 0
        total_params = 0

        for name, param in pruned_model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                if name not in selection or not selection[name]:
                    param.data.zero_()
                    num_pruned += param.numel()
                total_params += param.numel()

        logger.info(
            f"Pruned {num_pruned}/{total_params} parameters "
            f"({100*num_pruned/total_params:.2f}%)"
        )

        return pruned_model

    def get_sparsity_mask(
        self,
        model: HookedTransformer,
        target_sparsity: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Get binary mask for pruning.

        Args:
            model: Model to mask
            target_sparsity: Target sparsity

        Returns:
            Dict mapping parameter names to binary masks
        """
        selection = self.select_parameters(model, target_sparsity)
        masks = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                mask = torch.ones_like(param)
                if name not in selection or not selection[name]:
                    mask.zero_()
                masks[name] = mask

        return masks

    def compute_layer_importance(
        self,
        model: HookedTransformer,
    ) -> Dict[str, float]:
        """
        Compute per-layer importance (average magnitude).

        Args:
            model: Model to analyze

        Returns:
            Dict mapping layer names to average magnitude
        """
        layer_importance = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                # Extract layer name (e.g., "transformer.h.0.attn.c_attn")
                layer_name = ".".join(name.split(".")[:-1])
                importance = torch.abs(param).mean().item()

                if layer_name not in layer_importance:
                    layer_importance[layer_name] = []
                layer_importance[layer_name].append(importance)

        # Average per layer
        layer_avg = {k: np.mean(v) for k, v in layer_importance.items()}

        return layer_avg
