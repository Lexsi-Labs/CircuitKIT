"""
Random baseline for pruning.

Randomly prunes parameters to achieve target sparsity.
This provides a simple sanity check baseline.
"""

import logging
from typing import Dict

import numpy as np
import torch
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


class RandomBaseline:
    """
    Random pruning baseline.

    Randomly selects parameters to prune, achieving target sparsity.
    This is the simplest possible baseline and provides a lower bound
    on what any structured pruning method should achieve.
    """

    def __init__(self, seed: int = 42, verbose: bool = False):
        """
        Initialize RandomBaseline.

        Args:
            seed: Random seed for reproducibility
            verbose: Print debug info
        """
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)

    def select_parameters(
        self,
        model: HookedTransformer,
        target_sparsity: float = 0.1,
    ) -> Dict[str, bool]:
        """
        Randomly select parameters to keep.

        Args:
            model: Model to prune
            target_sparsity: Fraction of parameters to remove

        Returns:
            Dict mapping parameter names to keep/prune boolean
        """
        param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.numel() > 0:
                param_names.append(name)

        # Randomly select which parameters to keep
        num_to_keep = int(len(param_names) * (1 - target_sparsity))
        keep_params = set(self.rng.choice(param_names, size=num_to_keep, replace=False))

        selection = {}
        for name in param_names:
            selection[name] = name in keep_params

        if self.verbose:
            num_kept = sum(selection.values())
            num_total = len(selection)
            actual_sparsity = 1.0 - (num_kept / num_total)
            logger.info(
                f"Randomly selected {num_kept}/{num_total} parameters "
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
        Prune model randomly.

        Args:
            model: Model to prune
            target_sparsity: Target sparsity level
            inplace: Modify model in-place

        Returns:
            Pruned model
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
            f"Random pruned {num_pruned}/{total_params} parameters "
            f"({100*num_pruned/total_params:.2f}%)"
        )

        return pruned_model

    def get_sparsity_mask(
        self,
        model: HookedTransformer,
        target_sparsity: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Get binary mask for random pruning.

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
