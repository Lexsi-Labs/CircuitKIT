"""
CircuitKitImportance: bridges circuit-discovery scores from circuitkit
into the LLM-Pruner importance interface.

Usage
-----
    from applications.pruning.importance import CircuitKitImportance

    # scores_dict maps each nn.Module (k_proj / gate_proj) to a 1-D tensor
    # of per-channel importance scores.  Higher score = more important = kept.
    imp = CircuitKitImportance(scores_dict)

    # Pass directly to MetaPruner as the 'importance' kwarg.
    pruner = tp.pruner.MetaPruner(model, forward_prompts, importance=imp, ...)
"""

import os
import sys

# Make LLM-Pruner importable regardless of working directory
_LLM_PRUNER_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "LLM-Pruner")
_LLM_PRUNER_ROOT = os.path.abspath(_LLM_PRUNER_ROOT)
if _LLM_PRUNER_ROOT not in sys.path:
    sys.path.insert(0, _LLM_PRUNER_ROOT)

from typing import Dict, Optional  # noqa: E402 - import after intentional pre-import setup

import LLMPruner.torch_pruning as tp  # noqa: E402 - import after intentional pre-import setup
import torch  # noqa: E402 - import after intentional pre-import setup
import torch.nn as nn  # noqa: E402 - import after intentional pre-import setup


class CircuitKitImportance(tp.importance.Importance):
    """
    Importance estimator backed by circuit-discovery scores produced by circuitkit.

    The constructor receives a mapping from nn.Module instances (typically k_proj
    for attention heads and gate_proj for MLP layers) to 1-D importance tensors.
    Scores are per output-channel and should be non-negative; higher = more important.

    For any pruning group whose root module is not in the pre-computed dict the
    estimator falls back to L2 weight magnitude so that the pruner can still make
    a decision (useful for layers excluded from circuit discovery).

    Parameters
    ----------
    scores : Dict[nn.Module, torch.Tensor]
        Map from root module → per-channel importance tensor (shape: out_channels).
        Typically built by score_extractor.build_importance_dict().
    """

    def __init__(self, scores: Dict[nn.Module, torch.Tensor]):
        self.scores = scores

    @torch.no_grad()
    def __call__(
        self,
        group,
        ch_groups: int = 1,
        consecutive_groups: int = 1,
    ) -> Optional[torch.Tensor]:
        # Walk the group looking for a module we have pre-computed scores for.
        # LLM-Pruner block-wise pruning always puts the root (k_proj / gate_proj)
        # as the first or one of the early entries in a group.
        for dep, idxs in group:
            module = dep.target.module
            if module in self.scores:
                imp = self.scores[module]
                device = module.weight.device
                return imp.float().to(device)

        # Fallback: L2 magnitude of the first linear layer in the group.
        # This ensures the pruner is never blocked when a module has no
        # circuit-discovery scores (e.g., embed / lm_head touched via deps).
        return self._magnitude_fallback(group)

    def _magnitude_fallback(self, group) -> Optional[torch.Tensor]:
        for dep, idxs in group:
            module = dep.target.module
            prune_fn = dep.handler

            if not isinstance(module, nn.Linear):
                continue

            if (
                prune_fn
                in [
                    tp.prune_linear_out_channels,
                ]
                or hasattr(prune_fn, "__name__")
                and "out_channels" in getattr(prune_fn, "__name__", "")
            ):
                w = module.weight.data  # (out, in)
                imp = w.norm(p=2, dim=1)  # (out,)
                return imp.float()

        return None
