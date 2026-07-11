"""Corruption strategies — convert a clean ContrastiveRecord into a
contrastive (clean, corrupt) pair.

Strategies are independent of dataset shape. They consume a
ContrastiveRecord (already normalized by an Adapter) and either:

  - return it unchanged (if it is already paired natively, e.g. CrowS-Pairs)
  - generate ``corrupt_prompt`` + ``corrupt_answer`` via the strategy's logic

Built-in strategies cover the 16 patterns surveyed in the
PLANS/02_DATA.md literature scan. New strategies self-register via
``@register_strategy(name)``.

Reference:
    Zhang & Nanda (ICLR 2024) — String Token Replacement (in-distribution
    counterfactual pairs) is preferred whenever feasible.
"""

from .base import (
    STRATEGY_REGISTRY,
    CorruptionResult,
    CorruptionStrategy,
    get_strategy,
    list_strategies,
    register_strategy,
)
from .instruction_swap import InstructionSwap, audit_instruction_swap_degeneracy

__all__ = [
    "CorruptionStrategy",
    "CorruptionResult",
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "STRATEGY_REGISTRY",
    "InstructionSwap",
    "audit_instruction_swap_degeneracy",
]
