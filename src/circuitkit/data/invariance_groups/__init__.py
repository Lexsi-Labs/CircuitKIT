"""
Invariance-grouped dataset infrastructure for circuit evaluation.

Public API:
    InvarianceGroup       — a base example + its typed, contracted variants
    InvarianceVariant     — one transformed variant with an explicit contract
    InvarianceContract    — specifies label/position/length invariance
    VariantType           — enum of supported transformation families
    InvarianceGroupBuilder — wraps corruption transforms to build groups
    DEFAULT_CONTRACTS     — default contract per VariantType
    register_paraphrase_transform — register a custom paraphrase function
"""

from .builder import InvarianceGroupBuilder, register_paraphrase_transform
from .schema import (
    DEFAULT_CONTRACTS,
    InvarianceContract,
    InvarianceGroup,
    InvarianceVariant,
    VariantType,
    new_group_id,
)

__all__ = [
    "VariantType",
    "InvarianceContract",
    "InvarianceVariant",
    "InvarianceGroup",
    "DEFAULT_CONTRACTS",
    "new_group_id",
    "InvarianceGroupBuilder",
    "register_paraphrase_transform",
]
