"""
CircuitKit Corruption Module

Provides corruption strategies for systematic example modification across tasks.
Corruption strategies implement the CorruptionStrategy protocol to enable:
- Behavioral analysis of model responses to corrupted inputs
- Circuit discovery under different corruption modes
- Robustness evaluation and stress testing
"""

from .base import CorruptionStrategy, CorruptionValidation
from .color_swap import ColorSwapCorruption
from .distractor import DistractorInjectionCorruption
from .distractor_variation import DistractorVariationCorruption
from .effectiveness import CorruptionEffectiveness, EffectivenessCalculator
from .entity_swap import EntitySwapCorruption
from .negation import NegationCorruption
from .paraphrase import ParaphraseCorruption
from .pipeline import CorruptionPipeline
from .position_shift import PositionShiftCorruption
from .role_swap import RoleSwapCorruption
from .token_swap import TokenSwapCorruption
from .validators import CorruptionValidationResult  # deprecated alias of CorruptionValidation
from .validators import (
    CompositeValidator,
    CorruptionValidator,
    LabelConsistencyValidator,
    LengthBudgetValidator,
    ModelRequirementValidator,
    SemanticShiftValidator,
    TokenConsistencyValidator,
    TokenizationValidator,
)
from .voice_swap import VoiceSwapCorruption

__all__ = [
    "CorruptionStrategy",
    "CorruptionValidation",
    "EntitySwapCorruption",
    "TokenSwapCorruption",
    "ParaphraseCorruption",
    "DistractorInjectionCorruption",
    "RoleSwapCorruption",
    "ColorSwapCorruption",
    "VoiceSwapCorruption",
    "NegationCorruption",
    "DistractorVariationCorruption",
    "PositionShiftCorruption",
    "CorruptionValidator",
    "CorruptionValidationResult",
    "LengthBudgetValidator",
    "LabelConsistencyValidator",
    "TokenizationValidator",
    "SemanticShiftValidator",
    "CompositeValidator",
    "ModelRequirementValidator",
    "TokenConsistencyValidator",
    "CorruptionPipeline",
    "CorruptionEffectiveness",
    "EffectivenessCalculator",
]
