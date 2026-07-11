# corruption

Corruption strategies that modify clean examples into corrupted counterparts, used for behavioral analysis, circuit discovery, and robustness evaluation.

## Key modules

- `base.py` — `CorruptionStrategy` protocol and `CorruptionValidation` dataclass defining the interface all strategies implement (meaning-preserving / meaning-altering / role-swap modes).
- `color_swap.py` — `ColorSwapCorruption`: swaps an answer color for another from a fixed pool (meaning-altering).
- `distractor.py` — `DistractorInjectionCorruption`: injects irrelevant-but-plausible distractor sentences into QA/MCQ/long-context prompts.
- `distractor_variation.py` — `DistractorVariationCorruption`: varies MCQ distractor difficulty (easy/hard/random) while keeping the correct answer.
- `entity_swap.py` — `EntitySwapCorruption`: spaCy-NER token-level entity swap preserving entity type (PERSON→PERSON, etc.).
- `negation.py` — `NegationCorruption`: adds/removes/toggles negation markers via dependency parsing.
- `paraphrase.py` — `ParaphraseCorruption`: meaning-preserving surface/semantic rewrites using a small local LLM (no API calls).
- `position_shift.py` — `PositionShiftCorruption`: shuffles or rotates sentence/clause order to test position sensitivity.
- `role_swap.py` — `RoleSwapCorruption`: swaps subject/object roles via spaCy dependency parsing (SVA, gender-bias tasks).
- `token_swap.py` — `TokenSwapCorruption`: POS-aware single-token replacement with tokenization validation.
- `voice_swap.py` — `VoiceSwapCorruption`: active↔passive voice transformation (meaning-preserving).
- `validators.py` — validator protocol and implementations (length budget, label consistency, tokenization, semantic shift, model requirement, token consistency, composite).
- `effectiveness.py` — `CorruptionEffectiveness` / `EffectivenessCalculator`: metrics for corruption impact (avg impact, label consistency, semantic shift, difficulty).
- `pipeline.py` — `CorruptionPipeline`: orchestrator that chains strategies, applies validators, scores by severity, and produces multi-variant filtered datasets with optional caching.

## Public API

`CorruptionStrategy`, `CorruptionValidation`, the strategy classes above, `CorruptionValidator` (plus `LengthBudget`, `LabelConsistency`, `Tokenization`, `SemanticShift`, `Composite`, `ModelRequirement`, `TokenConsistency` validators), `CorruptionPipeline`, `CorruptionEffectiveness`, `EffectivenessCalculator`.

## How it fits

Strategies plug into `CorruptionPipeline` to build corrupted datasets for circuit discovery. `datasets/invariance_groups` also wraps them to produce typed, contracted invariance variants.
