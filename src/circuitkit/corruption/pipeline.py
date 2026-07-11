"""
CorruptionPipeline: Orchestrator for corruption strategies with validation and filtering.

Chains multiple corruption strategies, applies validators, scores by severity,
and produces a multi-variant corrupted dataset. Handles deterministic RNG and
stratified filtering. Supports optional caching to avoid regenerating expensive
corruptions.
"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from .base import CorruptionStrategy, CorruptionValidation
from .validators import CorruptionValidator

if TYPE_CHECKING:
    from circuitkit.utils.dataset_cache import CorruptionCache


@dataclass
class CorruptionCandidate:
    """Internal representation of a corruption candidate during pipeline execution.

    Attributes:
        example: Corrupted example dict.
        strategy_name: Name of the strategy used.
        severity: Severity score [0, 1] from validators.
        validation_results: Dict mapping validator name to CorruptionValidation.
        is_valid: Whether all validators passed.
    """

    example: Dict[str, Any]
    strategy_name: str
    severity: float
    validation_results: Dict[str, CorruptionValidation]
    is_valid: bool


class CorruptionPipeline:
    """Orchestrate multiple corruption strategies with validation and filtering.

    Chains multiple CorruptionStrategy instances, applies validators to each
    candidate corruption, and produces a filtered dataset with severity scoring
    and multi-variant support.

    This is the high-level orchestrator that:
    1. Tries multiple strategies per example
    2. Validates each corruption against all validators
    3. Scores by severity (lower = less severe change)
    4. Keeps top-k variants per example
    5. Filters by sampling strategy (best or random)
    """

    def __init__(
        self,
        strategies: List[CorruptionStrategy],
        validators: Optional[List[CorruptionValidator]] = None,
        n_variants: int = 1,
        keep_top_k: int = 1,
        sampling: Literal["best", "random"] = "best",
        max_workers: Optional[int] = None,
        cache: Optional["CorruptionCache"] = None,
    ):
        """Initialize the CorruptionPipeline.

        Args:
            strategies: List of CorruptionStrategy instances to try. Each will be
                       applied to each example in order.
            validators: List of CorruptionValidator instances to apply. If None,
                       uses no validators (all candidates marked valid). Can also
                       be an empty list.
            n_variants: Number of corruption attempts per (example, strategy) pair.
                       Higher values increase diversity but cost per example.
            keep_top_k: Number of top-k variants to keep per example after filtering.
                       Ranking is by severity (ascending). If keep_top_k > all valid
                       candidates, keeps all.
            sampling: "best" - deterministically select lowest severity variants.
                     "random" - randomly sample from top-k variants.
            max_workers: Number of threads for batch processing. If None, uses
                        sequential processing. Set to positive int for parallel.
            cache: Optional CorruptionCache instance for caching corruption results.
                  If provided, checks cache before generating corruptions and saves
                  successful corruptions to cache. Significantly speeds up repeated
                  runs with same configuration.
        """
        self.strategies = strategies
        self.validators = validators if validators is not None else []
        self.n_variants = n_variants
        self.keep_top_k = keep_top_k
        self.sampling = sampling
        self.max_workers = max_workers
        self.cache = cache

        if sampling not in ("best", "random"):
            raise ValueError(f"sampling must be 'best' or 'random', got {sampling}")

    def corrupt_example(
        self, example: Dict[str, Any], rng: random.Random, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Corrupt a single example and return the best variant.

        Algorithm:
        1. If cache enabled, check cache for each (strategy, seed) pair.
        2. For each strategy:
           - For each of n_variants attempts:
             * Check cache (if available)
             * If cache hit, use cached corruption
             * If cache miss:
               - Apply strategy to example
               - Run all validators
               - Score by severity
               - Save to cache (if enabled)
        3. Filter to valid corruptions only
        4. Sort by severity (ascending - lower is better)
        5. Keep top keep_top_k variants
        6. Sample one according to sampling strategy

        Args:
            example: Clean example dictionary.
            rng: Random number generator for reproducibility.
            metadata: Optional task-specific metadata (e.g., entity pools).

        Returns:
            Corrupted example dict with additional keys:
            - strategy_used: str, name of the strategy applied
            - severity: float, [0, 1], severity score
            - validation_results: dict, mapping validator name to results
            - cache_hit: bool, whether result came from cache
        """
        candidates: List[CorruptionCandidate] = []

        # Extract clean text from example for cache keying
        clean_text = example.get("prompt", str(example))

        # Generate candidates from all strategies
        for strategy in self.strategies:
            for variant_idx in range(self.n_variants):
                # Create deterministic seed for this variant
                variant_seed = rng.randint(0, 2**31 - 1)

                # Check cache if enabled
                cache_hit = False
                cached_result = None
                if self.cache is not None:
                    strategy_config = {
                        "strategy_name": strategy.name,
                        "variant_idx": variant_idx,
                    }
                    cache_key = self.cache.get_key(clean_text, strategy_config, variant_seed)
                    cached_result = self.cache.load(cache_key)

                    if cached_result is not None:
                        cache_hit = True

                # Use cached result or generate new one
                if cache_hit and cached_result is not None:
                    # Use cached corruption
                    corrupted = cached_result["corrupted"]
                    # Extract validation results from cache metadata if available
                    validation_results = cached_result.get("metadata", {}).get(
                        "validation_results", {}
                    )
                    severity = cached_result.get("metadata", {}).get("severity", 0.0)
                    all_valid = cached_result.get("metadata", {}).get("all_valid", True)
                else:
                    # Generate new corruption
                    try:
                        # Create RNG for this variant with deterministic seed
                        variant_rng = random.Random(variant_seed)
                        corrupted = strategy.corrupt(example, rng=variant_rng, metadata=metadata)
                    except Exception:
                        # Skip corruptions that fail
                        continue

                    # Validate the corruption
                    validation_results = {}
                    all_valid = True
                    max_severity = 0.0

                    for validator in self.validators:
                        try:
                            result = validator.validate(example, corrupted)
                            validator_name = validator.__class__.__name__
                            validation_results[validator_name] = result

                            if not result.is_valid:
                                all_valid = False
                            # Aggregate severity as max across validators
                            max_severity = max(max_severity, result.severity)
                        except Exception:
                            # If validator fails, mark as invalid
                            all_valid = False

                    severity = max_severity

                    # Save to cache if enabled and valid
                    if self.cache is not None and all_valid:
                        strategy_config = {
                            "strategy_name": strategy.name,
                            "variant_idx": variant_idx,
                        }
                        cache_key = self.cache.get_key(clean_text, strategy_config, variant_seed)

                        # Prepare metadata for caching
                        cache_metadata = {
                            "severity": severity,
                            "all_valid": all_valid,
                            "validation_results": {
                                k: v.reason for k, v in validation_results.items()
                            },
                        }
                        self.cache.save(
                            cache_key,
                            corrupted,
                            strategy_name=strategy.name,
                            metadata=cache_metadata,
                        )

                # Create candidate
                candidate = CorruptionCandidate(
                    example=corrupted,
                    strategy_name=strategy.name,
                    severity=severity,
                    validation_results=validation_results,
                    is_valid=all_valid,
                )
                candidates.append(candidate)

        # Filter to valid candidates only
        valid_candidates = [c for c in candidates if c.is_valid]

        if not valid_candidates:
            # If no valid candidates, return original example unmodified
            # (with empty metadata to indicate failure)
            return {
                **example,
                "strategy_used": "none",
                "severity": 0.0,
                "validation_results": {},
            }

        # Sort by severity (ascending - lower is better)
        valid_candidates.sort(key=lambda c: c.severity)

        # Keep top-k
        top_k_candidates = valid_candidates[: self.keep_top_k]

        # Sample one according to strategy
        if self.sampling == "best":
            # Return the best (lowest severity)
            selected = top_k_candidates[0]
        else:  # random
            # Randomly select from top-k
            selected = rng.choice(top_k_candidates)

        # Build output with metadata
        return {
            **selected.example,
            "strategy_used": selected.strategy_name,
            "severity": selected.severity,
            "validation_results": selected.validation_results,
        }

    def corrupt_dataset(
        self,
        examples: List[Dict[str, Any]],
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Corrupt a batch of examples (potentially in parallel).

        Args:
            examples: List of clean examples.
            rng: Random number generator. Will create independent RNG states for
                 each example to ensure reproducibility.
            metadata: Optional task-specific metadata.

        Returns:
            List of corrupted examples with metadata fields.
        """
        if self.max_workers is None or self.max_workers <= 1:
            # Sequential processing
            results = []
            for example in examples:
                # Create independent RNG state for each example
                example_seed = rng.randint(0, 2**31 - 1)
                example_rng = random.Random(example_seed)
                corrupted = self.corrupt_example(example, example_rng, metadata)
                results.append(corrupted)
            return results
        else:
            # Parallel processing with ThreadPoolExecutor
            results = [None] * len(examples)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for idx, example in enumerate(examples):
                    example_seed = rng.randint(0, 2**31 - 1)
                    example_rng = random.Random(example_seed)
                    future = executor.submit(self.corrupt_example, example, example_rng, metadata)
                    futures[future] = idx

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception:
                        # On error, return original example
                        results[idx] = {
                            **examples[idx],
                            "strategy_used": "error",
                            "severity": 0.0,
                            "validation_results": {},
                        }

            return results

    def get_strategy_names(self) -> List[str]:
        """Return list of strategy names.

        Returns:
            List of strategy name strings.
        """
        return [s.name for s in self.strategies]

    def get_validator_names(self) -> List[str]:
        """Return list of validator names.

        Returns:
            List of validator class names.
        """
        return [v.__class__.__name__ for v in self.validators]
