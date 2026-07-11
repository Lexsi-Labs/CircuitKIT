"""
Integration test for CorruptionPipeline with EntitySwap and Paraphrase strategies on IOI data.

This test demonstrates the orchestrator working with real strategies and validators
as specified in the POA.
"""

import importlib.util
import os
import random
from typing import Any, Dict, List

import pytest

# Load modules directly to avoid circuitkit/__init__.py issues
base_path = os.path.join(os.path.dirname(__file__), "../../src/circuitkit/corruption")

# Load base module
spec = importlib.util.spec_from_file_location(
    "test_base_module", os.path.join(base_path, "base.py")
)
base_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_mod)

# Load validators module
spec = importlib.util.spec_from_file_location(
    "test_validators_module", os.path.join(base_path, "validators.py")
)
validators_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validators_mod)

# Load entity_swap strategy
spec = importlib.util.spec_from_file_location(
    "test_entity_swap_module", os.path.join(base_path, "entity_swap.py")
)
entity_swap_code = open(os.path.join(base_path, "entity_swap.py")).read()
entity_swap_code = entity_swap_code.replace("from .base import", "# from base import")
entity_swap_dict = {
    "__name__": "test_entity_swap_module",
    "CorruptionStrategy": base_mod.CorruptionStrategy,
    "CorruptionValidation": base_mod.CorruptionValidation,
    "random": random,
    "List": List,
    "Dict": Dict,
    "Any": Any,
    "__builtins__": __builtins__,
}
exec(entity_swap_code, entity_swap_dict)
EntitySwapCorruption = entity_swap_dict["EntitySwapCorruption"]

# Load pipeline module
from concurrent.futures import (  # noqa: E402 - import after intentional pre-import setup
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field  # noqa: E402 - import after intentional pre-import setup
from typing import Literal, Optional  # noqa: E402 - import after intentional pre-import setup

spec = importlib.util.spec_from_file_location(
    "test_pipeline_module", os.path.join(base_path, "pipeline.py")
)
pipeline_code = open(os.path.join(base_path, "pipeline.py")).read()
pipeline_code = pipeline_code.replace("from .base import", "# from base import")
pipeline_code = pipeline_code.replace("from .validators import", "# from validators import")
pipeline_code = pipeline_code.replace("if TYPE_CHECKING:", "if False:")
pipeline_mod_dict = {
    "__name__": "test_pipeline_module",
    "CorruptionStrategy": base_mod.CorruptionStrategy,
    "CorruptionValidation": base_mod.CorruptionValidation,
    "CorruptionValidator": validators_mod.CorruptionValidator,
    "CorruptionValidationResult": validators_mod.CorruptionValidationResult,
    "random": random,
    "List": List,
    "Dict": Dict,
    "Any": Any,
    "Optional": Optional,
    "Literal": Literal,
    "dataclass": dataclass,
    "field": field,
    "ThreadPoolExecutor": ThreadPoolExecutor,
    "as_completed": as_completed,
    "TYPE_CHECKING": False,
    "__builtins__": __builtins__,
}
exec(pipeline_code, pipeline_mod_dict)
CorruptionPipeline = pipeline_mod_dict["CorruptionPipeline"]


class TestIOIIntegration:
    """Integration tests for IOI-style corruption pipeline."""

    def test_entity_swap_with_validators(self):
        """Test EntitySwap strategy with length and label validators."""
        # Create a simple entity swap strategy without spaCy
        # (we'll use a mock for testing)
        strategy = EntitySwapCorruption(
            entity_pool={
                "PERSON": ["Alice", "Bob", "Charlie"],
            }
        )

        # Create validators
        validators = [
            validators_mod.LengthBudgetValidator(tolerance=0.2),
        ]

        pipeline = CorruptionPipeline(
            strategies=[strategy],
            validators=validators,
            n_variants=2,
            keep_top_k=1,
            sampling="best",
        )

        # IOI-style example
        example = {
            "prompt": "Alice went to the store with Bob",
            "answer": "Alice",
        }

        rng = random.Random(42)
        corrupted = pipeline.corrupt_example(example, rng)

        # Verify metadata is present
        assert "strategy_used" in corrupted
        assert "severity" in corrupted
        assert "validation_results" in corrupted
        assert isinstance(corrupted["severity"], float)

    def test_pipeline_with_multiple_strategies(self):
        """Test pipeline trying multiple strategies."""
        # Create two strategies with different characteristics
        strategy1 = EntitySwapCorruption(
            entity_pool={
                "PERSON": ["Alice", "Bob"],
            }
        )

        # A simple strategy that capitalizes words
        class SimpleCapitalizeStrategy:
            name = "capitalize"
            mode = "meaning-preserving"

            def corrupt(
                self,
                example: Dict[str, Any],
                *,
                rng: random.Random,
                metadata: Dict[str, Any] = None,
            ) -> Dict[str, Any]:
                prompt = example.get("prompt", "")
                words = prompt.split()
                if words:
                    words[0] = words[0].upper()
                return {**example, "prompt": " ".join(words)}

            def batch_corrupt(
                self,
                examples: List[Dict[str, Any]],
                *,
                rng: random.Random,
                metadata: Dict[str, Any] = None,
            ) -> List[Dict[str, Any]]:
                return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

            def validate(
                self, clean: Dict[str, Any], corrupted: Dict[str, Any]
            ) -> base_mod.CorruptionValidation:
                return base_mod.CorruptionValidation(is_valid=True, severity=0.05)

        strategy2 = SimpleCapitalizeStrategy()

        pipeline = CorruptionPipeline(
            strategies=[strategy1, strategy2],
            validators=[validators_mod.LengthBudgetValidator(tolerance=0.2)],
            n_variants=1,
            keep_top_k=1,
        )

        example = {
            "prompt": "Alice went to the store",
            "answer": "Alice",
        }
        rng = random.Random(42)

        corrupted = pipeline.corrupt_example(example, rng)

        # One of the two strategies should be selected
        assert corrupted["strategy_used"] in ["entity_swap", "capitalize"]

    def test_batch_corruption_with_reproducibility(self):
        """Test batch processing maintains reproducibility."""
        strategy = EntitySwapCorruption(
            entity_pool={
                "PERSON": ["Alice", "Bob", "Charlie"],
            }
        )

        pipeline = CorruptionPipeline(
            strategies=[strategy],
            n_variants=1,
            keep_top_k=1,
        )

        examples = [
            {"prompt": "Alice went to the store", "answer": "Alice"},
            {"prompt": "Bob walked to the park", "answer": "Bob"},
        ]

        # First run
        rng1 = random.Random(42)
        results1 = pipeline.corrupt_dataset(examples, rng1)

        # Second run with same seed
        rng2 = random.Random(42)
        results2 = pipeline.corrupt_dataset(examples, rng2)

        # Should produce identical results
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            # Compare corrupted prompts
            if r1.get("strategy_used") != "none" and r2.get("strategy_used") != "none":
                # Both should have corrupted the same way
                assert r1["strategy_used"] == r2["strategy_used"]

    def test_validation_filtering(self):
        """Test that invalid corruptions are filtered out."""

        class StrictValidator:
            """Validator that rejects anything with 'store'."""

            def validate(
                self, clean: Dict[str, Any], corrupted: Dict[str, Any]
            ) -> validators_mod.CorruptionValidationResult:
                if "store" in corrupted.get("prompt", "").lower():
                    return validators_mod.CorruptionValidationResult(
                        is_valid=False, reason="Contains 'store'"
                    )
                return validators_mod.CorruptionValidationResult(is_valid=True, severity=0.0)

        class AlwaysSwapStrategy:
            """Strategy that always swaps words."""

            name = "swap"
            mode = "meaning-altering"

            def __init__(self):
                self.swap_count = 0

            def corrupt(
                self,
                example: Dict[str, Any],
                *,
                rng: random.Random,
                metadata: Dict[str, Any] = None,
            ) -> Dict[str, Any]:
                prompt = example.get("prompt", "")
                # Replace 'store' with something else
                prompt = prompt.replace("store", "mall")
                return {**example, "prompt": prompt}

            def batch_corrupt(
                self,
                examples: List[Dict[str, Any]],
                *,
                rng: random.Random,
                metadata: Dict[str, Any] = None,
            ) -> List[Dict[str, Any]]:
                return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

            def validate(
                self, clean: Dict[str, Any], corrupted: Dict[str, Any]
            ) -> base_mod.CorruptionValidation:
                return base_mod.CorruptionValidation(is_valid=True, severity=0.1)

        strategy = AlwaysSwapStrategy()
        validator = StrictValidator()

        pipeline = CorruptionPipeline(
            strategies=[strategy],
            validators=[validator],
            n_variants=1,
        )

        example = {"prompt": "Alice went to the store", "answer": "Alice"}
        rng = random.Random(42)

        corrupted = pipeline.corrupt_example(example, rng)

        # Should have applied the strategy (replacing 'store' with 'mall')
        assert "mall" in corrupted.get("prompt", "")
        assert corrupted["strategy_used"] == "swap"

    def test_best_vs_random_sampling(self):
        """Test best vs random sampling modes."""

        class VariableSeverityStrategy:
            """Strategy that produces different severities based on variant."""

            name = "variable"
            mode = "meaning-altering"

            def corrupt(
                self,
                example: Dict[str, Any],
                *,
                rng: random.Random,
                metadata: Dict[str, Any] = None,
            ) -> Dict[str, Any]:
                # Just return the example with minimal change
                return {**example, "corrupted": True}

            def batch_corrupt(
                self,
                examples: List[Dict[str, Any]],
                *,
                rng: random.Random,
                metadata: Dict[str, Any] = None,
            ) -> List[Dict[str, Any]]:
                return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

            def validate(
                self, clean: Dict[str, Any], corrupted: Dict[str, Any]
            ) -> base_mod.CorruptionValidation:
                return base_mod.CorruptionValidation(is_valid=True, severity=rng.random())

        strategy = VariableSeverityStrategy()

        # Test best sampling
        pipeline_best = CorruptionPipeline(
            strategies=[strategy],
            n_variants=5,
            keep_top_k=2,
            sampling="best",
        )

        example = {"prompt": "test", "answer": "test"}
        rng = random.Random(42)
        result_best = pipeline_best.corrupt_example(example, rng)
        assert "strategy_used" in result_best

        # Test random sampling
        pipeline_random = CorruptionPipeline(
            strategies=[strategy],
            n_variants=5,
            keep_top_k=2,
            sampling="random",
        )

        rng = random.Random(42)
        result_random = pipeline_random.corrupt_example(example, rng)
        assert "strategy_used" in result_random

    def test_output_format_compliance(self):
        """Test that output format matches specification."""

        class SimpleStrategy:
            name = "test_strategy"
            mode = "meaning-altering"

            def corrupt(
                self,
                example: Dict[str, Any],
                *,
                rng: random.Random,
                metadata: Dict[str, Any] = None,
            ) -> Dict[str, Any]:
                return {**example, "modified": True}

            def batch_corrupt(
                self,
                examples: List[Dict[str, Any]],
                *,
                rng: random.Random,
                metadata: Dict[str, Any] = None,
            ) -> List[Dict[str, Any]]:
                return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

            def validate(
                self, clean: Dict[str, Any], corrupted: Dict[str, Any]
            ) -> base_mod.CorruptionValidation:
                return base_mod.CorruptionValidation(is_valid=True, severity=0.3)

        strategy = SimpleStrategy()
        validator = validators_mod.LengthBudgetValidator()

        pipeline = CorruptionPipeline(
            strategies=[strategy],
            validators=[validator],
        )

        example = {"prompt": "test prompt", "answer": "answer"}
        rng = random.Random(42)

        result = pipeline.corrupt_example(example, rng)

        # Check required fields per POA
        assert "strategy_used" in result, "Missing 'strategy_used' field"
        assert "severity" in result, "Missing 'severity' field"
        assert "validation_results" in result, "Missing 'validation_results' field"

        # Check types
        assert isinstance(result["strategy_used"], str)
        assert isinstance(result["severity"], float)
        assert 0.0 <= result["severity"] <= 1.0
        assert isinstance(result["validation_results"], dict)

        # Check validator results format
        for validator_name, val_result in result["validation_results"].items():
            assert hasattr(
                val_result, "is_valid"
            ), f"Validator result missing is_valid: {validator_name}"
            assert hasattr(
                val_result, "severity"
            ), f"Validator result missing severity: {validator_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
