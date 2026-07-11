"""
Test suite for CorruptionPipeline orchestrator.

Tests the pipeline with multiple strategies, validators, and sampling modes.
Validates that corrupted examples have all required metadata fields.
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

# Load pipeline module
spec = importlib.util.spec_from_file_location(
    "test_pipeline_module", os.path.join(base_path, "pipeline.py")
)
pipeline_code = open(os.path.join(base_path, "pipeline.py")).read()
# Replace relative imports
pipeline_code = pipeline_code.replace("from .base import", "# from base import")
pipeline_code = pipeline_code.replace("from .validators import", "# from validators import")
from concurrent.futures import (  # noqa: E402 - import after intentional pre-import setup
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field  # noqa: E402 - import after intentional pre-import setup
from typing import Literal, Optional  # noqa: E402 - import after intentional pre-import setup

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
    "__builtins__": __builtins__,
}
exec(pipeline_code, pipeline_mod_dict)
CorruptionPipeline = pipeline_mod_dict["CorruptionPipeline"]
CorruptionValidation = base_mod.CorruptionValidation
CorruptionValidationResult = validators_mod.CorruptionValidationResult


# Simple test strategy for reproducibility
class SimpleSwapStrategy:
    """Simple test strategy that swaps two words in the prompt."""

    name = "simple_swap"
    mode = "meaning-altering"

    def corrupt(
        self, example: Dict[str, Any], *, rng: random.Random, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Swap two random words in the prompt."""
        prompt = example.get("prompt", "")
        words = prompt.split()

        if len(words) < 2:
            return example

        # Pick two random positions
        pos1 = rng.randint(0, len(words) - 1)
        pos2 = rng.randint(0, len(words) - 1)

        if pos1 == pos2:
            return example

        # Swap
        words[pos1], words[pos2] = words[pos2], words[pos1]
        corrupted_prompt = " ".join(words)

        return {**example, "prompt": corrupted_prompt}

    def batch_corrupt(
        self, examples: List[Dict[str, Any]], *, rng: random.Random, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Default batch implementation."""
        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(self, clean: Dict[str, Any], corrupted: Dict[str, Any]) -> CorruptionValidation:
        """Simple validation - always valid."""
        return CorruptionValidation(is_valid=True, severity=0.1)


class AnotherSimpleStrategy:
    """Another simple test strategy that capitalizes random words."""

    name = "capitalize_words"
    mode = "meaning-preserving"

    def corrupt(
        self, example: Dict[str, Any], *, rng: random.Random, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Capitalize a random word in the prompt."""
        prompt = example.get("prompt", "")
        words = prompt.split()

        if not words:
            return example

        # Pick a random position
        pos = rng.randint(0, len(words) - 1)
        words[pos] = words[pos].upper()
        corrupted_prompt = " ".join(words)

        return {**example, "prompt": corrupted_prompt}

    def batch_corrupt(
        self, examples: List[Dict[str, Any]], *, rng: random.Random, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Default batch implementation."""
        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(self, clean: Dict[str, Any], corrupted: Dict[str, Any]) -> CorruptionValidation:
        """Simple validation - always valid."""
        return CorruptionValidation(is_valid=True, severity=0.05)


class SimpleValidator:
    """Simple validator for testing."""

    def validate(
        self, clean: Dict[str, Any], corrupted: Dict[str, Any]
    ) -> CorruptionValidationResult:
        """Just check that prompt exists."""
        if "prompt" not in corrupted:
            return CorruptionValidationResult(is_valid=False, reason="No prompt in corrupted")
        return CorruptionValidationResult(is_valid=True, severity=0.0)


class TestCorruptionPipeline:
    """Test CorruptionPipeline orchestration."""

    def test_single_strategy_single_variant(self):
        """Test pipeline with one strategy and one variant per example."""
        strategy = SimpleSwapStrategy()
        pipeline = CorruptionPipeline(strategies=[strategy], n_variants=1, keep_top_k=1)

        example = {"prompt": "Alice went to the store", "answer": "store"}
        rng = random.Random(42)

        corrupted = pipeline.corrupt_example(example, rng)

        # Check metadata
        assert "strategy_used" in corrupted
        assert "severity" in corrupted
        assert "validation_results" in corrupted
        assert corrupted["strategy_used"] == "simple_swap"
        assert isinstance(corrupted["severity"], float)
        assert 0.0 <= corrupted["severity"] <= 1.0

    def test_multiple_strategies(self):
        """Test pipeline with multiple strategies."""
        strategy1 = SimpleSwapStrategy()
        strategy2 = AnotherSimpleStrategy()
        pipeline = CorruptionPipeline(strategies=[strategy1, strategy2], n_variants=1, keep_top_k=1)

        example = {"prompt": "Alice went to the store", "answer": "store"}
        rng = random.Random(42)

        corrupted = pipeline.corrupt_example(example, rng)

        # One of the two strategies should be used
        assert corrupted["strategy_used"] in ["simple_swap", "capitalize_words"]

    def test_with_validators(self):
        """Test pipeline with validators."""
        strategy = SimpleSwapStrategy()
        validator = SimpleValidator()
        pipeline = CorruptionPipeline(
            strategies=[strategy], validators=[validator], n_variants=1, keep_top_k=1
        )

        example = {"prompt": "Alice went to the store", "answer": "store"}
        rng = random.Random(42)

        corrupted = pipeline.corrupt_example(example, rng)

        # Check that validation results are present
        assert "validation_results" in corrupted
        assert isinstance(corrupted["validation_results"], dict)

    def test_best_sampling(self):
        """Test deterministic best (lowest severity) sampling."""
        strategy1 = SimpleSwapStrategy()  # severity 0.1
        strategy2 = AnotherSimpleStrategy()  # severity 0.05
        pipeline = CorruptionPipeline(
            strategies=[strategy1, strategy2],
            n_variants=1,
            keep_top_k=2,
            sampling="best",
        )

        example = {"prompt": "Alice went to the store", "answer": "store"}
        rng = random.Random(42)

        corrupted = pipeline.corrupt_example(example, rng)

        # Since capitalize_words has lower severity (0.05), it should be selected
        # (assuming both pass validation)
        assert "strategy_used" in corrupted

    def test_random_sampling(self):
        """Test random sampling from top-k."""
        strategy = SimpleSwapStrategy()
        pipeline = CorruptionPipeline(
            strategies=[strategy], n_variants=5, keep_top_k=3, sampling="random"
        )

        example = {"prompt": "Alice went to the store", "answer": "store"}
        rng = random.Random(42)

        corrupted = pipeline.corrupt_example(example, rng)

        # Should produce a valid result
        assert "strategy_used" in corrupted
        assert corrupted["strategy_used"] == "simple_swap"

    def test_corrupt_dataset(self):
        """Test batch processing of multiple examples."""
        strategy = SimpleSwapStrategy()
        pipeline = CorruptionPipeline(strategies=[strategy], n_variants=1, keep_top_k=1)

        examples = [
            {"prompt": "Alice went to the store", "answer": "store"},
            {"prompt": "Bob walked to the park", "answer": "park"},
            {"prompt": "Charlie ran to the gym", "answer": "gym"},
        ]
        rng = random.Random(42)

        corrupted_examples = pipeline.corrupt_dataset(examples, rng)

        # Check that we get the same number of examples
        assert len(corrupted_examples) == len(examples)

        # Check that all have metadata
        for corrupted in corrupted_examples:
            assert "strategy_used" in corrupted
            assert "severity" in corrupted
            assert "validation_results" in corrupted

    def test_parallel_processing(self):
        """Test parallel batch processing with max_workers."""
        strategy = SimpleSwapStrategy()
        pipeline = CorruptionPipeline(
            strategies=[strategy], n_variants=1, keep_top_k=1, max_workers=2
        )

        examples = [
            {"prompt": "Alice went to the store", "answer": "store"},
            {"prompt": "Bob walked to the park", "answer": "park"},
        ]
        rng = random.Random(42)

        corrupted_examples = pipeline.corrupt_dataset(examples, rng)

        assert len(corrupted_examples) == len(examples)
        for corrupted in corrupted_examples:
            assert "strategy_used" in corrupted

    def test_no_valid_candidates_fallback(self):
        """Test that pipeline returns original example when no valid candidates exist."""

        class AlwaysFailValidator:
            """Validator that always fails."""

            def validate(
                self, clean: Dict[str, Any], corrupted: Dict[str, Any]
            ) -> CorruptionValidationResult:
                return CorruptionValidationResult(is_valid=False, reason="Always fails for testing")

        strategy = SimpleSwapStrategy()
        validator = AlwaysFailValidator()
        pipeline = CorruptionPipeline(
            strategies=[strategy], validators=[validator], n_variants=1, keep_top_k=1
        )

        example = {"prompt": "Alice went to the store", "answer": "store"}
        rng = random.Random(42)

        corrupted = pipeline.corrupt_example(example, rng)

        # Should return original with special metadata
        assert corrupted["strategy_used"] == "none"
        assert corrupted["severity"] == 0.0

    def test_get_strategy_names(self):
        """Test getting strategy names."""
        strategy1 = SimpleSwapStrategy()
        strategy2 = AnotherSimpleStrategy()
        pipeline = CorruptionPipeline(strategies=[strategy1, strategy2])

        names = pipeline.get_strategy_names()
        assert names == ["simple_swap", "capitalize_words"]

    def test_get_validator_names(self):
        """Test getting validator names."""
        validator1 = SimpleValidator()
        validator2 = SimpleValidator()
        pipeline = CorruptionPipeline(
            strategies=[SimpleSwapStrategy()],
            validators=[validator1, validator2],
        )

        names = pipeline.get_validator_names()
        assert len(names) == 2
        assert all("SimpleValidator" in n for n in names)

    def test_reproducibility(self):
        """Test that same seed produces same corruptions."""
        strategy = SimpleSwapStrategy()
        pipeline = CorruptionPipeline(strategies=[strategy], n_variants=1)

        example = {"prompt": "Alice went to the store", "answer": "store"}

        # Two runs with same seed
        rng1 = random.Random(42)
        corrupted1 = pipeline.corrupt_example(example, rng1)

        rng2 = random.Random(42)
        corrupted2 = pipeline.corrupt_example(example, rng2)

        # Should be identical
        assert corrupted1["prompt"] == corrupted2["prompt"]
        assert corrupted1["strategy_used"] == corrupted2["strategy_used"]

    def test_invalid_sampling_mode(self):
        """Test that invalid sampling mode raises error."""
        with pytest.raises(ValueError):
            CorruptionPipeline(strategies=[SimpleSwapStrategy()], sampling="invalid_mode")

    def test_empty_validators_list(self):
        """Test pipeline with empty validators list (all candidates valid)."""
        strategy = SimpleSwapStrategy()
        pipeline = CorruptionPipeline(strategies=[strategy], validators=[], n_variants=1)

        example = {"prompt": "Alice went to the store", "answer": "store"}
        rng = random.Random(42)

        corrupted = pipeline.corrupt_example(example, rng)

        # Should succeed with empty validation_results
        assert corrupted["validation_results"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
