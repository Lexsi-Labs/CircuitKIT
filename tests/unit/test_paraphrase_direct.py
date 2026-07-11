"""
Direct test for ParaphraseCorruption (bypass package init).
Run with: python test_paraphrase_direct.py
"""

import json
import os
import random
import sys
import tempfile
from unittest.mock import patch

from circuitkit.corruption.paraphrase import ParaphraseCorruption


def test_cache_key_generation():
    """Test cache key generation."""
    print("Testing cache key generation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
            with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                prompt = "What is the capital of France?"
                key1 = paraphrase._get_cache_key(prompt)
                key2 = paraphrase._get_cache_key(prompt)

                assert key1 == key2
                print("  [OK] Same prompt produces same key")

                key3 = paraphrase._get_cache_key("Different prompt")
                assert key1 != key3
                print("  [OK] Different prompts produce different keys")


def test_cache_operations():
    """Test cache loading and saving."""
    print("\nTesting cache operations...")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "paraphrase_cache.json")

        # Pre-create cache
        cache_data = {"key1": "value1", "key2": "value2"}
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
            with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                assert paraphrase.cache == cache_data
                print("  [OK] Cache loading works")

                # Save new cache entry
                paraphrase.cache["new_key"] = "new_value"
                paraphrase._save_cache()

                # Verify it was saved
                with open(cache_file, "r") as f:
                    saved = json.load(f)

                assert saved["new_key"] == "new_value"
                print("  [OK] Cache saving works")


def test_surface_paraphrase():
    """Test surface paraphrase."""
    print("\nTesting surface-level paraphrase...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
            with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                paraphrase = ParaphraseCorruption(mode="surface", cache_dir=tmpdir)

                result = paraphrase._paraphrase_surface("What is this?")
                assert isinstance(result, str)
                assert len(result) > 0
                print("  [OK] Surface paraphrase generates output")


def test_corrupt_with_cache():
    """Test corrupt method uses cache."""
    print("\nTesting corrupt method...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
            with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                prompt = "Test prompt"
                example = {"prompt": prompt, "answer": "answer"}

                # Pre-populate cache
                cache_key = paraphrase._get_cache_key(prompt)
                paraphrase.cache[cache_key] = "Cached paraphrase"

                rng = random.Random(42)
                result = paraphrase.corrupt(example, rng=rng)

                assert result["prompt"] == "Cached paraphrase"
                assert result["answer"] == "answer"
                print("  [OK] Corrupt uses cache correctly")


def test_corrupt_missing_prompt():
    """Test corrupt raises error on missing prompt."""
    print("\nTesting error handling...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
            with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                example = {"answer": "answer"}
                rng = random.Random(42)

                try:
                    paraphrase.corrupt(example, rng=rng)
                    assert False, "Should have raised ValueError"
                except ValueError as e:
                    assert "prompt" in str(e).lower()
                    print("  [OK] Raises error on missing prompt")


def test_batch_corrupt():
    """Test batch_corrupt method."""
    print("\nTesting batch_corrupt...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
            with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                examples = [
                    {"prompt": "Test1", "answer": "A1"},
                    {"prompt": "Test2", "answer": "A2"},
                ]

                # Pre-populate cache
                for ex in examples:
                    cache_key = paraphrase._get_cache_key(ex["prompt"])
                    paraphrase.cache[cache_key] = f"Modified {ex['prompt']}"

                rng = random.Random(42)
                results = paraphrase.batch_corrupt(examples, rng=rng)

                assert isinstance(results, list)
                assert len(results) == 2
                assert results[0]["prompt"] == "Modified Test1"
                assert results[1]["prompt"] == "Modified Test2"
                print("  [OK] Batch corrupt works and maintains order")


def test_validate():
    """Test validate method."""
    print("\nTesting validate method...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
            with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                # Identical prompts should fail
                clean = {"prompt": "Same", "answer": "ans"}
                corrupted = {"prompt": "Same", "answer": "ans"}
                result = paraphrase.validate(clean, corrupted)
                assert result.is_valid is False
                print("  [OK] Rejects identical prompts")

                # Different prompts should pass
                clean = {"prompt": "The quick brown fox", "answer": "ans"}
                corrupted = {"prompt": "The speedy orange fox", "answer": "ans"}
                result = paraphrase.validate(clean, corrupted)
                assert result.is_valid is True
                print("  [OK] Accepts different prompts")

                # Modified answer should fail
                clean = {"prompt": "Test", "answer": "ans1"}
                corrupted = {"prompt": "Modified", "answer": "ans2"}
                result = paraphrase.validate(clean, corrupted)
                assert result.is_valid is False
                print("  [OK] Rejects modified answers")


def test_protocol_compliance():
    """Test that ParaphraseCorruption has required protocol attributes."""
    print("\nTesting protocol compliance...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
            with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                # Check required attributes
                assert hasattr(paraphrase, "name")
                assert paraphrase.name == "paraphrase"
                print("  [OK] Has 'name' attribute")

                # `mode` selects the paraphrasing technique; it defaults to
                # "semantic" (see ParaphraseCorruption.__init__).
                assert hasattr(paraphrase, "mode")
                assert paraphrase.mode == "semantic"
                print("  [OK] Has 'mode' attribute")

                assert hasattr(paraphrase, "corrupt")
                print("  [OK] Has 'corrupt' method")

                assert hasattr(paraphrase, "batch_corrupt")
                print("  [OK] Has 'batch_corrupt' method")

                assert hasattr(paraphrase, "validate")
                print("  [OK] Has 'validate' method")


if __name__ == "__main__":
    try:
        test_cache_key_generation()
        test_cache_operations()
        test_surface_paraphrase()
        test_corrupt_with_cache()
        test_corrupt_missing_prompt()
        test_batch_corrupt()
        test_validate()
        test_protocol_compliance()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
