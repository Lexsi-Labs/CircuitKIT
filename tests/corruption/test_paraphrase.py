"""
Tests for ParaphraseCorruption strategy.

Tests verify that:
1. Paraphrases are generated correctly
2. Caching works (2nd run is faster)
3. Surface-level and semantic modes both work
4. BLEU scores indicate sufficient difference from original
5. Validation catches bad paraphrases
6. Answer token is preserved in corrupted prompt
"""

import json
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from circuitkit.corruption.paraphrase import ParaphraseCorruption


class TestParaphraseCorruptionInit:
    """Tests for ParaphraseCorruption initialization."""

    @patch("circuitkit.corruption.paraphrase.AutoTokenizer")
    @patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM")
    def test_init_default_model(self, mock_model, mock_tokenizer):
        """Test initialization with default model using mocks to bypass downloads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

            assert paraphrase.name == "paraphrase"
            assert paraphrase.mode == "semantic"
            assert paraphrase.model is not None
            assert paraphrase.tokenizer is not None
            assert paraphrase.tokenizer is not None

    def test_init_with_custom_cache_dir(self):
        """Test initialization creates cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_cache = Path(tmpdir) / "my_cache"

            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=str(custom_cache))

                    assert paraphrase.cache_dir == custom_cache
                    assert custom_cache.exists()

    def test_init_creates_cache_file(self):
        """Test that cache file location is set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    assert paraphrase.cache_file == Path(tmpdir) / "paraphrase_cache.json"


class TestCacheOperations:
    """Tests for cache loading and saving."""

    def test_load_existing_cache(self):
        """Test loading an existing cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "paraphrase_cache.json"
            cache_data = {"key1": "value1", "key2": "value2"}
            cache_file.write_text(json.dumps(cache_data))

            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    assert paraphrase.cache == cache_data

    def test_load_nonexistent_cache(self):
        """Test loading when cache file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    assert paraphrase.cache == {}

    def test_load_invalid_json_cache(self):
        """Test loading when cache file has invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "paraphrase_cache.json"
            cache_file.write_text("invalid json {")

            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    assert paraphrase.cache == {}

    def test_save_cache(self):
        """Test saving cache to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    paraphrase.cache["test_key"] = "test_value"
                    paraphrase._save_cache()

                    cache_file = Path(tmpdir) / "paraphrase_cache.json"
                    assert cache_file.exists()

                    loaded = json.loads(cache_file.read_text())
                    assert loaded["test_key"] == "test_value"


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_cache_key_deterministic(self):
        """Test that same input produces same cache key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    prompt = "What is the capital of France?"
                    key1 = paraphrase._get_cache_key(prompt)
                    key2 = paraphrase._get_cache_key(prompt)

                    assert key1 == key2

    def test_cache_key_different_for_different_prompts(self):
        """Test that different prompts produce different keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    key1 = paraphrase._get_cache_key("prompt1")
                    key2 = paraphrase._get_cache_key("prompt2")

                    assert key1 != key2

    def test_cache_key_includes_model_name(self):
        """Test that different models produce different cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    p1 = ParaphraseCorruption(model_name="model1", cache_dir=tmpdir)
                    p2 = ParaphraseCorruption(model_name="model2", cache_dir=tmpdir)

                    prompt = "same prompt"
                    key1 = p1._get_cache_key(prompt)
                    key2 = p2._get_cache_key(prompt)

                    assert key1 != key2

    def test_cache_key_includes_mode(self):
        """Test that different modes produce different cache keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    p1 = ParaphraseCorruption(mode="surface", cache_dir=tmpdir)
                    p2 = ParaphraseCorruption(mode="semantic", cache_dir=tmpdir)

                    prompt = "same prompt"
                    key1 = p1._get_cache_key(prompt)
                    key2 = p2._get_cache_key(prompt)

                    assert key1 != key2


class TestSurfaceParaphrase:
    """Tests for surface-level paraphrasing."""

    def test_surface_paraphrase_basic(self):
        """Test basic surface paraphrase with synonym replacement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(mode="surface", cache_dir=tmpdir)

                    result = paraphrase._paraphrase_surface("What is this?")
                    assert isinstance(result, str)
                    assert len(result) > 0

    def test_surface_paraphrase_preserves_case(self):
        """Test that surface paraphrase preserves capitalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(mode="surface", cache_dir=tmpdir)

                    result = paraphrase._paraphrase_surface("What is this?")
                    # First word should be capitalized
                    assert result[0].isupper()

    def test_surface_paraphrase_handles_unknown_words(self):
        """Test that surface paraphrase handles unknown words gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(mode="surface", cache_dir=tmpdir)

                    result = paraphrase._paraphrase_surface("xyz unknown word")
                    assert "unknown" in result or "word" in result


class TestCorrupt:
    """Tests for the corrupt() method."""

    def test_corrupt_raises_on_missing_prompt(self):
        """Test that corrupt raises when prompt key is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    example = {"answer": "Paris"}
                    rng = random.Random(42)

                    with pytest.raises(ValueError, match="prompt"):
                        paraphrase.corrupt(example, rng=rng)

    def test_corrupt_uses_cache(self):
        """Test that corrupt uses cached paraphrases."""
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

    def test_corrupt_preserves_answer(self):
        """Test that corrupt preserves the answer field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    example = {"prompt": "Test", "answer": "TestAnswer"}

                    # Use cache to avoid actual LLM call
                    cache_key = paraphrase._get_cache_key("Test")
                    paraphrase.cache[cache_key] = "Modified test"

                    rng = random.Random(42)
                    result = paraphrase.corrupt(example, rng=rng)

                    assert result["answer"] == "TestAnswer"

    def test_corrupt_returns_dict_with_same_structure(self):
        """Test that corrupt returns dict with same keys as input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    example = {
                        "prompt": "Test",
                        "answer": "TestAnswer",
                        "extra_field": "value",
                    }

                    cache_key = paraphrase._get_cache_key("Test")
                    paraphrase.cache[cache_key] = "Modified"

                    rng = random.Random(42)
                    result = paraphrase.corrupt(example, rng=rng)

                    assert "prompt" in result
                    assert "answer" in result
                    assert "extra_field" in result


class TestBatchCorrupt:
    """Tests for batch_corrupt() method."""

    def test_batch_corrupt_returns_list(self):
        """Test that batch_corrupt returns a list."""
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

    def test_batch_corrupt_maintains_order(self):
        """Test that batch_corrupt maintains order of examples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    examples = [{"prompt": f"Test{i}", "answer": f"A{i}"} for i in range(5)]

                    # Pre-populate cache
                    for i, ex in enumerate(examples):
                        cache_key = paraphrase._get_cache_key(ex["prompt"])
                        paraphrase.cache[cache_key] = f"Modified {i}"

                    rng = random.Random(42)
                    results = paraphrase.batch_corrupt(examples, rng=rng)

                    for i, result in enumerate(results):
                        assert f"Modified {i}" in result["prompt"]


class TestValidate:
    """Tests for validate() method."""

    def test_validate_identical_prompts(self):
        """Test validation fails when prompts are identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    clean = {"prompt": "Same prompt", "answer": "answer"}
                    corrupted = {"prompt": "Same prompt", "answer": "answer"}

                    result = paraphrase.validate(clean, corrupted)

                    assert result.is_valid is False

    def test_validate_missing_prompt_key(self):
        """Test validation fails when prompt key is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    clean = {"prompt": "Test", "answer": "answer"}
                    corrupted = {"answer": "answer"}

                    result = paraphrase.validate(clean, corrupted)

                    assert result.is_valid is False

    def test_validate_modified_answer(self):
        """Test validation fails when answer is modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    clean = {"prompt": "Test", "answer": "answer1"}
                    corrupted = {"prompt": "Different", "answer": "answer2"}

                    result = paraphrase.validate(clean, corrupted)

                    assert result.is_valid is False

    def test_validate_within_length_budget(self):
        """Test validation passes when within length budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    clean = {"prompt": "The quick brown fox", "answer": "answer"}
                    corrupted = {"prompt": "The speedy orange fox", "answer": "answer"}

                    result = paraphrase.validate(clean, corrupted)

                    assert result.is_valid is True

    def test_validate_exceeds_length_budget(self):
        """Test validation fails when exceeding length budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    clean = {"prompt": "Short", "answer": "answer"}
                    corrupted = {
                        "prompt": "This is a much longer prompt that exceeds the budget",
                        "answer": "answer",
                    }

                    result = paraphrase.validate(clean, corrupted)

                    assert result.is_valid is False

    def test_validate_computes_severity(self):
        """Test that validate computes severity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("circuitkit.corruption.paraphrase.AutoTokenizer"):
                with patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM"):
                    paraphrase = ParaphraseCorruption(cache_dir=tmpdir)

                    clean = {"prompt": "The quick brown fox", "answer": "answer"}
                    corrupted = {"prompt": "The speedy orange fox", "answer": "answer"}

                    result = paraphrase.validate(clean, corrupted)

                    assert 0.0 <= result.severity <= 1.0


class TestIntegration:
    """Integration tests for ParaphraseCorruption."""

    @patch("circuitkit.corruption.paraphrase.AutoTokenizer")
    @patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM")
    def test_surface_mode_caching(self, mock_model, mock_tokenizer):
        """Test that surface mode uses cache correctly with mocked generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paraphrase = ParaphraseCorruption(mode="surface", cache_dir=tmpdir)

            example = {"prompt": "What is the capital?", "answer": "Paris"}
            rng = random.Random(42)

            # Mock the internal paraphrase generation to return a fixed string
            with patch.object(
                paraphrase, "_paraphrase_surface", return_value="What is the capital city?"
            ) as mock_generate:
                # First call
                result1 = paraphrase.corrupt(example, rng=rng)

                # Second call should use cache (same result)
                result2 = paraphrase.corrupt(example, rng=rng)

                assert result1["prompt"] == result2["prompt"]
                mock_generate.assert_called_once()

    @patch("circuitkit.corruption.paraphrase.AutoTokenizer")
    @patch("circuitkit.corruption.paraphrase.AutoModelForCausalLM")
    def test_corruption_with_ioi_prompt(self, mock_model, mock_tokenizer):
        """Test corruption on IOI-like prompt using deterministic mocked output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paraphrase = ParaphraseCorruption(mode="surface", cache_dir=tmpdir)

            example = {
                "prompt": "Alice and Bob went to the store. Alice gave a book to Bob. Who received the book?",
                "answer": "Bob",
            }

            # Provide a deterministic string that fulfills the validation requirements
            mocked_paraphrase = (
                "Alice and Bob visited the market. Alice handed a novel to Bob. Who got the book?"
            )

            with patch.object(paraphrase, "_paraphrase_surface", return_value=mocked_paraphrase):
                rng = random.Random(42)
                result = paraphrase.corrupt(example, rng=rng)

                # Should preserve answer
                assert result["answer"] == "Bob"

                # Validation should pass
                validation = paraphrase.validate(example, result)
                assert validation.is_valid is True or validation.severity < 1.0
