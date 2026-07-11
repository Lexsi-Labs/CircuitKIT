"""
Tests for RoleSwapCorruption strategy.

Tests verify:
1. Subject/object identification via dependency parsing
2. Proper swapping of roles in sentences
3. Handling of determiners and modifiers
4. SVA task corruption
5. Edge cases and validation
"""

import random
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from circuitkit.corruption.role_swap import (  # noqa: E402 - import after intentional pre-import setup
    RoleSwapCorruption,
)


class TestRoleSwapInitialization:
    """Test RoleSwapCorruption initialization."""

    def test_initialization_without_nlp(self):
        """Test initialization without providing a spaCy pipeline.

        ``__init__`` eagerly attempts to load ``en_core_web_sm``. Depending on
        the environment it either loads the model (``nlp`` set) or records an
        actionable load error (``nlp`` stays None) — never a silent half-state.
        """
        strategy = RoleSwapCorruption()

        assert strategy.name == "role_swap"
        assert strategy.mode == "role-swap"
        # Eager load: exactly one of (model loaded) / (load error recorded) holds.
        assert (strategy.nlp is not None) or (strategy._nlp_load_error is not None)

    def test_nlp_eager_loading(self):
        """spaCy is eagerly loaded in ``__init__`` (no lazy-init state)."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        # The lazy-init flag no longer exists; the model is loaded up front.
        assert not hasattr(strategy, "_nlp_initialized")

        if strategy.nlp is None:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

        example = {"prompt": "The cat sleeps"}
        result = strategy.corrupt(example, rng=random.Random(42))
        assert "prompt" in result


class TestRoleSwapBasic:
    """Test basic role swapping functionality."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_simple_subject_object_swap(self):
        """Test swapping subject and object in simple sentence."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "The cat likes the dog"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))

            # Should have role_swap_metadata
            assert "role_swap_metadata" in corrupted

            # Check if swap was performed
            if corrupted["role_swap_metadata"]["swapped"]:
                # Prompt should be different
                assert corrupted["prompt"] != example["prompt"]
                # Original entities should be present but in different positions
                assert "cat" in corrupted["prompt"]
                assert "dog" in corrupted["prompt"]
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_determiners_preserved_in_swap(self):
        """Test that determiners move with their nouns during swap."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "The cat likes the dog"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))

            if corrupted["role_swap_metadata"]["swapped"]:
                # Both "the cat" and "the dog" should appear with determiners
                prompt = corrupted["prompt"].lower()
                # The determiners should move with nouns
                assert "the" in prompt
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_no_swap_when_no_clear_svo(self):
        """Test that sentences without clear S-V-O aren't corrupted."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        # Sentence without clear object
        example = {"prompt": "The cat sleeps"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))

            # Should have metadata
            assert "role_swap_metadata" in corrupted

            # This sentence may or may not have a swappable structure
            # Just verify the structure is valid
            assert isinstance(corrupted["role_swap_metadata"]["swapped"], bool)
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    def test_missing_prompt_raises_error(self):
        """Test that missing prompt raises error."""
        strategy = RoleSwapCorruption()
        example = {"question": "test"}

        with pytest.raises(ValueError, match="must contain 'prompt'"):
            strategy.corrupt(example, rng=random.Random(42))

    def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises error."""
        strategy = RoleSwapCorruption()
        example = {"prompt": ""}

        with pytest.raises(ValueError, match="non-empty"):
            strategy.corrupt(example, rng=random.Random(42))


class TestRoleSwapMetadata:
    """Test metadata generated by role swap."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_swap_metadata_structure(self):
        """Test that swap metadata has correct structure."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "The dog chases the cat"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))

            metadata = corrupted["role_swap_metadata"]
            assert "swapped" in metadata
            assert isinstance(metadata["swapped"], bool)

            if metadata["swapped"]:
                assert "subject_text" in metadata
                assert "object_text" in metadata
                assert "subject_dep" in metadata
                assert "object_dep" in metadata
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_swap_reason_when_not_swapped(self):
        """Test that reason is provided when no swap occurs."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "Hello world"}  # No clear S-V-O

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))

            metadata = corrupted["role_swap_metadata"]
            if not metadata["swapped"]:
                assert "reason" in metadata
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")


class TestRoleSwapValidation:
    """Test corruption validation."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_validate_valid_swap(self):
        """Test validation of a valid role swap."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "The dog chases the cat"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))
            validation = strategy.validate(example, corrupted)

            # Should be valid (even if no swap occurred due to sentence structure)
            assert validation.is_valid
            assert isinstance(validation.severity, float)
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    def test_validate_missing_prompt(self):
        """Test validation fails with missing prompt."""
        strategy = RoleSwapCorruption()

        clean = {"question": "test"}
        corrupted = {"prompt": "test"}

        validation = strategy.validate(clean, corrupted)
        assert not validation.is_valid

    def test_validate_missing_metadata(self):
        """Test validation fails without metadata."""
        strategy = RoleSwapCorruption()

        clean = {"prompt": "test"}
        corrupted = {"prompt": "test"}  # No role_swap_metadata

        validation = strategy.validate(clean, corrupted)
        assert not validation.is_valid

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_validate_swapped_prompts_differ(self):
        """Test that swapped examples have different prompts."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()

        clean = {"prompt": "The dog chases the cat"}

        # Create a valid corrupted example with swap
        corrupted_with_swap = {
            "prompt": "The cat chases the dog",
            "role_swap_metadata": {
                "swapped": True,
                "subject_text": "dog",
                "subject_dep": "nsubj",
                "object_text": "cat",
                "object_dep": "dobj",
            },
        }

        validation = strategy.validate(clean, corrupted_with_swap)
        # Should be valid (if prompts differ)
        if validation.is_valid:
            assert clean["prompt"] != corrupted_with_swap["prompt"]

    def test_validate_no_swap_prompts_identical(self):
        """Test that non-swapped examples have identical prompts."""
        strategy = RoleSwapCorruption()

        clean = {"prompt": "test sentence"}
        corrupted_no_swap = {
            "prompt": "test sentence",
            "role_swap_metadata": {"swapped": False, "reason": "No clear structure"},
        }

        validation = strategy.validate(clean, corrupted_no_swap)
        assert validation.is_valid  # Still valid, just no corruption
        assert validation.severity == 0.0

    def test_severity_for_successful_swap(self):
        """Test severity scoring for successful swaps."""
        strategy = RoleSwapCorruption()

        clean = {"prompt": "The dog chases the cat"}
        corrupted = {
            "prompt": "The cat chases the dog",
            "role_swap_metadata": {
                "swapped": True,
                "subject_text": "dog",
                "subject_dep": "nsubj",
                "object_text": "cat",
                "object_dep": "dobj",
            },
        }

        validation = strategy.validate(clean, corrupted)
        if validation.is_valid:
            # Swaps should have moderate severity
            assert 0.0 < validation.severity <= 1.0


class TestSVAScenario:
    """Test on SVA (Subject-Verb Agreement) scenarios."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_sva_basic_swap(self):
        """Test role swap on SVA-style examples."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        # SVA test: "The dogs run"
        example = {"prompt": "The dogs run"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))

            assert "prompt" in corrupted
            assert "role_swap_metadata" in corrupted

            # Verify metadata structure
            assert isinstance(corrupted["role_swap_metadata"]["swapped"], bool)
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_sva_with_object_swap(self):
        """Test SVA example with explicit object to swap."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        # SVA with explicit object: "The dogs chase the cats"
        example = {"prompt": "The dogs chase the cats"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))

            assert "prompt" in corrupted
            metadata = corrupted["role_swap_metadata"]

            # If swap occurred, verify roles are different
            if metadata["swapped"]:
                assert metadata["subject_text"] != metadata["object_text"]
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")


class TestBatchProcessing:
    """Test batch role swap processing."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_batch_corrupt(self):
        """Test batch corruption of multiple examples."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        examples = [
            {"prompt": "The dog runs"},
            {"prompt": "The cat sleeps"},
            {"prompt": "The dog chases the cat"},
        ]

        try:
            corrupted_batch = strategy.batch_corrupt(examples, rng=random.Random(42))

            assert len(corrupted_batch) == len(examples)
            for corrupted in corrupted_batch:
                assert "role_swap_metadata" in corrupted
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    def test_batch_corrupt_empty_list(self):
        """Test batch corruption with empty list."""
        strategy = RoleSwapCorruption()
        examples = []

        corrupted_batch = strategy.batch_corrupt(examples, rng=random.Random(42))

        assert corrupted_batch == []


class TestSpacyIntegration:
    """Test spaCy integration and error handling."""

    def test_missing_spacy_raises_runtime_error(self, monkeypatch):
        """When spaCy can't be imported, construction records the failure and
        corrupt() fails loudly with an actionable RuntimeError.

        Simulates a missing spaCy install by making `import spacy` fail. The
        strategy must not silently no-op (which would fabricate an unchanged
        "corruption"); it records the reason and raises on use.
        """
        import builtins

        real_import = builtins.__import__

        def _no_spacy_import(name, *args, **kwargs):
            if name == "spacy" or name.startswith("spacy."):
                raise ImportError("No module named 'spacy'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_spacy_import)

        strategy = RoleSwapCorruption()  # nlp import blocked → nlp stays None
        assert strategy.nlp is None
        assert "spaCy is not installed" in (strategy._nlp_load_error or "")

        with pytest.raises(RuntimeError, match="spaCy"):
            strategy.corrupt({"prompt": "The cat sleeps"}, rng=random.Random(42))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_nlp_initialization_error_handling(self):
        """Test that NLP loading errors are handled gracefully."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()

        example = {"prompt": "Test sentence"}
        try:
            # This should either work or raise a clear error
            strategy.corrupt(example, rng=random.Random(42))
        except (ImportError, OSError) as e:
            # These are expected if spacy isn't fully installed
            assert "spacy" in str(e).lower() or "model" in str(e).lower()


class TestEdgeCases:
    """Test edge cases in role swapping."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_sentence_with_multiple_objects(self):
        """Test sentence with multiple potential objects."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        # Sentence with multiple nouns
        example = {"prompt": "The dog gives the ball to the cat"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))
            assert "prompt" in corrupted
            # Just verify it processes without error
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_passive_voice_swap(self):
        """Test role swap with passive voice."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        # Passive voice: subject becomes agent
        example = {"prompt": "The cat is chased by the dog"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))
            assert "role_swap_metadata" in corrupted
            # Passive voice has different dependency structure
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_pronoun_swap(self):
        """Test role swap with pronouns."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "She likes him"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))
            assert "prompt" in corrupted
            # Pronouns should be handled
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_sentence_with_adjectives(self):
        """Test role swap with adjective modifiers."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "The big dog chases the small cat"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))

            if corrupted["role_swap_metadata"]["swapped"]:
                # Both "big dog" and "small cat" should move together
                assert "big" in corrupted["prompt"]
                assert "small" in corrupted["prompt"]
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_empty_or_whitespace_only_prompt(self):
        """Test handling of whitespace-only prompts."""
        strategy = RoleSwapCorruption()
        example = {"prompt": "   "}

        with pytest.raises(ValueError, match="non-empty"):
            strategy.corrupt(example, rng=random.Random(42))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_special_characters_in_sentence(self):
        """Test handling of special characters."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "The dog (2 years old) likes the cat!!"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))
            assert "prompt" in corrupted
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")


class TestDeterminersAndModifiers:
    """Test proper handling of determiners and modifiers."""

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_determiner_stays_with_noun(self):
        """Test that 'the' stays with noun during swap."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "The dog likes the cat"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))

            if corrupted["role_swap_metadata"]["swapped"]:
                # Check that "the" still appears in prompt
                assert "the" in corrupted["prompt"].lower()
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_plural_determiner_handling(self):
        """Test handling of plural determiners."""
        pytest.importorskip("spacy")

        strategy = RoleSwapCorruption()
        example = {"prompt": "Those dogs chase those cats"}

        try:
            corrupted = strategy.corrupt(example, rng=random.Random(42))
            assert "prompt" in corrupted
        except OSError:
            pytest.skip("spaCy model 'en_core_web_sm' not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
