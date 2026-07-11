"""
RoleSwapCorruption Strategy

Swaps subject and object roles in sentences using dependency parsing for SVA,
gender-bias, and similar tasks that rely on entity role relationships.

Uses spaCy dependency parser to identify SVO (Subject-Verb-Object) structures
and swaps subject/object positions while preserving grammatical markers.

Corruption Mode: role-swap (role-preserving structural manipulation)
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from .base import CorruptionValidation


class RoleSwapCorruption:
    """Swaps subject and object roles in sentences using dependency parsing.

    This strategy is designed for tasks where:
    - Subject-verb agreement (SVA) must be evaluated
    - Gender bias and pronoun agreement matter
    - Role relationships determine correctness

    The strategy uses spaCy's dependency parser to:
    1. Identify the subject (nsubj, nsubjpass)
    2. Identify the object (dobj, iobj, attr, etc.)
    3. Swap their positions in the sentence
    4. Preserve grammatical structure where possible

    Example:
        Input: "The cat likes the dog"
        Output: "The dog likes the cat"
    """

    name = "role_swap"
    mode = "role-swap"

    def __init__(self, nlp=None):
        """Initialize RoleSwapCorruption with optional spaCy pipeline.

        Args:
            nlp: Optional spaCy Language model. If None, attempts to load
                en_core_web_sm. If loading fails, the failure reason is
                recorded and surfaced as a clear, actionable error on first
                use (or via set_nlp() to recover).
        """
        self.nlp = nlp
        # Records why spaCy could not be loaded so `_require_nlp()` can give an
        # actionable message instead of a cryptic NoneType error later.
        self._nlp_load_error: Optional[str] = None

        if self.nlp is None:
            try:
                import spacy
            except ImportError:
                self._nlp_load_error = (
                    "spaCy is not installed. Install it with:\n"
                    "    pip install spacy\n"
                    "    python -m spacy download en_core_web_sm"
                )
            except Exception as e:
                # `import spacy` can raise beyond ImportError — e.g. a ValueError
                # from a numpy/thinc ABI mismatch in spaCy's Cython extensions.
                self._nlp_load_error = (
                    f"spaCy failed to import: {e}\n"
                    "This usually indicates a binary incompatibility "
                    "(e.g. numpy ABI mismatch with thinc). Try reinstalling "
                    "spacy and numpy in a clean environment, or pass a "
                    "loaded spaCy model via the `nlp` argument / set_nlp()."
                )
            else:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self._nlp_load_error = (
                        "spaCy is installed but the 'en_core_web_sm' model is "
                        "missing. Download it with:\n"
                        "    python -m spacy download en_core_web_sm\n"
                        "or pass a loaded spaCy model via the `nlp` argument / set_nlp()."
                    )
                except Exception as e:
                    # Covers binary-incompatibility failures (e.g. thinc's
                    # Cython extensions built against a different numpy ABI),
                    # which surface as ValueError rather than ImportError/OSError.
                    self._nlp_load_error = (
                        f"spaCy failed to load 'en_core_web_sm': {e}\n"
                        "This usually indicates a binary incompatibility "
                        "(e.g. numpy ABI mismatch with thinc). Try reinstalling "
                        "spacy and numpy in a clean environment, or pass a "
                        "loaded spaCy model via the `nlp` argument / set_nlp()."
                    )

    def set_nlp(self, nlp):
        """Set the spaCy NLP model after initialization."""
        self.nlp = nlp
        self._nlp_load_error = None

    def _require_nlp(self):
        """Return the spaCy model, or fail fast with an actionable message.

        Raises:
            RuntimeError: If no spaCy model is available, including the specific
                reason (spaCy not installed vs. model missing vs. load error).
        """
        if self.nlp is None:
            detail = self._nlp_load_error or (
                "No spaCy model available. Install spacy and download "
                "en_core_web_sm:\n"
                "    pip install spacy\n"
                "    python -m spacy download en_core_web_sm\n"
                "or pass a loaded spaCy model via the `nlp` argument / set_nlp()."
            )
            raise RuntimeError(f"RoleSwapCorruption requires a spaCy model. {detail}")
        return self.nlp

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Corrupt an example by swapping subject and object roles.

        Args:
            example: Clean example dictionary with 'prompt' key.
            rng: Random number generator (unused in this deterministic strategy,
                but required by CorruptionStrategy protocol).
            metadata: Optional metadata (unused in base implementation).

        Returns:
            Corrupted example with subject/object swapped.
            Includes metadata about swap: 'role_swap_subject', 'role_swap_object'

        Raises:
            ValueError: If prompt is missing, not a string, or cannot be parsed.
            RuntimeError: If spaCy is not available (not installed, model missing,
                or ABI mismatch). The message includes install instructions.
        """
        if "prompt" not in example:
            raise ValueError("Example must contain 'prompt' key")

        prompt = example["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        nlp = self._require_nlp()
        try:
            doc = nlp(prompt)
        except Exception as e:
            raise ValueError(f"Failed to parse prompt with spaCy: {e}")

        # Find subject and object to swap
        subject_token, object_token = self._find_swap_candidates(doc)

        if subject_token is None or object_token is None:
            # Cannot swap if no clear S-V-O structure found
            # Return unchanged example with metadata indicating no swap
            corrupted_example = example.copy()
            if "role_swap_metadata" not in corrupted_example:
                corrupted_example["role_swap_metadata"] = {}
            corrupted_example["role_swap_metadata"]["swapped"] = False
            corrupted_example["role_swap_metadata"]["reason"] = "No clear S-V-O structure found"
            return corrupted_example

        # Perform the swap
        corrupted_prompt = self._swap_tokens_in_text(prompt, doc, subject_token, object_token)

        # Create corrupted example with metadata
        corrupted_example = example.copy()
        corrupted_example["prompt"] = corrupted_prompt

        if "role_swap_metadata" not in corrupted_example:
            corrupted_example["role_swap_metadata"] = {}

        corrupted_example["role_swap_metadata"]["swapped"] = True
        corrupted_example["role_swap_metadata"]["subject_text"] = subject_token.text
        corrupted_example["role_swap_metadata"]["subject_dep"] = subject_token.dep_
        corrupted_example["role_swap_metadata"]["object_text"] = object_token.text
        corrupted_example["role_swap_metadata"]["object_dep"] = object_token.dep_

        return corrupted_example

    def _find_swap_candidates(self, doc) -> Tuple[Optional[object], Optional[object]]:
        """Find subject and object tokens that can be swapped.

        Looks for:
        - Subject: token with dependency tag 'nsubj' (nominal subject) or 'nsubjpass'
        - Object: token with dependency tag 'dobj' (direct object), 'iobj' (indirect object),
                 'attr' (attribute), 'oprd' (object predicate)

        Args:
            doc: Parsed spaCy Doc object.

        Returns:
            Tuple of (subject_token, object_token) or (None, None) if not found.
        """
        subject_token = None
        object_token = None

        # Find subject
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subject_token = token
                break

        # Find object (prefer dobj, then iobj, then attr)
        for token in doc:
            if token.dep_ in ("dobj", "iobj", "attr", "oprd"):
                object_token = token
                break

        return subject_token, object_token

    def _swap_tokens_in_text(self, original_text: str, doc, subject_token, object_token) -> str:
        """Swap subject and object tokens in the text.

        This is a simplified swap that replaces token texts directly.
        For more complex cases (e.g., determiners), a more sophisticated
        approach would reconstruct the sentence with proper morphology.

        Args:
            original_text: Original text string.
            doc: Parsed spaCy Doc.
            subject_token: Subject token to swap out.
            object_token: Object token to swap out.

        Returns:
            Text with subject and object swapped.
        """
        # Get token texts including any attached determiners/modifiers
        subj_text, subj_char_span = self._get_token_with_modifiers(subject_token)
        obj_text, obj_char_span = self._get_token_with_modifiers(object_token)

        # Reconstruct text with swapped segments
        if subj_char_span[0] < obj_char_span[0]:
            # Subject comes before object
            swapped_text = (
                original_text[: subj_char_span[0]]
                + obj_text
                + original_text[subj_char_span[1] : obj_char_span[0]]
                + subj_text
                + original_text[obj_char_span[1] :]
            )
        else:
            # Object comes before subject
            swapped_text = (
                original_text[: obj_char_span[0]]
                + subj_text
                + original_text[obj_char_span[1] : subj_char_span[0]]
                + obj_text
                + original_text[subj_char_span[1] :]
            )

        return swapped_text

    def _get_token_with_modifiers(self, token) -> Tuple[str, Tuple[int, int]]:
        """Get token text including determiners and modifiers.

        For noun phrases like "the cat", we want to swap the entire phrase,
        not just the noun itself. This method looks for attached det/amod children.

        Args:
            token: spaCy Token.

        Returns:
            Tuple of (combined_text, (start_char, end_char))
        """
        # Start with the token's character span
        start_char = token.idx
        end_char = token.idx + len(token.text)

        # Look for determiners and adjectives attached to this noun
        for child in token.children:
            if child.dep_ in ("det", "amod", "nummod"):
                # Extend span to include modifiers
                child_start = child.idx
                child_end = child.idx + len(child.text)

                start_char = min(start_char, child_start)
                end_char = max(end_char, child_end)

        # Get the text for this span
        combined_text = token.doc.text[start_char:end_char]

        return combined_text, (start_char, end_char)

    def batch_corrupt(
        self,
        examples: List[Dict[str, Any]],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Corrupt a batch of examples.

        Default implementation calls corrupt() on each example.

        Args:
            examples: List of clean examples.
            rng: Random number generator.
            metadata: Optional metadata.

        Returns:
            List of corrupted examples in same order as input.
        """
        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
    ) -> CorruptionValidation:
        """Validate corruption is well-formed.

        Checks:
        1. Both examples have 'prompt' key.
        2. Prompts are non-empty strings.
        3. Role swap metadata is present.
        4. If swapped, prompts should differ.
        5. If not swapped, corruption is still valid (some sentences may not have clear S-V-O).

        Args:
            clean: Original clean example.
            corrupted: Corrupted example from corrupt().

        Returns:
            CorruptionValidation with validity and severity score.
        """
        # Check required fields
        if "prompt" not in clean or "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Missing 'prompt' key in clean or corrupted example"
            )

        clean_prompt = clean.get("prompt", "")
        corrupted_prompt = corrupted.get("prompt", "")

        if not isinstance(clean_prompt, str) or not isinstance(corrupted_prompt, str):
            return CorruptionValidation(is_valid=False, reason="Prompts must be strings")

        if not clean_prompt.strip() or not corrupted_prompt.strip():
            return CorruptionValidation(is_valid=False, reason="Prompts must be non-empty")

        # Check for role_swap metadata
        has_metadata = "role_swap_metadata" in corrupted and isinstance(
            corrupted.get("role_swap_metadata"), dict
        )

        if not has_metadata:
            return CorruptionValidation(
                is_valid=False, reason="Missing role_swap_metadata in corrupted example"
            )

        # Check if swap was successful
        swapped = corrupted["role_swap_metadata"].get("swapped", False)

        if swapped:
            # If swap happened, prompts should differ
            if clean_prompt == corrupted_prompt:
                return CorruptionValidation(
                    is_valid=False, reason="Prompts marked as swapped but are identical"
                )

            # Compute severity as length-normalized difference
            # For role swaps, severity is moderate (structural change but same length usually)
            severity = 0.5  # Default for successful swaps

        else:
            # If no swap happened (no clear S-V-O), still valid but lower severity
            # Prompts should be identical in this case
            if clean_prompt != corrupted_prompt:
                return CorruptionValidation(
                    is_valid=False, reason="Prompts marked as not swapped but differ"
                )
            severity = 0.0  # No corruption occurred

        return CorruptionValidation(is_valid=True, reason=None, severity=severity)
