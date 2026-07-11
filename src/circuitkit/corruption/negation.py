"""
NegationCorruption: Add/remove negations while preserving valid answer format.

Implements a meaning-altering corruption strategy that introduces or removes
negation markers (not, no, never, etc.) in text. Preserves sentence structure
and validity by targeting specific syntactic positions.

Example:
    Original:  "The cat is sleeping."
    Negated:   "The cat is not sleeping."
    Removed:   "The cat is sleeping."
"""

import random
from typing import Any, Dict, List, Literal, Optional

from .base import CorruptionValidation


class NegationCorruption:
    """Add/remove negations using dependency parsing.

    Attributes:
        name: Strategy identifier, "negation".
        mode: "meaning-altering" (changes semantic content).
    """

    name = "negation"
    mode: Literal["meaning-altering"] = "meaning-altering"

    def __init__(
        self,
        nlp=None,
        operation: Optional[str] = None,
        negation_words: Optional[List[str]] = None,
    ):
        """Initialize NegationCorruption.

        Args:
            nlp: Optional spaCy Language model. If None, attempts to load en_core_web_sm.
            operation: Optional operation ("add", "remove", or "toggle"). If None,
                      randomly toggles between add/remove.
            negation_words: Optional list of negation markers to use/remove.
                           Default: ["not", "no", "never", "neither", "nobody", "nothing"]
        """
        self.nlp = nlp
        self.operation = operation
        self.negation_words = negation_words or [
            "not",
            "no",
            "never",
            "neither",
            "nobody",
            "nothing",
            "nowhere",
            "nope",
            "can't",
            "won't",
            "don't",
            "didn't",
        ]
        # Records why spaCy could not be loaded so `_require_nlp()` can give an
        # actionable message instead of silently no-op'ing on every corrupt() call.
        self._nlp_load_error: Optional[str] = None

        # Try to load spaCy model if nlp not provided
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
                # `import spacy` itself can raise something other than
                # ImportError (e.g. ValueError from a numpy/thinc ABI
                # mismatch in spaCy's compiled Cython extensions).
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
            raise RuntimeError(f"NegationCorruption requires a spaCy model. {detail}")
        return self.nlp

    def _has_negation(self, text: str) -> bool:
        """Check if text already contains negation.

        Args:
            text: Text to check

        Returns:
            True if text contains negation markers
        """
        text_lower = text.lower()
        return any(
            f" {neg} " in f" {text_lower} "
            or text_lower.startswith(f"{neg} ")
            or text_lower.endswith(f" {neg}")
            for neg in self.negation_words
        )

    def _find_negation_position(self, doc) -> Optional[int]:
        """Find position of negation in doc.

        Args:
            doc: spaCy Doc object

        Returns:
            Token index of negation marker or None
        """
        for i, token in enumerate(doc):
            if token.text.lower() in self.negation_words:
                return i
        return None

    def _find_auxiliary_verb(self, doc) -> Optional[int]:
        """Find auxiliary verb position for inserting negation.

        In English, negation typically goes after auxiliary verb:
        "can not", "do not", "will not", etc.

        Args:
            doc: spaCy Doc object

        Returns:
            Token index of auxiliary verb or None
        """
        for i, token in enumerate(doc):
            if token.pos_ in ["AUX", "VERB"] and token.dep_ in ["aux", "ROOT"]:
                return i
        return None

    def _add_negation(self, text: str, doc) -> str:
        """Add negation to text.

        Args:
            text: Original text
            doc: spaCy Doc object

        Returns:
            Text with negation added
        """
        # Find position to insert negation
        aux_pos = self._find_auxiliary_verb(doc)

        if aux_pos is None:
            # No auxiliary verb; try prepending
            return f"not {text}"

        # Insert after auxiliary verb
        text.split()
        negation = "not"

        # Handle contractions (can't, won't, etc.)
        next_token_idx = aux_pos + 1
        if next_token_idx < len(doc):
            next_token = doc[next_token_idx].text.lower()
            if next_token in ["have", "has", "be", "been", "being"]:
                # Use full form "not"
                pass

        # Reconstruct with negation
        try:
            doc_tokens = [t.text for t in doc]
            doc_tokens.insert(aux_pos + 1, negation)
            return " ".join(doc_tokens)
        except Exception:
            return f"not {text}"

    def _remove_negation(self, text: str, doc) -> str:
        """Remove negation from text.

        Args:
            text: Original text
            doc: spaCy Doc object

        Returns:
            Text with negation removed
        """
        neg_pos = self._find_negation_position(doc)

        if neg_pos is None:
            return text

        # Remove negation token
        try:
            doc_tokens = [t.text for t in doc]
            doc_tokens.pop(neg_pos)
            result = " ".join(doc_tokens)

            # Clean up double spaces
            result = " ".join(result.split())
            return result

        except Exception:
            return text

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Corrupt example by adding or removing negation.

        Args:
            example: Dict with 'prompt' field.
            rng: Random number generator for reproducibility.
            metadata: Optional task-specific metadata.

        Returns:
            Corrupted example with negation toggled.

        Raises:
            RuntimeError: If no spaCy model is available (see ``_require_nlp``).
        """
        nlp = self._require_nlp()

        prompt = example.get("prompt", "")
        if not prompt:
            return example

        try:
            doc = nlp(prompt)
            has_neg = self._has_negation(prompt)

            # Determine operation
            op = self.operation
            if op is None:
                # Toggle: if has negation, remove; otherwise add
                op = "remove" if has_neg else "add"

            # Apply transformation
            if op == "add" and not has_neg:
                corrupted_text = self._add_negation(prompt, doc)
            elif op == "remove" and has_neg:
                corrupted_text = self._remove_negation(prompt, doc)
            elif op == "toggle":
                # Explicit toggle
                if has_neg:
                    corrupted_text = self._remove_negation(prompt, doc)
                else:
                    corrupted_text = self._add_negation(prompt, doc)
            else:
                corrupted_text = prompt

            # Build corrupted example
            result = example.copy()
            result["prompt"] = corrupted_text

            return result

        except Exception:
            return example

    def batch_corrupt(
        self,
        examples: List[Dict[str, Any]],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Corrupt a batch of examples.

        Args:
            examples: List of clean example dictionaries.
            rng: Random number generator for reproducibility.
            metadata: Task-specific metadata.

        Returns:
            List of corrupted examples.
        """
        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
    ) -> CorruptionValidation:
        """Validate negation corruption.

        Args:
            clean: Original example.
            corrupted: Corrupted example.

        Returns:
            CorruptionValidation result.
        """
        if "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Missing 'prompt' field in corrupted example", severity=1.0
            )

        clean_prompt = clean.get("prompt", "")
        corrupted_prompt = corrupted.get("prompt", "")

        # Ensure prompts are different
        if clean_prompt == corrupted_prompt:
            return CorruptionValidation(
                is_valid=False, reason="Negation corruption did not modify prompt", severity=0.0
            )

        # Ensure result is non-empty
        if not corrupted_prompt or len(corrupted_prompt) < 2:
            return CorruptionValidation(
                is_valid=False, reason="Corrupted prompt is too short", severity=1.0
            )

        # Calculate severity
        if len(clean_prompt) > 0:
            diff_len = abs(len(corrupted_prompt) - len(clean_prompt))
            severity = min(1.0, diff_len / len(clean_prompt))
        else:
            severity = 0.5

        return CorruptionValidation(is_valid=True, reason=None, severity=severity)
