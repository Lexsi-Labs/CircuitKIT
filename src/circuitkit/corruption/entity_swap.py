"""
EntitySwapCorruption: Token-level entity swapping using NER.

Implements a deterministic corruption strategy that identifies entities
in text using spaCy NER, selects a target entity, and swaps it with a
replacement from a pool of entities of the same type.

Useful for IOI-style tasks where you want to swap person names while
preserving entity types (PERSON → PERSON, GPE → GPE, etc.).
"""

import random
from typing import Any, Dict, List, Literal, Optional

from .base import CorruptionValidation


class EntitySwapCorruption:
    """Token-level entity swap using spaCy NER for entity type preservation.

    Attributes:
        name: Strategy identifier, "entity_swap".
        mode: "meaning-altering" (changes semantic content).
    """

    name = "entity_swap"
    mode: Literal["meaning-altering"] = "meaning-altering"

    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        entity_pool: Optional[Dict[str, List[str]]] = None,
        nlp=None,
    ):
        """Initialize EntitySwapCorruption.

        Args:
            entity_types: Optional list of NER entity types to target (e.g., ["PERSON"]).
                         If None, targets all entity types found in text.
            entity_pool: Optional dict mapping entity type to list of replacement entities.
                        If None, entities will be extracted during first corrupt() call.
                        If "auto", automatically build pool from dataset.
            nlp: Optional spaCy Language model. If None, attempts to load en_core_web_sm.
                If loading fails, the failure reason is recorded and surfaced as a
                clear, actionable error on first use (or via set_nlp() to recover).
        """
        self.entity_types = entity_types
        self.entity_pool = entity_pool if entity_pool != "auto" else None
        self.nlp = nlp
        self._pool_built = entity_pool != "auto" and entity_pool is not None
        # Records why spaCy could not be loaded so `_require_nlp()` can give an
        # actionable message instead of a cryptic NoneType error later.
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
                reason (spaCy not installed vs. en_core_web_sm not downloaded).
        """
        if self.nlp is None:
            detail = self._nlp_load_error or (
                "No spaCy model available. Install spacy and download "
                "en_core_web_sm:\n"
                "    pip install spacy\n"
                "    python -m spacy download en_core_web_sm\n"
                "or pass a loaded spaCy model via the `nlp` argument / set_nlp()."
            )
            raise RuntimeError(f"EntitySwapCorruption requires a spaCy model. {detail}")
        return self.nlp

    def _build_pool_from_examples(self, examples: List[Dict[str, Any]]) -> None:
        """Build entity pool from a list of examples.

        Args:
            examples: List of example dicts containing 'prompt' field.
        """
        nlp = self._require_nlp()

        pool = {}
        for example in examples:
            prompt = example.get("prompt", "")
            doc = nlp(prompt)
            for ent in doc.ents:
                ent_type = ent.label_
                if self.entity_types and ent_type not in self.entity_types:
                    continue
                if ent_type not in pool:
                    pool[ent_type] = []
                if ent.text not in pool[ent_type]:
                    pool[ent_type].append(ent.text)

        self.entity_pool = pool
        self._pool_built = True

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Corrupt example by swapping an entity with another of the same type.

        Args:
            example: Dict with 'prompt' and optionally other fields.
            rng: Random generator for reproducibility.
            metadata: Optional dict with 'entity_pool' override or 'target_entity'.

        Returns:
            Corrupted example with modified 'prompt' field.

        Raises:
            RuntimeError: If spaCy model not loaded.
            ValueError: If no suitable entities found in prompt.
        """
        nlp = self._require_nlp()

        prompt = example.get("prompt", "")
        doc = nlp(prompt)

        # Use metadata override pool if provided
        pool = metadata.get("entity_pool", self.entity_pool) if metadata else self.entity_pool

        # Find candidate entities to swap
        candidates = []
        for ent in doc.ents:
            ent_type = ent.label_
            # Filter by entity_types if specified
            if self.entity_types and ent_type not in self.entity_types:
                continue
            # Check if we have replacements for this type
            if pool and ent_type in pool and len(pool[ent_type]) > 0:
                candidates.append((ent.text, ent_type, ent.start_char, ent.end_char))

        if not candidates:
            # No valid entities to swap; return unchanged
            return example.copy()

        # Select a random entity to swap
        target_entity_text, target_type, start, end = candidates[
            rng.randint(0, len(candidates) - 1)
        ]

        # Select replacement from pool of same type
        if not pool or target_type not in pool:
            return example.copy()

        replacements = pool[target_type]
        # Avoid replacing with the same entity
        valid_replacements = [e for e in replacements if e != target_entity_text]
        if not valid_replacements:
            return example.copy()

        replacement = rng.choice(valid_replacements)

        # Swap in prompt
        corrupted_prompt = prompt[:start] + replacement + prompt[end:]

        result = example.copy()
        result["prompt"] = corrupted_prompt
        result["_corruption_info"] = {
            "swapped_entity": target_entity_text,
            "replacement": replacement,
            "entity_type": target_type,
        }
        return result

    def batch_corrupt(
        self,
        examples: List[Dict[str, Any]],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Corrupt a batch of examples (optional optimization).

        Default implementation calls corrupt() on each example.

        Args:
            examples: List of clean example dicts.
            rng: Random generator for reproducibility.
            metadata: Task-specific metadata.

        Returns:
            List of corrupted examples.
        """
        # If entity pool not yet built and we have "auto" mode, build it now
        if not self._pool_built and self.entity_pool is None and self.nlp is not None:
            self._build_pool_from_examples(examples)

        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
    ) -> CorruptionValidation:
        """Validate that corruption is well-formed.

        Args:
            clean: Original example.
            corrupted: Result of corrupt().

        Returns:
            CorruptionValidation with is_valid=True if corruption is usable.
        """
        # Check required fields exist
        if "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False,
                reason="Missing 'prompt' field in corrupted example",
            )

        clean_prompt = clean.get("prompt", "")
        corrupted_prompt = corrupted.get("prompt", "")

        # Check that prompt is non-empty
        if not corrupted_prompt:
            return CorruptionValidation(
                is_valid=False,
                reason="Corrupted prompt is empty",
            )

        # Compute severity as character-level difference
        if clean_prompt == corrupted_prompt:
            severity = 0.0  # No change
        else:
            # Simple Levenshtein-like severity: fraction of chars that differ
            import difflib

            matcher = difflib.SequenceMatcher(None, clean_prompt, corrupted_prompt)
            ratio = matcher.ratio()
            severity = 1.0 - ratio

        return CorruptionValidation(
            is_valid=True,
            reason=None,
            severity=severity,
        )
