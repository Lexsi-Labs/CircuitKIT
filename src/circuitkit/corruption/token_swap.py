"""
TokenSwapCorruption: POS-aware token-level swapping.

Implements a corruption strategy that identifies tokens matching specific
parts-of-speech (POS), and replaces them with other tokens of the same POS.
Validates that replacements tokenize to a single token (no subword splitting).

Useful for tasks like greater_than where you want to swap numbers, or
more generally where NER is insufficient and you need POS-aware swaps.
"""

import random
from typing import Any, Dict, List, Literal, Optional

from .base import CorruptionValidation


class TokenSwapCorruption:
    """POS-aware token swap for general datasets.

    Attributes:
        name: Strategy identifier, "token_swap".
        mode: "meaning-altering" (changes semantic content).
    """

    name = "token_swap"
    mode: Literal["meaning-altering"] = "meaning-altering"

    def __init__(
        self,
        pos_tags: Optional[List[str]] = None,
        tokenizer=None,
        vocab: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize TokenSwapCorruption.

        Args:
            pos_tags: Optional list of POS tags to target (e.g., ["NUM", "NN", "VB"]).
                     If None, all tokens are eligible.
            tokenizer: Optional tokenizer (e.g., from transformers).
                      If provided, used to validate single-token replacements.
            vocab: Optional dict mapping POS tag to list of replacement tokens.
                  If None, will be built from examples during batch_corrupt().
        """
        self.pos_tags = pos_tags
        self.tokenizer = tokenizer
        self.vocab = vocab or {}
        self._vocab_built = vocab is not None

    def set_tokenizer(self, tokenizer):
        """Set tokenizer after initialization."""
        self.tokenizer = tokenizer

    def _build_vocab_from_examples(
        self,
        examples: List[Dict[str, Any]],
        tagger,
    ) -> None:
        """Build POS-filtered vocabulary from examples.

        Args:
            examples: List of example dicts containing 'prompt' field.
            tagger: Function that takes text and returns (tokens, pos_tags).
        """
        vocab = {}
        for example in examples:
            prompt = example.get("prompt", "")
            tokens, pos_list = tagger(prompt)
            for token, pos in zip(tokens, pos_list):
                # Filter by POS tags if specified
                if self.pos_tags and pos not in self.pos_tags:
                    continue
                # Skip if tokenizer is set and token doesn't tokenize to 1 token
                if self.tokenizer is not None:
                    token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                    if len(token_ids) != 1:
                        continue
                if pos not in vocab:
                    vocab[pos] = []
                if token not in vocab[pos]:
                    vocab[pos].append(token)

        self.vocab = vocab
        self._vocab_built = True

    def _tokenize_and_tag(self, text: str, tagger) -> tuple:
        """Tokenize and tag text with POS.

        Args:
            text: Text to tokenize and tag.
            tagger: Function that takes text and returns (tokens, pos_tags).

        Returns:
            Tuple of (tokens, pos_tags).
        """
        return tagger(text)

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Corrupt example by swapping a token with another of the same POS.

        Args:
            example: Dict with 'prompt' and optionally other fields.
            rng: Random generator for reproducibility.
            metadata: Optional dict with 'tagger' function and vocab overrides.

        Returns:
            Corrupted example with modified 'prompt' field.

        Raises:
            ValueError: If tagger not provided in metadata and no default tagger available.
            ValueError: If no suitable tokens found in prompt.
        """
        if metadata is None or "tagger" not in metadata:
            raise ValueError(
                "metadata with 'tagger' function required for TokenSwapCorruption.corrupt()"
            )

        tagger = metadata["tagger"]
        prompt = example.get("prompt", "")

        # Tokenize and tag
        tokens, pos_tags = self._tokenize_and_tag(prompt, tagger)

        # Use metadata vocab override if provided
        vocab = metadata.get("vocab", self.vocab) if metadata else self.vocab

        # Find candidate tokens to swap
        candidates = []
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            # Filter by POS tags if specified
            if self.pos_tags and pos not in self.pos_tags:
                continue
            # Check if we have replacements for this POS
            if vocab and pos in vocab and len(vocab[pos]) > 0:
                candidates.append((i, token, pos))

        if not candidates:
            # No valid tokens to swap; return unchanged
            return example.copy()

        # Select a random token to swap
        idx, target_token, target_pos = candidates[rng.randint(0, len(candidates) - 1)]

        # Select replacement from vocab of same POS
        if not vocab or target_pos not in vocab:
            return example.copy()

        replacements = vocab[target_pos]
        # Avoid replacing with the same token
        valid_replacements = [t for t in replacements if t != target_token]
        if not valid_replacements:
            return example.copy()

        replacement = rng.choice(valid_replacements)

        # Validate replacement tokenizes to single token (if tokenizer provided)
        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(replacement, add_special_tokens=False)
            if len(token_ids) != 1:
                # Skip this replacement
                return example.copy()

        # Reconstruct prompt with replacement
        corrupted_tokens = tokens.copy()
        corrupted_tokens[idx] = replacement
        corrupted_prompt = " ".join(corrupted_tokens)

        result = example.copy()
        result["prompt"] = corrupted_prompt
        result["_corruption_info"] = {
            "swapped_token": target_token,
            "replacement": replacement,
            "pos": target_pos,
            "token_index": idx,
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
            metadata: Task-specific metadata with 'tagger' function.

        Returns:
            List of corrupted examples.
        """
        # If vocab not yet built, build it now from examples
        if not self._vocab_built and self.vocab == {} and metadata:
            tagger = metadata.get("tagger")
            if tagger:
                self._build_vocab_from_examples(examples, tagger)

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
