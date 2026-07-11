"""
Model-agnostic token ID generation utilities.

This module provides utilities for generating token IDs that work with any
TransformerLens-supported model, ensuring no hardcoded assumptions or defaults.
"""

from typing import Any, Dict, List, Tuple


class TokenIDGenerator:
    """Model-agnostic token ID generation - NO DEFAULTS."""

    def __init__(self, model):
        """
        Initialize token ID generator with a model.

        Args:
            model: Model name string (e.g., 'gpt2') or HookedTransformer model instance

        Raises:
            ValueError: If model is None or cannot resolve tokenizer
        """
        if model is None:
            raise ValueError(
                "TokenIDGenerator requires a model. "
                "Provide model name string (e.g., 'gpt2') or model instance."
            )

        self.model = model

        # Handle model name string
        if isinstance(model, str):
            from transformers import AutoTokenizer

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model)
                self.vocab_size = len(self.tokenizer)
                self.model_name = model
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer for model '{model}': {e}")
        else:
            # Handle model object
            self.tokenizer = model.tokenizer
            self.vocab_size = len(self.tokenizer)
            self.model_name = getattr(model.cfg, "model_name", "unknown")

    def get_token_id(self, text: str, prepend_space: bool = True) -> int:
        """
        Get token ID for a single token using the model's tokenizer.

        Args:
            text: Text to tokenize
            prepend_space: Whether to prepend a space (for proper tokenization)

        Returns:
            Token ID as integer

        Raises:
            ValueError: If tokenization fails or returns empty result
        """
        if prepend_space:
            text = " " + text

        # Use model's to_tokens method for consistency
        tokens = self.model.to_tokens(text, prepend_bos=False)
        if tokens.numel() == 0:
            raise ValueError(f"Failed to tokenize text: '{text}'")

        # Return the last token ID (for single token queries)
        return int(tokens[0, -1].item())

    def get_token_ids_batch(self, texts: List[str], prepend_space: bool = True) -> List[int]:
        """
        Get token IDs for multiple texts.

        Args:
            texts: List of texts to tokenize
            prepend_space: Whether to prepend a space to each text

        Returns:
            List of token IDs

        Raises:
            ValueError: If any tokenization fails
        """
        token_ids = []
        for text in texts:
            token_id = self.get_token_id(text, prepend_space)
            token_ids.append(token_id)
        return token_ids

    def find_differing_tokens(self, clean_text: str, corrupted_text: str) -> Tuple[int, int]:
        """
        Find the first differing token between clean and corrupted text.

        Args:
            clean_text: Original text
            corrupted_text: Corrupted version of the text

        Returns:
            Tuple of (correct_token_id, incorrect_token_id)

        Raises:
            ValueError: If no differences found or tokenization fails
        """
        # Tokenize both texts
        clean_tokens = self.tokenizer.encode(clean_text, add_special_tokens=False)
        corrupted_tokens = self.tokenizer.encode(corrupted_text, add_special_tokens=False)

        # Find first differing token
        for i in range(min(len(clean_tokens), len(corrupted_tokens))):
            if clean_tokens[i] != corrupted_tokens[i]:
                return clean_tokens[i], corrupted_tokens[i]

        # If no differences found, this is an error
        raise ValueError(
            f"No token differences found between clean and corrupted text. "
            f"This suggests the corruption didn't change the tokenization. "
            f"Clean: '{clean_text}', Corrupted: '{corrupted_text}'"
        )

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return tokenizer metadata for caching validation.

        Returns:
            Dictionary containing model and tokenizer metadata
        """
        return {
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", "unknown"),
        }

    def validate_compatibility(self, other_metadata: Dict[str, Any]) -> bool:
        """
        Validate compatibility with another tokenizer's metadata.

        Args:
            other_metadata: Metadata from another TokenIDGenerator

        Returns:
            True if compatible, False otherwise
        """
        current_metadata = self.get_metadata()

        # Check critical compatibility factors
        if current_metadata["model_name"] != other_metadata["model_name"]:
            return False

        if current_metadata["vocab_size"] != other_metadata["vocab_size"]:
            return False

        if current_metadata["tokenizer_class"] != other_metadata["tokenizer_class"]:
            return False

        return True
