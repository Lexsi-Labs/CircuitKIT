"""
Paraphrase Corruption Strategy

Implements semantic-preserving and surface-level corruption using a small local LLM.
No API calls; uses local model inference for meaning-preserving rewriting of prompts.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import CorruptionValidation


import logging

logger = logging.getLogger(__name__)

class ParaphraseCorruption:
    """
    Paraphrase corruption strategy using a small local model.

    Corrupts prompts by generating semantic-preserving paraphrases using
    a small local language model (e.g., Qwen2.5-0.5B or Phi-3-mini).
    Supports both surface-level (syntactic) and semantic (template-driven) modes.

    Attributes:
        name: "paraphrase"
        mode: "meaning-preserving"
    """

    name = "paraphrase"
    mode = "meaning-preserving"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        mode: str = "semantic",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 150,
    ):
        """
        Initialize paraphrase corruption strategy.

        Args:
            model_name: HuggingFace model identifier for local paraphrasing.
                       Default: "Qwen/Qwen2.5-0.5B-Instruct" (small instruction
                       -tuned model, ~0.5B params). An instruction-tuned model is
                       required: base completion models tend to echo the input
                       verbatim rather than rephrase it.
                       Alternative: "microsoft/phi-2" or similar compact models.
            mode: "surface" (syntactic synonym replacement) or
                  "semantic" (template-driven, more meaningful rewrites).
            cache_dir: Directory for caching paraphrases. If None, uses ./cache/corruptions/.
            device: Device to run model on ("cpu" or "cuda").
            max_length: Maximum tokens for generated paraphrase.
        """
        self.model_name = model_name
        self.mode = mode
        self.device = device
        self.max_length = max_length

        # Setup cache directory
        if cache_dir is None:
            cache_dir = "./cache/corruptions"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "paraphrase_cache.json"

        # Load cache
        self.cache = self._load_cache()

        # Model and tokenizer are loaded lazily on first semantic paraphrase
        # (see `_ensure_model_loaded`). This keeps construction cheap and avoids
        # consuming VRAM when the strategy is only used in "surface" mode or
        # when all results are served from cache.
        self._tokenizer = None
        self._model = None

    def _ensure_model_loaded(self) -> None:
        """Lazily load the local model and tokenizer on first use.

        Idempotent: subsequent calls are no-ops once the model is loaded.
        """
        if self._model is not None:
            return

        logger.info(f"Loading model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        model.eval()

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._tokenizer = tokenizer
        self._model = model

    @property
    def tokenizer(self):
        """Tokenizer for the local paraphrase model (loaded lazily on access)."""
        self._ensure_model_loaded()
        return self._tokenizer

    @property
    def model(self):
        """Local paraphrase model (loaded lazily on access)."""
        self._ensure_model_loaded()
        return self._model

    def _load_cache(self) -> Dict[str, str]:
        """Load paraphrase cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save paraphrase cache to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key: sha256(prompt + model_name + mode)."""
        key_str = f"{prompt}|{self.model_name}|{self.mode}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _paraphrase_surface(self, prompt: str) -> str:
        """
        Surface-level paraphrase using simple synonym replacement.

        This is a lightweight fallback when LLM inference is too slow.
        Uses basic word substitutions.
        """
        # Simple synonym dictionary for common words
        synonyms = {
            "who": "which person",
            "what": "which thing",
            "where": "in what location",
            "when": "at what time",
            "why": "for what reason",
            "how": "in what manner",
            "is": "happens to be",
            "was": "happened to be",
            "are": "happen to be",
            "were": "happened to be",
            "has": "possesses",
            "have": "possess",
            "had": "possessed",
        }

        words = prompt.split()
        paraphrased_words = []
        for word in words:
            lower_word = word.lower().strip(".,!?;:")
            if lower_word in synonyms:
                replacement = synonyms[lower_word]
                # Preserve original casing and punctuation
                if word[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                paraphrased_words.append(replacement)
            else:
                paraphrased_words.append(word)

        return " ".join(paraphrased_words)

    def _paraphrase_semantic(self, prompt: str) -> str:
        """
        Semantic paraphrase using instruction-driven rewriting with local LLM.

        Uses the loaded local instruction-tuned model to generate a semantic
        paraphrase. The model's chat template is used when available (instruct
        models); otherwise a plain completion template is used as a fallback.
        """
        tokenizer = self.tokenizer  # triggers lazy load

        # Build the model input. Instruction-tuned models expose a chat
        # template; using it (instead of a raw "Rephrased:" completion prompt)
        # is what makes the model actually rephrase rather than echo the input.
        use_chat = getattr(tokenizer, "chat_template", None) is not None
        if use_chat:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that rephrases sentences "
                        "while preserving their exact meaning."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Rephrase the following sentence in a different way "
                        "while keeping the meaning the same. Output only the "
                        f'rephrased sentence, nothing else.\n\n"{prompt}"'
                    ),
                },
            ]
            model_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            model_input = (
                "Rephrase the following sentence in a different way while "
                f'keeping the meaning the same: "{prompt}"\n\nRephrased: '
            )

        # Tokenize input
        inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        input_len = inputs["input_ids"].shape[1]

        # Generate paraphrase. Cap generated tokens relative to the input so a
        # long prompt does not get truncated, and use greedy decoding for
        # determinism.
        max_new = max(32, int(input_len * 1.5))
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated continuation.
        gen_ids = outputs[0][input_len:]
        paraphrased = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Some models still echo a "Rephrased:" marker; strip it.
        if "Rephrased:" in paraphrased:
            paraphrased = paraphrased.split("Rephrased:")[-1].strip()

        # Strip wrapping quotes the model sometimes adds.
        if len(paraphrased) >= 2 and paraphrased[0] in "\"'" and paraphrased[-1] in "\"'":
            paraphrased = paraphrased[1:-1].strip()

        # Keep only the first line/sentence-ish chunk to avoid trailing chatter.
        paraphrased = paraphrased.split("\n")[0].strip()

        # Fallback to original if empty
        if not paraphrased or len(paraphrased) < 5:
            paraphrased = prompt

        return paraphrased

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Corrupt an example by paraphrasing its prompt.

        Args:
            example: Example dict with 'prompt' and 'answer' keys.
            rng: Random number generator (unused for deterministic paraphrasing).
            metadata: Optional metadata (unused).

        Returns:
            Corrupted example with paraphrased prompt, answer unchanged.
        """
        if "prompt" not in example:
            raise ValueError("Example must contain 'prompt' key")

        prompt = example["prompt"]

        # Check cache
        cache_key = self._get_cache_key(prompt)
        if cache_key in self.cache:
            paraphrased_prompt = self.cache[cache_key]
        else:
            # Generate paraphrase
            if self.mode == "surface":
                paraphrased_prompt = self._paraphrase_surface(prompt)
            elif self.mode == "semantic":
                paraphrased_prompt = self._paraphrase_semantic(prompt)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            # Cache result
            self.cache[cache_key] = paraphrased_prompt
            self._save_cache()

        # Return corrupted example with paraphrased prompt
        corrupted_example = example.copy()
        corrupted_example["prompt"] = paraphrased_prompt

        return corrupted_example

    def batch_corrupt(
        self,
        examples: List[Dict[str, Any]],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Corrupt a batch of examples.

        Default implementation calls corrupt() sequentially.
        Could be optimized for vectorized inference.

        Args:
            examples: List of example dicts.
            rng: Random number generator.
            metadata: Optional metadata.

        Returns:
            List of corrupted examples.
        """
        return [self.corrupt(ex, rng=rng, metadata=metadata) for ex in examples]

    def validate(
        self,
        clean: Dict[str, Any],
        corrupted: Dict[str, Any],
    ) -> CorruptionValidation:
        """
        Validate that a corruption is well-formed.

        Checks:
        1. Both dicts have 'prompt' key
        2. Prompts are different
        3. Answer unchanged
        4. Prompt not too short/long (within ±30% of clean length)

        Args:
            clean: Original example.
            corrupted: Paraphrased example.

        Returns:
            CorruptionValidation result.
        """
        # Check required fields
        if "prompt" not in clean or "prompt" not in corrupted:
            return CorruptionValidation(is_valid=False, reason="Missing 'prompt' key")

        clean_prompt = clean["prompt"]
        corrupted_prompt = corrupted["prompt"]

        # Check prompts are different
        if clean_prompt == corrupted_prompt:
            return CorruptionValidation(
                is_valid=False, reason="Paraphrased prompt identical to original"
            )

        # Check answer unchanged (if present)
        if "answer" in clean and "answer" in corrupted:
            if clean["answer"] != corrupted["answer"]:
                return CorruptionValidation(is_valid=False, reason="Answer was modified")

        # Check length within ±30%
        clean_len = len(clean_prompt.split())
        corrupted_len = len(corrupted_prompt.split())
        if clean_len > 0:
            length_ratio = corrupted_len / clean_len
            if not (0.7 <= length_ratio <= 1.3):
                return CorruptionValidation(
                    is_valid=False,
                    reason=f"Length ratio {length_ratio:.2f} outside ±30% budget",
                )

        # Compute severity as word-level difference ratio
        clean_words = set(clean_prompt.lower().split())
        corrupted_words = set(corrupted_prompt.lower().split())
        if clean_words or corrupted_words:
            union_size = len(clean_words | corrupted_words)
            diff_size = len(clean_words ^ corrupted_words)
            severity = diff_size / union_size if union_size > 0 else 0.0
        else:
            severity = 0.0

        return CorruptionValidation(is_valid=True, severity=severity)
