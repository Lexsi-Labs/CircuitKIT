"""
VoiceSwapCorruption: Active/Passive voice transformation using dependency parsing.

Implements a meaning-preserving corruption strategy that converts active voice
sentences to passive voice and vice versa. Uses spaCy dependency parsing to
identify verb phrases and reconstruct sentences in the opposite voice.

Example:
    Active:  "The cat ate the mouse."
    Passive: "The mouse was eaten by the cat."
"""

import random
from typing import Any, Dict, List, Literal, Optional

from .base import CorruptionValidation


class VoiceSwapCorruption:
    """Active/Passive voice swapping using dependency parsing.

    Attributes:
        name: Strategy identifier, "voice_swap".
        mode: "meaning-preserving" (changes surface form, preserves semantics).
    """

    name = "voice_swap"
    mode: Literal["meaning-preserving"] = "meaning-preserving"

    def __init__(self, nlp=None, target_voice: Optional[str] = None):
        """Initialize VoiceSwapCorruption.

        Args:
            nlp: Optional spaCy Language model. If None, attempts to load en_core_web_sm.
            target_voice: Optional target voice ("active" or "passive"). If None,
                         automatically swaps from current voice.
        """
        self.nlp = nlp
        self.target_voice = target_voice
        self._voices = {"active", "passive"}
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
            raise RuntimeError(f"VoiceSwapCorruption requires a spaCy model. {detail}")
        return self.nlp

    def _detect_voice(self, doc) -> str:
        """Detect if a sentence is in active or passive voice.

        Uses heuristic: presence of "be" auxiliary + past participle suggests passive.

        Args:
            doc: spaCy Doc object

        Returns:
            "active" or "passive"
        """
        # Check for passive voice patterns: aux verb (be) + past participle
        aux_verbs = {"be", "been", "being", "am", "is", "are", "was", "were"}
        has_aux = any(token.lemma_.lower() in aux_verbs and token.pos_ == "AUX" for token in doc)

        # Look for past participle
        has_participle = any(token.tag_ in ["VBN"] for token in doc)

        if has_aux and has_participle:
            return "passive"
        return "active"

    def _extract_verb_phrase(self, doc, verb_idx: int) -> Optional[Dict[str, Any]]:
        """Extract verb phrase structure for voice transformation.

        Args:
            doc: spaCy Doc object
            verb_idx: Index of the main verb token

        Returns:
            Dict with verb phrase components or None if unable to extract
        """
        verb_token = doc[verb_idx]

        # Find subject (nsubj) and object (dobj/obj)
        subject = None
        obj = None
        aux_tokens = []

        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                subject = child
            elif child.dep_ in ["dobj", "obj"]:
                obj = child
            elif child.pos_ == "AUX":
                aux_tokens.append(child)

        if not subject or not obj:
            return None

        return {
            "verb": verb_token,
            "subject": subject,
            "object": obj,
            "aux_tokens": aux_tokens,
        }

    def _transform_to_passive(self, text: str, doc) -> str:
        """Transform active voice sentence to passive voice.

        Args:
            text: Original sentence text
            doc: spaCy Doc object

        Returns:
            Passive voice sentence
        """
        # Find main verb
        verb_idx = None
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "conj"]:
                verb_idx = token.i
                break

        if verb_idx is None:
            return text

        phrase = self._extract_verb_phrase(doc, verb_idx)
        if not phrase:
            return text

        verb = phrase["verb"]
        subject = phrase["subject"]
        obj = phrase["object"]

        # Simple transformation: "Subject Verb Object" → "Object was Verbed by Subject"
        # This is a simplified version; full implementation would handle more cases
        try:
            passive_verb = self._form_past_participle(verb.text)
            result = f"{obj.text} was {passive_verb} by {subject.text}"
            return result
        except Exception:
            return text

    def _transform_to_active(self, text: str, doc) -> str:
        """Transform passive voice sentence to active voice.

        Args:
            text: Original sentence text
            doc: spaCy Doc object

        Returns:
            Active voice sentence
        """
        # Find main verb
        verb_idx = None
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "conj"]:
                verb_idx = token.i
                break

        if verb_idx is None:
            return text

        phrase = self._extract_verb_phrase(doc, verb_idx)
        if not phrase:
            return text

        # In passive voice, agent is in "by" prepositional phrase
        # Subject becomes object, agent becomes subject
        verb = phrase["verb"]
        subject = phrase["subject"]

        # Find "by" phrase
        agent = None
        for token in doc:
            if token.text.lower() == "by" and token.dep_ == "case":
                # Find the object of "by"
                for child in token.head.children:
                    if child.dep_ in ["pobj", "obj"]:
                        agent = child
                        break

        if not agent:
            return text

        try:
            base_verb = self._form_base_verb(verb.text)
            result = f"{agent.text} {base_verb} {subject.text}"
            return result
        except Exception:
            return text

    def _form_past_participle(self, verb: str) -> str:
        """Form past participle from base verb (simplified).

        Args:
            verb: Base verb form

        Returns:
            Past participle form
        """
        # Very simplified; real implementation would use lemmatizer
        if verb.endswith("e"):
            return verb + "d"
        elif verb.endswith(("s", "x", "z", "ch", "sh")):
            return verb + "ed"
        else:
            return verb + "ed"

    def _form_base_verb(self, verb: str) -> str:
        """Form base verb from past participle (simplified).

        Args:
            verb: Past participle form

        Returns:
            Base verb form
        """
        # Very simplified; real implementation would use lemmatizer
        if verb.endswith("ed"):
            return verb[:-2]
        return verb

    def corrupt(
        self,
        example: Dict[str, Any],
        *,
        rng: random.Random,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Corrupt example by swapping voice.

        Args:
            example: Dict with 'prompt' field (and optionally other fields).
            rng: Random number generator for reproducibility.
            metadata: Optional task-specific metadata.

        Returns:
            Corrupted example with voice swapped.

        Raises:
            RuntimeError: If no spaCy model is available (see ``_require_nlp``).
        """
        nlp = self._require_nlp()

        prompt = example.get("prompt", "")
        if not prompt:
            return example

        try:
            doc = nlp(prompt)
            detected_voice = self._detect_voice(doc)

            # Determine target voice
            target = self.target_voice
            if target is None:
                target = "passive" if detected_voice == "active" else "active"

            # Transform
            if target == "passive" and detected_voice == "active":
                corrupted_text = self._transform_to_passive(prompt, doc)
            elif target == "active" and detected_voice == "passive":
                corrupted_text = self._transform_to_active(prompt, doc)
            else:
                # Already in target voice
                corrupted_text = prompt

            # Build corrupted example
            result = example.copy()
            result["prompt"] = corrupted_text

            return result

        except Exception:
            # On any error, return original
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
        """Validate voice swap corruption.

        Args:
            clean: Original example.
            corrupted: Corrupted example.

        Returns:
            CorruptionValidation result.
        """
        # Check that required fields exist
        if "prompt" not in corrupted:
            return CorruptionValidation(
                is_valid=False, reason="Missing 'prompt' field in corrupted example", severity=1.0
            )

        clean_prompt = clean.get("prompt", "")
        corrupted_prompt = corrupted.get("prompt", "")

        # Ensure prompts are different
        if clean_prompt == corrupted_prompt:
            return CorruptionValidation(
                is_valid=False, reason="Voice swap did not modify prompt", severity=0.0
            )

        # Calculate severity as character-level difference
        if len(clean_prompt) > 0:
            diff_len = abs(len(corrupted_prompt) - len(clean_prompt))
            severity = min(1.0, diff_len / len(clean_prompt))
        else:
            severity = 0.0

        return CorruptionValidation(is_valid=True, reason=None, severity=severity)
