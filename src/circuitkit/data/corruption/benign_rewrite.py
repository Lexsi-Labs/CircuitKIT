"""benign_rewrite — symbolic harmful->benign instruction rewrite.

For safety datasets (AdvBench / HarmBench / WildJailbreak) the natural
contrastive pair is a harmful instruction paired with a benign-but-
similarly-structured rewrite. This strategy generates benign rewrites
*without* an LLM: it operates by swapping known harmful keywords with
benign equivalents drawn from a curated mapping table.

This is the "no-external-LLM" fallback. When an instruction-tuned LLM
is available, ``llm_counterfactual`` produces higher-quality rewrites.

Length contract: UNKNOWN — keyword pairs aren't always same-length.

Reference: SafeSeek (arxiv:2603.23268), Refusal Direction (Arditi 2024,
arxiv:2406.11717), and CAA (Rimsky et al. 2024).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

# A curated harmful->benign keyword mapping. Designed to convert AdvBench/
# HarmBench-style instructions into structurally similar but harmless
# requests. Order matters: longer phrases are checked before shorter ones
# to avoid partial-overlap rewrites.
_HARMFUL_TO_BENIGN: Dict[str, str] = {
    # Violence
    "build a bomb": "bake a cake",
    "make a bomb": "make a cake",
    "how to kill": "how to celebrate",
    "how to murder": "how to compliment",
    "assassinate": "congratulate",
    "shoot someone": "greet someone",
    "stab someone": "hug someone",
    "poison": "season",
    # Hacking / cyber
    "hack into": "log into",
    "exploit a vulnerability": "fix a vulnerability",
    "phishing": "marketing",
    "malware": "software",
    "ransomware": "newsletter",
    "ddos": "ping",
    "steal credentials": "share contact info",
    "bypass security": "follow security",
    # Drugs / illegal
    "synthesize drugs": "synthesize sugar",
    "manufacture drugs": "manufacture cookies",
    "make meth": "make tea",
    "make heroin": "make pasta",
    "buy drugs": "buy snacks",
    # Weapons
    "build a gun": "build a chair",
    "manufacture a weapon": "manufacture a tool",
    "ghost gun": "wooden chair",
    # Bio
    "synthesize a virus": "synthesize a salad",
    "weaponize anthrax": "package coffee",
    "biological weapon": "biological textbook",
    # Generic harmful framings
    "harm a child": "help a child",
    "harm someone": "help someone",
    "destroy": "decorate",
    "abduct": "invite",
    "manipulate": "encourage",
    # Generic instruction openers
    "Write instructions for ": "Write instructions for safely cleaning ",
    "Write a guide to ": "Write a guide to politely ",
    "Explain how to ": "Explain how to safely ",
}


@register_strategy("benign_rewrite")
class BenignRewrite(CorruptionStrategy):
    description = (
        "Symbolic harmful->benign keyword rewrite for safety datasets. "
        "No external LLM required. For higher-quality rewrites, prefer "
        "llm_counterfactual when an instruction-tuned generator is available."
    )
    length_contract = LengthContract.UNKNOWN

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        extra_mapping: Optional[Dict[str, str]] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        prompt = record.clean_prompt
        mapping = dict(_HARMFUL_TO_BENIGN)
        if extra_mapping:
            mapping.update(extra_mapping)
        # Apply longest-key-first to avoid partial-overlap issues
        replaced = False
        notes_parts = []
        for harmful in sorted(mapping, key=len, reverse=True):
            benign = mapping[harmful]
            pattern = re.compile(re.escape(harmful), re.IGNORECASE)
            if pattern.search(prompt):
                prompt = pattern.sub(benign, prompt, count=1)
                notes_parts.append(f"{harmful!r}->{benign!r}")
                replaced = True
                break  # one rewrite per record
        if not replaced:
            return CorruptionResult(
                None,
                None,
                notes=(
                    "no harmful keyword from the built-in mapping found; "
                    "use --extra-mapping to provide domain-specific terms, "
                    "or use llm_counterfactual"
                ),
                succeeded=False,
            )
        return CorruptionResult(
            corrupt_prompt=prompt,
            # For refusal datasets, the clean answer is typically a refusal
            # phrase ("I cannot...") and the corrupt answer should be a
            # comply phrase. We default to the clean answer; caller can
            # override via meta.
            corrupt_answer=record.clean_answer,
            notes="rewrote: " + " ; ".join(notes_parts),
            succeeded=True,
        )
