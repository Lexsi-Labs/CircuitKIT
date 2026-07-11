"""instruction_swap — pair instructions for circuit discovery on
instruction-following tasks (IFEval-style).

For instruction-following datasets, each example has a directive
("Generate a 3-paragraph essay") plus a content prompt. Standard
EAP needs single-token answer differentiation, which generation
tasks don't naturally provide. This strategy approximates a
contrastive pair by:

1. Picking another record with a DIFFERENT directive verb
   (generate / list / explain / summarize / compare ...)
2. Swapping the directive verb in the prompt
3. Using a stop-token (period) as both the clean and corrupt
   answer's first token — but ONLY if the prompts ARE different
   under tokenisation. NormalizedTaskSpec then drops degenerate
   pairs at CSV materialisation.

This is a coarse approximation. A proper IFEval circuit would
need per-instruction compliance probes (see Mueller et al. 2024
mediator-survey for the framework).
"""

from __future__ import annotations

import random
import re
from typing import Any, Iterable, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

_DIRECTIVE_VERBS = [
    "generate",
    "write",
    "create",
    "list",
    "explain",
    "describe",
    "summarize",
    "compare",
    "translate",
    "convert",
    "rewrite",
]
_DIRECTIVE_RE = re.compile(r"\b(" + "|".join(_DIRECTIVE_VERBS) + r")\b", re.IGNORECASE)


@register_strategy("instruction_swap")
class InstructionSwap(CorruptionStrategy):
    description = (
        "Swap the directive verb in an instruction-style prompt "
        "(generate <-> list, explain <-> summarize, etc.). Coarse "
        "approximation for IFEval-style circuit discovery; a proper "
        "implementation needs per-instruction compliance probes."
    )
    length_contract = LengthContract.PRESERVE

    def fits(self, record: ContrastiveRecord) -> bool:
        if not super().fits(record):
            return False
        return _DIRECTIVE_RE.search(record.clean_prompt) is not None

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        rng: Optional[random.Random] = None,
        **_unused: Any,
    ) -> CorruptionResult:
        rng = rng or random.Random(hash(record.record_id) & 0xFFFFFFFF)
        m = _DIRECTIVE_RE.search(record.clean_prompt)
        if not m:
            return CorruptionResult(None, None, notes="no directive verb", succeeded=False)
        original = m.group(1)
        candidates = [v for v in _DIRECTIVE_VERBS if v.lower() != original.lower()]
        new_verb = rng.choice(candidates)
        # Preserve original capitalisation.
        if original[0].isupper():
            new_verb = new_verb.capitalize()
        new_prompt = record.clean_prompt[: m.start()] + new_verb + record.clean_prompt[m.end() :]
        # Answer stays the clean answer's first token; if both halves
        # tokenise to the same first token, NormalizedTaskSpec will drop.
        return CorruptionResult(
            corrupt_prompt=new_prompt,
            corrupt_answer=record.clean_answer,
            notes=f"directive {original!r} -> {new_verb!r}",
            succeeded=True,
        )


def audit_instruction_swap_degeneracy(
    records: Iterable[ContrastiveRecord],
    tokenizer: Any,
) -> dict[str, Any]:
    """Report how often instruction-swap pairs collapse at tokenization time.

    The current instruction-swap strategy preserves the clean answer. That is
    fine for paired-data bookkeeping, but EAP-style materialisation drops any
    pair whose answer tokens collapse to the same first meaningful token.

    This helper counts that collapse rate so it can be audited explicitly.
    """

    def _first_meaningful_token(text: str):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return None
        if not ids:
            return None

        try:
            ws_probe = tokenizer.encode(" ", add_special_tokens=False)
            ws_token_id = ws_probe[0] if len(ws_probe) == 1 else None
        except Exception:
            ws_token_id = None

        if ws_token_id is not None and ids[0] == ws_token_id and len(ids) > 1:
            return ids[1]
        return ids[0]

    total = 0
    paired = 0
    same_first_token = 0
    skipped_unpaired = 0

    for record in records:
        total += 1
        if not record.is_paired:
            skipped_unpaired += 1
            continue
        paired += 1
        clean_idx = _first_meaningful_token(record.clean_answer)
        corrupt_idx = _first_meaningful_token(record.corrupt_answer or "")
        if clean_idx is None or corrupt_idx is None:
            skipped_unpaired += 1
            continue
        if clean_idx == corrupt_idx:
            same_first_token += 1

    return {
        "total": total,
        "paired": paired,
        "same_first_token": same_first_token,
        "same_first_token_frac": round(same_first_token / paired, 4) if paired else 0,
        "kept": paired - same_first_token,
        "dropped": same_first_token,
        "skipped_unpaired": skipped_unpaired,
    }
