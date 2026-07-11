"""Instruction adapter — Alpaca / Dolly / OpenAssistant -style datasets.

Schema variants handled:
  - Alpaca (tatsu-lab/alpaca):    {instruction, input, output, text?}
  - Dolly (databricks/dolly-15k): {instruction, context, response, category}
  - OpenAssistant (OASST):        {instruction, response} or {prompt, response}

Output shape: clean_prompt = formatted instruction (with optional input
context); clean_answer = first token of the desired response. The corrupt
half is NOT supplied here (instruction tasks have no native counter-factual);
a CorruptionStrategy must be applied before discovery — typically
``llm_counterfactual`` (rewrite to a different instruction) or
``token_swap`` on a salient noun in the instruction.

The adapter records ``contrast_source = NOT_PAIRED_YET``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from .base import DataAdapter, register_adapter
from .pairwise import _iter_rows, _peek_columns

# Detection keys, in order of preference.
_INSTRUCTION_KEYS = ("instruction", "prompt")
_CONTEXT_KEYS = ("input", "context")
_RESPONSE_KEYS = ("output", "response", "answer", "completion")


@register_adapter(DatasetShape.INSTRUCTION)
class InstructionAdapter(DataAdapter):
    """Adapter for Alpaca / Dolly / OASST -style single-turn instruction data."""

    description = (
        "Single-turn instruction-tuning datasets (Alpaca / Dolly / OASST): "
        "{instruction, input?, output} -> ContrastiveRecord with no native "
        "pair (a corruption strategy must be applied before discovery)."
    )

    @classmethod
    def fits(cls, raw: Any) -> bool:
        cols = _peek_columns(raw)
        if not cols:
            return False
        has_instr = any(k in cols for k in _INSTRUCTION_KEYS)
        has_response = any(k in cols for k in _RESPONSE_KEYS)
        # Conversational shape (messages list) wins over instruction.
        if "messages" in cols or "conversations" in cols:
            return False
        return has_instr and has_response

    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        instruction_template: str = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n{context_block}### Response:\n"
        ),
        first_token_only: bool = True,
        **_unused: Any,
    ) -> NormalizedDataset:
        cols = _peek_columns(raw)
        instr_key = next((k for k in _INSTRUCTION_KEYS if k in cols), None)
        ctx_key = next((k for k in _CONTEXT_KEYS if k in cols), None)
        resp_key = next((k for k in _RESPONSE_KEYS if k in cols), None)
        if not instr_key or not resp_key:
            raise ValueError(
                f"Instruction adapter requires one of {_INSTRUCTION_KEYS} "
                f"and one of {_RESPONSE_KEYS}; got cols={cols}"
            )

        records: List[ContrastiveRecord] = []
        for i, row in enumerate(_iter_rows(raw)):
            instr = (row.get(instr_key) or "").strip()
            ctx = (row.get(ctx_key) or "").strip() if ctx_key else ""
            resp = (row.get(resp_key) or "").strip()
            if not instr or not resp:
                continue
            ctx_block = f"### Input:\n{ctx}\n\n" if ctx else ""
            prompt = instruction_template.format(
                instruction=instr,
                context_block=ctx_block,
            )
            if first_token_only:
                first_word = resp.split()[0] if resp.split() else resp
                answer = " " + first_word
            else:
                answer = " " + resp
            records.append(
                ContrastiveRecord(
                    record_id=f"{i:06d}",
                    clean_prompt=prompt,
                    clean_answer=answer,
                    contrast_source=ContrastSource.NOT_PAIRED_YET,
                    meta={
                        "category": row.get("category"),
                        "has_context": bool(ctx),
                        "instruction_chars": len(instr),
                        "response_chars": len(resp),
                    },
                    target_field="first_response_token",
                )
            )
            if max_records and len(records) >= max_records:
                break

        return NormalizedDataset(
            name=name or "instruction",
            shape=DatasetShape.INSTRUCTION,
            records=records,
            source=source or "raw",
            meta={
                "instruction_key": instr_key,
                "context_key": ctx_key,
                "response_key": resp_key,
                "n_loaded": len(records),
            },
        )
