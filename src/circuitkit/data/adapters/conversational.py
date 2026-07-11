"""Conversational adapter — ShareGPT / OpenAI / multi-turn chat datasets.

Schema variants handled (auto-detected by column names):

  - ShareGPT (anon8231489123/ShareGPT_Vicuna_unfiltered, ShareGPT52K, etc.):
      {id, conversations: [{from: "human"|"gpt", value: "..."}, ...]}

  - OpenAI / HuggingFace chat messages format
    (Mistral, Llama-Instruct training data, etc.):
      {messages: [{role: "system"|"user"|"assistant", content: "..."}, ...]}

  - PIPPA-shareGPT, OpenChat, similar.

For circuit discovery we treat each (user-turn, assistant-response) pair
in the conversation as one ContrastiveRecord:
    clean_prompt   = full conversation history concatenated up to and
                     including the user's last message, in chat-template
                     form (Vicuna-style by default).
    clean_answer   = first token of the assistant's reply.
    corrupt_*      = NOT_PAIRED_YET (a CorruptionStrategy must fill them in).
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from .base import DataAdapter, register_adapter
from .pairwise import _iter_rows, _peek_columns

# Map ShareGPT "from" values to canonical roles.
_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def _vicuna_template(messages: List[dict]) -> str:
    """Render a list of {role, content} messages in Vicuna format up to the
    last user turn. The assistant's response is the target token to predict.
    """
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        if role == "system":
            lines.append(content)
        elif role == "user":
            lines.append(f"USER: {content}")
        elif role == "assistant":
            lines.append(f"ASSISTANT: {content}")
        else:
            lines.append(f"{role.upper()}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


@register_adapter(DatasetShape.CONVERSATIONAL)
class ConversationalAdapter(DataAdapter):
    """Adapter for ShareGPT / OpenAI messages format multi-turn data."""

    description = (
        "Multi-turn chat datasets (ShareGPT / OpenAI messages / OpenChat / "
        "PIPPA): each (history+user_turn, assistant_response) pair becomes "
        "a ContrastiveRecord. No native pair; apply a CorruptionStrategy "
        "before discovery."
    )

    @classmethod
    def fits(cls, raw: Any) -> bool:
        cols = _peek_columns(raw)
        if not cols:
            return False
        return ("conversations" in cols) or ("messages" in cols)

    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        max_turns_per_conv: int = 1,
        first_token_only: bool = True,
        chat_template: str = "vicuna",
        **_unused: Any,
    ) -> NormalizedDataset:
        cols = _peek_columns(raw)
        records: List[ContrastiveRecord] = []
        for i, row in enumerate(_iter_rows(raw)):
            convo = row.get("conversations") or row.get("messages") or []
            normalized: List[dict] = []
            for turn in convo:
                # ShareGPT-style {from, value}
                if "from" in turn and "value" in turn:
                    role = _SHAREGPT_ROLE_MAP.get(str(turn["from"]).lower(), "user")
                    content = str(turn.get("value") or "")
                # OpenAI-style {role, content}
                elif "role" in turn and "content" in turn:
                    role = str(turn["role"]).lower()
                    content = str(turn.get("content") or "")
                else:
                    continue
                normalized.append({"role": role, "content": content})

            # Yield one record per (history-up-to-user-turn, assistant-reply) pair.
            n_emitted = 0
            for idx, msg in enumerate(normalized):
                if msg["role"] != "assistant":
                    continue
                if idx == 0:
                    continue
                history = normalized[:idx]
                if history[-1]["role"] != "user":
                    continue
                if chat_template == "vicuna":
                    prompt = _vicuna_template(history)
                else:
                    raise ValueError(f"Unknown chat_template: {chat_template}")
                response = msg["content"].strip()
                if not response:
                    continue
                if first_token_only:
                    answer = " " + response.split()[0]
                else:
                    answer = " " + response
                records.append(
                    ContrastiveRecord(
                        record_id=f"{i:06d}-t{idx}",
                        clean_prompt=prompt,
                        clean_answer=answer,
                        contrast_source=ContrastSource.NOT_PAIRED_YET,
                        target_field="first_assistant_token",
                        meta={
                            "conv_id": row.get("id", i),
                            "turn_index": idx,
                            "history_turns": len(history),
                            "response_chars": len(response),
                        },
                    )
                )
                n_emitted += 1
                if n_emitted >= max_turns_per_conv:
                    break
                if max_records and len(records) >= max_records:
                    break
            if max_records and len(records) >= max_records:
                break

        return NormalizedDataset(
            name=name or "conversational",
            shape=DatasetShape.CONVERSATIONAL,
            records=records,
            source=source or "raw",
            meta={
                "chat_template": chat_template,
                "format_columns": [c for c in cols if c in ("conversations", "messages")],
                "n_loaded": len(records),
                "max_turns_per_conv": max_turns_per_conv,
            },
        )
