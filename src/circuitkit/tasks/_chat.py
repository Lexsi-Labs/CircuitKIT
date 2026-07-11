"""Chat-template handling for circuit discovery on instruction-tuned models.

Circuit discovery and every downstream stage (pruning, quantization export,
lm-eval benchmarking) must use the **same** prompt formatting. If discovery
sees raw text but the model is later used/evaluated with its chat template
(or vice versa), the discovered circuit is misattributed — it explains a
behavior on a prompt distribution the model is never actually run on.

Each task declares a ``chat_template_mode``:

* ``"auto"`` — wrap prompts in the model's chat template iff the model is an
  instruction-tuned / chat model (its tokenizer ships a ``chat_template``).
  This is the default for *downstream-behavior* tasks (MMLU, BoolQ, GSM8K,
  TruthfulQA, ...) and for custom user-defined tasks.
* ``"on"``  — always wrap. (Equivalent to a ``--use-chat-template`` flag.)
* ``"off"`` — never wrap. This is correct for *diagnostic minimal-pair*
  tasks (the IOI family, greater-than, SVA, ...), which the circuit-discovery
  literature studies raw, and for cloze tasks (WinoGrande) that have no
  user/assistant turn structure.

:func:`resolve_chat_template` collapses the declared mode against a concrete
model into a single boolean. That boolean is **not** persisted into the
discovery artifact — ``chat_template_mode`` is resolved fresh on every call
from the mode you pass and the model in hand, so discovery and every later
stage must be handed the *same* mode (and the same model type) to stay
consistent; a mismatch surfaces only at run time, not via stored metadata.

Known limitation of ``"auto"``
------------------------------
``"auto"`` detection keys off the tokenizer carrying a non-empty
``chat_template`` — the HuggingFace-standard signal (the same one
``lm-eval``'s ``--apply_chat_template`` uses). Name-matching (``"instruct"``
in the model name) is deliberately *not* used; it is fragile and not
authoritative. The one residual blind spot: a *base* (non-instruction-tuned)
model whose tokenizer ships a stray/default ``chat_template`` will be
``auto``-templated, which is wrong — base models should be discovered raw. An
empty or whitespace-only template is already guarded against, but a base
model carrying a real-looking stray template cannot be distinguished from a
genuine chat model by template presence alone. If you hit this, set
``chat_template_mode="off"`` explicitly (via the task spec, ``discovery_cfg``,
or ``--chat-template-mode``) to override ``auto``. The one-time log line
emitted when ``auto`` resolves to True makes such a misfire visible.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Accepted values for a task's ``chat_template_mode``.
VALID_MODES = ("auto", "on", "off")

#: Model/tokenizer names already logged for an ``auto`` chat-model detection,
#: so the informative log line is emitted at most once per name per process.
_AUTO_DETECTED_SEEN: set[str] = set()


def _has_chat_template(obj: Any) -> bool:
    """Return True iff ``obj`` carries a non-empty ``chat_template``.

    A ``chat_template`` of ``None``, ``""`` or whitespace-only text does not
    count: some tokenizers ship a stray/trivial value that must not be
    mistaken for a genuine chat model.
    """
    template = getattr(obj, "chat_template", None)
    return isinstance(template, str) and template.strip() != ""


def _log_auto_detection(name: str) -> None:
    """Emit the one-time ``auto``-detected log line for ``name``."""
    if name in _AUTO_DETECTED_SEEN:
        return
    _AUTO_DETECTED_SEEN.add(name)
    logger.info(
        "chat_template_mode=auto: detected chat model %r, "
        "wrapping discovery prompts in its chat template",
        name,
    )


def model_is_chat(model: Any) -> bool:
    """Return True iff ``model`` is an instruction-tuned / chat model.

    The reliable signal is the tokenizer carrying a non-empty ``chat_template``
    — far more robust than matching ``"instruct"`` in the model name. An empty
    or whitespace-only template is treated as *not* a chat model.
    """
    tok = getattr(model, "tokenizer", None)
    return _has_chat_template(tok)


def resolve_chat_template(mode: str, model: Any) -> bool:
    """Collapse a declared ``chat_template_mode`` against a concrete model.

    Args:
        mode: one of :data:`VALID_MODES`.
        model: a HookedTransformer (or anything exposing ``.tokenizer``).

    Returns:
        ``True`` if task prompts should be wrapped in the model's chat
        template, ``False`` for raw text.

    Raises:
        ValueError: if ``mode`` is not a recognized value.

    Note:
        Under ``"auto"`` a *base* model whose tokenizer ships a stray
        ``chat_template`` is indistinguishable from a genuine chat model and
        will be templated. Set ``chat_template_mode="off"`` (task spec,
        ``discovery_cfg``, or ``--chat-template-mode``) to override.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"chat_template_mode must be one of {VALID_MODES}, got {mode!r}")
    if mode == "on":
        return True
    if mode == "off":
        return False
    # "auto"
    detected = model_is_chat(model)
    if detected:
        name = getattr(getattr(model, "cfg", None), "model_name", None)
        if not name:
            name = getattr(getattr(model, "tokenizer", None), "name_or_path", None)
        _log_auto_detection(name or "<unknown>")
    return detected


def wrap_prompt(
    model: Any,
    user_text: str,
    assistant_prefix: str = "",
    *,
    apply: bool,
) -> str:
    """Format one task prompt for ``model``.

    Args:
        model: a HookedTransformer (or anything exposing ``.tokenizer``).
        user_text: the task prompt that belongs in the user turn.
        assistant_prefix: text placed at the *start of the assistant turn*,
            i.e. immediately after the generation prompt. Use this to keep the
            answer the immediate next token (e.g. ``"The answer is"``) so the
            single-token logit-difference metric stays valid — a chat model
            asked a bare question otherwise emits a verbose reply.
        apply: the resolved boolean from :func:`resolve_chat_template`.

    Returns:
        The string to tokenize. When ``apply`` is False this is plain
        ``user_text + assistant_prefix`` — the legacy raw-text behavior, so
        base models and ``"off"`` tasks are completely unaffected.

    Note:
        Contrastive (clean / corrupted) pairs stay token-aligned as long as
        both are wrapped through this function with the *same*
        ``assistant_prefix``: the template adds an identical prefix/suffix to
        each, so they still differ only in the original differing span.
    """
    if not apply:
        return user_text + assistant_prefix
    tok = model.tokenizer
    templated = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return templated + assistant_prefix


def tokenizer_is_chat(tokenizer: Any) -> bool:
    """Return True iff ``tokenizer`` belongs to an instruction-tuned / chat model.

    The finetuning path is handed a bare HuggingFace tokenizer (no
    HookedTransformer), so it cannot use :func:`model_is_chat`. The same signal
    applies directly: a tokenizer that ships a non-empty ``chat_template`` is a
    chat model's tokenizer. An empty or whitespace-only template is treated as
    *not* a chat model.
    """
    return _has_chat_template(tokenizer)


def resolve_chat_template_from_tokenizer(mode: str, tokenizer: Any) -> bool:
    """Collapse a declared ``chat_template_mode`` against a bare tokenizer.

    The tokenizer-only analogue of :func:`resolve_chat_template`, used by
    ``build_finetuning_dataset`` (which receives a tokenizer, not a model). The
    resolved boolean must match what discovery resolved for the same model so
    circuit-tuning trains on the same prompt distribution discovery used.

    Args:
        mode: one of :data:`VALID_MODES`.
        tokenizer: a HuggingFace tokenizer.

    Returns:
        ``True`` if task prompts should be wrapped in the model's chat
        template, ``False`` for raw text.

    Raises:
        ValueError: if ``mode`` is not a recognized value.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"chat_template_mode must be one of {VALID_MODES}, got {mode!r}")
    if mode == "on":
        return True
    if mode == "off":
        return False
    # "auto"
    detected = tokenizer_is_chat(tokenizer)
    if detected:
        name = getattr(tokenizer, "name_or_path", None)
        _log_auto_detection(name or "<unknown>")
    return detected


def wrap_prompt_with_tokenizer(
    tokenizer: Any,
    user_text: str,
    assistant_prefix: str = "",
    *,
    apply: bool,
) -> str:
    """Format one task prompt using a bare tokenizer.

    The tokenizer-only analogue of :func:`wrap_prompt`, used by
    ``build_finetuning_dataset``. Wrapping the finetuning prompt with the SAME
    ``assistant_prefix`` the task uses at discovery time keeps the circuit-tuning
    text byte-identical to the discovery text.

    Args:
        tokenizer: a HuggingFace tokenizer.
        user_text: the task prompt that belongs in the user turn.
        assistant_prefix: text placed at the start of the assistant turn, i.e.
            immediately after the generation prompt — pass the same value the
            task uses at discovery time.
        apply: the resolved boolean from :func:`resolve_chat_template_from_tokenizer`.

    Returns:
        The string to tokenize. When ``apply`` is False this is plain
        ``user_text + assistant_prefix`` — the legacy raw-text behavior, so
        base models and ``"off"`` tasks are completely unaffected.
    """
    if not apply:
        return user_text + assistant_prefix
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return templated + assistant_prefix


def to_tokens(model: Any, text: str, *, templated: bool):
    """Tokenize ``text`` for ``model`` with BOS handled correctly.

    A chat template renders its own beginning-of-text token into the string
    (``<|begin_of_text|>``, ``<bos>``, ...). Tokenizing that with
    ``prepend_bos=True`` would inject a *second* BOS, shifting every position
    by one and corrupting attribution. So templated text is tokenized with
    ``prepend_bos=False``; raw text keeps the usual ``prepend_bos=True``.
    """
    return model.to_tokens(text, prepend_bos=not templated)
