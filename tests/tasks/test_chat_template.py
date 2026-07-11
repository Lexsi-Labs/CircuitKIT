"""
Test suite for chat-template handling (circuitkit.tasks._chat).

Tests the pure helpers without requiring a real model:
- resolve_chat_template across all three modes (incl. "auto" both ways)
- ValueError on an unrecognized mode
- wrap_prompt no-op when apply=False
- wrap_prompt invokes apply_chat_template when apply=True
- to_tokens passes prepend_bos correctly for templated / raw text
"""

import pytest

from circuitkit.tasks._chat import (
    VALID_MODES,
    model_is_chat,
    resolve_chat_template,
    to_tokens,
    wrap_prompt,
)

# ===== Fixtures =====


class FakeTokenizer:
    """Minimal tokenizer stand-in with an optional chat_template."""

    def __init__(self, chat_template=None):
        if chat_template is not None:
            self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Render a deterministic, recognizable templated string.
        user = messages[0]["content"]
        return f"<bos><user>{user}</user><assistant>"


class FakeModel:
    """Minimal model stand-in exposing .tokenizer and .to_tokens."""

    def __init__(self, chat_template=None):
        self.tokenizer = FakeTokenizer(chat_template=chat_template)
        self.to_tokens_calls = []

    def to_tokens(self, text, prepend_bos=True):
        self.to_tokens_calls.append((text, prepend_bos))
        return text  # identity is enough to inspect call args


@pytest.fixture
def chat_model():
    """An instruction-tuned model: tokenizer carries a chat_template."""
    return FakeModel(chat_template="{{ messages }}")


@pytest.fixture
def base_model():
    """A base model: tokenizer has no chat_template."""
    return FakeModel(chat_template=None)


# ===== model_is_chat =====


def test_model_is_chat_true(chat_model):
    assert model_is_chat(chat_model) is True


def test_model_is_chat_false(base_model):
    assert model_is_chat(base_model) is False


def test_model_is_chat_empty_template_is_false():
    """A stray empty/whitespace chat_template must not count as a chat model."""
    assert model_is_chat(FakeModel(chat_template="")) is False
    assert model_is_chat(FakeModel(chat_template="   \n\t ")) is False


def test_resolve_auto_empty_template_is_false():
    """'auto' treats an empty chat_template as a base model (no wrapping)."""
    assert resolve_chat_template("auto", FakeModel(chat_template="")) is False


# ===== resolve_chat_template =====


def test_resolve_on_always_true(base_model, chat_model):
    assert resolve_chat_template("on", base_model) is True
    assert resolve_chat_template("on", chat_model) is True


def test_resolve_off_always_false(base_model, chat_model):
    assert resolve_chat_template("off", base_model) is False
    assert resolve_chat_template("off", chat_model) is False


def test_resolve_auto_chat_model(chat_model):
    """'auto' wraps when the model is instruction-tuned."""
    assert resolve_chat_template("auto", chat_model) is True


def test_resolve_auto_base_model(base_model):
    """'auto' does not wrap for a base model."""
    assert resolve_chat_template("auto", base_model) is False


def test_resolve_bad_mode_raises(base_model):
    with pytest.raises(ValueError):
        resolve_chat_template("sometimes", base_model)


def test_valid_modes_constant():
    assert VALID_MODES == ("auto", "on", "off")


# ===== wrap_prompt =====


def test_wrap_prompt_noop_when_apply_false(base_model):
    """apply=False is the legacy raw-text path: plain concatenation."""
    out = wrap_prompt(base_model, "What is 2+2?", apply=False)
    assert out == "What is 2+2?"


def test_wrap_prompt_noop_with_assistant_prefix(base_model):
    out = wrap_prompt(base_model, "What is 2+2?", " The answer is", apply=False)
    assert out == "What is 2+2? The answer is"


def test_wrap_prompt_applies_chat_template(chat_model):
    """apply=True routes the prompt through apply_chat_template."""
    out = wrap_prompt(chat_model, "What is 2+2?", apply=True)
    assert out == "<bos><user>What is 2+2?</user><assistant>"


def test_wrap_prompt_applies_template_with_prefix(chat_model):
    """assistant_prefix is appended after the generation prompt."""
    out = wrap_prompt(chat_model, "What is 2+2?", " The answer is", apply=True)
    assert out == "<bos><user>What is 2+2?</user><assistant> The answer is"


# ===== to_tokens =====


def test_to_tokens_raw_prepends_bos(base_model):
    """Raw text keeps the usual prepend_bos=True."""
    to_tokens(base_model, "hello", templated=False)
    assert base_model.to_tokens_calls == [("hello", True)]


def test_to_tokens_templated_skips_bos(chat_model):
    """Templated text already carries its own BOS, so prepend_bos=False."""
    to_tokens(chat_model, "<bos>hello", templated=True)
    assert chat_model.to_tokens_calls == [("<bos>hello", False)]
