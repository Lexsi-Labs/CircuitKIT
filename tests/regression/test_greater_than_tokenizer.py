"""Regression test — greater_than task non-GPT-2 tokenizer bug.

Historical bug
--------------
The Greater-Than task assumed GPT-2-style BPE tokenization, where a number
with a leading space (``" 42"``) is a single token. Many SentencePiece-based
tokenizers (Llama-3, etc.) instead encode the *unspaced* form (``"42"``) as a
single token while ``" 42"`` splits into two. The task hard-coded the spaced
form, so on a non-GPT-2 tokenizer the answer-token ids were wrong (multi-token)
and discovery data was silently corrupted / empty.

The fix probes the tokenizer (``_number_prefix``) and picks whichever form —
spaced or unspaced — yields more single-token numbers.

This test fails if the tokenizer probe is removed and the spaced form is
re-hard-coded: a synthetic tokenizer whose single-token form is *unspaced*
must make the task select the ``""`` prefix.
"""

from __future__ import annotations

import pytest


class _FakeUnspacedTokenizer:
    """Tokenizer where '42' is one token but ' 42' splits — the Llama-style case."""

    def encode(self, text, add_special_tokens=False):  # noqa: D401
        if text.startswith(" "):
            # Leading-space form always splits into >= 2 tokens.
            return [0, 1]
        # Unspaced number string -> single token; everything else -> multi.
        if text.strip().isdigit():
            return [42]
        return [0, 1, 2]


class _FakeSpacedTokenizer:
    """GPT-2-style tokenizer: ' 42' is one token, '42' splits."""

    def encode(self, text, add_special_tokens=False):  # noqa: D401
        if text.startswith(" ") and text.strip().isdigit():
            return [99]
        if text.strip().isdigit():
            return [0, 1]
        return [0, 1, 2]


class _FakeModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        class _Cfg:
            model_name = "fake"

        self.cfg = _Cfg()


def test_number_prefix_picks_unspaced_for_sentencepiece_style_tokenizer():
    """A tokenizer that single-tokenizes '42' (not ' 42') must yield prefix ''."""
    from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec

    spec = GreaterThanTaskSpec()
    prefix = spec._number_prefix(_FakeModel(_FakeUnspacedTokenizer()))
    assert prefix == "", (
        "greater_than hard-coded the GPT-2 spaced number form — a non-GPT-2 "
        "tokenizer whose single-token numbers are unspaced was mishandled."
    )


def test_number_prefix_picks_spaced_for_gpt2_style_tokenizer():
    """A GPT-2-style tokenizer (' 42' single-token) must still yield prefix ' '."""
    from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec

    spec = GreaterThanTaskSpec()
    prefix = spec._number_prefix(_FakeModel(_FakeSpacedTokenizer()))
    assert prefix == " "


def test_greater_than_works_with_real_gpt2_tokenizer():
    """End-to-end-ish: number pool building must succeed on a real tokenizer."""
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer

    from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    spec = GreaterThanTaskSpec()

    prefix = spec._number_prefix(model)
    assert prefix in (" ", "")

    numbers = spec._get_single_token_numbers(model)
    # The probe must yield a usable pool of single-token numbers.
    assert len(numbers) >= 2
    # Each number, with the chosen prefix, must really be a single token.
    for n in numbers[:10]:
        toks = model.tokenizer.encode(f"{prefix}{n}", add_special_tokens=False)
        assert len(toks) == 1, f"number {n!r} is not single-token with prefix {prefix!r}"
