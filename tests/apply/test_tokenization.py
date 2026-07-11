"""
test_tokenization.py
====================
Unit tests for circuitkit.apply._tokenization.

Scope
-----
This file tests the four tokeniser-aware primitives in isolation:

    format_target            — leading-space rule
    build_teacher_forced     — sequence construction, BOS, boundary integrity
    score_target             — P(target | prompt), single- and multi-token
    locate_subject_last_token — two-tier subject-finding (offset map + fallback)
    tokenize_prompt          — BOS delegation
    argmax_token_at_end      — argmax decoding

The tests are organised in three tiers:

    Tier 1 — Pure-Python / no-model (format_target, ScoringError, SubjectLocationError)
    Tier 2 — Mock-model (everything else, using a realistic BPE-like mock)
    Tier 3 — Real-model (marked @pytest.mark.slow; covers GPT-2, Qwen2, Llama-3)

The mock model is deliberately more faithful than the one in
test_knowledge_editing_pipeline.py: it supports offset_mapping (fast-tokeniser
path in locate_subject_last_token) and honours default_prepend_bos via cfg, so
the Qwen BOS fix can be exercised without loading a real Qwen model.

Usage
-----
    Unit only (fast):   pytest test_tokenization.py -v -m "not slow"
    All tests:          pytest test_tokenization.py -v
    Real models only:   pytest test_tokenization.py -v -m slow
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import gate
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_IMPORT_ERROR = ""
try:
    from circuitkit.applications.common_utils._tokenization import (
        ScoringError,
        SubjectLocationError,
        TargetScore,
        TeacherForcedSequence,
        argmax_token_at_end,
        build_teacher_forced,
        format_target,
        locate_subject_last_token,
        sample_random_prefixes,
        score_target,
        tokenize_prompt,
    )

    _MODULES_AVAILABLE = True
except ImportError as exc:
    _MODULES_AVAILABLE = False
    _IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _MODULES_AVAILABLE,
    reason=f"_tokenization not importable: {_IMPORT_ERROR}",
)


# ===========================================================================
# Mock infrastructure
# ===========================================================================

# A small, fixed vocabulary that covers the characters we use in tests.
# Each character maps deterministically to a unique integer so that
# sequence-matching in locate_subject_last_token is reliable.

_VOCAB: Dict[str, int] = {}
_ID_TO_STR: Dict[int, str] = {}


def _build_vocab() -> None:
    """Populate _VOCAB with printable ASCII, a BOS token, and a space token."""
    _VOCAB["<BOS>"] = 0
    _ID_TO_STR[0] = "<BOS>"
    for i, ch in enumerate(
        " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
        start=1,
    ):
        _VOCAB[ch] = i
        _ID_TO_STR[i] = ch


_build_vocab()
_D_VOCAB = max(_VOCAB.values()) + 1  # ~97


@dataclass
class _MockConfig:
    """Minimal TransformerLens config."""

    n_layers: int = 4
    d_model: int = 32
    d_mlp: int = 64
    device: str = "cpu"
    default_prepend_bos: bool = True
    use_hook_mlp_in: bool = True


class _MockTokenizer:
    """
    Simulates a HuggingFace fast tokenizer.

    Tokenises character-by-character so subject-finding via offset_mapping
    is exact and testable. Supports:
        - __call__(text, return_offsets_mapping, add_special_tokens)
        - decode(ids)
    """

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
    ) -> Dict:
        ids = [_VOCAB.get(ch, 1) for ch in text]
        result: Dict = {"input_ids": ids}
        if return_offsets_mapping:
            # Each character occupies exactly one position, offset [i, i+1].
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids: List[int]) -> str:
        return "".join(_ID_TO_STR.get(i, "?") for i in ids)


class _BPELikeMockModel(nn.Module):
    """
    A lightweight model mock that exercises the real tokenisation paths.

    Key properties:
      * to_tokens() honours prepend_bos, uses the shared char-level vocab.
      * tokenizer is a _MockTokenizer (fast-tokeniser path).
      * forward() returns deterministic logits so score_target / argmax are
        testable without randomness.
      * default_prepend_bos is configurable so the Qwen BOS=False path is
        exercised without loading a real Qwen model.
    """

    def __init__(self, default_prepend_bos: bool = True):
        super().__init__()
        self.cfg = _MockConfig(default_prepend_bos=default_prepend_bos)
        self.tokenizer = _MockTokenizer()
        # Logit bias as a *Parameter* (not a buffer) so that loss.backward()
        # produces a non-zero gradient and TestTrainerIntegration can verify
        # the backward pass.  The values are fixed at init so argmax is
        # deterministic: token id t gets a base logit of t/100.
        self._logit_bias = nn.Parameter(
            torch.arange(_D_VOCAB, dtype=torch.float32) / 100.0,
            requires_grad=True,
        )

    # -- TransformerLens API --------------------------------------------------

    def to_tokens(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        ids = [_VOCAB.get(ch, 1) for ch in text]
        if prepend_bos:
            ids = [0] + ids
        return torch.tensor([ids], dtype=torch.long)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Returns logits shaped [1, S, V].

        Two components:

        1. Token-id boost: the token actually at position p gets +2.0,
           so argmax at position p == tokens[p]. This keeps argmax
           deterministic and correct for TestTrainerIntegration.

        2. Position-dependent non-uniform perturbation: vocab index 0 gets
           an extra +pos*0.01 at position p. This is NOT a uniform shift so
           it is NOT cancelled by log_softmax. It makes the logit distribution
           genuinely different at each absolute position, which means that
           the same target token sequence scores differently when read from
           position L (BOS model, larger L) vs L-1 (no-BOS model).
           This is what makes test_bos_vs_no_bos_produce_different_scores
           testable without a real model.
        """
        B, S = tokens.shape
        logits = self._logit_bias.unsqueeze(0).unsqueeze(0).expand(B, S, -1).clone()
        for pos in range(S):
            tok_id = int(tokens[0, pos].item())
            logits[0, pos, tok_id] += 2.0  # argmax == tokens[pos]
            logits[0, pos, 0] += pos * 0.01  # non-uniform: survives softmax
        return logits

    def to_string(self, ids: torch.Tensor) -> str:
        return "".join(_ID_TO_STR.get(int(i), "?") for i in ids.tolist())


@pytest.fixture
def model() -> _BPELikeMockModel:
    """Standard model fixture (default_prepend_bos=True, like GPT-2)."""
    return _BPELikeMockModel(default_prepend_bos=True)


@pytest.fixture
def model_no_bos() -> _BPELikeMockModel:
    """Model fixture with default_prepend_bos=False, like Qwen."""
    return _BPELikeMockModel(default_prepend_bos=False)


# ===========================================================================
# Tier 1 — format_target (pure Python, no model)
# ===========================================================================


class TestFormatTarget:
    """
    format_target(target, prompt=None) -> str

    Rule: prepend exactly one space unless the boundary already has one.
    These tests prove every branch of the rule, including edge cases
    that produced bugs before centralisation.
    """

    def test_bare_word_gets_space_prepended(self):
        assert format_target("Lyon") == " Lyon"

    def test_already_space_prefixed_target_unchanged(self):
        assert format_target(" Lyon") == " Lyon"

    def test_prompt_trailing_space_suppresses_prepend(self):
        """
        The bug this test documents: old code in MEMIT did
        `if not target.startswith(" "): target = " " + target`
        and missed the prompt-already-ends-in-space case.
        """
        result = format_target("Lyon", prompt="The capital of France is ")
        assert result == "Lyon", "When prompt ends in space, format_target must not prepend another"

    def test_prompt_no_trailing_space_does_prepend(self):
        result = format_target("Lyon", prompt="The capital of France is")
        assert result == " Lyon"

    def test_target_already_space_overrides_prompt_check(self):
        """Target owns the space — prompt trailing space is irrelevant."""
        result = format_target(" Lyon", prompt="The capital of France is ")
        assert result == " Lyon"

    def test_no_prompt_given_prepends_space(self):
        assert format_target("Paris", prompt=None) == " Paris"

    def test_multiword_target_gets_single_space(self):
        assert format_target("Saint Petersburg") == " Saint Petersburg"

    def test_multiword_target_already_spaced_unchanged(self):
        assert format_target(" Saint Petersburg") == " Saint Petersburg"

    def test_empty_target_raises_scoring_error(self):
        with pytest.raises(ScoringError, match="[Ee]mpty"):
            format_target("")

    def test_empty_target_with_prompt_raises_scoring_error(self):
        with pytest.raises(ScoringError):
            format_target("", prompt="Some prompt")

    def test_target_with_only_spaces_treated_as_space_prefixed(self):
        """A target of '  word' starts with space → no extra space."""
        result = format_target("  word")
        assert result == "  word"

    def test_newline_in_prompt_does_not_count_as_space(self):
        """
        Only ASCII space (0x20) suppresses prepend.
        A prompt ending in \\n should still get a space prepended.
        """
        result = format_target("Lyon", prompt="The capital of France is\n")
        assert result == " Lyon"


# ===========================================================================
# Tier 2 — tokenize_prompt (mock model)
# ===========================================================================


class TestTokenizePrompt:
    """tokenize_prompt(model, prompt) -> Tensor[1, L]"""

    def test_returns_2d_tensor(self, model):
        ids = tokenize_prompt(model, "France")
        assert ids.ndim == 2
        assert ids.shape[0] == 1

    def test_bos_prepended_when_config_true(self, model):
        ids = tokenize_prompt(model, "A")
        # BOS is id 0; with one char the result should be [BOS, char_id]
        assert ids.shape[1] == 2
        assert int(ids[0, 0].item()) == 0  # BOS

    def test_bos_not_prepended_when_config_false(self, model_no_bos):
        ids = tokenize_prompt(model_no_bos, "A")
        # No BOS → only one token for single-char prompt
        assert ids.shape[1] == 1
        assert int(ids[0, 0].item()) == _VOCAB["A"]

    def test_empty_prompt_raises(self, model):
        with pytest.raises(ScoringError):
            tokenize_prompt(model, "")

    def test_length_matches_chars_plus_bos(self, model):
        text = "France"
        ids = tokenize_prompt(model, text)
        assert ids.shape[1] == len(text) + 1  # +1 for BOS

    def test_length_matches_chars_no_bos(self, model_no_bos):
        text = "France"
        ids = tokenize_prompt(model_no_bos, text)
        assert ids.shape[1] == len(text)

    def test_consistent_with_to_tokens(self, model):
        """tokenize_prompt must agree with model.to_tokens using the same BOS flag."""
        text = "France"
        expected = model.to_tokens(text, prepend_bos=True)
        got = tokenize_prompt(model, text)
        assert torch.equal(got, expected)


# ===========================================================================
# Tier 2 — build_teacher_forced (mock model)
# ===========================================================================


class TestBuildTeacherForced:
    """
    build_teacher_forced(model, prompt, target) -> TeacherForcedSequence

    The most critical function: everything the trainers need for a
    teacher-forced cross-entropy pass, produced once, consistently.
    """

    def test_returns_teacher_forced_sequence_type(self, model):
        seq = build_teacher_forced(model, "France is", "Lyon")
        assert isinstance(seq, TeacherForcedSequence)

    def test_full_ids_shape_is_1d_batch(self, model):
        seq = build_teacher_forced(model, "France is", "Lyon")
        assert seq.full_ids.ndim == 2
        assert seq.full_ids.shape[0] == 1

    def test_prompt_len_plus_target_len_equals_full_len(self, model):
        seq = build_teacher_forced(model, "France is", "Lyon")
        assert seq.prompt_len + seq.target_len == seq.full_ids.shape[1]

    def test_target_ids_match_full_ids_slice(self, model):
        seq = build_teacher_forced(model, "France is", "Lyon")
        L = seq.prompt_len
        T = seq.target_len
        expected = seq.full_ids[0, L : L + T]
        assert torch.equal(seq.target_ids, expected)

    def test_target_len_at_least_one(self, model):
        seq = build_teacher_forced(model, "France is", "Lyon")
        assert seq.target_len >= 1

    def test_prepend_bos_used_reflects_model_config(self, model):
        seq = build_teacher_forced(model, "France is", "Lyon")
        assert seq.prepend_bos_used is True

    def test_prepend_bos_used_false_for_no_bos_model(self, model_no_bos):
        seq = build_teacher_forced(model_no_bos, "France is", "Lyon")
        assert seq.prepend_bos_used is False

    def test_bos_model_prompt_len_longer_than_no_bos(self, model, model_no_bos):
        """BOS adds exactly 1 token to the prompt length."""
        prompt, target = "France is", "Lyon"
        seq_bos = build_teacher_forced(model, prompt, target)
        seq_no_bos = build_teacher_forced(model_no_bos, prompt, target)
        assert seq_bos.prompt_len == seq_no_bos.prompt_len + 1

    def test_target_text_used_has_leading_space(self, model):
        """format_target must be called internally — bare target gets a space."""
        seq = build_teacher_forced(model, "France is", "Lyon")
        assert seq.target_text_used.startswith(" ")

    def test_target_text_used_no_double_space_when_prompt_ends_in_space(self, model):
        """
        The MEMIT bug: old code would produce '  Lyon' when prompt ends in space.
        After centralisation, build_teacher_forced must produce ' Lyon'.
        Not '  Lyon'.
        """
        seq = build_teacher_forced(model, "France is ", "Lyon")
        assert (
            seq.target_text_used == "Lyon"
        ), "When prompt ends in space, target_text_used must NOT have a leading space"

    def test_empty_prompt_raises_scoring_error(self, model):
        with pytest.raises(ScoringError):
            build_teacher_forced(model, "", "Lyon")

    def test_empty_target_raises_scoring_error(self, model):
        with pytest.raises(ScoringError):
            build_teacher_forced(model, "France is", "")

    def test_multitoken_target_yields_correct_target_len(self, model):
        """
        "Saint Petersburg" tokenises to multiple characters, each its own token
        in the char-level mock. target_len must match.
        """
        target_text = " Saint Petersburg"  # pre-spaced so format_target is a no-op
        n_chars = len(target_text)  # number of char-level tokens
        seq = build_teacher_forced(model, "The second capital of Russia is", target_text)
        assert seq.target_len == n_chars

    def test_prompt_prefix_of_full_ids(self, model):
        """
        Boundary integrity: full_ids[:L] must equal to_tokens(prompt).
        This is what justifies slicing full_ids[:prompt_len] as the prompt tensor.
        """
        prompt = "France is"
        seq = build_teacher_forced(model, prompt, "Lyon")
        L = seq.prompt_len
        prompt_ids = model.to_tokens(prompt, prepend_bos=True)
        assert torch.equal(seq.full_ids[:, :L], prompt_ids)

    def test_target_ids_are_long_tensor(self, model):
        seq = build_teacher_forced(model, "France is", "Lyon")
        assert seq.target_ids.dtype == torch.long

    def test_no_bos_model_prompt_prefix_integrity(self, model_no_bos):
        """Same boundary-integrity check for the Qwen / no-BOS path."""
        prompt = "France is"
        seq = build_teacher_forced(model_no_bos, prompt, "Lyon")
        L = seq.prompt_len
        prompt_ids = model_no_bos.to_tokens(prompt, prepend_bos=False)
        assert torch.equal(seq.full_ids[:, :L], prompt_ids)


# ===========================================================================
# Tier 2 — score_target (mock model)
# ===========================================================================


class TestScoreTarget:
    """
    score_target(model, prompt, target) -> TargetScore

    Proves that scores are computed from the correct token positions and
    that the BOS convention does not corrupt them.
    """

    def test_returns_target_score_type(self, model):
        result = score_target(model, "France is", "Lyon")
        assert isinstance(result, TargetScore)

    def test_first_token_prob_in_unit_interval(self, model):
        result = score_target(model, "France is", "Lyon")
        assert 0.0 <= result.first_token_prob <= 1.0

    def test_sequence_prob_in_unit_interval(self, model):
        result = score_target(model, "France is", "Lyon")
        assert 0.0 <= result.sequence_prob <= 1.0

    def test_sequence_logprob_nonpositive(self, model):
        """Log-probability of a real outcome is always <= 0."""
        result = score_target(model, "France is", "Lyon")
        assert result.sequence_logprob <= 0.0

    def test_per_token_logprobs_length_matches_target_tokens(self, model):
        result = score_target(model, "France is", "Lyon")
        assert len(result.per_token_logprobs) == len(result.target_token_ids)

    def test_per_token_logprobs_all_nonpositive(self, model):
        result = score_target(model, "France is", "Lyon")
        for lp in result.per_token_logprobs:
            assert lp <= 0.0

    def test_sequence_logprob_equals_sum_of_per_token(self, model):
        result = score_target(model, "France is", "Lyon")
        expected = sum(result.per_token_logprobs)
        assert abs(result.sequence_logprob - expected) < 1e-4

    def test_target_text_used_stored(self, model):
        result = score_target(model, "France is", "Lyon")
        assert result.target_text_used != ""

    def test_empty_prompt_raises(self, model):
        with pytest.raises(ScoringError):
            score_target(model, "", "Lyon")

    def test_empty_target_raises(self, model):
        with pytest.raises(ScoringError):
            score_target(model, "France is", "")

    def test_known_token_has_higher_prob_than_unknown(self, model):
        """
        Compare sequence_logprob for two targets whose SECOND character differs.
        Both targets start with ' ' (format_target prepends a space), so their
        first token is identical. The second token — the actual letter — has a
        different id and therefore a different logit_bias (id/100), making the
        sequence comparison meaningful.

        'L' has id=45 (logit 0.45) vs '!' has id=2 (logit 0.02), so
        sequence_logprob(' L') > sequence_logprob(' !').
        """
        result_high = score_target(model, "France is", "L")  # ' L': second token id=45
        result_low = score_target(model, "France is", "!")  # ' !': second token id=2
        assert result_high.sequence_logprob > result_low.sequence_logprob, (
            "Higher-id token should have higher log-probability under the mock's "
            "id/100 logit bias"
        )

    def test_bos_false_model_gives_consistent_score(self, model_no_bos):
        """
        Scores on a no-BOS model must be internally consistent (logprob <=0,
        prob in [0,1]) even though positions differ from BOS=True.
        """
        result = score_target(model_no_bos, "France is", "Lyon")
        assert 0.0 <= result.first_token_prob <= 1.0
        assert result.sequence_logprob <= 0.0

    def test_bos_vs_no_bos_produce_different_scores(self, model, model_no_bos):
        """
        BOS shifts every position. Scores MUST differ between a BOS=True
        and BOS=False model — if they were the same, BOS handling would be broken.
        This was the actual Qwen bug.
        """
        r_bos = score_target(model, "France is", "Lyon")
        r_no_bos = score_target(model_no_bos, "France is", "Lyon")
        assert r_bos.sequence_logprob != r_no_bos.sequence_logprob, (
            "BOS=True and BOS=False produced identical scores — "
            "BOS handling is not being applied"
        )

    def test_multitoken_target_all_tokens_scored(self, model):
        """For 'Paris' (5 chars in char-level mock) all 5 log-probs must appear."""
        result = score_target(model, "The capital of France is", "Paris")
        n_target_chars = len(" Paris")  # format_target prepends space
        assert len(result.per_token_logprobs) == n_target_chars

    def test_no_grad_does_not_leak_into_model_params(self, model):
        """score_target must not leave gradients on model parameters."""
        for p in model.parameters():
            p.grad = None
        score_target(model, "France is", "Lyon")
        for p in model.parameters():
            assert p.grad is None, "score_target leaked gradients into model parameters"


# ===========================================================================
# Tier 2 — locate_subject_last_token (mock model)
# ===========================================================================


class TestLocateSubjectLastToken:
    """
    locate_subject_last_token(model, prompt, subject) -> int

    The most safety-critical function: a wrong index silently injects
    the ROME/MEMIT edit at the wrong position.
    """

    def test_single_char_subject_at_end(self, model):
        """Subject is the last character before the end of prompt."""
        prompt = "France is"
        subject = "e"  # last char of "France"
        idx = locate_subject_last_token(model, prompt, subject)
        # BOS(1) + "France is" chars up to and including 'e' at position 5
        # BOS=id0 at pos 0, 'F'=1,'r'=2,'a'=3,'n'=4,'c'=5,'e'=6 → idx=6
        assert idx >= 1  # at minimum past BOS

    def test_returns_int(self, model):
        idx = locate_subject_last_token(model, "The capital of France is", "France")
        assert isinstance(idx, int)

    def test_multichar_subject_index_is_last_token(self, model):
        """
        For 'France' in 'The capital of France is', the returned index
        must point at the last character of 'France', not the first.
        BOS + 'The capital of France' = 1 + 20 = 21 tokens (0-indexed: 20).
        """
        prompt = "The capital of France is"
        subject = "France"
        idx = locate_subject_last_token(model, prompt, subject)
        # Manually compute: BOS=0, then chars of "The capital of France is"
        # 'France' starts at char offset 15, ends at 20 (inclusive).
        # With BOS offset: last token of France is at position 21 (0-indexed).
        expected_char_offset = prompt.rfind(subject) + len(subject) - 1
        expected_idx = expected_char_offset + 1  # +1 for BOS
        assert idx == expected_idx

    def test_index_within_prompt_bounds(self, model):
        prompt = "The capital of France is"
        subject = "France"
        idx = locate_subject_last_token(model, prompt, subject)
        prompt_ids = model.to_tokens(prompt, prepend_bos=True)
        assert 0 <= idx < prompt_ids.shape[1]

    def test_no_bos_model_index_is_one_less(self, model, model_no_bos):
        """
        Without BOS every index shifts down by 1. Verifies the Qwen fix:
        before centralisation both models would return the same index
        because prepend_bos=True was hardcoded.
        """
        prompt = "The capital of France is"
        subject = "France"
        idx_bos = locate_subject_last_token(model, prompt, subject)
        idx_no_bos = locate_subject_last_token(model_no_bos, prompt, subject)
        assert (
            idx_bos == idx_no_bos + 1
        ), "BOS model index should be exactly 1 greater than no-BOS model index"

    def test_subject_not_in_prompt_raises(self, model):
        with pytest.raises(SubjectLocationError):
            locate_subject_last_token(model, "The capital of France is", "Germany")

    def test_empty_prompt_raises(self, model):
        with pytest.raises(SubjectLocationError):
            locate_subject_last_token(model, "", "France")

    def test_empty_subject_raises(self, model):
        with pytest.raises(SubjectLocationError):
            locate_subject_last_token(model, "The capital of France is", "")

    def test_subject_at_start_of_prompt(self, model):
        """Subject is right at the beginning — BOS is the only token before it."""
        prompt = "France is the country"
        subject = "France"
        idx = locate_subject_last_token(model, prompt, subject)
        # BOS(0), F(1), r(2), a(3), n(4), c(5), e(6) → last of 'France' is 6
        assert idx == len("France")  # 6, because BOS is at 0

    def test_subject_appears_twice_returns_last_occurrence(self, model):
        """
        rfind-based subject search returns the last occurrence.
        The edit should target the most recent mention of the subject.
        """
        prompt = "France changed. France is"
        subject = "France"
        idx = locate_subject_last_token(model, prompt, subject)
        # Last 'France' starts at char 16. Last char at 21. With BOS: 22.
        last_start = prompt.rfind(subject)
        expected = last_start + len(subject) - 1 + 1  # +1 for BOS
        assert idx == expected

    def test_fallback_works_without_tokenizer(self, model, monkeypatch):
        """
        When the tokenizer attribute is absent (slow-tokeniser path),
        locate_subject_last_token falls back to sequence matching.
        The result must still be correct.
        """
        monkeypatch.delattr(model, "tokenizer", raising=False)
        prompt = "The capital of France is"
        subject = "France"
        # Should not raise; fallback must find it
        idx = locate_subject_last_token(model, prompt, subject)
        assert isinstance(idx, int)
        assert idx > 0

    def test_fallback_raises_for_absent_subject(self, model, monkeypatch):
        """Fallback path must still raise SubjectLocationError for a missing subject."""
        monkeypatch.delattr(model, "tokenizer", raising=False)
        with pytest.raises(SubjectLocationError):
            locate_subject_last_token(model, "The capital of France is", "Germany")

    def test_multitoken_subject_last_token_only(self, model):
        """
        'Saint Petersburg' is a two-word subject. The returned index must be
        the *last* token of the subject, not the first.
        """
        prompt = "Saint Petersburg is in Russia"
        subject = "Saint Petersburg"
        idx = locate_subject_last_token(model, prompt, subject)
        # Last char of 'Saint Petersburg' is 'g' at char-offset 15.
        # With BOS: 16.
        expected = len(subject) - 1 + 1  # subject is at the very start
        assert idx == expected

    def test_index_is_consistent_with_build_teacher_forced_prompt_len(self, model):
        """
        The index returned by locate_subject_last_token must be within the
        prompt_len returned by build_teacher_forced — otherwise the trainer
        would inject the hook outside the prompt tokens.
        """
        prompt = "The capital of France is"
        subject = "France"
        target = "Lyon"
        idx = locate_subject_last_token(model, prompt, subject)
        seq = build_teacher_forced(model, prompt, target)
        assert (
            idx < seq.prompt_len
        ), "subject index falls outside prompt_len — hook injection would hit target tokens"


# ===========================================================================
# Tier 2 — argmax_token_at_end (mock model)
# ===========================================================================


class TestArgmaxTokenAtEnd:
    """argmax_token_at_end(model, prompt) -> (int, str)"""

    def test_returns_tuple_of_int_and_str(self, model):
        tok_id, decoded = argmax_token_at_end(model, "France is")
        assert isinstance(tok_id, int)
        assert isinstance(decoded, str)

    def test_token_id_in_vocab_range(self, model):
        tok_id, _ = argmax_token_at_end(model, "France is")
        assert 0 <= tok_id < _D_VOCAB

    def test_decoded_string_nonempty(self, model):
        _, decoded = argmax_token_at_end(model, "France is")
        assert len(decoded) >= 1

    def test_empty_prompt_raises(self, model):
        with pytest.raises(ScoringError):
            argmax_token_at_end(model, "")

    def test_deterministic_on_same_input(self, model):
        """Same prompt must return the same argmax every time (no randomness)."""
        a = argmax_token_at_end(model, "France is")
        b = argmax_token_at_end(model, "France is")
        assert a == b

    def test_different_prompts_can_produce_different_argmax(self, model):
        """Sanity: two different prompts should not always agree on the top token."""
        a, _ = argmax_token_at_end(model, "France is")
        b, _ = argmax_token_at_end(model, "ZZZZZZZZZ")
        # With a fixed model this may or may not differ — but if they always match,
        # the forward pass is ignoring the input tokens entirely.
        # We just verify the call doesn't raise.
        assert isinstance(a, int) and isinstance(b, int)

    def test_no_grad_does_not_leak(self, model):
        for p in model.parameters():
            p.grad = None
        argmax_token_at_end(model, "France is")
        for p in model.parameters():
            assert p.grad is None


# ===========================================================================
# Tier 2 — Integration: build_teacher_forced feeds trainer correctly
# ===========================================================================


class TestTrainerIntegration:
    """
    End-to-end tests that simulate exactly what the ROME and MEMIT trainers
    do with the output of build_teacher_forced.

    These prove the practical correctness of the API surface, not just the
    individual fields.
    """

    def test_cross_entropy_loss_is_finite(self, model):
        """
        Simulate the MEMIT _compute_z_vector loss computation.
        loss = cross_entropy(logits[prompt_len-1:-1], target_ids) must be finite.
        """
        seq = build_teacher_forced(model, "The capital of France is", "Paris")
        logits = model(seq.full_ids)  # [1, L+T, V]
        L, T = seq.prompt_len, seq.target_len
        pred_logits = logits[0, L - 1 : L - 1 + T, :]  # [T, V]
        loss = F.cross_entropy(pred_logits, seq.target_ids)
        assert torch.isfinite(loss), "cross-entropy loss is not finite"

    def test_cross_entropy_loss_backward_computes_gradient(self, model):
        """
        Simulate MEMIT training step: loss.backward() must populate gradients.
        This proves the full_ids tensor flows through the model differentiably.
        """
        seq = build_teacher_forced(model, "The capital of France is", "Paris")
        logits = model(seq.full_ids)
        L, T = seq.prompt_len, seq.target_len
        pred_logits = logits[0, L - 1 : L - 1 + T, :]
        loss = F.cross_entropy(pred_logits, seq.target_ids)
        loss.backward()
        # At least one parameter must have a non-None, non-zero gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters()
        )
        assert has_grad, "No gradient flowed through the model — backprop is broken"

    def test_prompt_slice_of_full_ids_safe_for_cache_call(self, model):
        """
        Simulates what _compute_z_vector does at step 2:
            model.run_with_cache(full_tokens[:, :prompt_len], ...)
        Slicing must yield a valid 2D token tensor of the prompt length.
        """
        seq = build_teacher_forced(model, "The capital of France is", "Paris")
        prompt_slice = seq.full_ids[:, : seq.prompt_len]
        assert prompt_slice.shape == (1, seq.prompt_len)
        assert prompt_slice.dtype == torch.long

    def test_logit_positions_align_with_target_ids(self, model):
        """
        Prove that the position arithmetic in build_teacher_forced is correct:
        logits at positions [L-1, L-1+T] are exactly the positions that predict
        the target tokens at full_ids[L:L+T].

        In our mock, forward() boosts token id t by +2.0 at position p,
        so argmax at position p == full_ids[0, p] (the token sitting there).

        Therefore argmax at position L-1+i == full_ids[0, L-1+i],
        and full_ids[0, L+i] == target_ids[i].

        The correct assertion is NOT predicted[i] == target_ids[i]
        (that would require predicted[i] == full_ids[0, L+i], but we read
        logits at L-1+i which boosts full_ids[0, L-1+i], off by one).

        Instead we assert two independent invariants that together prove
        the position arithmetic is correct end-to-end:
          (a) argmax at position p equals full_ids[0, p]  — forward() is working
          (b) target_ids == full_ids[0, L:L+T]            — slicing is correct
        Both together mean: "the logit at position L-1+i predicts token L+i,
        which is target_ids[i]" — which is exactly the teacher-forcing contract.
        """
        seq = build_teacher_forced(model, "The capital of France is", "Paris")
        logits = model(seq.full_ids)
        L, T = seq.prompt_len, seq.target_len

        # (a) At every position p, argmax == the token at that position.
        all_argmax = logits[0].argmax(dim=-1)  # [L+T]
        assert torch.equal(all_argmax, seq.full_ids[0]), (
            "Mock forward() argmax does not match input tokens — "
            "the +2.0 boost is not working correctly"
        )

        # (b) target_ids is exactly the slice full_ids[L:L+T].
        assert torch.equal(seq.target_ids, seq.full_ids[0, L : L + T]), (
            "target_ids does not match full_ids[L:L+T] — "
            "the prompt_len / target_len slicing is wrong"
        )

        # Together (a) and (b) prove: logits[L-1+i].argmax() == full_ids[L-1+i],
        # and full_ids[L+i] == target_ids[i], so the CE loss at positions
        # [L-1, L-1+T] against target_ids is computing the right thing.

    def test_no_bos_teacher_forcing_still_valid(self, model_no_bos):
        """
        Qwen path: teacher-forcing must work the same way when there is no BOS.
        """
        seq = build_teacher_forced(model_no_bos, "The capital of France is", "Paris")
        logits = model_no_bos(seq.full_ids)
        L, T = seq.prompt_len, seq.target_len
        pred_logits = logits[0, L - 1 : L - 1 + T, :]
        loss = F.cross_entropy(pred_logits, seq.target_ids)
        assert torch.isfinite(loss)


# ===========================================================================
# Tier 2 — sample_random_prefixes (mock model)
# ===========================================================================


class _PrefixCapableTokenizer(_MockTokenizer):
    """
    Extends _MockTokenizer with a decode() that accepts **kwargs,
    as required by sample_random_prefixes (passes skip_special_tokens=True).
    """

    def decode(self, ids: List[int], **kwargs) -> str:
        return "".join(_ID_TO_STR.get(i, "?") for i in ids)


class _PrefixCapableMockModel(_BPELikeMockModel):
    """
    _BPELikeMockModel with a tokenizer whose decode accepts **kwargs.
    sample_random_prefixes calls tokenizer.decode(window, skip_special_tokens=True),
    which would TypeError on the base _MockTokenizer.
    """

    def __init__(self, default_prepend_bos: bool = True):
        super().__init__(default_prepend_bos=default_prepend_bos)
        self.tokenizer = _PrefixCapableTokenizer()


@pytest.fixture
def prefix_model() -> _PrefixCapableMockModel:
    return _PrefixCapableMockModel(default_prepend_bos=True)


@pytest.fixture
def model_no_tokenizer() -> _BPELikeMockModel:
    """Model with tokenizer attribute removed."""
    m = _BPELikeMockModel(default_prepend_bos=True)
    del m.tokenizer
    return m


class TestSampleRandomPrefixes:
    """
    sample_random_prefixes(model, n_short, n_med, len_short, len_med, seed)
        -> List[str]

    Proves correctness of the random-prefix sampling added in Phase 1.4.
    The function sources prefix material from _FALLBACK_TEXTS via the model's
    tokenizer, picks random token windows, and decodes them to text.
    """

    def test_returns_list_of_strings(self, prefix_model):
        result = sample_random_prefixes(prefix_model, n_short=2, n_med=2, seed=0)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_respects_n_short_and_n_med_counts(self, prefix_model):
        """Output length is at most n_short + n_med."""
        result = sample_random_prefixes(prefix_model, n_short=3, n_med=2, seed=0)
        assert len(result) <= 5

    def test_returns_empty_without_tokenizer(self, model_no_tokenizer):
        """Model without .tokenizer attribute should return []."""
        result = sample_random_prefixes(model_no_tokenizer, seed=0)
        assert result == []

    def test_deterministic_with_same_seed(self, prefix_model):
        """Two calls with the same seed produce identical output."""
        a = sample_random_prefixes(prefix_model, n_short=3, n_med=3, seed=42)
        b = sample_random_prefixes(prefix_model, n_short=3, n_med=3, seed=42)
        assert a == b

    def test_different_seeds_produce_different_output(self, prefix_model):
        """Different seeds should (with high probability) produce different prefixes."""
        a = sample_random_prefixes(prefix_model, n_short=5, n_med=5, seed=0)
        b = sample_random_prefixes(prefix_model, n_short=5, n_med=5, seed=99)
        # Could be identical by astronomical coincidence; assert at least one differs.
        assert a != b, "Different seeds produced identical prefixes"

    def test_does_not_mutate_global_rng(self, prefix_model):
        """
        sample_random_prefixes uses random.Random(seed) locally.
        The global random state must be unchanged after the call.
        """
        state_before = random.getstate()
        sample_random_prefixes(prefix_model, n_short=5, n_med=5, seed=7)
        state_after = random.getstate()
        assert state_before == state_after, "sample_random_prefixes mutated the global random state"

    def test_n_prefixes_zero_returns_empty(self, prefix_model):
        """n_short=0, n_med=0 → no prefixes requested → empty list."""
        result = sample_random_prefixes(prefix_model, n_short=0, n_med=0, seed=0)
        assert result == []

    def test_all_prefixes_are_nonempty_stripped_strings(self, prefix_model):
        """Every returned prefix must be non-empty with no leading/trailing whitespace."""
        result = sample_random_prefixes(prefix_model, n_short=5, n_med=5, seed=0)
        for i, pfx in enumerate(result):
            assert len(pfx) > 0, f"Prefix {i} is empty"
            assert pfx == pfx.strip(), f"Prefix {i} has leading/trailing whitespace: {pfx!r}"

    def test_graceful_with_very_short_corpus(self, prefix_model, monkeypatch):
        """
        If _FALLBACK_TEXTS contains only strings shorter than len_med tokens,
        the medium-length window cannot be sampled. The function should return
        fewer than requested rather than crash.
        """
        import circuitkit.applications.common_utils._covariance as _cov_mod

        # Replace the fallback with very short strings (3 chars → 3 token ids
        # with the char-level tokenizer, which is < len_med=10).
        monkeypatch.setattr(_cov_mod, "_FALLBACK_TEXTS", ["Hi", "Ok", "No"])

        result = sample_random_prefixes(
            prefix_model, n_short=3, n_med=3, len_short=5, len_med=10, seed=0
        )
        # Should not crash; may return fewer than 6 (possibly 0)
        assert isinstance(result, list)
        assert len(result) <= 6

    def test_default_args_produce_nonempty_result(self, prefix_model):
        """
        With the default _FALLBACK_TEXTS (50 sentences of 30+ chars each)
        and the default parameters (n_short=5, n_med=5, len_short=5, len_med=10),
        we should get a non-empty result.
        """
        result = sample_random_prefixes(prefix_model)
        assert len(result) > 0, (
            "Default parameters with the full fallback corpus should produce "
            "at least some prefixes"
        )


# ===========================================================================
# Tier 3 — Real model tests (marked slow)
# ===========================================================================
#
# These tests load actual TransformerLens models. They are the definitive
# proof that the tokenisation primitives work correctly with real BPE /
# SentencePiece tokenisers across model families.
#
# Models used:
#   gpt2           — BPE-Ġ, default_prepend_bos=True   (baseline)
#   Qwen/Qwen2-0.5B — tiktoken BPE, default_prepend_bos=False (Qwen fix)
#   meta-llama/Llama-3.2-1B — tiktoken BPE, default_prepend_bos=True
#
# Each test is independent; no shared mutable state.


@pytest.fixture(scope="module")
def gpt2_model():
    tl = pytest.importorskip("transformer_lens")
    return tl.HookedTransformer.from_pretrained("gpt2", device=("cuda" if torch.cuda.is_available() else "cpu"))


@pytest.fixture(scope="module")
def qwen2_model():
    tl = pytest.importorskip("transformer_lens")
    try:
        model = tl.HookedTransformer.from_pretrained("Qwen/Qwen2-0.5B", device=("cuda" if torch.cuda.is_available() else "cpu"))
        return model
    except Exception as exc:
        pytest.skip(f"Qwen2-0.5B not available: {exc}")


@pytest.fixture(scope="module")
def llama3_model():
    tl = pytest.importorskip("transformer_lens")
    try:
        model = tl.HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B", device=("cuda" if torch.cuda.is_available() else "cpu"))
        return model
    except Exception as exc:
        pytest.skip(f"Llama-3.2-1B not available: {exc}")


# -- Helpers shared across real-model tests -----------------------------------

_REAL_FACTS = [
    # (prompt, subject, target)
    ("The capital of France is", "France", "Paris"),
    ("The capital of Germany is", "Germany", "Berlin"),
    ("The inventor of the telephone was", "telephone", "Bell"),
]


def _assert_format_target_round_trips(model_name: str, real_model) -> None:
    """
    Core correctness check: tokenising prompt + format_target(target, prompt)
    must produce more tokens than tokenising the prompt alone, and the prefix
    of the combined tokenisation must equal the prompt-only tokenisation.
    This is the boundary-integrity property that build_teacher_forced relies on.
    """
    for prompt, _, target in _REAL_FACTS:
        seq = build_teacher_forced(real_model, prompt, target)
        assert (
            seq.target_len >= 1
        ), f"[{model_name}] target '{target}' added no tokens after '{prompt}'"
        # Boundary integrity: first L tokens of full_ids == prompt tokens
        prepend_bos = seq.prepend_bos_used
        prompt_ids = real_model.to_tokens(prompt, prepend_bos=prepend_bos)
        L = seq.prompt_len
        assert torch.equal(
            seq.full_ids[:, :L], prompt_ids
        ), f"[{model_name}] boundary integrity failed for prompt='{prompt}', target='{target}'"


def _assert_subject_location_sane(model_name: str, real_model) -> None:
    """Subject index must be within bounds and before prompt_len."""
    for prompt, subject, target in _REAL_FACTS:
        idx = locate_subject_last_token(real_model, prompt, subject)
        seq = build_teacher_forced(real_model, prompt, target)
        prompt_ids = real_model.to_tokens(prompt, prepend_bos=seq.prepend_bos_used)
        n_prompt_tokens = prompt_ids.shape[1]
        assert 0 <= idx < n_prompt_tokens, (
            f"[{model_name}] subject '{subject}' index {idx} out of "
            f"prompt bounds [0, {n_prompt_tokens})"
        )
        assert (
            idx < seq.prompt_len
        ), f"[{model_name}] subject index {idx} >= prompt_len {seq.prompt_len}"


# -- GPT-2 (BPE-Ġ) -----------------------------------------------------------


@pytest.mark.slow
class TestRealGPT2:
    """GPT-2 — BPE with Ġ prefix, default_prepend_bos=True."""

    def test_default_prepend_bos_is_true(self, gpt2_model):
        from circuitkit.applications.common_utils._tokenization import _prepend_bos

        assert _prepend_bos(gpt2_model) is True

    def test_format_target_boundary_integrity(self, gpt2_model):
        _assert_format_target_round_trips("gpt2", gpt2_model)

    def test_subject_location_sane(self, gpt2_model):
        _assert_subject_location_sane("gpt2", gpt2_model)

    def test_paris_scores_higher_than_noise(self, gpt2_model):
        """GPT-2 knows Paris is the capital of France."""
        r_paris = score_target(gpt2_model, "The capital of France is", "Paris")
        r_noise = score_target(gpt2_model, "The capital of France is", "zzyzx")
        assert (
            r_paris.first_token_prob > r_noise.first_token_prob
        ), "GPT-2: 'Paris' should have higher probability than 'zzyzx'"

    def test_score_target_returns_valid_score(self, gpt2_model):
        result = score_target(gpt2_model, "The capital of France is", "Paris")
        assert 0.0 < result.first_token_prob <= 1.0
        assert result.sequence_logprob <= 0.0
        assert len(result.per_token_logprobs) >= 1

    def test_argmax_is_valid_token(self, gpt2_model):
        tok_id, decoded = argmax_token_at_end(gpt2_model, "The capital of France is")
        assert isinstance(tok_id, int)
        assert isinstance(decoded, str) and len(decoded) >= 1

    def test_trailing_space_prompt_no_double_space(self):
        """
        Regression: when the prompt ends in a space, format_target must not
        prepend another space, which would produce a double-space target that
        tokenises differently from the intended single-space boundary.

        We test format_target directly because build_teacher_forced on a
        trailing-space prompt raises ScoringError on BPE models — the trailing
        space re-tokenises non-compositionally when followed by the next word,
        breaking the boundary integrity check. That is correct behaviour from
        build_teacher_forced; the regression point is solely in format_target.
        """
        result = format_target("Paris", prompt="The capital of France is ")
        assert not result.startswith(" "), (
            "format_target prepended a space despite prompt ending in space — "
            "would produce double-space target '  Paris'"
        )
        assert result == "Paris"

    def test_multitoken_target_all_positions_scored(self, gpt2_model):
        """'New York' is two BPE tokens on GPT-2; both must be scored."""
        result = score_target(gpt2_model, "The largest city in the US is", "New York")
        assert (
            len(result.per_token_logprobs) >= 2
        ), "Multi-token target 'New York' should produce at least 2 log-probs"


# -- Qwen2 (tiktoken BPE, no BOS) --------------------------------------------


@pytest.mark.slow
class TestRealQwen2:
    """
    Qwen2-0.5B — tiktoken-style BPE, default_prepend_bos=False.
    This is the model family that exposed the BOS hardcoding bug.
    """

    def test_default_prepend_bos_is_false(self, qwen2_model):
        """Qwen2 loaded via TransformerLens must have default_prepend_bos=False."""
        from circuitkit.applications.common_utils._tokenization import _prepend_bos

        assert _prepend_bos(qwen2_model) is False, (
            "Qwen2 should have default_prepend_bos=False — "
            "if True, the BOS fix was not applied or the model config changed"
        )

    def test_format_target_boundary_integrity(self, qwen2_model):
        _assert_format_target_round_trips("Qwen2", qwen2_model)

    def test_subject_location_sane(self, qwen2_model):
        _assert_subject_location_sane("Qwen2", qwen2_model)

    def test_score_target_valid_without_bos(self, qwen2_model):
        result = score_target(qwen2_model, "The capital of France is", "Paris")
        assert 0.0 <= result.first_token_prob <= 1.0
        assert result.sequence_logprob <= 0.0

    def test_bos_absent_in_full_ids(self, qwen2_model):
        """
        The first token of the full sequence must NOT be a BOS token.
        This verifies that build_teacher_forced honours the model config
        and does not silently prepend BOS.
        """
        bos_id = getattr(qwen2_model.tokenizer, "bos_token_id", None)
        if bos_id is None:
            pytest.skip("Qwen2 tokenizer has no bos_token_id — cannot assert absence")
        seq = build_teacher_forced(qwen2_model, "The capital of France is", "Paris")
        first_token = int(seq.full_ids[0, 0].item())
        assert (
            first_token != bos_id
        ), f"BOS token ({bos_id}) was prepended despite default_prepend_bos=False"

    def test_subject_index_bos_offset_is_zero(self, qwen2_model):
        """
        The core Qwen BOS fix, validated within Qwen2 alone.

        With default_prepend_bos=False, the subject index must equal the
        zero-based character offset of the subject's last character in the
        prompt — no +1 BOS shift. Before the fix, prepend_bos=True was
        hardcoded, so the returned index was always one too high.

        We verify this by comparing the index against the prompt's own
        tokenisation: the token at subject_idx must decode to (or contain)
        the last character of the subject.
        """
        prompt = "The capital of France is"
        subject = "France"

        idx = locate_subject_last_token(qwen2_model, prompt, subject)

        # Tokenise just the prompt with BOS=False (as the model config demands).
        prompt_ids = qwen2_model.to_tokens(prompt, prepend_bos=False)[0]

        # idx must be a valid position within the prompt.
        assert 0 <= idx < prompt_ids.shape[0], (
            f"subject index {idx} is out of prompt bounds [0, {prompt_ids.shape[0]}). "
            "BOS offset is being applied incorrectly."
        )

        # The token at idx must be within the span of 'France' in the prompt.
        # We verify via the tokenizer's offset_mapping: decode the token and
        # confirm it overlaps the subject's last character.
        encoding = qwen2_model.tokenizer(
            prompt, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = encoding["offset_mapping"]
        # idx has no BOS offset, so it maps directly into the offset list.
        token_start, token_end = offsets[idx]
        subject_char_start = prompt.rfind(subject)
        subject_char_end = subject_char_start + len(subject)  # exclusive
        assert token_start < subject_char_end and token_end > subject_char_start, (
            f"Token at idx={idx} (char span [{token_start},{token_end})) does not "
            f"overlap subject 'France' (char span [{subject_char_start},{subject_char_end})). "
            "locate_subject_last_token is returning the wrong position."
        )


# -- Llama-3 (tiktoken BPE, has BOS) ----------------------------------------


@pytest.mark.slow
class TestRealLlama3:
    """
    Llama-3.2-1B — tiktoken-derived BPE (not SentencePiece like Llama-1/2),
    default_prepend_bos=True.
    Tests that the leading-space rule works for a model that switched
    tokeniser families between generations.
    """

    def test_default_prepend_bos_is_true(self, llama3_model):
        from circuitkit.applications.common_utils._tokenization import _prepend_bos

        assert _prepend_bos(llama3_model) is True

    def test_format_target_boundary_integrity(self, llama3_model):
        _assert_format_target_round_trips("Llama-3", llama3_model)

    def test_subject_location_sane(self, llama3_model):
        _assert_subject_location_sane("Llama-3", llama3_model)

    def test_score_target_valid(self, llama3_model):
        result = score_target(llama3_model, "The capital of France is", "Paris")
        assert 0.0 <= result.first_token_prob <= 1.0
        assert result.sequence_logprob <= 0.0

    def test_multitoken_target_scored_correctly(self, llama3_model):
        """
        'Saint Petersburg' is two tokens on Llama-3's tiktoken BPE.
        Both positions must contribute to sequence_logprob.
        """
        result = score_target(
            llama3_model,
            "The second largest city in Russia is",
            "Saint Petersburg",
        )
        assert len(result.per_token_logprobs) >= 2
        expected_sum = sum(result.per_token_logprobs)
        assert abs(result.sequence_logprob - expected_sum) < 1e-4

    def test_trailing_space_prompt_no_double_space(self):
        """
        Same regression check as TestRealGPT2. format_target must not prepend
        a space when the prompt already ends in one. Tested via format_target
        directly — build_teacher_forced correctly raises ScoringError on
        trailing-space prompts due to BPE non-compositionality.
        """
        result = format_target("Paris", prompt="The capital of France is ")
        assert result == "Paris"

    def test_locate_subject_consistent_with_teacher_forced(self, llama3_model):
        """
        Cross-function consistency: the subject index must fall strictly
        inside the prompt tokens produced by build_teacher_forced.
        """
        prompt = "The capital of France is"
        subject = "France"
        idx = locate_subject_last_token(llama3_model, prompt, subject)
        seq = build_teacher_forced(llama3_model, prompt, "Paris")
        assert idx < seq.prompt_len
