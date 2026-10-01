"""Group E — task / tokenization compatibility for tiny-aya (cohere2).

The single-answer-position discovery metrics (EAP/EAP-IG and the IOI /
greater-than / SVA logit-difference metrics) score one vocabulary entry at the
answer position. That silently assumes every answer is a single token. A name
or number that is one token under GPT-2's 50k BPE can split under a larger
vocabulary (Cohere's 262k), in which case the derived answer id is a subword
and attribution is corrupted with no error raised.

Group E adds a guard that fails fast (or warns) when the *loaded* tokenizer
splits a task's answers, and unifies the single-token predicate the diagnostic
tasks already used. These tests pin that behavior. They are fully offline —
no gated weights, no network, no GPU — using duck-typed fake tokenizers/models
that reproduce the split-vs-single-token failure mode. The literal Cohere
tokenizer facts (BOS id 2, ``add_bos_token=true``, ``add_eos_token=false``)
require the gated weights and are validated by the Group F real-weight gate;
here we pin the codepath contracts those facts flow through.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --------------------------------------------------------------------------- #
# Fakes — reproduce "one token here, splits there" without a real tokenizer.   #
# --------------------------------------------------------------------------- #
class FakeTokenizer:
    """Minimal HF-like tokenizer for the single-token guards.

    Strings in ``single_tokens`` encode to one id; everything else encodes to
    two. ``encode(text, add_special_tokens=...)`` is the only surface the
    guards touch. ``chat_template`` defaults to ``None`` (a base model).
    """

    def __init__(self, single_tokens, *, name_or_path="fake/tok", chat_template=None):
        self._single = set(single_tokens)
        self.name_or_path = name_or_path
        self.chat_template = chat_template

    def encode(self, text, add_special_tokens=True):
        # add_special_tokens accepted for signature parity; the guards always
        # pass add_special_tokens=False.
        return [1000] if text in self._single else [1000, 1001]


class RecordingModel:
    """Fake model exposing ``.to_tokens`` that records its ``prepend_bos`` arg."""

    def __init__(self, tokenizer=None, model_name="fake-model"):
        self.tokenizer = tokenizer or FakeTokenizer(single_tokens=set())
        self.cfg = SimpleNamespace(model_name=model_name)
        self.calls = []

    def to_tokens(self, text, prepend_bos=True):
        self.calls.append({"text": text, "prepend_bos": prepend_bos})
        # Return a shape-only stand-in; the guards/chat code only pass it on.
        return SimpleNamespace(shape=(1, len(text.split())))


def _ioi_names():
    from circuitkit.data.task_data.tasks.ioi.ioi_dataset import NAMES

    return NAMES


def _all_single_tokenizer(**kwargs):
    """Tokenizer that keeps every space-prefixed IOI name single-token."""
    return FakeTokenizer({" " + n for n in _ioi_names()}, **kwargs)


# --------------------------------------------------------------------------- #
# encodes_to_single_token                                                      #
# --------------------------------------------------------------------------- #
class TestEncodesToSingleToken:
    def test_single_token_string(self):
        from circuitkit.tasks.specs import encodes_to_single_token

        tok = FakeTokenizer({" Michael"})
        assert encodes_to_single_token(tok, " Michael") is True

    def test_multi_token_string(self):
        from circuitkit.tasks.specs import encodes_to_single_token

        tok = FakeTokenizer({" Michael"})
        assert encodes_to_single_token(tok, " Zbigniew") is False

    def test_uses_add_special_tokens_false(self):
        """The predicate must never let special tokens inflate the count."""
        from circuitkit.tasks.specs import encodes_to_single_token

        seen = {}

        class SpyTokenizer:
            def encode(self, text, add_special_tokens=True):
                seen["add_special_tokens"] = add_special_tokens
                return [1]

        assert encodes_to_single_token(SpyTokenizer(), "x") is True
        assert seen["add_special_tokens"] is False


# --------------------------------------------------------------------------- #
# require_single_token_answers                                                 #
# --------------------------------------------------------------------------- #
class TestRequireSingleTokenAnswers:
    def test_returns_only_single_token_answers(self):
        from circuitkit.tasks.specs import require_single_token_answers

        tok = FakeTokenizer({" a", " c"})
        out = require_single_token_answers(tok, [" a", " bb", " c"], task_name="t", min_valid=1)
        assert out == [" a", " c"]

    def test_raises_when_too_few_single_token(self):
        from circuitkit.tasks.specs import require_single_token_answers

        tok = FakeTokenizer({" a"})  # only one single-token answer
        with pytest.raises(ValueError) as exc:
            require_single_token_answers(
                tok,
                [" a", " bb", " cc"],
                task_name="mytask",
                min_valid=2,
                tokenizer_name="fake/tok",
            )
        msg = str(exc.value)
        # Message is actionable: names the task, the tokenizer, the shortfall,
        # concrete multi-token examples and a remedy.
        assert "mytask" in msg
        assert "fake/tok" in msg
        assert "1 of 3" in msg
        assert "' bb'" in msg and "' cc'" in msg
        assert "greater_than" in msg

    def test_all_single_token_passes(self):
        from circuitkit.tasks.specs import require_single_token_answers

        tok = FakeTokenizer({" a", " b", " c"})
        out = require_single_token_answers(tok, [" a", " b", " c"], task_name="t", min_valid=2)
        assert out == [" a", " b", " c"]

    def test_example_limit_truncates_and_counts_remainder(self):
        from circuitkit.tasks.specs import require_single_token_answers

        multis = [f" m{i}" for i in range(20)]  # all multi-token
        tok = FakeTokenizer(set())
        with pytest.raises(ValueError) as exc:
            require_single_token_answers(tok, multis, task_name="t", min_valid=2, example_limit=8)
        msg = str(exc.value)
        assert "(+12 more)" in msg  # 20 multi - 8 shown = 12
        # Only the first 8 examples are shown verbatim.
        assert "' m7'" in msg
        assert "' m8'" not in msg.split("(+12 more)")[0]


# --------------------------------------------------------------------------- #
# IOI guard                                                                    #
# --------------------------------------------------------------------------- #
class TestIOISingleTokenGuard:
    def test_all_single_token_no_warning(self, monkeypatch):
        # The task logger sets propagate=False, so capture via the module logger
        # rather than caplog (which only sees propagating loggers).
        from circuitkit.tasks.builtins import ioi as ioi_mod

        fake_logger = MagicMock()
        monkeypatch.setattr(ioi_mod, "logger", fake_logger)
        model = RecordingModel(_all_single_tokenizer())
        ioi_mod.IOITaskSpec()._verify_single_token_answers(model)
        fake_logger.warning.assert_not_called()

    def test_all_split_raises(self):
        from circuitkit.tasks.builtins.ioi import IOITaskSpec

        model = RecordingModel(FakeTokenizer(set(), name_or_path="cohere/tiny-aya"))
        with pytest.raises(ValueError) as exc:
            IOITaskSpec()._verify_single_token_answers(model)
        msg = str(exc.value)
        assert "ioi" in msg
        assert "cohere/tiny-aya" in msg

    def test_partial_split_warns_but_proceeds(self, monkeypatch):
        from circuitkit.tasks.builtins import ioi as ioi_mod

        fake_logger = MagicMock()
        monkeypatch.setattr(ioi_mod, "logger", fake_logger)

        names = _ioi_names()
        # Keep only half the names single-token: enough to pass min_valid=2 but
        # trip the partial-degeneracy warning.
        keep = {" " + n for n in names[: max(2, len(names) // 2)]}
        model = RecordingModel(FakeTokenizer(keep, name_or_path="partial/tok"))
        ioi_mod.IOITaskSpec()._verify_single_token_answers(model)

        fake_logger.warning.assert_called_once()
        warning_msg = fake_logger.warning.call_args.args[0]
        assert "partial/tok" in warning_msg
        assert "multi-token" in warning_msg

    def test_build_dataloader_guards_before_data_generation(self):
        """A splitting tokenizer must fail in build_dataloader, not mid-generation."""
        from circuitkit.tasks.builtins.ioi import IOITaskSpec

        model = RecordingModel(FakeTokenizer(set()))  # every name splits
        cfg = {"algorithm": "eap", "level": "node"}
        with pytest.raises(ValueError, match="single-token"):
            IOITaskSpec().build_dataloader(model, cfg, "cpu")
        # The guard runs before any tokenization of prompts into tensors.
        assert model.calls == []

    def test_prefixes_names_with_leading_space(self):
        """The guard must check the space-prefixed form the metric actually reads."""
        from circuitkit.tasks.builtins.ioi import IOITaskSpec

        names = _ioi_names()
        # Bare (unspaced) names single-token, but space-prefixed ones split.
        model = RecordingModel(FakeTokenizer(set(names)))
        with pytest.raises(ValueError):
            IOITaskSpec()._verify_single_token_answers(model)


# --------------------------------------------------------------------------- #
# greater_than — already single-token by construction (validate the wiring).   #
# --------------------------------------------------------------------------- #
class TestGreaterThanSingleToken:
    def test_number_prefix_uses_shared_predicate(self):
        """_number_prefix picks whichever number form the tokenizer keeps single."""
        from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec

        # Only the space-prefixed numbers 0..999 are single-token here.
        spaced = {f" {n}" for n in range(1000)}
        model = RecordingModel(FakeTokenizer(spaced))
        assert GreaterThanTaskSpec()._number_prefix(model) == " "

        unspaced = {f"{n}" for n in range(1000)}
        model2 = RecordingModel(FakeTokenizer(unspaced))
        assert GreaterThanTaskSpec()._number_prefix(model2) == ""

    def test_single_token_number_pool_rejects_incompatible_tokenizer(self):
        """A tokenizer with < 2 single-token numbers must fail loudly."""
        from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec

        model = RecordingModel(FakeTokenizer(set()))  # nothing is single-token
        with pytest.raises(ValueError):
            GreaterThanTaskSpec()._get_single_token_numbers(model)


# --------------------------------------------------------------------------- #
# Raw-prompt path — diagnostic tasks stay un-templated (BOS prepend contract). #
# --------------------------------------------------------------------------- #
class TestRawPromptPath:
    def test_diagnostic_tasks_default_to_off(self):
        from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec
        from circuitkit.tasks.builtins.ioi import IOITaskSpec
        from circuitkit.tasks.builtins.sva import SVATaskSpec

        assert IOITaskSpec.chat_template_mode == "off"
        assert GreaterThanTaskSpec.chat_template_mode == "off"
        assert SVATaskSpec.chat_template_mode == "off"

    def test_padding_side_is_right(self):
        from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec
        from circuitkit.tasks.builtins.ioi import IOITaskSpec

        assert IOITaskSpec.pair_padding_side == "right"
        assert GreaterThanTaskSpec.pair_padding_side == "right"

    def test_to_tokens_prepends_bos_for_raw_text(self):
        """Raw (un-templated) prompts are tokenized with prepend_bos=True.

        Cohere's tokenizer sets add_bos_token=true (BOS id 2); the raw-text
        discovery path relies on ``to_tokens`` prepending exactly one BOS.
        """
        from circuitkit.tasks._chat import to_tokens

        model = RecordingModel()
        to_tokens(model, "hello world", templated=False)
        assert model.calls[-1]["prepend_bos"] is True

    def test_to_tokens_skips_bos_for_templated_text(self):
        """Chat-templated text already carries its own BOS, so prepend_bos=False."""
        from circuitkit.tasks._chat import to_tokens

        model = RecordingModel()
        to_tokens(model, "<bos>hello", templated=True)
        assert model.calls[-1]["prepend_bos"] is False

    def test_ioi_rejects_chat_template_override(self):
        """The off-only diagnostic task fails loudly on an 'on' override."""
        from circuitkit.tasks.builtins.ioi import IOITaskSpec

        chat_tok = _all_single_tokenizer(chat_template="{{ messages }}")
        model = RecordingModel(chat_tok)
        cfg = {"algorithm": "eap", "level": "node", "chat_template_mode": "on"}
        with pytest.raises(NotImplementedError):
            IOITaskSpec().build_dataloader(model, cfg, "cpu")
