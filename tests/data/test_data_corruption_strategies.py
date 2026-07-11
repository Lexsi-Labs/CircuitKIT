"""Unit tests for the ``circuitkit.data.corruption`` strategy family.

These strategies were previously untested (coverage ~0%). Each one consumes a
:class:`~circuitkit.data.normalized.ContrastiveRecord` and produces a
:class:`~circuitkit.data.corruption.base.CorruptionResult`. The tests below,
for every strategy:

  * instantiate it and confirm it self-registered in ``STRATEGY_REGISTRY``;
  * feed a representative example the strategy *can* corrupt and assert the
    result shape (``succeeded`` / ``corrupt_prompt`` / ``corrupt_answer``)
    and that it actually changes the prompt (or the answer, where the
    strategy is meant to flip it);
  * feed a no-op / edge input and assert ``succeeded is False`` with the
    prompt/answer left as ``None``.

Every target strategy here is pure-Python (regex + built-in lexicons); none
require spaCy or an LLM at runtime, so these tests are CPU-only and never load
a model. ``pytest.importorskip`` is used defensively for the package import so
the module skips cleanly rather than erroring if the data layer is absent.
"""

from __future__ import annotations

import random

import pytest

# The corruption package is pure-Python but lives under the (optional) data
# layer; skip the whole module rather than error if it cannot be imported.
pytest.importorskip("circuitkit.data.corruption")

from circuitkit.data.corruption.base import (  # noqa: E402
    STRATEGY_REGISTRY,
    CorruptionResult,
    CorruptionStrategy,
    get_strategy,
    list_strategies,
)
from circuitkit.data.corruption.benign_rewrite import BenignRewrite  # noqa: E402
from circuitkit.data.corruption.code_syntax_corrupt import CodeSyntaxCorrupt  # noqa: E402
from circuitkit.data.corruption.entity_swap import EntitySwap  # noqa: E402
from circuitkit.data.corruption.final_answer_swap import FinalAnswerSwap  # noqa: E402
from circuitkit.data.corruption.instruction_swap import (  # noqa: E402
    InstructionSwap,
    audit_instruction_swap_degeneracy,
)
from circuitkit.data.corruption.logical_negation import LogicalNegation  # noqa: E402
from circuitkit.data.corruption.math_step_corrupt import MathStepCorrupt  # noqa: E402
from circuitkit.data.corruption.mcq_choice_swap import MCQChoiceSwap  # noqa: E402
from circuitkit.data.corruption.operand_swap import OperandSwap  # noqa: E402
from circuitkit.data.corruption.profession_swap import ProfessionSwap  # noqa: E402
from circuitkit.data.corruption.resample import Resample  # noqa: E402
from circuitkit.data.corruption.token_swap import TokenSwap  # noqa: E402
from circuitkit.data.normalized import (  # noqa: E402
    ContrastiveRecord,
    ContrastSource,
    DatasetShape,
    NormalizedDataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(**kw) -> ContrastiveRecord:
    """Build a GENERATED ContrastiveRecord with sensible defaults."""
    base = dict(
        record_id="r1",
        clean_prompt="",
        clean_answer="",
        contrast_source=ContrastSource.GENERATED,
    )
    base.update(kw)
    return ContrastiveRecord(**base)


def _assert_result_shape(res: CorruptionResult) -> None:
    """Every corrupt() must return a CorruptionResult with a bool succeeded."""
    assert isinstance(res, CorruptionResult)
    assert isinstance(res.succeeded, bool)
    assert isinstance(res.notes, str)


def _assert_success(res: CorruptionResult) -> None:
    _assert_result_shape(res)
    assert res.succeeded is True
    assert res.corrupt_prompt is not None
    assert res.corrupt_answer is not None


def _assert_noop(res: CorruptionResult) -> None:
    _assert_result_shape(res)
    assert res.succeeded is False
    assert res.corrupt_prompt is None
    assert res.corrupt_answer is None
    assert res.notes  # a no-op must explain itself


# ---------------------------------------------------------------------------
# Registry / protocol
# ---------------------------------------------------------------------------

_ALL_TARGET_STRATEGIES = [
    "benign_rewrite",
    "code_syntax_corrupt",
    "entity_swap",
    "final_answer_swap",
    "instruction_swap",
    "logical_negation",
    "math_step_corrupt",
    "mcq_choice_swap",
    "operand_swap",
    "profession_swap",
    "resample",
    "token_swap",
]


class TestRegistry:
    """The @register_strategy decorator wires classes into the registry."""

    @pytest.mark.parametrize("name", _ALL_TARGET_STRATEGIES)
    def test_strategy_is_registered(self, name):
        assert name in STRATEGY_REGISTRY
        assert name in list_strategies()

    @pytest.mark.parametrize("name", _ALL_TARGET_STRATEGIES)
    def test_get_strategy_roundtrip(self, name):
        cls = get_strategy(name)
        assert issubclass(cls, CorruptionStrategy)
        inst = cls()
        # register_strategy sets the class .name to the registry key.
        assert inst.name == name

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(KeyError):
            get_strategy("no_such_strategy_xyz")


# ---------------------------------------------------------------------------
# entity_swap
# ---------------------------------------------------------------------------


class TestEntitySwap:
    def test_swaps_capitalised_entity(self):
        rec = _rec(clean_prompt="Mary went to the store with John.", clean_answer="Mary")
        res = EntitySwap().corrupt(rec, rng=random.Random(1))
        _assert_success(res)
        assert res.corrupt_prompt != rec.clean_prompt
        # A same-length name replaced "Mary"; "Mary" no longer opens the prompt.
        assert not res.corrupt_prompt.startswith("Mary ")
        # When the answer equals the swapped entity it is swapped there too.
        assert res.corrupt_answer != "Mary"

    def test_preserves_length_of_swapped_token(self):
        rec = _rec(clean_prompt="Mary went home.", clean_answer="x")
        res = EntitySwap().corrupt(rec, rng=random.Random(1))
        _assert_success(res)
        # Same-length substitution => overall prompt length unchanged.
        assert len(res.corrupt_prompt) == len(rec.clean_prompt)

    def test_no_capitalised_entity_is_noop(self):
        rec = _rec(clean_prompt="the sky is blue today", clean_answer="blue")
        res = EntitySwap().corrupt(rec, rng=random.Random(1))
        _assert_noop(res)


# ---------------------------------------------------------------------------
# logical_negation
# ---------------------------------------------------------------------------


class TestLogicalNegation:
    def test_inserts_not_after_auxiliary(self):
        rec = _rec(clean_prompt="The door is open.", clean_answer="open")
        res = LogicalNegation().corrupt(rec, rng=random.Random(0))
        _assert_success(res)
        assert "not" in res.corrupt_prompt.lower()
        assert res.corrupt_prompt != rec.clean_prompt
        # Answer is left to the caller — negation only rewrites the prompt.
        assert res.corrupt_answer == rec.clean_answer

    def test_removes_existing_negation(self):
        rec = _rec(clean_prompt="The door is not open.", clean_answer="open")
        res = LogicalNegation().corrupt(rec, rng=random.Random(0))
        _assert_success(res)
        assert "not" not in res.corrupt_prompt.lower().split()
        assert res.corrupt_prompt != rec.clean_prompt
        # No double spaces left behind after removal.
        assert "  " not in res.corrupt_prompt

    def test_no_aux_and_no_negation_is_noop(self):
        rec = _rec(clean_prompt="Blue sky today", clean_answer="x")
        res = LogicalNegation().corrupt(rec, rng=random.Random(0))
        _assert_noop(res)


# ---------------------------------------------------------------------------
# token_swap
# ---------------------------------------------------------------------------


class TestTokenSwap:
    def test_swaps_content_token(self):
        rec = _rec(
            clean_prompt="The quick brown fox jumps over lazy dogs",
            clean_answer="fox",
        )
        res = TokenSwap().corrupt(rec, rng=random.Random(3))
        _assert_success(res)
        assert res.corrupt_prompt != rec.clean_prompt
        # STR replaces one content token with another already present in the
        # prompt, so the whitespace-separated word count is unchanged.
        assert len(res.corrupt_prompt.split()) == len(rec.clean_prompt.split())
        # Every replacement token comes from the original prompt's vocabulary.
        assert set(res.corrupt_prompt.split()) <= set(rec.clean_prompt.split())

    def test_too_few_content_tokens_is_noop(self):
        rec = _rec(clean_prompt="a the of", clean_answer="x")
        res = TokenSwap().corrupt(rec, rng=random.Random(0))
        _assert_noop(res)


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_pairs_with_a_peer(self):
        me = _rec(record_id="a", clean_prompt="prompt a", clean_answer="A")
        pool = [me, _rec(record_id="b", clean_prompt="prompt b", clean_answer="B")]
        res = Resample().corrupt(me, rng=random.Random(0), pool=pool)
        _assert_success(res)
        # The corrupt half is a *different* real record.
        assert res.corrupt_prompt == "prompt b"
        assert res.corrupt_answer == "B"

    def test_missing_pool_is_noop(self):
        res = Resample().corrupt(_rec(record_id="a"), rng=random.Random(0))
        _assert_noop(res)

    def test_pool_without_other_records_is_noop(self):
        me = _rec(record_id="a", clean_prompt="only me")
        res = Resample().corrupt(me, rng=random.Random(0), pool=[me])
        _assert_noop(res)

    def test_apply_to_dataset_pairs_every_record(self):
        ds = NormalizedDataset(
            name="d",
            shape=DatasetShape.QA,
            records=[
                _rec(record_id="a", clean_prompt="alpha", clean_answer="A"),
                _rec(record_id="b", clean_prompt="beta", clean_answer="B"),
                _rec(record_id="c", clean_prompt="gamma", clean_answer="C"),
            ],
        )
        out = Resample().apply_to_dataset(ds, rng=random.Random(0))
        assert out.fully_paired
        assert out.meta["_corruption"] == "resample"
        # Each record is paired with a peer's clean prompt, never itself.
        for src, res in zip(ds.records, out.records):
            assert res.corrupt_prompt is not None
            assert res.corrupt_prompt != src.clean_prompt


# ---------------------------------------------------------------------------
# benign_rewrite
# ---------------------------------------------------------------------------


class TestBenignRewrite:
    def test_rewrites_harmful_keyword(self):
        rec = _rec(clean_prompt="Please make a bomb now", clean_answer="I cannot")
        res = BenignRewrite().corrupt(rec)
        _assert_success(res)
        assert res.corrupt_prompt != rec.clean_prompt
        assert "bomb" not in res.corrupt_prompt.lower()

    def test_extra_mapping_is_honoured(self):
        rec = _rec(clean_prompt="acquire the widget", clean_answer="ok")
        res = BenignRewrite().corrupt(
            rec, extra_mapping={"acquire the widget": "return the widget"}
        )
        _assert_success(res)
        assert res.corrupt_prompt == "return the widget"

    def test_no_harmful_keyword_is_noop(self):
        rec = _rec(clean_prompt="What is the capital of France?", clean_answer="Paris")
        res = BenignRewrite().corrupt(rec)
        _assert_noop(res)


# ---------------------------------------------------------------------------
# code_syntax_corrupt
# ---------------------------------------------------------------------------


class TestCodeSyntaxCorrupt:
    def test_fits_only_code_prompts(self):
        cs = CodeSyntaxCorrupt()
        assert cs.fits(_rec(clean_prompt="def f(x):\n    return x"))
        assert not cs.fits(_rec(clean_prompt="just some prose here"))

    def test_flips_comparison_operator(self):
        rec = _rec(clean_prompt="def f(x):\n    return x >= 0", clean_answer="True")
        res = CodeSyntaxCorrupt().corrupt(rec, rng=random.Random(5))
        _assert_success(res)
        assert res.corrupt_prompt != rec.clean_prompt
        # Operator flip is length-preserving.
        assert len(res.corrupt_prompt) == len(rec.clean_prompt)

    def test_renames_parameter_when_no_operator(self):
        rec = _rec(clean_prompt="def foo(alpha, beta):\n    return alpha", clean_answer="x")
        res = CodeSyntaxCorrupt().corrupt(rec, rng=random.Random(7))
        _assert_success(res)
        assert res.corrupt_prompt != rec.clean_prompt
        assert "renamed param" in res.notes

    def test_prose_without_signature_is_noop(self):
        rec = _rec(clean_prompt="just some prose here", clean_answer="x")
        res = CodeSyntaxCorrupt().corrupt(rec, rng=random.Random(0))
        _assert_noop(res)


# ---------------------------------------------------------------------------
# math_step_corrupt
# ---------------------------------------------------------------------------


class TestMathStepCorrupt:
    def test_corrupts_equation_in_solution_text(self):
        rec = _rec(
            clean_prompt="Q: solve it",
            clean_answer="10",
            meta={"solution_text": "First 2 + 3 = 5 then done"},
        )
        ms = MathStepCorrupt()
        assert ms.fits(rec)
        res = ms.corrupt(rec, rng=random.Random(2))
        _assert_success(res)
        # In solution_text mode the *prompt* is intentionally left unchanged;
        # the corruption lives in the (recorded) solution equation.
        assert res.corrupt_prompt == rec.clean_prompt
        assert res.corrupt_answer == rec.clean_answer
        assert "corrupted equation" in res.notes

    def test_corrupts_equation_in_prompt_when_no_solution(self):
        rec = _rec(clean_prompt="We know 12 + 34 = 46 exactly.", clean_answer="46")
        res = MathStepCorrupt().corrupt(rec, rng=random.Random(4))
        _assert_success(res)
        # No solution_text => the equation in the prompt itself is edited.
        assert res.corrupt_prompt != rec.clean_prompt
        assert "= 46" not in res.corrupt_prompt

    def test_no_equation_is_noop(self):
        rec = _rec(clean_prompt="no equations here", clean_answer="x")
        ms = MathStepCorrupt()
        assert not ms.fits(rec)
        res = ms.corrupt(rec, rng=random.Random(1))
        _assert_noop(res)


# ---------------------------------------------------------------------------
# mcq_choice_swap
# ---------------------------------------------------------------------------


class TestMCQChoiceSwap:
    def _mcq_record(self):
        return _rec(
            clean_prompt=(
                "What is 2+2?\nA. three\nB. four\nC. five\nD. six\nAnswer:"
            ),
            clean_answer=" B",
            meta={"choices": ["three", "four", "five", "six"], "correct_idx": 1},
        )

    def test_swaps_correct_choice_and_flips_answer_letter(self):
        rec = self._mcq_record()
        mc = MCQChoiceSwap()
        assert mc.fits(rec)
        res = mc.corrupt(rec, rng=random.Random(9))
        _assert_success(res)
        assert res.corrupt_prompt != rec.clean_prompt
        # The correct answer letter must change (the correct content moved).
        assert res.corrupt_answer.strip() != rec.clean_answer.strip()
        # New answer is still a single letter label.
        assert res.corrupt_answer.strip() in {"A", "C", "D"}
        # The correct content "four" now sits behind the new answer letter.
        new_letter = res.corrupt_answer.strip()
        assert f"{new_letter}. four" in res.corrupt_prompt

    def test_missing_meta_is_noop(self):
        res = MCQChoiceSwap().corrupt(_rec(clean_prompt="hi", clean_answer="x"), rng=random.Random(1))
        _assert_noop(res)

    def test_fits_requires_two_choices(self):
        mc = MCQChoiceSwap()
        one = _rec(clean_prompt="q", meta={"choices": ["only"], "correct_idx": 0})
        assert not mc.fits(one)


# ---------------------------------------------------------------------------
# operand_swap
# ---------------------------------------------------------------------------


class TestOperandSwap:
    def test_swaps_operands_and_updates_answer(self):
        rec = _rec(clean_prompt="Compute: 2 + 3 =", clean_answer=" 5")
        op = OperandSwap()
        assert op.fits(rec)
        res = op.corrupt(rec, rng=random.Random(11))
        _assert_success(res)
        assert res.corrupt_prompt != rec.clean_prompt
        # A genuinely different arithmetic result must be produced.
        assert res.corrupt_answer.strip() != "5"
        # Same operator template is retained.
        assert "+" in res.corrupt_prompt and "=" in res.corrupt_prompt

    def test_non_arithmetic_is_noop(self):
        rec = _rec(clean_prompt="no arithmetic here", clean_answer="x")
        op = OperandSwap()
        assert not op.fits(rec)
        res = op.corrupt(rec, rng=random.Random(1))
        _assert_noop(res)


# ---------------------------------------------------------------------------
# profession_swap
# ---------------------------------------------------------------------------


class TestProfessionSwap:
    def test_swaps_profession_and_pronoun(self):
        rec = _rec(clean_prompt="The accountant said that", clean_answer=" he")
        ps = ProfessionSwap()
        assert ps.fits(rec)
        res = ps.corrupt(rec, rng=random.Random(1))
        _assert_success(res)
        assert res.corrupt_prompt != rec.clean_prompt
        assert "accountant" not in res.corrupt_prompt
        # The expected pronoun flips he -> she.
        assert res.corrupt_answer.strip() == "she"

    def test_no_profession_is_noop(self):
        rec = _rec(clean_prompt="The dog ran fast", clean_answer=" it")
        ps = ProfessionSwap()
        assert not ps.fits(rec)
        res = ps.corrupt(rec, rng=random.Random(1))
        _assert_noop(res)


# ---------------------------------------------------------------------------
# final_answer_swap
# ---------------------------------------------------------------------------


class TestFinalAnswerSwap:
    def test_math_reasoning_mode_edits_prompt_and_answer(self):
        sol = "She had 90 and gained 18 so <<90+18=108>>108 total. #### 108"
        rec = _rec(clean_prompt=sol, clean_answer=" 108", meta={"solution_text": sol})
        res = FinalAnswerSwap().corrupt(rec, rng=random.Random(3))
        _assert_success(res)
        # Both the reasoning trace in the prompt AND the answer change.
        assert res.corrupt_prompt != rec.clean_prompt
        assert res.corrupt_answer.strip() != "108"
        # The trailing "#### N" marker is kept consistent with the new answer.
        assert res.corrupt_answer.strip() in res.corrupt_prompt

    def test_answer_only_numeric_mode(self):
        rec = _rec(clean_prompt="The answer is", clean_answer="42")
        res = FinalAnswerSwap().corrupt(rec, rng=random.Random(3))
        _assert_success(res)
        # Prompt is untouched in answer-only mode; only the target moves.
        assert res.corrupt_prompt == rec.clean_prompt
        assert res.corrupt_answer != "42"
        assert res.corrupt_answer.lstrip("-").isdigit()

    def test_answer_only_boolean_mode(self):
        rec = _rec(clean_prompt="Is the sky blue?", clean_answer="yes")
        res = FinalAnswerSwap().corrupt(rec, rng=random.Random(3))
        _assert_success(res)
        assert res.corrupt_answer == "no"

    def test_answer_only_letter_mode(self):
        rec = _rec(clean_prompt="Pick one", clean_answer="B")
        res = FinalAnswerSwap().corrupt(rec, rng=random.Random(3))
        _assert_success(res)
        assert res.corrupt_answer != "B"
        assert res.corrupt_answer.isalpha() and res.corrupt_answer.isupper()

    def test_non_flippable_answer_is_noop(self):
        rec = _rec(clean_prompt="Describe cats", clean_answer="cats are nice")
        res = FinalAnswerSwap().corrupt(rec, rng=random.Random(3))
        _assert_noop(res)


# ---------------------------------------------------------------------------
# instruction_swap
# ---------------------------------------------------------------------------


class TestInstructionSwap:
    def test_swaps_directive_verb(self):
        rec = _rec(clean_prompt="Generate a 3-paragraph essay about dogs.", clean_answer=".")
        isw = InstructionSwap()
        assert isw.fits(rec)
        res = isw.corrupt(rec, rng=random.Random(2))
        _assert_success(res)
        assert res.corrupt_prompt != rec.clean_prompt
        assert not res.corrupt_prompt.lower().startswith("generate")
        # Answer is intentionally preserved (first stop-token approximation).
        assert res.corrupt_answer == rec.clean_answer

    def test_no_directive_verb_is_noop(self):
        rec = _rec(clean_prompt="The weather is nice today.", clean_answer=".")
        isw = InstructionSwap()
        assert not isw.fits(rec)
        res = isw.corrupt(rec, rng=random.Random(2))
        _assert_noop(res)

    def test_audit_degeneracy_counts_collapsing_pairs(self):
        """The audit helper reports how many paired records share a first token."""

        class _StubTokenizer:
            # Encodes each character to its ordinal; a leading space is a
            # distinct single "whitespace" token so the helper's ws-strip
            # branch is exercised.
            def encode(self, text, add_special_tokens=False):
                return [ord(c) for c in text]

        records = [
            # paired, identical answers -> collapses on first token
            ContrastiveRecord(
                record_id="1",
                clean_prompt="Generate x",
                clean_answer="A",
                corrupt_prompt="List x",
                corrupt_answer="A",
                contrast_source=ContrastSource.GENERATED,
            ),
            # paired, different first token -> kept
            ContrastiveRecord(
                record_id="2",
                clean_prompt="Generate y",
                clean_answer="A",
                corrupt_prompt="List y",
                corrupt_answer="B",
                contrast_source=ContrastSource.GENERATED,
            ),
            # unpaired -> skipped
            _rec(record_id="3", clean_prompt="Generate z", clean_answer="A"),
        ]
        report = audit_instruction_swap_degeneracy(records, _StubTokenizer())
        assert report["total"] == 3
        assert report["paired"] == 2
        assert report["same_first_token"] == 1
        assert report["kept"] == 1
        assert report["dropped"] == 1
        assert report["skipped_unpaired"] == 1


# ---------------------------------------------------------------------------
# Cross-cutting: the apply() / corrupt_example() bridges on the base class
# ---------------------------------------------------------------------------


class TestBaseBridges:
    def test_apply_success_marks_generated_and_pairs(self):
        rec = _rec(clean_prompt="2 + 3 =", clean_answer=" 5")
        out = OperandSwap().apply(rec, rng=random.Random(1))
        assert out.contrast_source == ContrastSource.GENERATED
        assert out.is_paired
        assert out.meta["_strategy_name"] == "operand_swap"

    def test_apply_failure_records_strategy_error(self):
        rec = _rec(clean_prompt="no math here", clean_answer="x")
        out = OperandSwap().apply(rec, rng=random.Random(1))
        # Failure leaves the record unpaired but tags the error for debugging.
        assert out.corrupt_prompt is None
        assert out.meta["_strategy_error"]
        assert out.meta["_strategy_name"] == "operand_swap"

    def test_apply_short_circuits_native_pairs(self):
        paired = ContrastiveRecord(
            record_id="p",
            clean_prompt="clean",
            clean_answer="a",
            corrupt_prompt="corrupt",
            corrupt_answer="b",
            contrast_source=ContrastSource.NATIVE_PAIR,
        )
        out = TokenSwap().apply(paired, rng=random.Random(1))
        # Native pairs are returned untouched (same object identity).
        assert out is paired

    def test_corrupt_example_dict_bridge_preserves_extra_keys(self):
        example = {"prompt": "The accountant said that", "answer": " he", "extra": 123}
        out = ProfessionSwap().corrupt_example(example, rng=random.Random(1))
        assert out["prompt"] != example["prompt"]
        assert out["answer"].strip() == "she"
        # Non-prompt/answer keys survive the round-trip.
        assert out["extra"] == 123
