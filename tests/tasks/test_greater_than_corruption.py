"""Regression test — greater_than label-flipping corruption.

Historical bug
--------------
The Greater-Than task built its corrupt prompt by merely *swapping operand
order*::

    clean     : "Which number is greater: 114 or 25? Answer:"
    corrupted : "Which number is greater: 25 or 114? Answer:"

Both prompts contain the SAME two numbers, so the larger (correct) number is
identical in clean and corrupt, and ``correct_idx`` / ``incorrect_idx`` were
unchanged across the pair. Patching clean->corrupt therefore never flips the
target: EAP/EAP-IG saw an almost-flat metric and produced circuit scores
~100x weaker than IOI/SVA (max ~7e-5) -- a degenerate, low-signal circuit.

The fix
-------
The corruption now draws three distinct operands ``lo < mid < hi`` and
replaces the *answer* operand so the correct number flips ``hi -> mid``
(``lo`` stays fixed). ``correct_idx`` is the clean answer (``hi``);
``incorrect_idx`` is the corrupt prompt's true answer (``mid``) -- the exact
contract SVA uses. The default metric is the (unbounded) logit difference,
which does not saturate the way softmax probabilities of number tokens do.

These tests fail if the operand-swap corruption is reintroduced or the
prob-saturating metric is made the default again.
"""

from __future__ import annotations

import csv
import re

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _nums(text: str):
    return [int(x) for x in re.findall(r"\d+", text)]


@pytest.fixture(scope="module")
def gpt2_model():
    """Real (CPU) GPT-2 — the CSV synthesizer needs a genuine tokenizer."""
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")


@pytest.fixture(scope="module")
def synth_rows(gpt2_model, tmp_path_factory):
    """Synthesize a greater_than CSV once and reuse across tests."""
    from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec

    out = tmp_path_factory.mktemp("gt") / "gt.csv"
    GreaterThanTaskSpec()._synthesize_csv(gpt2_model, str(out), num_samples=24, seed=42)
    with open(out) as f:
        return list(csv.DictReader(f))


def test_corruption_flips_the_correct_answer(synth_rows):
    """Each clean/corrupt pair must have a DIFFERENT correct number."""
    assert len(synth_rows) == 24
    for row in synth_rows:
        clean_nums = _nums(row["clean"])
        corrupt_nums = _nums(row["corrupted"])
        assert len(clean_nums) == 2 and len(corrupt_nums) == 2

        # The whole point: the corrupt prompt's true answer is a DIFFERENT
        # number than the clean prompt's.  (Old bug: these were equal.)
        assert max(clean_nums) != max(corrupt_nums), (
            f"corrupt prompt did not flip the answer: clean={row['clean']!r} "
            f"corrupt={row['corrupted']!r} -- operand-swap corruption regressed"
        )
        # Clean and corrupt must NOT be the same two numbers in swapped order.
        assert sorted(clean_nums) != sorted(corrupt_nums), (
            "clean/corrupt share the same operand multiset -- this is the "
            "old order-swap corruption, which gives EAP no signal"
        )


def test_labels_match_clean_and_corrupt_answers(gpt2_model, synth_rows):
    """correct_idx is the clean answer; incorrect_idx is the corrupt answer."""
    from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec
    from circuitkit.utils.token_utils import TokenIDGenerator

    spec = GreaterThanTaskSpec()
    prefix = spec._number_prefix(gpt2_model)
    token_gen = TokenIDGenerator(gpt2_model)

    def answer_id(n):
        return token_gen.get_token_id(prefix + str(n), prepend_space=False)

    for row in synth_rows:
        clean_answer = max(_nums(row["clean"]))
        corrupt_answer = max(_nums(row["corrupted"]))
        assert int(row["correct_idx"]) == answer_id(
            clean_answer
        ), "correct_idx must be the CLEAN prompt's true answer token"
        assert int(row["incorrect_idx"]) == answer_id(
            corrupt_answer
        ), "incorrect_idx must be the CORRUPT prompt's true answer token"
        assert row["correct_idx"] != row["incorrect_idx"]


def test_clean_corrupt_token_length_aligned(gpt2_model, synth_rows):
    """EAP requires clean and corrupt prompts to tokenize to equal length."""
    for row in synth_rows:
        clean_len = gpt2_model.to_tokens(row["clean"]).shape[1]
        corrupt_len = gpt2_model.to_tokens(row["corrupted"]).shape[1]
        assert clean_len == corrupt_len, (
            f"clean/corrupt token-length mismatch: {row['clean']!r} vs " f"{row['corrupted']!r}"
        )


def test_answers_are_single_token(gpt2_model, synth_rows):
    """Answer tokens must be single tokens for the gather-based metrics."""
    tok = gpt2_model.tokenizer
    for row in synth_rows:
        for col in ("correct_idx", "incorrect_idx"):
            decoded = tok.decode([int(row[col])]).strip()
            assert decoded.isdigit(), f"{col} did not decode to a number"


def test_answer_slot_is_not_positionally_degenerate(synth_rows):
    """Both first-slot and second-slot answers must appear across rows."""
    first_slot = second_slot = 0
    for row in synth_rows:
        a, b = _nums(row["clean"])
        if a > b:
            first_slot += 1
        else:
            second_slot += 1
    assert first_slot > 0 and second_slot > 0, (
        "answer is always in the same operand slot -- corruption is " "positionally degenerate"
    )


def test_default_metric_is_logit_diff():
    """Default metric must be the (non-saturating) logit difference."""
    from circuitkit.tasks.builtins.greater_than import GreaterThanTaskSpec

    spec = GreaterThanTaskSpec()
    default_fn = spec.metric_fn()
    logit_fn = spec.metric_fn("logit_diff")
    # Both partials must wrap the same underlying _logit_diff function.
    assert default_fn.func == logit_fn.func == GreaterThanTaskSpec._logit_diff, (
        "greater_than default metric is not logit_diff -- a prob_diff default "
        "saturates near zero for number tokens and kills EAP signal"
    )


@pytest.mark.slow
def test_greater_than_circuit_has_real_signal():
    """End-to-end: discovered circuit must have IOI/SVA-magnitude scores."""
    pytest.importorskip("transformer_lens")
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import torch

    from circuitkit.api import discover_circuit

    workdir = Path(tempfile.mkdtemp())
    try:
        for algo in ("eap", "eap-ig"):
            out = workdir / f"{algo}.pt"
            cfg = {
                "model": {"name": "gpt2", "precision": "float32"},
                "discovery": {
                    "algorithm": algo,
                    "task": "greater_than",
                    "level": "node",
                    "batch_size": 4,
                    "ig_steps": 5,
                    "data_params": {
                        "num_examples": 12,
                        "seed": 42,
                        "cache_dir": str(workdir / "cache"),
                    },
                    "cache_dir": str(workdir / "cache"),
                },
                "pruning": {"target_sparsity": 0.2, "scope": "heads"},
                "output_path": str(out),
            }
            discover_circuit(cfg)

            d = torch.load(
                str(out).replace(".pt", "_scores.pt"),
                map_location="cpu",
                weights_only=False,
            )
            scores = np.array([float(v) for v in d["node_scores"].values()])
            absmax = float(np.abs(scores).max())

            # Gradients are real (finite) and non-degenerate.
            assert np.isfinite(scores).all(), f"{algo}: non-finite scores"
            # Real signal: not the old ~1e-4 near-flat regime. IOI ~1, SVA
            # ~8e-3; require at least SVA order of magnitude.
            assert absmax > 1e-3, (
                f"{algo}: circuit absmax {absmax:.2e} is ~100x weaker than "
                f"IOI/SVA -- corruption/metric regressed to near-flat signal"
            )
            # Heavy-tailed: a few nodes dominate (a real circuit, not noise).
            assert np.abs(scores).std() > 1e-4
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
