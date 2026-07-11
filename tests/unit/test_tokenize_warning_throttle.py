"""The misaligned-pair warning in tokenize_batch_pair must not be silenced forever.

A previous once-per-process latch (has_warned) meant the FIRST misaligned batch
silenced the warning for every later dataset in the same process. The throttle
now warns on the 1st misaligned batch and every 100th thereafter, keeps running
totals, and never permanently goes quiet. Alignment/padding behaviour unchanged.
"""

import logging

import pytest
import torch

from circuitkit.backends.eap import eap_utils


class _Tok:
    padding_side = "right"
    pad_token_id = 0


class _Model:
    tokenizer = _Tok()

    def to_str_tokens(self, tokens):  # used by a one-time debug log in the function
        return [str(int(t)) for t in tokens]


def _fake_tokenize_plus(model, inputs, padding_side=None, templated=False, max_length=None):
    """Clean gets length 5, corrupted gets length 4 -> every pair misaligned."""
    batch = len(inputs)
    is_clean = inputs[0].startswith("clean")
    seq_len = 5 if is_clean else 4
    tokens = torch.ones(batch, seq_len, dtype=torch.long)
    mask = torch.ones(batch, seq_len, dtype=torch.long)
    lengths = torch.full((batch,), seq_len, dtype=torch.long)
    return tokens, mask, lengths, seq_len


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    monkeypatch.setattr(eap_utils, "tokenize_plus", _fake_tokenize_plus)
    for attr in ("_misaligned_batches", "_misaligned_pairs", "has_warned", "_pad_verified"):
        if hasattr(eap_utils.tokenize_batch_pair, attr):
            delattr(eap_utils.tokenize_batch_pair, attr)
    yield


def _run_batches(n):
    for _ in range(n):
        eap_utils.tokenize_batch_pair(_Model(), ["clean a", "clean b"], ["corr a", "corr b"])


def test_warns_on_first_and_101st_batch_not_in_between(caplog):
    with caplog.at_level(logging.WARNING, logger=eap_utils.logger.name):
        _run_batches(101)
    warnings = [r for r in caplog.records if "unequal clean/corrupt" in r.getMessage()]
    # 1st and 101st batch warn; 2..100 are throttled — but never permanently silenced.
    assert len(warnings) == 2, [w.getMessage()[:60] for w in warnings]
    assert "202 misaligned pair(s) across 101 batch(es)" in warnings[-1].getMessage()


def test_later_dataset_still_warns_after_earlier_one(caplog):
    """The old has_warned latch failed exactly this: dataset B after dataset A."""
    with caplog.at_level(logging.WARNING, logger=eap_utils.logger.name):
        _run_batches(100)  # "dataset A": warns once (batch 1)
        _run_batches(1)  # "dataset B": batch 101 -> warns again
    warnings = [r for r in caplog.records if "unequal clean/corrupt" in r.getMessage()]
    assert len(warnings) == 2


def test_aligned_batches_never_warn_or_count(caplog):
    def _aligned(model, inputs, padding_side=None, templated=False, max_length=None):
        batch = len(inputs)
        tokens = torch.ones(batch, 5, dtype=torch.long)
        mask = torch.ones(batch, 5, dtype=torch.long)
        return tokens, mask, torch.full((batch,), 5, dtype=torch.long), 5

    import unittest.mock as m

    with m.patch.object(eap_utils, "tokenize_plus", _aligned):
        with caplog.at_level(logging.WARNING, logger=eap_utils.logger.name):
            for _ in range(3):
                eap_utils.tokenize_batch_pair(_Model(), ["clean"], ["corr"])
    assert not [r for r in caplog.records if "unequal clean/corrupt" in r.getMessage()]
    assert getattr(eap_utils.tokenize_batch_pair, "_misaligned_batches", 0) == 0
