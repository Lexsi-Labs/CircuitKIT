"""tokenize_batch_pair forwards max_length to tokenize_plus and stays unbounded by default."""

from unittest.mock import patch

import torch

from circuitkit.backends.eap import eap_utils


class _Tok:
    padding_side = "right"
    pad_token_id = 0


class _Model:
    tokenizer = _Tok()

    def to_str_tokens(self, tokens):  # tokenize_batch_pair's one-time pad-verified debug log
        return [str(int(t)) for t in tokens]


def _stub(seen):
    def tokenize_plus(model, inputs, max_length=None, padding_side=None, templated=False):
        seen.append(max_length)
        n = len(inputs)
        return torch.ones(n, 3, dtype=torch.long), torch.ones(n, 3, dtype=torch.long), torch.full((n,), 3), 3

    return tokenize_plus


def test_max_length_none_by_default():
    seen = []
    with patch.object(eap_utils, "tokenize_plus", _stub(seen)):
        eap_utils.tokenize_batch_pair(_Model(), ["a"], ["b"])
    assert seen == [None, None]


def test_max_length_forwarded_to_both_sides():
    seen = []
    with patch.object(eap_utils, "tokenize_plus", _stub(seen)):
        eap_utils.tokenize_batch_pair(_Model(), ["a"], ["b"], max_length=128)
    assert seen == [128, 128]
