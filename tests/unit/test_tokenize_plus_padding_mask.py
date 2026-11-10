"""tokenize_plus masks the pads on the side it actually padded, not the tokenizer's default side."""

import torch

from circuitkit.backends.eap import eap_utils

PAD, BOS = 0, 2


class _Tok:
    padding_side = "right"  # HF default for most tokenizers (Cohere2, Llama, GPT-2 ...)
    pad_token_id = PAD
    bos_token_id = BOS


class _Model:
    """to_tokens pads on the requested side, as HookedTransformer.to_tokens does."""

    tokenizer = _Tok()

    class cfg:
        n_ctx = 64

    def to_tokens(self, inputs, prepend_bos=True, padding_side=None, truncate=False):
        rows = [([BOS] if prepend_bos else []) + [10 + len(w) for w in s.split()] for s in inputs]
        n = max(len(r) for r in rows)
        side = padding_side or self.tokenizer.padding_side
        return torch.tensor(
            [([PAD] * (n - len(r)) + r) if side == "left" else (r + [PAD] * (n - len(r))) for r in rows]
        )


def test_left_padding_masks_leading_pads_when_tokenizer_default_is_right():
    tokens, mask, lengths, n_pos = eap_utils.tokenize_plus(
        _Model(), ["a b", "a b c d e"], padding_side="left"
    )
    assert tokens[0, :3].tolist() == [PAD, PAD, PAD]
    assert mask[0].tolist() == [0, 0, 0, 1, 1, 1]  # 3 leading pads, then BOS a b
    assert mask[1].tolist() == [1] * 6


def test_right_padding_unchanged():
    tokens, mask, lengths, n_pos = eap_utils.tokenize_plus(
        _Model(), ["a b", "a b c d e"], padding_side="right"
    )
    assert mask[0].tolist() == [1, 1, 1, 0, 0, 0]
    assert lengths.tolist() == [3, 6]
