import torch

from circuitkit.api import _validate_ibcircuit_dataloader
from circuitkit.backends.ibcircuit.ib_noise import apply_ib_noise
from circuitkit.data.normalized import (
    ContrastiveRecord,
    ContrastSource,
    DatasetShape,
    NormalizedDataset,
)
from circuitkit.data.normalized_task import NormalizedTaskSpec


class _ToyTokenizer:
    pad_token_id = 99
    eos_token_id = 98

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == " ":
            return [0]
        if text.startswith(" "):
            return [0, self._token_id(text.strip())]
        return [self._token_id(text)]

    def _token_id(self, text):
        return {
            "yes": 11,
            "no": 12,
            "maybe": 13,
        }.get(text, 50 + len(text))


class _ToyModel:
    tokenizer = _ToyTokenizer()

    def to_tokens(self, text, prepend_bos=True):
        base = [1] if prepend_bos else []
        base.extend(20 + (ord(ch) % 17) for ch in text if ch != " ")
        return torch.tensor([base], dtype=torch.long)


def _paired_dataset(n_records=5):
    return NormalizedDataset(
        name="toy_ib_nan_regression",
        shape=DatasetShape.QA,
        source="unit-test",
        records=[
            ContrastiveRecord(
                record_id=f"r{i}",
                clean_prompt=f"Question {i}?",
                clean_answer=" yes",
                corrupt_prompt=f"Counterfactual {i}?",
                corrupt_answer=" no",
                contrast_source=ContrastSource.GENERATED,
            )
            for i in range(n_records)
        ],
    )


class _JointEncodingTokenizer:
    """Tokenizer where ' No'/'No' are distinct tokens and joint encoding works.

    Mimics BPE tokenizers (e.g. tiktoken/Qwen) where the space is merged
    into the following token (' No' != 'No') and concatenation is composable
    (encode(A + B) starts with encode(A)).
    """

    pad_token_id = 99
    eos_token_id = 98

    # Fixed vocabulary: space-prefixed and bare forms are different IDs.
    _vocab = {
        "Assistant:": 200,
        " No": 300,   # space merged into token — what the model actually predicts
        "No": 301,    # bare form — wrong label if model generates ' No'
        " Yes": 400,
        "Yes": 401,
    }

    def encode(self, text, add_special_tokens=False):
        """Greedy longest-match tokenization on the fixed vocab."""
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for tok, tid in sorted(self._vocab.items(), key=lambda x: -len(x[0])):
                if text[i:].startswith(tok):
                    ids.append(tid)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ids.append(ord(text[i]) % 900 + 100)  # fallback: char ID
                i += 1
        return ids


class _JointEncodingModel:
    tokenizer = _JointEncodingTokenizer()

    def to_tokens(self, text, prepend_bos=True):
        base = [1] if prepend_bos else []
        base.extend(self.tokenizer.encode(text))
        return torch.tensor([base], dtype=torch.long)


def _joint_encoding_dataset(n_records=3):
    return NormalizedDataset(
        name="joint_encoding_test",
        shape=DatasetShape.QA,
        source="unit-test",
        records=[
            ContrastiveRecord(
                record_id=f"r{i}",
                clean_prompt="Assistant:",
                clean_answer=" No",   # space-prefixed: what the model generates
                corrupt_prompt="Assistant:",
                corrupt_answer=" Yes",
                contrast_source=ContrastSource.GENERATED,
            )
            for i in range(n_records)
        ],
    )


def test_ibcircuit_joint_encoding_uses_context_token_not_bare_token(tmp_path):
    """Joint encoding resolves ' No' (token 300) not 'No' (token 301).

    This guards the fix for the leading-space ambiguity bug: standalone
    encode('No') would give token 301, but the model actually predicts
    token 300 (' No') after 'Assistant:'. Joint encode('Assistant: No')
    correctly extracts 300 as the first continuation token.
    """
    task = NormalizedTaskSpec(_joint_encoding_dataset(), cache_dir=str(tmp_path))

    dataloader = task.build_dataloader(
        _JointEncodingModel(),
        {"algorithm": "ibcircuit", "batch_size": 1, "data_params": {"num_examples": 3}},
        device="cpu",
    )

    batch = next(iter(dataloader))
    # All labels should be 300 (' No' token), NOT 301 ('No' token).
    assert batch["labels"].tolist() == [300, 300, 300]


def test_ibcircuit_normalized_task_uses_one_multi_example_batch_for_nan_stability(tmp_path):
    task = NormalizedTaskSpec(_paired_dataset(), cache_dir=str(tmp_path))

    dataloader = task.build_dataloader(
        _ToyModel(),
        {
            "algorithm": "ibcircuit",
            "batch_size": 1,
            "data_params": {"num_examples": 5},
        },
        device="cpu",
    )

    assert len(dataloader) == 1
    batch = next(iter(dataloader))
    assert batch["tokens"].shape[0] == 5
    assert batch["labels"].tolist() == [11, 11, 11, 11, 11]
    assert batch["answer_positions"].shape == (5,)
    _validate_ibcircuit_dataloader(dataloader)

    activation = batch["tokens"].float().unsqueeze(-1).repeat(1, 1, 3)
    ib_weight = torch.zeros_like(activation)
    noisy_activation, kl_loss = apply_ib_noise(
        activation,
        ib_weight,
        mask_type="raw",
    )

    assert torch.isfinite(noisy_activation).all()
    assert torch.isfinite(kl_loss)
