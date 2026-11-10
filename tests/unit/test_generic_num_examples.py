"""GenericTaskSpec honours n_examples from ck.discover / Pipeline.discover (data_params.num_examples)."""

import json

import pytest
import torch

from circuitkit.quick import build_discovery_config
from circuitkit.tasks import GenericTaskSpec


class _Tok:
    padding_side = "right"
    pad_token_id = 0
    bos_token_id = 1
    chat_template = None


class _Model:
    """Enough of HookedTransformer for GenericTaskSpec.build_dataloader and build_discovery_config."""

    tokenizer = _Tok()

    class cfg:
        tokenizer_name = "toy"
        model_name = "toy"
        dtype = torch.float32

    def to_tokens(self, text, prepend_bos=True, padding_side=None, **_):
        ids = ([1] if prepend_bos else []) + [2 + (sum(map(ord, w)) % 50) for w in str(text).split()]
        return torch.tensor([ids])


@pytest.fixture
def ten_pair_jsonl(tmp_path):
    p = tmp_path / "pairs.jsonl"
    with open(p, "w") as f:
        for i in range(10):
            row = {"prompt": f"q{i} a", "answer": "yes", "corrupted_prompt": f"q{i} b", "corrupted_answer": "no"}
            f.write(json.dumps(row) + "\n")
    return str(p)


def _spec(path):
    return GenericTaskSpec(
        name="toy_pairs",
        source={"type": "jsonl", "path_or_id": path},
        schema={"prompt": "prompt", "answer": "answer",
                "corrupted_prompt": "corrupted_prompt", "corrupted_answer": "corrupted_answer"},
        chat_template_mode="off",
    )


def test_n_examples_from_discover_config_caps_examples(ten_pair_jsonl):
    discovery_cfg = build_discovery_config(_Model(), "toy_pairs", algorithm="eap-ig", n_examples=3, batch_size=1)["discovery"]
    assert "num_examples" not in discovery_cfg  # ck.discover nests it under data_params
    loader = _spec(ten_pair_jsonl).build_dataloader(_Model(), discovery_cfg, "cpu")
    assert len(loader.dataset) == 3


def test_top_level_num_examples_still_wins(ten_pair_jsonl):
    cfg = {"algorithm": "eap", "batch_size": 1, "num_examples": 4, "data_params": {"num_examples": 7}}
    loader = _spec(ten_pair_jsonl).build_dataloader(_Model(), cfg, "cpu")
    assert len(loader.dataset) == 4


def test_invalid_nested_num_examples_rejected():
    with pytest.raises(ValueError, match="num_examples"):
        GenericTaskSpec.validate_discovery_config(
            None, {"algorithm": "eap", "data_params": {"num_examples": 0}}
        )
