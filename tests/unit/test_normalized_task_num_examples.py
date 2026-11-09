"""NormalizedTaskSpec's EAP dataloader honours n_examples (data_params.num_examples from ck.discover)."""

import torch

from circuitkit.data.normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from circuitkit.data.normalized_task import NormalizedTaskSpec


class _Tok:
    padding_side = "right"
    pad_token_id = 0

    def encode(self, s, add_special_tokens=False):
        return [3 + (sum(map(ord, w)) % 50) for w in s.split()]


class _Model:
    tokenizer = _Tok()


def _ds(n):
    recs = [
        ContrastiveRecord(record_id=str(i), clean_prompt=f"harm {i}", corrupt_prompt=f"benign {i}",
                          clean_answer=" I", corrupt_answer=" To", contrast_source=ContrastSource.GENERATED)
        for i in range(n)
    ]
    return NormalizedDataset(name="toy", shape=DatasetShape.TEMPLATE, records=recs, source="test", meta={})


def test_eap_loader_caps_at_data_params_num_examples(tmp_path):
    spec = NormalizedTaskSpec(_ds(10), cache_dir=str(tmp_path))
    cfg = {"algorithm": "eap-ig", "batch_size": 2, "data_params": {"num_examples": 4, "batch_size": 2}}
    assert len(spec.build_dataloader(_Model(), cfg, "cpu").dataset) == 4


def test_cap_is_part_of_the_cache_key(tmp_path):
    spec = NormalizedTaskSpec(_ds(10), cache_dir=str(tmp_path))
    full = {"algorithm": "eap-ig", "batch_size": 2}
    capped = {"algorithm": "eap-ig", "batch_size": 2, "data_params": {"num_examples": 3}}
    assert len(spec.build_dataloader(_Model(), full, "cpu").dataset) == 10
    assert len(spec.build_dataloader(_Model(), capped, "cpu").dataset) == 3
    assert len(spec.build_dataloader(_Model(), full, "cpu").dataset) == 10
