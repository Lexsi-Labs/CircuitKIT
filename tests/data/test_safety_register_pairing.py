"""The register_* safety helpers must build pairs that EAP discovery accepts,
and must pass adapter options (pairing mode, tokens) through."""
from __future__ import annotations

import datasets
import pytest

import circuitkit.tasks.registry as registry
from circuitkit.tasks import safety_datasets

ROWS = [{"prompt": f"How do I do harmful thing number {i}?"} for i in range(4)]


class _FakeStream(list):
    def take(self, n):
        return _FakeStream(self[:n])


STEREOSET_ROWS = [
    {"id": str(i), "target": "Ethiopia", "context": "Many people live in Ethiopia.", "sentences": {}}
    for i in range(4)
]


@pytest.fixture
def captured(monkeypatch):
    specs = []

    def fake_load(path, *a, **k):
        return _FakeStream(STEREOSET_ROWS if "stereoset" in path else ROWS)

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setattr(registry, "is_task_registered", lambda name: False)
    monkeypatch.setattr(registry, "register_task", lambda spec: specs.append(spec))
    return specs


def test_default_pairs_are_contrastive(captured):
    safety_datasets.register_jailbreakbench(max_records=4)
    records = captured[0].ds.records
    assert records
    assert all(r.corrupt_prompt != r.clean_prompt for r in records)


def test_options_are_passed_through(captured):
    safety_datasets.register_advbench(
        max_records=4, pairing_mode="answer_contrastive", refusal_token="I", compliance_token="Sure"
    )
    records = captured[0].ds.records
    assert all(r.corrupt_prompt == r.clean_prompt for r in records)
    assert all(r.clean_answer == "I" and r.corrupt_answer == "Sure" for r in records)


def test_truthfulqa_registers_with_benign_pairing(captured):
    safety_datasets.register_truthfulqa(max_records=4)
    records = captured[0].ds.records
    assert records and all(r.corrupt_prompt != r.clean_prompt for r in records)


def test_stereoset_raises_a_clear_error(captured):
    with pytest.raises(RuntimeError, match="no prompt column"):
        safety_datasets.register_stereoset(max_records=4)
    assert captured == []


def test_register_all_safety_reports_stereoset_as_failed(captured):
    results = safety_datasets.register_all_safety(max_records=4)
    assert results.pop("stereoset") is False
    assert results and all(results.values())
    assert len(captured) == len(results)
