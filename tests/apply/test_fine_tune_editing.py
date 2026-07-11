import types

import torch
import torch.nn as nn

from circuitkit.applications.editing import fine_tune_editing as fte


class TinyEditableModel(nn.Module):
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, vocab_size))
        self.cfg = types.SimpleNamespace(device="cpu")

    def forward(self, input_ids):
        return self.weight[input_ids]


class FakeTeacherForced:
    def __init__(self, prompt: str, target: str, vocab_size: int = 32):
        prompt_id = (sum(map(ord, prompt)) % (vocab_size - 1)) + 1
        target_id = (sum(map(ord, target)) % (vocab_size - 1)) + 1
        self.full_ids = torch.tensor([[prompt_id, target_id]], dtype=torch.long)
        self.prompt_len = 1
        self.target_len = 1
        self.target_ids = torch.tensor([target_id], dtype=torch.long)


def _patch_token_helpers(monkeypatch, vocab_size: int = 32):
    def fake_build_teacher_forced(model, prompt, target):
        if target == "boom":
            raise fte.ScoringError("bad target")
        return FakeTeacherForced(prompt, target, vocab_size=vocab_size)

    def fake_score_target(model, prompt, target):
        seq = fake_build_teacher_forced(model, prompt, target)
        logits = model(seq.full_ids)
        probs = torch.softmax(logits[0, 0], dim=-1)
        return types.SimpleNamespace(first_token_prob=float(probs[seq.target_ids[0]].item()))

    monkeypatch.setattr(fte, "build_teacher_forced", fake_build_teacher_forced)
    monkeypatch.setattr(fte, "score_target", fake_score_target)


def test_init(monkeypatch):
    _patch_token_helpers(monkeypatch)
    model = TinyEditableModel()
    handler = fte.FineTuneEditHandler(model, steps=1, lr=0.1)
    assert handler.model is model
    assert handler.device == "cpu"
    assert handler.edit_history == []


def test_single_edit_improves_target_probability(monkeypatch):
    _patch_token_helpers(monkeypatch)
    model = TinyEditableModel()
    handler = fte.FineTuneEditHandler(model, steps=5, lr=0.5)

    before = handler._get_fact_confidence("The capital of France is", "Paris")
    result = handler.edit_single_fact(
        prompt="The capital of France is",
        subject="France",
        target="Paris",
        verify=True,
    )
    after = handler._get_fact_confidence("The capital of France is", "Paris")

    assert result.success is True
    assert after >= before
    assert result.confidence_after == after
    assert result.edit_magnitude > 0.0


def test_batch_edit(monkeypatch):
    _patch_token_helpers(monkeypatch)
    model = TinyEditableModel()
    handler = fte.FineTuneEditHandler(model, steps=3, lr=0.4, batch_size=2)

    facts = [
        ("The capital of France is", "France", "Paris"),
        ("The capital of Germany is", "Germany", "Berlin"),
    ]
    results = handler.edit_multiple_facts(facts, verify=True)

    assert len(results) == 2
    assert all(r.success for r in results)
    assert all(r.edit_magnitude > 0.0 for r in results)


def test_rollback_on_failure(monkeypatch):
    _patch_token_helpers(monkeypatch)
    model = TinyEditableModel()
    handler = fte.FineTuneEditHandler(model, steps=1, lr=0.2)
    original = {k: v.clone() for k, v in model.state_dict().items()}

    result = handler.edit_single_fact(
        prompt="The capital of France is",
        subject="France",
        target="boom",
        rollback_on_failure=True,
    )

    assert result.success is False
    assert result.error_message
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
