"""Pillar 2 must reuse clean/corrupt baselines that Pillar 1 already computed."""

from types import SimpleNamespace

import pytest
import torch

from circuitkit.evaluation.pillars import ablation as ab


def _run(monkeypatch, **kw):
    calls = []
    monkeypatch.setattr(ab, "evaluate_baseline", lambda *a, **k: calls.append(k) or torch.zeros(2))
    monkeypatch.setattr(ab, "evaluate_graph", lambda **k: torch.tensor([0.5, 0.5]))
    model = SimpleNamespace(cfg=SimpleNamespace(use_attn_result=True))
    result = ab.Pillar2_Ablation.run(
        model, graph=object(), dataloader=None, metric_fn=None, quiet=True, **kw
    )
    return result, calls


def test_precomputed_baselines_skip_recomputation(monkeypatch):
    result, calls = _run(monkeypatch, clean_score=1.0, corrupt_score=0.0)
    assert calls == []
    assert result["clean_score"] == 1.0 and result["corrupt_score"] == 0.0
    assert result["score"] == pytest.approx(0.5)


def test_baselines_computed_when_only_one_is_given(monkeypatch):
    _, calls = _run(monkeypatch, clean_score=1.0)
    assert len(calls) == 2
