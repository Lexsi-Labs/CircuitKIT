"""Regression tests — steering / baseline wiring bugs (hardening pass).

Covers:
  * Bug 8  — Pillar5_Baselines must actually RUN the 'wanda' baseline. It was
             previously commented out, so requesting 'wanda' silently fell
             through to the "Unknown baseline type" branch and produced no
             wanda result.
  * Bug 10 — Constructing ``ActivationSteering(model, ...)`` must enable
             ``model.cfg.use_attn_result = True`` (steering hooks attach to
             ``attn.hook_result``, which only materialises when that flag is
             set).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Bug 8 — Pillar5_Baselines actually runs the wanda baseline
# ---------------------------------------------------------------------------


def _mock_model(use_attn_result=True):
    model = MagicMock()
    model.cfg.use_attn_result = use_attn_result
    model.cfg.device = "cpu"
    model.cfg.dtype = torch.float32
    return model


def _mock_graph(n_nodes=6):
    class MockNode:
        def __init__(self, name, score):
            self.name = name
            self.score = torch.tensor(score)
            self.in_graph = True

    class MockGraph:
        def __init__(self):
            self.nodes = {
                f"node_{i}": MockNode(f"node_{i}", float(i) * 0.1) for i in range(n_nodes)
            }
            self.neurons_in_graph = None
            self.neurons_scores = None

    return MockGraph()


def test_pillar5_dispatches_to_wanda_baseline(monkeypatch):
    """run(baseline_types=['wanda']) must call _evaluate_wanda_baseline.

    The historical bug commented out the ``elif baseline_type == 'wanda'``
    branch, so 'wanda' hit the ``else: logger.warning("Unknown baseline ...")``
    path and produced no wanda result. We spy on the three baseline evaluators
    and assert ONLY the wanda one fired and that 'wanda' appears in the result.
    """
    from circuitkit.evaluation.pillars.baselines import Pillar5_Baselines

    calls = {"wanda": 0, "random": 0, "magnitude": 0}

    monkeypatch.setattr(
        Pillar5_Baselines,
        "_evaluate_circuit",
        staticmethod(lambda *a, **k: 0.8),
    )

    def _spy(name, score):
        def _inner(*a, **k):
            calls[name] += 1
            return score

        return staticmethod(_inner)

    monkeypatch.setattr(Pillar5_Baselines, "_evaluate_wanda_baseline", _spy("wanda", 0.5))
    monkeypatch.setattr(Pillar5_Baselines, "_evaluate_random_baseline", _spy("random", 0.4))
    monkeypatch.setattr(Pillar5_Baselines, "_evaluate_magnitude_baseline", _spy("magnitude", 0.45))
    # Keep sparsity helpers cheap / deterministic.
    monkeypatch.setattr(Pillar5_Baselines, "_is_neuron_level", staticmethod(lambda g: False))
    monkeypatch.setattr(Pillar5_Baselines, "_extract_circuit_nodes", staticmethod(lambda g: {}))

    result = Pillar5_Baselines.run(
        model=_mock_model(),
        graph=_mock_graph(),
        dataloader=MagicMock(),
        metric_fn=MagicMock(),
        baseline_types=["wanda"],
        device="cpu",
        quiet=True,
    )

    assert calls["wanda"] == 1, (
        "Pillar5_Baselines did not invoke _evaluate_wanda_baseline — the "
        "'wanda' branch is missing/commented out (it fell through to the "
        "'Unknown baseline type' path)."
    )
    assert calls["random"] == 0 and calls["magnitude"] == 0
    assert "wanda" in result.get(
        "baselines", {}
    ), "'wanda' baseline absent from the results — it was silently skipped."
    assert result["baselines"]["wanda"]["score"] == 0.5


# ---------------------------------------------------------------------------
# Bug 10 — ActivationSteering enables use_attn_result
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_no_attn_result():
    """Fresh gpt2 with use_attn_result explicitly OFF — the bug precondition."""
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer

    m = HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.float32)
    m.cfg.use_attn_result = False  # explicitly off so the test is meaningful
    return m


def test_activation_steering_enables_use_attn_result(gpt2_no_attn_result):
    """Constructing ActivationSteering must flip use_attn_result to True.

    Steering hooks attach to ``attn.hook_result``; without ``use_attn_result``
    that hook point never materialises and steering is a silent no-op.
    """
    from circuitkit.applications.steering.steering import ActivationSteering

    model = gpt2_no_attn_result
    assert model.cfg.use_attn_result is False, "precondition not set up"

    ActivationSteering(model, {"A0.0": 0.9}, score_threshold=0.5)

    assert model.cfg.use_attn_result is True, (
        "ActivationSteering.__init__ did not enable model.cfg.use_attn_result — "
        "steering hooks would attach to a non-existent hook_result."
    )
