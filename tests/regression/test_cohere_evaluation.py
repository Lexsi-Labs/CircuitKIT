"""Group F — gated real-weight evaluation regression for tiny-aya (cohere2).

Stage 5 (interventions & evaluation validation): proves the second of the
three required CircuitKIT surfaces -- discovery (Stage 1) and interventions
(``test_cohere_interventions.py``) are the other two -- by running circuit
*evaluation* (faithfulness) on a real discovered circuit.

Runs ``cdt`` discovery (already validated for this model in
``test_cohere_discovery.py`` -- no qkv-flag activation blow-up, no
OOM-preflight concerns) on ``greater_than``, then evaluates the resulting
circuit's faithfulness via ``circuitkit.api.evaluate_circuit`` and asserts
the returned ``FaithfulnessReport``'s ``patching_score``/``ablation_score``
are present and finite.

Double-load avoidance (the OOM audit's Stage 5 instruction): the model is
loaded exactly ONCE via ``HookedTransformer.from_pretrained`` and the same
handle is threaded through both ``discover_circuit(..., _model=model)`` and
``evaluate_circuit(..., _model=model)``. Without the explicit ``_model=``
handle, ``evaluate_circuit`` would call ``HookedTransformer.from_pretrained``
internally a second time, loading a second full copy of the gated checkpoint.

Gating mirrors ``test_cohere_discovery.py``: opt in with
``CIRCUITKIT_RUN_TINY_AYA=1`` and provide ``HF_TOKEN``. Marked ``slow``.

    CIRCUITKIT_RUN_TINY_AYA=1 HF_TOKEN=... \
        python -m pytest tests/regression/test_cohere_evaluation.py -v
"""

from __future__ import annotations

import math
import os

import pytest
import torch

MODEL_NAME = "CohereLabs/tiny-aya-base"

_OPT_IN = os.environ.get("CIRCUITKIT_RUN_TINY_AYA", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_OPT_IN and _HAS_TOKEN),
        reason=(
            "gated tiny-aya evaluation: set CIRCUITKIT_RUN_TINY_AYA=1 and HF_TOKEN "
            "to run (loads a 3.35B gated checkpoint once, shared across discovery "
            "+ evaluation)"
        ),
    ),
]


def test_faithfulness_report_is_finite_on_greater_than(tmp_path):
    from transformer_lens import HookedTransformer

    from circuitkit.api import discover_circuit, evaluate_circuit
    from circuitkit.evaluation.report import FaithfulnessReport

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
    try:
        out = tmp_path / "cdt.pt"
        cache_dir = str(tmp_path / "cache")
        cfg = {
            "model": {"name": MODEL_NAME, "precision": "bfloat16"},
            "discovery": {
                "algorithm": "cdt",
                "task": "greater_than",
                "level": "node",
                "batch_size": 2,
                "data_params": {"num_examples": 4, "seed": 42, "cache_dir": cache_dir},
                "cache_dir": cache_dir,
            },
            "pruning": {"target_sparsity": 0.3, "scope": "heads"},
            "eval": {"pillars": ["patching", "ablation"], "num_examples": 4},
            "output_path": str(out),
        }
        discover_circuit(cfg, _model=model)

        report = evaluate_circuit(cfg, pruned_artifact_path=str(out), _model=model)

        assert isinstance(report, FaithfulnessReport)
        assert report.patching_score is not None, "patching_score (Pillar 1) missing"
        assert math.isfinite(
            report.patching_score
        ), f"patching_score is non-finite: {report.patching_score}"
        assert report.ablation_score is not None, "ablation_score (Pillar 2) missing"
        assert math.isfinite(
            report.ablation_score
        ), f"ablation_score is non-finite: {report.ablation_score}"
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
