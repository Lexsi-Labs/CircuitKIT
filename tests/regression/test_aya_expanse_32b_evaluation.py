"""Group F — gated real-weight evaluation regression for Aya Expanse 32B (cohere1).

Mirrors ``test_aya_expanse_evaluation.py``: runs ``cdt`` discovery (the same
case as in ``test_aya_expanse_32b_discovery.py``; not yet passing on 32B, see ``docs/advanced/experimental-models.md``) on
``greater_than``, then evaluates the resulting circuit's faithfulness via
``circuitkit.api.evaluate_circuit`` and asserts the returned
``FaithfulnessReport``'s ``patching_score``/``ablation_score`` are present
and finite.

Double-load avoidance: the model is loaded exactly ONCE via
``circuitkit.load_model`` (``n_devices=2`` -- see the discovery file's
module docstring for the measured hardware this choice is based on) and the
same handle is threaded through both ``discover_circuit(..., _model=model)``
and ``evaluate_circuit(..., _model=model)``.

Gating mirrors ``test_aya_expanse_32b_discovery.py``: opt in with
``CIRCUITKIT_RUN_AYA_EXPANSE_32B=1`` and provide ``HF_TOKEN``. Marked ``slow``.

    CIRCUITKIT_RUN_AYA_EXPANSE_32B=1 HF_TOKEN=... \
        python -m pytest tests/regression/test_aya_expanse_32b_evaluation.py -v

**Status as of this run: blocked at the same ``n_devices=2`` model load as
discovery** -- see ``test_aya_expanse_32b_discovery.py``'s module docstring
for the two confirmed, independent root causes (a transformer-lens==3.8.0
multi-GPU block-placement bug, and this model's activation memory needs
exceeding this run's hardware regardless). Not run to completion here.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

MODEL_NAME = "CohereLabs/aya-expanse-32b"

_OPT_IN = os.environ.get("CIRCUITKIT_RUN_AYA_EXPANSE_32B", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))
_FULL_DEPTH_OPT_IN = os.environ.get(
    "CIRCUITKIT_RUN_AYA_EXPANSE_32B_FULL", ""
).strip().lower() not in ("", "0", "false", "no")
# The full 40-layer model does not fit one 48 GiB GPU, and transformer-lens 3.8.0's multi-GPU
# (n_devices) block placement is broken, so everything that loads it at full depth through
# TransformerLens fails today. Kept behind a second flag so the documented opt-in runs only
# what can pass. See docs/advanced/experimental-models.md#tests.
_NEEDS_FULL_DEPTH = pytest.mark.skipif(
    not _FULL_DEPTH_OPT_IN,
    reason=(
        "needs the full 40-layer Aya Expanse 32B loaded through TransformerLens across GPUs, "
        "which transformer-lens 3.8.0 cannot do (n_devices block placement bug); set "
        "CIRCUITKIT_RUN_AYA_EXPANSE_32B_FULL=1 to run it anyway"
    ),
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_OPT_IN and _HAS_TOKEN),
        reason=(
            "real-weight Aya Expanse 32B evaluation: set CIRCUITKIT_RUN_AYA_EXPANSE_32B=1 "
            "and HF_TOKEN to run (loads a ~32B/60GB checkpoint once, shared across "
            "discovery + evaluation)"
        ),
    ),
    _NEEDS_FULL_DEPTH,
]


def test_faithfulness_report_is_finite_on_greater_than(tmp_path):
    from circuitkit import load_model
    from circuitkit.api import discover_circuit, evaluate_circuit
    from circuitkit.evaluation.report import FaithfulnessReport

    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    model = load_model(MODEL_NAME, dtype="bfloat16", n_devices=max(1, min(2, n_devices)))
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
