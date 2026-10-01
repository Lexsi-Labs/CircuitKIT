"""Group F — gated end-to-end discovery regression for Aya Expanse 8B (cohere1).

Runs the full ``discover_circuit`` pipeline on the real Aya Expanse checkpoint
and asserts the produced node scores are finite and non-degenerate, for five
of the six *stable* discovery algorithms (``eap``, ``eap-ig``, ``eap-gp``,
``ibcircuit``, ``cdt`` -- see ``circuitkit.backends.ALGORITHMS``) on
``greater_than``, which builds its own single-token operand pool against the
loaded tokenizer and so is valid on Aya Expanse's 256k vocab without the IOI
name-splitting caveat (same tokenizer family/size as Command R7B).

``acdc`` (the sixth stable algorithm) is intentionally excluded from the
standard gate, per the same reasoning ``test_command_r7b_discovery.py``
documents: it needs the same per-head qkv-flag activation blow-up as the EAP
family, which the repo's own OOM preflight correctly refuses on a single 47
GB GPU for a ~8B model (an even larger footprint than Command R7B's 7B); the
CPU fallback took over 2.5 hours on the smaller 7B model without finishing.
That is not a practical regression gate, and per explicit instruction ACDC is
excluded from real discovery regression runs for large models. The
``test_acdc_...`` case below is kept (structurally identical to Command R7B's
-- it does work) but is skipped by default behind a second, separate opt-in
so it doesn't silently eat hours of every real-weight run.

Gating mirrors ``test_aya_expanse_parity.py``: opt in with
``CIRCUITKIT_RUN_AYA_EXPANSE=1`` and provide ``HF_TOKEN``. Each case loads the
checkpoint fresh (discovery mutates ``model.cfg``), so run this with adequate
VRAM/RAM. Marked ``slow``.

    CIRCUITKIT_RUN_AYA_EXPANSE=1 HF_TOKEN=... \
        python -m pytest tests/regression/test_aya_expanse_discovery.py -v

    # ACDC only, opted in separately (multi-hour, not run as part of the
    # Stage 3 gate -- see the Stage 3 handoff notes):
    CIRCUITKIT_RUN_AYA_EXPANSE=1 CIRCUITKIT_RUN_AYA_EXPANSE_ACDC=1 HF_TOKEN=... \
        python -m pytest tests/regression/test_aya_expanse_discovery.py -v -k acdc
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

MODEL_NAME = "CohereLabs/aya-expanse-8b"

_OPT_IN = os.environ.get("CIRCUITKIT_RUN_AYA_EXPANSE", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))
_ACDC_OPT_IN = os.environ.get("CIRCUITKIT_RUN_AYA_EXPANSE_ACDC", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_OPT_IN and _HAS_TOKEN),
        reason=(
            "real-weight Aya Expanse discovery: set CIRCUITKIT_RUN_AYA_EXPANSE=1 and "
            "HF_TOKEN to run (loads a ~8B checkpoint per algorithm)"
        ),
    ),
]


def _run_discovery(tmp_path, algorithm, task, **discovery_extra):
    from circuitkit.api import discover_circuit

    out = tmp_path / f"{algorithm}.pt"
    cache_dir = str(tmp_path / "cache")
    cfg = {
        "model": {"name": MODEL_NAME, "precision": "bfloat16"},
        "discovery": {
            "algorithm": algorithm,
            "task": task,
            "level": "node",
            "batch_size": 2,
            "data_params": {"num_examples": 4, "seed": 42, "cache_dir": cache_dir},
            "cache_dir": cache_dir,
            **discovery_extra,
        },
        "pruning": {"target_sparsity": 0.3, "scope": "heads"},
        "output_path": str(out),
    }
    discover_circuit(cfg)

    scores_path = str(out).replace(".pt", "_scores.pt")
    data = torch.load(scores_path, map_location="cpu", weights_only=False)
    return np.array([float(v) for v in data["node_scores"].values()])


class TestDiscoveryProducesFiniteScores:
    @pytest.mark.parametrize(
        "algorithm,extra",
        [
            ("eap", {}),
            ("eap-ig", {"ig_steps": 5}),
            ("eap-gp", {}),
            ("cdt", {}),
            # IBCircuit trains a fixed-batch mask (unlike the inference-only
            # EAP family); on an 8B model at bf16 its own memory guard needs
            # batch_size=1 to fit a single 47 GB GPU's free VRAM after weights
            # (same lever Command R7B's 7B gate needed).
            ("ibcircuit", {"scope": "heads", "batch_size": 1}),
        ],
    )
    def test_scores_are_finite_and_non_degenerate_on_greater_than(self, tmp_path, algorithm, extra):
        scores = _run_discovery(tmp_path, algorithm, "greater_than", **extra)

        assert scores.size > 0, f"{algorithm}: no node scores produced"
        assert np.isfinite(scores).all(), f"{algorithm}: non-finite node scores"
        assert np.abs(scores).max() > 0.0, (
            f"{algorithm}: all node scores are zero -- attribution produced no "
            f"signal on aya-expanse-8b"
        )

    @pytest.mark.skipif(
        not _ACDC_OPT_IN,
        reason=(
            "acdc on an 8B model needs the CPU fallback (OOM preflight refuses "
            "the qkv-flag blow-up on a single 47GB GPU) and is confirmed "
            "impractically slow there (2.5+ hours on the smaller 7B Command R7B "
            "model without finishing) -- opt in explicitly with "
            "CIRCUITKIT_RUN_AYA_EXPANSE_ACDC=1 to run it deliberately"
        ),
    )
    def test_acdc_scores_are_finite_and_non_degenerate_on_ioi(self, tmp_path):
        scores = _run_discovery(
            tmp_path,
            "acdc",
            "ioi",
            tao_exps=[-3],
            tao_bases=[1],
        )

        assert scores.size > 0, "acdc: no node scores produced"
        assert np.isfinite(scores).all(), "acdc: non-finite node scores"
        assert (
            np.abs(scores).max() > 0.0
        ), "acdc: all node scores are zero -- attribution produced no signal on aya-expanse-8b"
