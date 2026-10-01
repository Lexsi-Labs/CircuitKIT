"""Group F — gated end-to-end discovery regression for SmolLM3-3B.

Runs the full ``discover_circuit`` pipeline on the real SmolLM3-3B checkpoint
and asserts the produced node scores are finite and non-degenerate, for five
of the six *stable* discovery algorithms (``eap``, ``eap-ig``, ``eap-gp``,
``ibcircuit``, ``cdt`` -- see ``circuitkit.backends.ALGORITHMS``) on
``greater_than``, which builds its own single-token operand pool against the
loaded tokenizer. SmolLM3's tokenizer has a smaller vocab (128256) than
Command R7B / Aya Expanse's 256000, so this was verified rather than assumed
to still work: all 100 unspaced two-digit-or-fewer numbers ("0".."99") are
single tokens under this tokenizer (the spaced form is not, which is exactly
why ``greater_than``'s own tokenizer probe -- ``tasks/specs.py::
encodes_to_single_token`` -- picks the unspaced form dynamically rather than
hard-coding GPT-2's spaced convention; see ``test_greater_than_tokenizer.py``
for the historical bug this probe fixes). All 99 IOI names are also
single-token under this tokenizer, for the ACDC/IOI case below.

``acdc`` (the sixth stable algorithm) is intentionally excluded from the
standard gate, per the same reasoning ``test_command_r7b_discovery.py`` /
``test_aya_expanse_discovery.py`` document and the user's own standing
instruction (recorded in this account's persistent memory): it needs the same
per-head qkv-flag activation blow-up as the EAP family, and was confirmed
impractically slow (2.5+ hours without finishing on a 7B model's CPU
fallback, worse at 8B) -- not a practical regression gate even though
SmolLM3-3B is the smallest of the three models. The ``test_acdc_...`` case
below is kept (structurally identical to the other two models' -- it does
work) but is skipped by default behind a second, separate opt-in so it
doesn't silently eat hours of every real-weight run.

Gating mirrors ``test_smollm3_parity.py``: opt in with
``CIRCUITKIT_RUN_SMOLLM3=1`` (SmolLM3 is a plain public repo, so no HF_TOKEN
is required, unlike the cohere-family models). Each case loads the checkpoint
fresh (discovery mutates ``model.cfg``), so run this with adequate VRAM/RAM.
Marked ``slow``.

    CIRCUITKIT_RUN_SMOLLM3=1 python -m pytest tests/regression/test_smollm3_discovery.py -v

    # ACDC only, opted in separately (multi-hour, not run as part of the
    # Stage 4 gate):
    CIRCUITKIT_RUN_SMOLLM3=1 CIRCUITKIT_RUN_SMOLLM3_ACDC=1 \
        python -m pytest tests/regression/test_smollm3_discovery.py -v -k acdc
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

_OPT_IN = os.environ.get("CIRCUITKIT_RUN_SMOLLM3", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_ACDC_OPT_IN = os.environ.get("CIRCUITKIT_RUN_SMOLLM3_ACDC", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _OPT_IN,
        reason=(
            "real-weight SmolLM3 discovery: set CIRCUITKIT_RUN_SMOLLM3=1 to run "
            "(loads a ~3B checkpoint per algorithm; public, no HF_TOKEN required)"
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
            ("ibcircuit", {"scope": "heads", "batch_size": 1}),
        ],
    )
    def test_scores_are_finite_and_non_degenerate_on_greater_than(self, tmp_path, algorithm, extra):
        scores = _run_discovery(tmp_path, algorithm, "greater_than", **extra)

        assert scores.size > 0, f"{algorithm}: no node scores produced"
        assert np.isfinite(scores).all(), f"{algorithm}: non-finite node scores"
        assert (
            np.abs(scores).max() > 0.0
        ), f"{algorithm}: all node scores are zero -- attribution produced no signal on SmolLM3-3B"

    @pytest.mark.skipif(
        not _ACDC_OPT_IN,
        reason=(
            "acdc needs the CPU fallback (OOM preflight refuses the qkv-flag "
            "blow-up) and is confirmed impractically slow there (2.5+ hours "
            "without finishing on a 7B model) -- opt in explicitly with "
            "CIRCUITKIT_RUN_SMOLLM3_ACDC=1 to run it deliberately"
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
        ), "acdc: all node scores are zero -- attribution produced no signal on SmolLM3-3B"
