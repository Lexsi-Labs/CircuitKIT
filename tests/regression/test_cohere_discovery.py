"""Group F — gated end-to-end discovery regression for tiny-aya (cohere2).

Runs the full ``discover_circuit`` pipeline on the real gated tiny-aya
checkpoint for each discovery backend and asserts the produced node scores are
finite and non-degenerate. This exercises everything the offline unit tests
mock: real model load through the TL port, tokenization, EAP graph
construction, GQA-aware attribution (Group C), and score serialization.

The task is ``greater_than`` on purpose:

* it builds its own single-token operand pool against the loaded tokenizer, so
  it is valid on Cohere's 262k vocab without the IOI name-splitting caveat
  (Group E), and
* it supports exactly the EAP family, CD-T, and IBCircuit — the four backends
  under test here.

ACDC is intentionally not covered: ``greater_than`` does not support it, and
the only ACDC-compatible built-in (IOI) goes through a GPT-2-shaped data path
whose viability on Cohere's tokenizer is a separate, pre-existing question
independent of this port.

Gating mirrors ``test_cohere_parity.py``: opt in with
``CIRCUITKIT_RUN_TINY_AYA=1`` and provide ``HF_TOKEN``. Each case loads the
3.35B checkpoint fresh (discovery mutates ``model.cfg`` — the qkv-input flags
and GQA ungroup — so a shared model would leak state across algorithms), and
EAP-family runs set ``use_attn_result`` etc., so run this with adequate VRAM or
on CPU high-RAM. Marked ``slow``.

    CIRCUITKIT_RUN_TINY_AYA=1 HF_TOKEN=... \
        python -m pytest tests/regression/test_cohere_discovery.py -v
"""

from __future__ import annotations

import os

import numpy as np
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
            "gated tiny-aya discovery: set CIRCUITKIT_RUN_TINY_AYA=1 and HF_TOKEN "
            "to run (loads a 3.35B gated checkpoint per algorithm)"
        ),
    ),
]


def _run_discovery(tmp_path, algorithm, **discovery_extra):
    """Run discovery on greater_than and return the node-score array."""
    from circuitkit.api import discover_circuit

    out = tmp_path / f"{algorithm}.pt"
    cache_dir = str(tmp_path / "cache")
    # bfloat16 matches circuitkit's own config recommendation ("avoid float32
    # to prevent OOM errors") and lets the EAP-family qkv-flag preflight
    # (api._check_qkv_flag_memory_headroom) fit within realistic VRAM: at
    # float32 the estimated 4x-weights activation headroom (~58 GB) does not
    # fit even A100-80GB after cache/reservations, causing the preflight to
    # (correctly) refuse eap/eap-ig. This test asserts functional properties
    # (finite, non-degenerate scores), not numerical parity, so the dtype
    # difference is immaterial.
    cfg = {
        "model": {"name": MODEL_NAME, "precision": "bfloat16"},
        "discovery": {
            "algorithm": algorithm,
            "task": "greater_than",
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
            ("cdt", {}),
            ("ibcircuit", {"scope": "heads"}),
        ],
    )
    def test_scores_are_finite_and_non_degenerate(self, tmp_path, algorithm, extra):
        scores = _run_discovery(tmp_path, algorithm, **extra)

        assert scores.size > 0, f"{algorithm}: no node scores produced"
        assert np.isfinite(scores).all(), f"{algorithm}: non-finite node scores"
        # A working attribution assigns some component nonzero importance; an
        # all-zero score vector means the pass produced no signal at all.
        assert np.abs(scores).max() > 0.0, (
            f"{algorithm}: all node scores are zero -- attribution produced no "
            f"signal on tiny-aya"
        )
