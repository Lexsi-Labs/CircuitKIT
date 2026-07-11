"""Regression test — CD-T all-zero scores bug (LayerNorm folding).

Historical bug
--------------
``backends/cdt/propagation.py`` decomposed TransformerLens LayerNorm by
reading ``ln.w`` / ``ln.b``. When ``discover_circuit`` enables
``use_attn_result`` / ``use_split_qkv_input`` / ``use_hook_mlp_in``,
TransformerLens *folds* the LayerNorm scale/bias into the neighbouring weight
matrices and swaps in a parameter-free ``LayerNormPre`` module that has **no**
``.w`` / ``.b`` attributes. Accessing ``ln.w`` raised ``AttributeError``,
which the adapter caught and turned into an all-zeros fallback — CD-T silently
returned a score of 0.0 for every node, with no crash.

The fix uses ``getattr(ln, "w", None)`` and treats a missing scale as identity
(and missing bias as zero), so ``LayerNormPre`` is handled correctly.

This test fails if the all-zero regression returns: it asserts the CD-T node
scores are NOT all zero and that they span a real (non-degenerate) range.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformer_lens")


@pytest.fixture(scope="module")
def gpt2_folded():
    """gpt2 with LN folded into a parameter-free ``LayerNormPre`` module.

    ``from_pretrained`` with the default ``fold_ln=True`` plus the discovery
    config flags reproduces exactly the model state CD-T sees inside
    ``discover_circuit`` — the state that triggered the original bug.
    """
    from transformer_lens import HookedTransformer

    m = HookedTransformer.from_pretrained("gpt2", device="cpu", dtype=torch.float32)
    m.cfg.use_attn_result = True
    m.cfg.use_split_qkv_input = True
    m.cfg.use_hook_mlp_in = True
    return m


def test_cdt_layernorm_is_folded_pre_variant(gpt2_folded):
    """Sanity check: the fixture really does use the folded ``LayerNormPre``.

    If TransformerLens stopped folding, the bug-trigger condition would no
    longer hold and the rest of the test would be vacuous.
    """
    ln1 = gpt2_folded.blocks[0].ln1
    assert "Pre" in type(ln1).__name__, f"expected a folded LayerNormPre, got {type(ln1).__name__}"
    assert (
        getattr(ln1, "w", None) is None
    ), "folded LayerNormPre unexpectedly exposes a .w attribute"


def test_cdt_scores_not_all_zero(gpt2_folded):
    """CD-T on gpt2/IOI must return real, non-degenerate node scores."""
    from torch.utils.data import DataLoader

    from circuitkit.backends.cdt.adapter import run_cdt_discovery

    # Tiny IOI-style clean/corrupted pairs — enough to exercise propagation.
    clean = [
        "When John and Mary went to the store, John gave a drink to",
        "When Tom and Sara went to the park, Tom gave a ball to",
    ]
    corrupted = [
        "When John and Mary went to the store, Mary gave a drink to",
        "When Tom and Sara went to the park, Sara gave a ball to",
    ]
    batch = list(zip(clean, corrupted, [0, 0]))

    def _collate(items):
        cl = [x[0] for x in items]
        co = [x[1] for x in items]
        lab = [x[2] for x in items]
        return cl, co, lab

    dataloader = DataLoader(batch, batch_size=2, collate_fn=_collate)

    scores = run_cdt_discovery(
        tl_model=gpt2_folded,
        dataloader=dataloader,
        device="cpu",
        n_examples=2,
        use_full_propagation=True,
    )

    assert scores, "CD-T returned an empty score dict"
    values = [float(v) for v in scores.values()]

    # The historical bug made EVERY score exactly 0.0.
    assert any(v != 0.0 for v in values), (
        "CD-T returned all-zero node scores — LayerNorm-folding regression "
        "(LayerNormPre has no .w; adapter fell back to all-zeros)."
    )
    # Scores must span a real range, not be a single repeated constant.
    assert max(values) > min(values), (
        f"CD-T scores are degenerate (all equal to {values[0]}); propagation "
        f"is not producing meaningful contributions."
    )
    assert all(v == v for v in values), "CD-T produced NaN scores"
