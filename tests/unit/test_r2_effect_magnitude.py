"""Pillar 7 R2: near-zero baselines must be skipped, not saturate the clamp.

The old computation normalized every seed with `d / (abs(b) + 1e-8)`: a seed
with baseline ~ 0 (e.g. delta=0.5, baseline=1e-9) produced norm_delta ~ 5e8,
which the [0,1] clamp silently saturated to r2 = 1.0 — a maximal "effect
magnitude" for an undefined normalized effect, feeding reliability_index.
"""

import pytest

from circuitkit.evaluation.pillars.intervention_reliability import _r2_effect_magnitude


def test_near_zero_baseline_seed_is_skipped_not_saturating():
    """One degenerate seed among healthy ones must not drag r2 to 1.0."""
    # healthy seeds: norm_deltas = 0.5/1.0 = 0.5 each -> r2_raw = 0.5 -> r2 = 0.75
    # degenerate seed (delta=0.5, baseline=1e-9): old code -> norm ~ 5e8 -> r2 = 1.0
    deltas = [0.5, 0.5, 0.5]
    baselines = [1.0, 1.0, 1e-9]
    r2 = _r2_effect_magnitude(deltas, baselines)
    assert abs(r2 - 0.75) < 1e-9, f"degenerate seed leaked into r2={r2}"


def test_all_baselines_degenerate_returns_neutral_midpoint():
    """No defined effect anywhere -> neutral 0.5, not a saturated extreme."""
    r2 = _r2_effect_magnitude([0.5, 0.3], [1e-9, 0.0])
    assert r2 == 0.5


def test_healthy_baselines_unchanged():
    """Sign handling via abs(b) is preserved for valid baselines."""
    # norm_deltas: 0.5/1.0 = 0.5 and -0.2/0.5 = -0.4 -> raw mean 0.05 -> r2 0.525
    r2 = _r2_effect_magnitude([0.5, -0.2], [1.0, -0.5])
    assert abs(r2 - 0.525) < 1e-9
