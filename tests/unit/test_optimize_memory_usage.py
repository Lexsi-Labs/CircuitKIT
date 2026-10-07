"""optimize_memory_usage's per-process cap scales with what's actually free.

The previous formula folded *other* processes' usage into the per-process cap
in a way that always landed back at max_fraction regardless of how much was
actually free on a shared GPU -- the exact failure its own comment claimed to
fix. Covers: idle GPU, a shared GPU with other processes holding 60% (the cap
must actually shrink, not stay at max_fraction), this process's own
reservation already exceeding max_fraction (must not drop below it), and no
CUDA.
"""

from unittest.mock import patch

import torch

from circuitkit.utils import memory as mem


def _run(mem_get_info, reserved, max_fraction=0.8):
    seen = {}
    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
        torch.cuda, "empty_cache"
    ), patch.object(torch.cuda, "synchronize"), patch.object(
        torch.cuda, "mem_get_info", return_value=mem_get_info
    ), patch.object(
        torch.cuda, "memory_reserved", return_value=reserved
    ), patch.object(
        torch.cuda,
        "set_per_process_memory_fraction",
        side_effect=lambda f, device=None: seen.__setitem__("fraction", f),
    ):
        mem.optimize_memory_usage(max_fraction=max_fraction)
    return seen.get("fraction")


def test_idle_gpu_is_capped_at_max_fraction():
    total = 100 * 1024**3
    fraction = _run(mem_get_info=(total, total), reserved=0)
    assert fraction == 0.8


def test_other_processes_holding_60_percent_shrinks_the_cap():
    """The regression this fixes: a shared GPU with 60% already used by
    *other* processes (this process has no reservation yet) must get a cap
    well below max_fraction, not max_fraction regardless."""
    total = 100 * 1024**3
    free = 40 * 1024**3  # 60% used by others, 0% by us
    fraction = _run(mem_get_info=(free, total), reserved=0)
    assert abs(fraction - 0.36) < 1e-9  # (0.9 * 40 + 0) / 100
    assert fraction < 0.8


def test_cap_never_drops_below_this_process_own_reservation():
    """This process already holds 85% (above max_fraction=0.8); the cap must
    not be set below that, which set_per_process_memory_fraction cannot honor
    (it would already be over the requested budget)."""
    total = 100 * 1024**3
    reserved = 85 * 1024**3
    free = 5 * 1024**3  # others hold the remaining 10%
    fraction = _run(mem_get_info=(free, total), reserved=reserved)
    assert fraction == 0.85
    assert fraction >= 0.85


def test_never_exceeds_max_fraction_argument():
    total = 100 * 1024**3
    fraction = _run(mem_get_info=(total, total), reserved=0, max_fraction=0.5)
    assert fraction == 0.5


def test_full_device_leaves_the_cap_unset():
    """Nothing free and nothing reserved by us would give a cap of 0.0, which torch accepts
    and which makes every later allocation fail. The cap must be left unset instead."""
    total = 100 * 1024**3
    assert _run(mem_get_info=(0, total), reserved=0) is None


def test_nearly_full_device_leaves_the_cap_unset():
    """1 GiB free of 100 would cap the process at under 1% for good, even after the other
    tenants exit."""
    total = 100 * 1024**3
    assert _run(mem_get_info=(1 * 1024**3, total), reserved=0) is None


def test_small_but_usable_share_is_still_capped():
    total = 100 * 1024**3
    fraction = _run(mem_get_info=(10 * 1024**3, total), reserved=0)
    assert abs(fraction - 0.09) < 1e-9


def test_no_cuda_does_not_raise():
    with patch.object(torch.cuda, "is_available", return_value=False):
        mem.optimize_memory_usage()  # must not raise
