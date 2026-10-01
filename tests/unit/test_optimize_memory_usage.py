"""optimize_memory_usage's per-process cap scales with what's actually free."""

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
    return seen["fraction"]


def test_idle_gpu_is_capped_at_max_fraction():
    total = 100 * 1024**3
    fraction = _run(mem_get_info=(total, total), reserved=0)
    assert fraction == 0.8


def test_cap_never_drops_below_what_is_already_used_even_past_max_fraction():
    total = 100 * 1024**3
    free = 10 * 1024**3  # 90% already used by other processes -- past max_fraction
    fraction = _run(mem_get_info=(free, total), reserved=0)
    # min(max_fraction, ...) would otherwise ask for LESS than what's already
    # committed to this device, which set_per_process_memory_fraction cannot
    # honor (it would immediately look over budget).
    assert fraction >= 0.9

def test_never_exceeds_max_fraction_argument():
    total = 100 * 1024**3
    fraction = _run(mem_get_info=(total, total), reserved=0, max_fraction=0.5)
    assert fraction == 0.5


def test_no_cuda_does_not_raise():
    with patch.object(torch.cuda, "is_available", return_value=False):
        mem.optimize_memory_usage()  # must not raise
