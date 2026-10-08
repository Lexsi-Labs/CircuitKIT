"""Device auto-detection: CUDA > MPS > CPU."""

import os
import sys
from typing import Optional

import torch

_expandable_segments_applied: Optional[bool] = None


def enable_expandable_segments() -> bool:
    """Turn on the CUDA caching allocator's ``expandable_segments`` (idempotent).

    Attribution and evaluation allocate differently sized activation buffers
    for every batch. With fixed-size segments those allocations fragment the
    pool and can OOM while gigabytes are nominally free; expandable segments
    grow and remap instead.

    Applied in-process through ``torch.cuda.memory._set_allocator_settings`` so
    child processes (e.g. vLLM) do not inherit it. If that private API is
    unavailable, falls back to ``PYTORCH_CUDA_ALLOC_CONF``, which only takes
    effect while CUDA has not allocated yet. Skipped when CUDA/ROCm is absent,
    when the user already set ``expandable_segments`` in
    ``PYTORCH_CUDA_ALLOC_CONF``/``PYTORCH_ALLOC_CONF``, when
    ``CIRCUITKIT_NO_EXPANDABLE_SEGMENTS`` is set, or on Windows (PyTorch does
    not support this allocator setting there and warns if asked).

    Returns:
        True if CircuitKIT applied the setting, False otherwise.
    """
    global _expandable_segments_applied
    if _expandable_segments_applied is not None:
        return _expandable_segments_applied
    _expandable_segments_applied = False

    user_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + os.environ.get(
        "PYTORCH_ALLOC_CONF", ""
    )
    if (
        os.environ.get("CIRCUITKIT_NO_EXPANDABLE_SEGMENTS")
        or "expandable_segments" in user_conf
        or not torch.cuda.is_available()
        or torch.version.hip
        or sys.platform == "win32"
    ):
        return False
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        _expandable_segments_applied = True
    except (AttributeError, RuntimeError):
        if not torch.cuda.is_initialized():
            existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"{existing},expandable_segments:True" if existing else "expandable_segments:True"
            )
            _expandable_segments_applied = True
    return _expandable_segments_applied


def get_device(prefer: str = "auto") -> str:
    """Resolve the best available compute device.

    Args:
        prefer: One of ``"auto"`` (default), ``"cuda"``,
            ``"mps"``, or ``"cpu"``.

    Returns:
        Device string (``"cuda"``, ``"mps"``, or ``"cpu"``).

    Examples::

        >>> device = get_device()            # auto-detect
        >>> device = get_device("cuda")      # force CUDA or raise
        >>> device = get_device("cpu")       # force CPU
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"

    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return "mps"

    if prefer == "cpu":
        return "cpu"

    raise ValueError(f"Unknown device: {prefer!r}. Use 'auto', 'cuda', 'mps', or 'cpu'.")


def empty_cache(device: str = "auto") -> None:
    """Clear cache on the current device (safe to call on all platforms)."""
    resolved = get_device(device)
    if resolved == "cuda":
        torch.cuda.empty_cache()
    elif resolved == "mps":
        torch.mps.empty_cache()
