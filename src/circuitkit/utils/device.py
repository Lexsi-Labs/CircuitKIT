"""Device auto-detection: CUDA > MPS > CPU."""

import torch


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
