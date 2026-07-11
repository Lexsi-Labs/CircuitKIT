"""Device-related runtime guards for the applications layer."""

import logging

logger = logging.getLogger(__name__)

_warned_mps_editing = False


def warn_if_mps_editing(device) -> None:
    """Warn once if knowledge editing (ROME/MEMIT) is run on the MPS backend.

    ROME/MEMIT rely on numerically-sensitive linear algebra — covariance
    estimation, solving ``(C + lambda*I) v = k``, and rank-one weight updates.
    PyTorch's MPS (Apple Silicon) backend computes these with different, and
    inconsistent, precision than CUDA/CPU, so the resulting edit magnitudes are
    unreliable and the edits can be **silently incorrect** (they may pass or fail
    a sanity check depending on op ordering). This is a backend limitation, not a
    CircuitKit bug. Run knowledge editing on CUDA or CPU for correct results.

    Emits at most one warning per process.
    """
    global _warned_mps_editing
    dev = str(getattr(device, "type", device)).lower()
    if "mps" in dev and not _warned_mps_editing:
        _warned_mps_editing = True
        logger.warning(
            "Knowledge editing (ROME/MEMIT) is running on the MPS (Apple Silicon) "
            "backend, which is numerically unreliable for these ops: the covariance / "
            "linear-solve / rank-one-update steps diverge from CUDA/CPU and edit "
            "magnitudes are inconsistent, so results can be SILENTLY INCORRECT. "
            "Run knowledge editing on CUDA or CPU instead."
        )
