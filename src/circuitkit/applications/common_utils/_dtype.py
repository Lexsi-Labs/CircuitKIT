"""Dtype guards for the applications layer."""

from typing import Iterable

import torch


def reject_float16(params: Iterable[torch.Tensor], what: str) -> None:
    """Raise ``ValueError`` if any parameter being trained is float16.

    The fine-tuning code here trains weights in place with Adam/AdamW, with no
    float32 master copy and no loss scaling. In float16 that fails on the first
    step: ``eps=1e-8`` and the squared gradient of any gradient below about 5e-3
    round to 0, so the update is 0/0 (NaN) or x/0 (inf) and the weights are
    destroyed. bfloat16 and float32 are fine. Call this before anything is changed.

    Args:
        params: The parameters the optimiser will update.
        what: Name of the caller, for the error message.
    """
    if any(p.dtype == torch.float16 for p in params):
        raise ValueError(
            f"{what} trains weights in place with Adam, which turns float16 weights "
            f"into NaN/inf after one step. Load the model in torch.bfloat16 or "
            f"torch.float32."
        )
