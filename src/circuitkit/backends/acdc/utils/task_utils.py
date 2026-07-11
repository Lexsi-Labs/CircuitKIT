import dataclasses
from typing import Any, Callable, Dict, Optional

import torch as t


@dataclasses.dataclass(frozen=False)
class AllDataThings:
    """A dataclass to hold all the data and metrics for a specific task."""

    # Fields without default values
    validation_data: t.Tensor
    validation_patch_data: t.Tensor
    validation_metric: Callable[[t.Tensor], t.Tensor]
    test_data: t.Tensor
    test_patch_data: t.Tensor
    test_metrics: Dict[str, Any]

    # Fields with default values
    validation_labels: Optional[t.Tensor] = None
    validation_wrong_labels: Optional[t.Tensor] = None
    validation_mask: Optional[t.Tensor] = None
    test_labels: Optional[t.Tensor] = None
    test_wrong_labels: Optional[t.Tensor] = None
    test_mask: Optional[t.Tensor] = None


def shuffle_tensor(tens: t.Tensor, seed: int = 42) -> t.Tensor:
    """Shuffle a tensor along the first dimension."""
    t.random.manual_seed(seed)
    return tens[t.randperm(tens.shape[0])]
