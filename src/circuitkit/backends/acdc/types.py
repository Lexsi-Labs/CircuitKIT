from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import plotly.io as pio
import torch as t

from .utils.misc import module_by_name

# ==============================================================================
# NEW: Data Structures Moved from data.py to Break Circular Import
# ==============================================================================
BatchKey = int
"""A unique key for a [`PromptPairBatch`][auto_circuit.types.PromptPairBatch]."""


@dataclass(frozen=True)
class PromptPair:
    """
    A pair of clean and corrupt prompts with correct and incorrect answers.

    Args:
        clean: The 'clean' prompt.
        corrupt: The 'corrupt' prompt.
        answers: The correct completions for the clean prompt.
        wrong_answers: The incorrect completions for the clean prompt.
    """

    clean: t.Tensor
    corrupt: t.Tensor
    answers: t.Tensor
    wrong_answers: t.Tensor


@dataclass(frozen=True)
class PromptPairBatch:
    """
    A batch of prompt pairs.

    Args:
        key: A unique integer that identifies the batch.
        clean: The 'clean' prompts in a 2D tensor.
        corrupt: The 'corrupt' prompts in a 2D tensor.
        answers: Correct completions. A tensor where each row corresponds to a prompt.
        wrong_answers: Incorrect completions. A tensor where each row corresponds to a prompt.
    """

    key: BatchKey
    clean: t.Tensor
    corrupt: t.Tensor
    answers: t.Tensor
    wrong_answers: t.Tensor


# ==============================================================================
# Original Content of types.py
# ==============================================================================
class PatchWrapper(t.nn.Module, ABC):
    """Abstract class for a wrapper around a module that can be patched."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any):
        pass


MaskFn = Optional[Literal["hard_concrete", "sigmoid"]]
"""
Determines how mask values are used to ablate edges. See documentation on the original
repo for more details on `"hard_concrete"` and `"sigmoid"`.
"""

# Define a colorblind-friendly palette
COLOR_PALETTE = [
    "rgb(55, 126, 184)",  # blue
    "rgb(255, 127, 0)",  # orange
    "rgb(77, 175, 74)",  # green
    "rgb(247, 129, 191)",  # pink
    "rgb(228, 26, 28)",  # red
    "rgb(152, 78, 163)",  # purple
    "rgb(166, 86, 40)",  # brown
    "rgb(153, 153, 153)",  # grey
    "rgb(222, 222, 0)",  # yellow
]

# Create or modify a template
template = pio.templates["plotly"]
template.layout.colorway = COLOR_PALETTE  # type: ignore
template.layout.font.size = 19  # type: ignore

# Set the template as the default
pio.templates.default = "plotly"


class EdgeCounts(Enum):
    """Special values for `TestEdges` that get computed at runtime."""

    ALL = 1
    LOGARITHMIC = 2
    GROUPS = 3


TestEdges = EdgeCounts | List[int | float]
"""Determines the set of [number of edges to prune] to test."""

OutputSlice = Optional[Literal["last_seq", "not_first_seq"]]
"""The slice of the model output to be considered for task evaluation."""


class PatchType(Enum):
    """Whether to patch the edges in the circuit or the complement of the circuit."""

    EDGE_PATCH = 1
    TREE_PATCH = 2

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


class AblationType(Enum):
    """Type of activation to replace an original activation with during a forward pass."""

    RESAMPLE = 1
    ZERO = 2
    TOKENWISE_MEAN_CLEAN = 3
    TOKENWISE_MEAN_CORRUPT = 4
    TOKENWISE_MEAN_CLEAN_AND_CORRUPT = 5
    BATCH_TOKENWISE_MEAN = 6
    BATCH_ALL_TOK_MEAN = 7

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()

    @property
    def mean_over_dataset(self) -> bool:
        return self in {
            AblationType.TOKENWISE_MEAN_CLEAN,
            AblationType.TOKENWISE_MEAN_CORRUPT,
            AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
        }

    @property
    def clean_dataset(self) -> bool:
        return self in {
            AblationType.TOKENWISE_MEAN_CLEAN,
            AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
        }

    @property
    def corrupt_dataset(self) -> bool:
        return self in {
            AblationType.TOKENWISE_MEAN_CORRUPT,
            AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
        }


@dataclass(frozen=True)
class Node:
    """A node in the computational graph of the model used for ablation."""

    name: str
    module_name: str
    layer: int
    head_idx: Optional[int] = None
    head_dim: Optional[int] = None
    weight: Optional[str] = None
    weight_head_dim: Optional[int] = None

    def module(self, model: Any) -> PatchWrapper:
        patch_wrapper = module_by_name(model, self.module_name)
        assert isinstance(patch_wrapper, PatchWrapper)
        return patch_wrapper

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class SrcNode(Node):
    """A node that is the source of an edge."""

    src_idx: int = 0


@dataclass(frozen=True)
class DestNode(Node):
    """A node that is the destination of an edge."""

    min_src_idx: int = 0


PruneScores = Dict[str, t.Tensor]
"""
Dictionary from module names of `DestNodes` to edge scores. The edge scores are
stored as a tensor where each value corresponds to the score of an incoming `Edge`.
"""


@dataclass(frozen=True)
class Edge:
    """A directed edge from a `SrcNode` to a `DestNode`."""

    src: SrcNode
    dest: DestNode
    seq_idx: Optional[int] = None

    @property
    def name(self) -> str:
        return f"{self.src.name}->{self.dest.name}"

    @property
    def patch_idx(self) -> Tuple[int, ...]:
        """Index of the edge in the `patch_mask` or `PruneScores` tensor."""
        seq_idx = [] if self.seq_idx is None else [self.seq_idx]
        head_idx = [] if self.dest.head_idx is None else [self.dest.head_idx]
        return tuple(seq_idx + head_idx + [self.src.src_idx - self.dest.min_src_idx])

    def patch_mask(self, model: Any) -> t.nn.Parameter:
        return self.dest.module(model).patch_mask

    def prune_score(self, prune_scores: PruneScores) -> t.Tensor:
        return prune_scores[self.dest.module_name][self.patch_idx]

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


TaskKey = str
AlgoKey = str
PruneMetricKey = str
Measurements = List[Tuple[int | float, int | float]]
BatchOutputs = Dict[BatchKey, t.Tensor]
CircuitOutputs = Dict[int, BatchOutputs]
AlgoPruneScores = Dict[AlgoKey, PruneScores]
TaskPruneScores = Dict[TaskKey, AlgoPruneScores]
AlgoMeasurements = Dict[AlgoKey, Measurements]
TaskMeasurements = Dict[TaskKey, AlgoMeasurements]
PruneMetricMeasurements = Dict[PruneMetricKey, TaskMeasurements]
AblationMeasurements = Dict[AblationType, PruneMetricMeasurements]
