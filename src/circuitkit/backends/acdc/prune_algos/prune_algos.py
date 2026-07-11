from dataclasses import dataclass
from typing import Callable, Optional, Set

from ..data import PromptDataLoader
from ..types import AlgoKey, Edge, PruneScores
from ..utils.patchable_model import PatchableModel


@dataclass(frozen=True)
class PruneAlgo:
    """
    An algorithm that finds the importance of each edge in a model for a given task.

    Args:
        key: A unique identifier for the algorithm.
        name: The name of the algorithm.
        func: The function that computes the importance of each edge.
        _short_name: A short name for the algorithm. If not provided, `name` is used.
    """

    key: AlgoKey
    name: str
    func: Callable[[PatchableModel, PromptDataLoader, Optional[Set[Edge]]], PruneScores]
    _short_name: Optional[str] = None

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PruneAlgo):
            return False
        return self.key == __value.key

    @property
    def short_name(self) -> str:
        return self._short_name if self._short_name is not None else self.name
