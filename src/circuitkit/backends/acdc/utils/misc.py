from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import Any, Iterator, Set

import torch as t
from torch.utils.hooks import RemovableHandle


def repo_path_to_abs_path(path: str) -> Path:
    """
    Convert a path relative to the repository root to an absolute path.

    Args:
        path: A path relative to the repository root.

    Returns:
        The absolute path.
    """
    repo_abs_path = Path(__file__).parent.parent.parent.absolute()
    return repo_abs_path / path


@contextmanager
def remove_hooks() -> Iterator[Set[RemovableHandle]]:
    """
    Context manager to safely add and remove temporary PyTorch hooks.

    Yields:
        An empty set to store hook handles, which are removed upon exiting the context.
    """
    handles: Set[RemovableHandle] = set()
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def module_by_name(model: Any, module_name: str) -> t.nn.Module:
    """
    Gets a module from a model by its string name.

    Args:
        model: The model to get the module from.
        module_name: The name of the module (e.g., "blocks.0.attn").

    Returns:
        The module.
    """
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    return reduce(getattr, init_mod + module_name.split("."))  # type: ignore


def set_module_by_name(model: Any, module_name: str, new_module: t.nn.Module):
    """
    Sets a module in a model by its string name. Modifies the model in place.

    Args:
        model: The model to set the module in.
        module_name: The name of the module to set.
        new_module: The module to replace the existing module with.
    """
    parent = model
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    if "." in module_name:
        parent = reduce(getattr, init_mod + module_name.split(".")[:-1])  # type: ignore
    setattr(parent, module_name.split(".")[-1], new_module)
