"""Dataset shape adapters.

Each adapter converts a raw dataset of a given shape into a
``NormalizedDataset[ContrastiveRecord]``. Adapters never know which
corruption strategy will be applied; that is the strategy's job.

Public registry: ``ADAPTER_REGISTRY: Dict[DatasetShape, Type[DataAdapter]]``.
Use ``get_adapter(shape)`` to look one up. New adapters self-register
via the ``@register_adapter`` decorator on their class definition.
"""

from .base import ADAPTER_REGISTRY, DataAdapter, get_adapter, list_adapters, register_adapter

__all__ = [
    "DataAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "ADAPTER_REGISTRY",
]
