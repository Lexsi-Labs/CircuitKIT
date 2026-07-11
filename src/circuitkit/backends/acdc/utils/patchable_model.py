from typing import Any, Collection, Dict, List, Optional, Set, Tuple

import torch as t
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

from ..types import DestNode, Edge, Node, PruneScores, SrcNode
from ..utils.patch_wrapper import PatchWrapperImpl


class PatchableModel(t.nn.Module):
    """
    A model that can be ablated along individual edges in its computation graph.

    This class has many of the same methods and attributes as TransformerLens'
    `HookedTransformer`s. These are simple wrappers which pass through to the
    implementation in the wrapped model.

    Args:
        nodes: The set of all nodes in the computation graph.
        srcs: The set of all source nodes in the computation graph.
        dests: The set of all destination nodes in the computation graph.
        edge_dict: A dictionary mapping sequence positions to the edges.
        edges: The set of all edges in the computation graph.
        seq_dim: The sequence dimension of the model's activations.
        seq_len: The sequence length of the model inputs.
        wrappers: The set of all `PatchWrapper`s in the model.
        out_slice: Specifies the slice of the model's output to be considered.
        is_factorized: Whether the model is factorized for Edge Patching.
        is_transformer: Whether the model is a transformer.
        separate_qkv: Whether the model has separate Q, K, and V inputs.
        kv_caches: A dictionary mapping batch sizes to past key-value caches.
        wrapped_model: The model being made patchable.
    """

    nodes: Set[Node]
    srcs: Set[SrcNode]
    dests: Set[DestNode]
    edge_dict: Dict[int | None, List[Edge]]  # Key is token position or None for all
    edges: Set[Edge]
    n_edges: int
    seq_dim: int
    seq_len: Optional[int]
    wrappers: Set[PatchWrapperImpl]
    src_wrappers: Set[PatchWrapperImpl]
    dest_wrappers: Set[PatchWrapperImpl]
    patch_masks: Dict[str, t.nn.Parameter]
    out_slice: Tuple[slice | int, ...]
    is_factorized: bool
    is_transformer: bool
    separate_qkv: Optional[bool]
    kv_caches: Optional[Dict[int, HookedTransformerKeyValueCache]]
    wrapped_model: t.nn.Module

    def __init__(
        self,
        nodes: Set[Node],
        srcs: Set[SrcNode],
        dests: Set[DestNode],
        edge_dict: Dict[int | None, List[Edge]],
        edges: Set[Edge],
        seq_dim: int,
        seq_len: Optional[int],
        wrappers: Set[PatchWrapperImpl],
        src_wrappers: Set[PatchWrapperImpl],
        dest_wrappers: Set[PatchWrapperImpl],
        out_slice: Tuple[slice | int, ...],
        is_factorized: bool,
        is_transformer: bool,
        separate_qkv: Optional[bool],
        kv_caches: Tuple[Optional[HookedTransformerKeyValueCache], ...],
        wrapped_model: t.nn.Module,
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.srcs = srcs
        self.dests = dests
        self.edge_dict = edge_dict
        self.edges = edges
        self.n_edges = len(edges)
        self.seq_dim = seq_dim
        self.seq_len = seq_len
        self.wrappers = wrappers
        self.src_wrappers = src_wrappers
        self.dest_wrappers = dest_wrappers
        self.patch_masks = {}
        for dest_wrapper in self.dest_wrappers:
            self.patch_masks[dest_wrapper.module_name] = dest_wrapper.patch_mask
        self.out_slice = out_slice
        self.is_factorized = is_factorized
        self.is_transformer = is_transformer
        if is_transformer:
            assert separate_qkv is not None
        self.separate_qkv = separate_qkv
        if all([kv_cache is None for kv_cache in kv_caches]) or len(kv_caches) == 0:
            self.kv_caches = None
        else:
            self.kv_caches = {}
            for kv_cache in kv_caches:
                if kv_cache is not None:
                    batch_size = kv_cache.previous_attention_mask.shape[0]
                    self.kv_caches[batch_size] = kv_cache
        self.wrapped_model = wrapped_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Wrapper around the forward method of the wrapped model."""
        if self.kv_caches is None or "past_kv_cache" in kwargs:
            return self.wrapped_model(*args, **kwargs)
        else:
            batch_size = args[0].shape[0]
            kv = self.kv_caches[batch_size]
            return self.wrapped_model(*args, past_kv_cache=kv, **kwargs)

    def new_prune_scores(self, init_val: float = 0.0) -> PruneScores:
        """A new `PruneScores` instance with zeroed values."""
        prune_scores: PruneScores = {}
        for mod_name, mask in self.patch_masks.items():
            prune_scores[mod_name] = t.full_like(mask.data, init_val)
        return prune_scores

    def circuit_prune_scores(self, edges: Collection[Edge]) -> PruneScores:
        """Convert a set of edges to a corresponding `PruneScores` object."""
        ps = self.new_prune_scores()
        for edge in self.edges:
            if edge in edges:
                ps[edge.dest.module_name][edge.patch_idx] = 1.0
        return ps

    def current_patch_masks_as_prune_scores(self) -> PruneScores:
        """Convert the current patch masks to a `PruneScores` object."""
        return dict([(mod, mask.data) for (mod, mask) in self.patch_masks.items()])

    def __getattr__(self, name):
        """Pass through attributes to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_model, name)

    def __str__(self) -> str:
        return self.wrapped_model.__str__()

    def __repr__(self) -> str:
        return self.wrapped_model.__repr__()
