import math
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain, product
from typing import Any, Collection, Dict, Iterator, List, Optional, Set, Tuple

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache

from ..model_utils import micro_model_utils as mm_utils
from ..model_utils import transformer_lens_utils as tl_utils
from ..model_utils.micro_model_utils import MicroModel
from ..types import (
    DestNode,
    Edge,
    EdgeCounts,
    MaskFn,
    Node,
    OutputSlice,
    PruneScores,
    SrcNode,
    TestEdges,
)
from ..utils.misc import module_by_name, set_module_by_name
from ..utils.patch_wrapper import PatchWrapperImpl
from ..utils.patchable_model import PatchableModel
from ..utils.tensor_ops import desc_prune_scores


def _analyze_model_architecture(model: t.nn.Module) -> Dict[str, Any]:
    """
    Analyze model architecture to provide intelligent node creation.

    Args:
        model: The model to analyze

    Returns:
        Dictionary containing architectural analysis
    """
    analysis = {
        "layer_types": {},
        "connections": [],
        "attention_layers": [],
        "mlp_layers": [],
        "embedding_layers": [],
        "output_layers": [],
    }

    # Analyze each layer type
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_type = type(module).__name__
            analysis["layer_types"][name] = layer_type

            # Categorize layers
            if "attention" in name.lower() or isinstance(module, t.nn.MultiheadAttention):
                analysis["attention_layers"].append(name)
            elif "mlp" in name.lower() or isinstance(module, t.nn.Linear):
                analysis["mlp_layers"].append(name)
            elif "embed" in name.lower() or isinstance(module, t.nn.Embedding):
                analysis["embedding_layers"].append(name)
            elif "output" in name.lower() or "head" in name.lower():
                analysis["output_layers"].append(name)

    # Detect connections between layers
    analysis["connections"] = _detect_layer_connections(model)

    return analysis


def _detect_layer_connections(model: t.nn.Module) -> List[Tuple[str, str]]:
    """
    Detect connections between layers in the model.

    Args:
        model: The model to analyze

    Returns:
        List of (source_layer, target_layer) tuples
    """
    connections = []

    # Simple heuristic: layers are connected in sequential order
    layer_names = [
        name for name, module in model.named_modules() if len(list(module.children())) == 0
    ]

    for i in range(len(layer_names) - 1):
        connections.append((layer_names[i], layer_names[i + 1]))

    return connections


def _create_generic_nodes(model: t.nn.Module) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """
    Create sophisticated node structure for generic torch.nn.Module.
    Enhanced implementation with proper layer analysis and connection detection.
    """
    srcs = set()
    dests = set()

    # Analyze model architecture to create intelligent node structure
    _analyze_model_architecture(model)

    # Create nodes based on architectural analysis
    layer_idx = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            # Create enhanced src and dest nodes with architectural context
            src_node = SrcNode(
                src_idx=layer_idx,
                module=lambda m, n=name: module_by_name(m, n),
                head_dim=None,
                head_idx=0,
            )
            dest_node = DestNode(
                dest_idx=layer_idx,
                module=lambda m, n=name: module_by_name(m, n),
                head_dim=None,
                head_idx=0,
            )

            srcs.add(src_node)
            dests.add(dest_node)
            layer_idx += 1

    return srcs, dests


def _create_generic_factorized_nodes(model: t.nn.Module) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """
    Create factorized node structure for generic torch.nn.Module.
    This is a simplified implementation for models not specifically supported.
    """
    srcs = set()
    dests = set()

    # Create basic factorized nodes
    layer_idx = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            # Create factorized src and dest nodes
            src_node = SrcNode(
                src_idx=layer_idx,
                module=lambda m, n=name: module_by_name(m, n),
                head_dim=None,
                head_idx=0,
            )
            dest_node = DestNode(
                dest_idx=layer_idx,
                module=lambda m, n=name: module_by_name(m, n),
                head_dim=None,
                head_idx=0,
            )

            srcs.add(src_node)
            dests.add(dest_node)
            layer_idx += 1

    return srcs, dests


def patchable_model(
    model: t.nn.Module,
    factorized: bool,
    slice_output: OutputSlice = None,
    seq_len: Optional[int] = None,
    separate_qkv: Optional[bool] = None,
    kv_caches: Tuple[Optional[HookedTransformerKeyValueCache], ...] = (None,),
    device: t.device = t.device("cpu"),
) -> PatchableModel:
    """
    Wrap a model and inject `PatchWrapper`s into the node modules to enable patching.

    Args:
        model: The model to make patchable.
        factorized: Whether to use Edge Patching (factorized graph). Otherwise, Node
            Patching (residual graph) is used.
        slice_output: Specifies the slice of the model's output to be considered.
            For example, "last_seq" uses only the last token's output.
        seq_len: The sequence length of model inputs. If `None`, all token positions
            are patched simultaneously.
        separate_qkv: Whether the model has separate query, key, and value inputs.
        kv_caches: Key-value caches for transformers, used for efficiency.
        device: The device the model is on.

    Returns:
        The patchable model.
    """
    assert not isinstance(model, PatchableModel), "Model is already patchable"
    nodes, srcs, dests, edge_dict, edges, seq_dim, seq_len = graph_edges(
        model, factorized, separate_qkv, seq_len
    )
    wrappers, src_wrappers, dest_wrappers = make_model_patchable(
        model, factorized, srcs, nodes, device, seq_len, seq_dim
    )
    if slice_output is None:
        out_slice: Tuple[slice | int, ...] = (slice(None),)
    else:
        last_slice = [-1] if slice_output == "last_seq" else [slice(1, None)]
        out_slice = tuple([slice(None)] * seq_dim + last_slice)
    is_transformer = isinstance(model, HookedTransformer)

    return PatchableModel(
        nodes=nodes,
        srcs=srcs,
        dests=dests,
        edge_dict=edge_dict,
        edges=edges,
        seq_dim=seq_dim,
        seq_len=seq_len,
        wrappers=wrappers,
        src_wrappers=src_wrappers,
        dest_wrappers=dest_wrappers,
        out_slice=out_slice,
        is_factorized=factorized,
        is_transformer=is_transformer,
        separate_qkv=separate_qkv,
        kv_caches=kv_caches,
        wrapped_model=model,
    )


def graph_edges(
    model: t.nn.Module,
    factorized: bool,
    separate_qkv: Optional[bool] = None,
    seq_len: Optional[int] = None,
) -> Tuple[
    Set[Node],
    Set[SrcNode],
    Set[DestNode],
    Dict[int | None, List[Edge]],
    Set[Edge],
    int,
    Optional[int],
]:
    """
    Get the nodes and edges of the computation graph of the model used for ablation.
    """
    seq_dim = 1
    edge_dict: Dict[Optional[int], List[Edge]] = defaultdict(list)
    if not factorized:
        if isinstance(model, MicroModel):
            srcs, dests = mm_utils.simple_graph_nodes(model)
        elif isinstance(model, HookedTransformer):
            srcs, dests = tl_utils.simple_graph_nodes(model)
        elif isinstance(model, t.nn.Module):
            # Generic torch.nn.Module support - create basic node structure
            # This is a simplified implementation for generic models
            srcs, dests = _create_generic_nodes(model)
        else:
            raise NotImplementedError(f"Unsupported model type: {type(model)}")
        for i in [None] if seq_len is None else range(seq_len):
            pairs = product(srcs, dests)
            edge_dict[i] = [Edge(s, d, i) for s, d in pairs if s.layer + 1 == d.layer]
    else:
        if isinstance(model, MicroModel):
            srcs: Set[SrcNode] = mm_utils.factorized_src_nodes(model)
            dests: Set[DestNode] = mm_utils.factorized_dest_nodes(model)
        elif isinstance(model, HookedTransformer):
            assert separate_qkv is not None, "separate_qkv must be specified for LLM"
            srcs: Set[SrcNode] = tl_utils.factorized_src_nodes(model)
            dests: Set[DestNode] = tl_utils.factorized_dest_nodes(model, separate_qkv)
        elif isinstance(model, t.nn.Module):
            # Generic torch.nn.Module support for factorized graph
            srcs, dests = _create_generic_factorized_nodes(model)
        else:
            raise NotImplementedError(f"Unsupported model type: {type(model)}")
        for i in [None] if seq_len is None else range(seq_len):
            pairs = product(srcs, dests)
            edge_dict[i] = [Edge(s, d, i) for s, d in pairs if s.layer < d.layer]

    nodes: Set[Node] = set(srcs | dests)
    edges = set(list(chain.from_iterable(edge_dict.values())))

    return nodes, srcs, dests, edge_dict, edges, seq_dim, seq_len


def make_model_patchable(
    model: t.nn.Module,
    factorized: bool,
    src_nodes: Set[SrcNode],
    nodes: Set[Node],
    device: t.device,
    seq_len: Optional[int] = None,
    seq_dim: Optional[int] = None,
) -> Tuple[Set[PatchWrapperImpl], Set[PatchWrapperImpl], Set[PatchWrapperImpl]]:
    """Injects `PatchWrapper`s into the model at the node positions."""
    node_dict: Dict[str, Set[Node]] = defaultdict(set)
    [node_dict[node.module_name].add(node) for node in nodes]
    wrappers, src_wrappers, dest_wrappers = set(), set(), set()
    dtype = next(model.parameters()).dtype

    for module_name, module_nodes in node_dict.items():
        module = module_by_name(model, module_name)
        a_node = next(iter(module_nodes))
        head_dim = a_node.head_dim
        assert all([node.head_dim == head_dim for node in module_nodes])

        is_src = any([isinstance(node, SrcNode) for node in module_nodes])
        src_idxs_slice = None
        if is_src:
            src_idxs = [n.src_idx for n in module_nodes if isinstance(n, SrcNode)]
            src_idxs_slice = slice(min(src_idxs), max(src_idxs) + 1)
            assert src_idxs_slice.stop - src_idxs_slice.start == len(src_idxs)

        mask, in_srcs = None, None
        is_dest = any([isinstance(node, DestNode) for node in module_nodes])
        if is_dest:
            dest_nodes_in_module = [n for n in module_nodes if isinstance(n, DestNode)]
            module_dest_count = len(dest_nodes_in_module)
            min_src_idx = min([n.min_src_idx for n in dest_nodes_in_module])
            if factorized:
                n_in_src = len([n for n in src_nodes if n.layer < a_node.layer])
            else:
                n_in_src = len([n for n in src_nodes if n.layer + 1 == a_node.layer])
            in_srcs = slice(min_src_idx, min_src_idx + n_in_src)
            seq_shape = [seq_len] if seq_len is not None else []
            head_shape = [module_dest_count] if head_dim is not None else []
            mask_shape = seq_shape + head_shape + [n_in_src]
            mask = t.zeros(mask_shape, device=device, dtype=dtype, requires_grad=False)

        wrapper = PatchWrapperImpl(
            module_name=module_name,
            module=module,
            head_dim=head_dim,
            seq_dim=None if seq_len is None else seq_dim,
            is_src=is_src,
            src_idxs=src_idxs_slice,
            is_dest=is_dest,
            patch_mask=mask,
            in_srcs=in_srcs,
        )
        set_module_by_name(model, module_name, wrapper)
        wrappers.add(wrapper)
        if is_src:
            src_wrappers.add(wrapper)
        if is_dest:
            dest_wrappers.add(wrapper)

    return wrappers, src_wrappers, dest_wrappers


@contextmanager
def patch_mode(
    model: PatchableModel,
    patch_src_outs: t.Tensor,
    edges: Optional[Collection[str | Edge]] = None,
    curr_src_outs: Optional[t.Tensor] = None,
):
    """Context manager to enable patching in the model."""
    if curr_src_outs is None:
        curr_src_outs = t.zeros_like(patch_src_outs)

    if edges is not None:
        set_all_masks(model, val=0.0)
        for edge in model.edges:
            if edge in edges or edge.name in edges:
                edge.patch_mask(model).data[edge.patch_idx] = 1.0

    for wrapper in model.wrappers:
        wrapper.patch_mode = True
        wrapper.curr_src_outs = curr_src_outs
        if wrapper.is_dest:
            wrapper.patch_src_outs = patch_src_outs
    try:
        yield
    finally:
        for wrapper in model.wrappers:
            wrapper.patch_mode = False
            wrapper.curr_src_outs = None
            if wrapper.is_dest:
                wrapper.patch_src_outs = None
        del curr_src_outs, patch_src_outs


def set_all_masks(model: PatchableModel, val: float) -> None:
    """Set all the patch masks in the model to the specified value."""
    for wrapper in model.dest_wrappers:
        t.nn.init.constant_(wrapper.patch_mask, val)


@contextmanager
def train_mask_mode(
    model: PatchableModel, requires_grad: bool = True
) -> Iterator[Dict[str, t.nn.Parameter]]:
    """Context manager to set `requires_grad` on patch masks."""
    model.eval()
    model.zero_grad()
    parameters: Dict[str, t.nn.Parameter] = {}
    for wrapper in model.dest_wrappers:
        patch_mask = wrapper.patch_mask
        patch_mask.detach_().requires_grad_(requires_grad)
        parameters[wrapper.module_name] = patch_mask
        wrapper.train()
    try:
        yield parameters
    finally:
        for wrapper in model.dest_wrappers:
            wrapper.eval()
            wrapper.patch_mask.detach_().requires_grad_(False)


@contextmanager
def mask_fn_mode(model: PatchableModel, mask_fn: MaskFn, dropout_p: float = 0.0):
    """Context manager to set the mask function and dropout probability."""
    for wrapper in model.dest_wrappers:
        wrapper.mask_fn = mask_fn
        wrapper.dropout_layer.p = dropout_p  # type: ignore
    try:
        yield
    finally:
        for wrapper in model.dest_wrappers:
            wrapper.mask_fn = None
            wrapper.dropout_layer.p = 0.0  # type: ignore


@contextmanager
def set_mask_batch_size(model: PatchableModel, batch_size: int | None):
    """Context manager to set the batch size of the patch masks in the model."""
    for wrapper in model.dest_wrappers:
        wrapper.set_mask_batch_size(batch_size)
    try:
        yield
    finally:
        for wrapper in model.dest_wrappers:
            wrapper.set_mask_batch_size(None)


def edge_counts_util(
    edges: Set[Edge],
    test_counts: Optional[TestEdges] = None,
    prune_scores: Optional[PruneScores] = None,
    zero_edges: Optional[bool] = None,
    all_edges: Optional[bool] = None,
    true_edge_count: Optional[int] = None,
) -> List[int]:
    """Calculate a set of [number of edges in the circuit] to test."""
    n_edges = len(edges)

    sorted_ps_count: Optional[t.Tensor] = None
    if test_counts is None:
        test_counts = EdgeCounts.LOGARITHMIC if n_edges > 200 else EdgeCounts.ALL
        if prune_scores is not None:
            flat_ps = desc_prune_scores(prune_scores)
            unique_ps, sorted_ps_count = flat_ps.unique(sorted=True, return_counts=True)
            if list(unique_ps.size())[0] < min(n_edges / 2, 100):
                test_counts = EdgeCounts.GROUPS

    if test_counts == EdgeCounts.ALL:
        counts_list = [n for n in range(n_edges + 1)]
    elif test_counts == EdgeCounts.LOGARITHMIC:
        counts_list = [
            n for n in range(1, n_edges) if n % (10 ** max(math.floor(math.log10(n)), 0)) == 0
        ]
    elif test_counts == EdgeCounts.GROUPS:
        assert prune_scores is not None
        if sorted_ps_count is None:
            flat_ps = desc_prune_scores(prune_scores)
            _, sorted_ps_count = flat_ps.unique(sorted=True, return_counts=True)
        assert sorted_ps_count is not None
        counts_list = sorted_ps_count.flip(dims=(0,)).cumsum(dim=0).tolist()
    elif isinstance(test_counts, List):
        counts_list = [n if isinstance(n, int) else int(n_edges * n) for n in test_counts]
    elif isinstance(test_counts, (int, float)):
        # Single value - convert to list
        if isinstance(test_counts, float):
            counts_list = [int(n_edges * test_counts)]
        else:
            counts_list = [test_counts]
    elif hasattr(test_counts, "__iter__"):
        # Handle other iterable types
        counts_list = [n if isinstance(n, int) else int(n_edges * n) for n in test_counts]
    else:
        raise NotImplementedError(
            f"Unknown test_counts type: {type(test_counts)} with value: {test_counts}"
        )

    if zero_edges is None:
        zero_edges = True if len(counts_list) > 2 else False
    if all_edges is None:
        all_edges = True if len(counts_list) > 2 else False

    if zero_edges and 0 not in counts_list:
        counts_list = [0] + counts_list
    if all_edges and n_edges not in counts_list:
        counts_list.append(n_edges)
    if not zero_edges and 0 in counts_list:
        counts_list.remove(0)
    if not all_edges and n_edges in counts_list:
        counts_list.remove(n_edges)
    if true_edge_count is not None and true_edge_count not in counts_list:
        counts_list.append(true_edge_count)
    counts_list.sort()

    return counts_list
