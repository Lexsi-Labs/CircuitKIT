from itertools import count
from typing import Set, Tuple

import transformer_lens as tl

from ..types import DestNode, SrcNode


def factorized_src_nodes(model: tl.HookedTransformer) -> Set[SrcNode]:
    """
    Get the source part of each edge in the factorized graph, grouped by layer.
    """
    assert model.cfg.use_attn_result
    if not model.cfg.attn_only:
        assert model.cfg.use_hook_mlp_in
    layers, idxs = count(), count()
    nodes = set()
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name="blocks.0.hook_resid_pre",
            layer=next(layers),
            src_idx=next(idxs),
            weight="embed.W_E",
        )
    )

    for block_idx in range(model.cfg.n_layers):
        layer = next(layers)
        for head_idx in range(model.cfg.n_heads):
            nodes.add(
                SrcNode(
                    name=f"A{block_idx}.{head_idx}",
                    module_name=f"blocks.{block_idx}.attn.hook_result",
                    layer=layer,
                    src_idx=next(idxs),
                    head_dim=2,
                    head_idx=head_idx,
                    weight=f"blocks.{block_idx}.attn.W_O",
                    weight_head_dim=0,
                )
            )
        if not model.cfg.attn_only:
            nodes.add(
                SrcNode(
                    name=f"MLP {block_idx}",
                    module_name=f"blocks.{block_idx}.hook_mlp_out",
                    layer=layer if model.cfg.parallel_attn_mlp else next(layers),
                    src_idx=next(idxs),
                    weight=f"blocks.{block_idx}.mlp.W_out",
                )
            )
    return nodes


def factorized_dest_nodes(model: tl.HookedTransformer, separate_qkv: bool) -> Set[DestNode]:
    """
    Get the destination part of each edge in the factorized graph, grouped by layer.
    This version correctly handles Grouped-Query Attention (GQA).
    """
    if separate_qkv:
        assert model.cfg.use_split_qkv_input
    else:
        assert model.cfg.use_attn_in
    if not model.cfg.attn_only:
        assert model.cfg.use_hook_mlp_in

    n_q_heads = model.cfg.n_heads
    # GQA handling: hook_k_input / hook_v_input are sized by the key/value head
    # count (`n_key_value_heads`), which is smaller than the query head count
    # (`n_heads`) for grouped-query-attention models such as Llama-3.2.
    #
    # However, when `ungroup_grouped_query_attention` is enabled, TransformerLens
    # expands hook_k_input / hook_v_input back up to `n_heads` so every query head
    # sees its own (repeated) key/value input. In that case the patch graph must
    # also use `n_heads` for the K/V destination nodes, otherwise the patch mask
    # is built at `n_key_value_heads` and the einsum in PatchWrapperImpl.forward
    # mismatches the (ungrouped) activation tensor.
    #
    # For non-GQA models `n_key_value_heads` equals `n_heads`, so both branches
    # are byte-identical to the original GPT-2 behaviour.
    cfg_n_kv_heads = getattr(model.cfg, "n_key_value_heads", None)
    ungroup_gqa = getattr(model.cfg, "ungroup_grouped_query_attention", False)
    if cfg_n_kv_heads is None or ungroup_gqa:
        n_kv_heads = n_q_heads
    else:
        n_kv_heads = cfg_n_kv_heads

    layers = count(1)
    nodes = set()
    for block_idx in range(model.cfg.n_layers):
        layer = next(layers)
        if separate_qkv:
            # Q heads
            for head_idx in range(n_q_heads):
                nodes.add(
                    DestNode(
                        name=f"A{block_idx}.{head_idx}.Q",
                        module_name=f"blocks.{block_idx}.hook_q_input",
                        layer=layer,
                        head_dim=2,
                        head_idx=head_idx,
                        weight=f"blocks.{block_idx}.attn.W_Q",
                        weight_head_dim=0,
                    )
                )
            # K and V heads (handling for GQA)
            for head_idx in range(n_kv_heads):
                for letter in ["K", "V"]:
                    nodes.add(
                        DestNode(
                            name=f"A{block_idx}.{head_idx}.{letter}",
                            module_name=f"blocks.{block_idx}.hook_{letter.lower()}_input",
                            layer=layer,
                            head_dim=2,
                            head_idx=head_idx,
                            weight=f"blocks.{block_idx}.attn.W_{letter}",
                            weight_head_dim=0,
                        )
                    )
        else:  # Not separate QKV
            for head_idx in range(n_q_heads):
                nodes.add(
                    DestNode(
                        name=f"A{block_idx}.{head_idx}",
                        module_name=f"blocks.{block_idx}.hook_attn_in",
                        layer=layer,
                        head_dim=2,
                        head_idx=head_idx,
                        weight=f"blocks.{block_idx}.attn.W_QKV",
                        weight_head_dim=0,
                    )
                )

        if not model.cfg.attn_only:
            nodes.add(
                DestNode(
                    name=f"MLP {block_idx}",
                    module_name=f"blocks.{block_idx}.hook_mlp_in",
                    layer=layer if model.cfg.parallel_attn_mlp else next(layers),
                    weight=f"blocks.{block_idx}.mlp.W_in",
                )
            )
    nodes.add(
        DestNode(
            name="Resid End",
            module_name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
            layer=next(layers),
            weight="unembed.W_U",
        )
    )
    return nodes


def simple_graph_nodes(
    model: tl.HookedTransformer,
) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """
    Get the nodes of the unfactorized graph (for node-level patching).
    """
    assert not model.cfg.parallel_attn_mlp
    layers, src_idxs = count(), count()
    src_nodes, dest_nodes = set(), set()
    layer, min_src_idx = next(layers), next(src_idxs)
    for block_idx in range(model.cfg.n_layers):
        first_block = block_idx == 0
        src_nodes.add(
            SrcNode(
                name="Resid Start" if first_block else f"Resid Post {block_idx - 1}",
                module_name=(
                    "blocks.0.hook_resid_pre"
                    if first_block
                    else f"blocks.{block_idx - 1}.hook_resid_post"
                ),
                layer=layer,
                src_idx=min_src_idx,
            )
        )
        for head_idx in range(model.cfg.n_heads):
            src_nodes.add(
                SrcNode(
                    name=f"A{block_idx}.{head_idx}",
                    module_name=f"blocks.{block_idx}.attn.hook_result",
                    layer=layer,
                    src_idx=next(src_idxs),
                    head_idx=head_idx,
                    head_dim=2,
                    weight=f"blocks.{block_idx}.attn.W_O",
                    weight_head_dim=0,
                )
            )
        if not (model.cfg.attn_only or model.cfg.parallel_attn_mlp):
            layer = next(layers)
            dest_nodes.add(
                DestNode(
                    name=f"Resid Mid {block_idx}",
                    module_name=f"blocks.{block_idx}.hook_resid_mid",
                    layer=layer,
                    min_src_idx=min_src_idx,
                )
            )
            min_src_idx = next(src_idxs)
            src_nodes.add(
                SrcNode(
                    name=f"Resid Mid {block_idx}",
                    module_name=f"blocks.{block_idx}.hook_resid_mid",
                    layer=layer,
                    src_idx=min_src_idx,
                )
            )
        if not model.cfg.attn_only:
            src_nodes.add(
                SrcNode(
                    name=f"MLP {block_idx}",
                    module_name=f"blocks.{block_idx}.hook_mlp_out",
                    layer=layer,
                    src_idx=next(src_idxs),
                    weight=f"blocks.{block_idx}.mlp.W_out",
                )
            )
        last_block = block_idx + 1 == model.cfg.n_layers
        layer = next(layers)
        dest_nodes.add(
            DestNode(
                name="Resid End" if last_block else f"Resid Post {block_idx}",
                module_name=f"blocks.{block_idx}.hook_resid_post",
                layer=layer,
                min_src_idx=min_src_idx,
            )
        )
        min_src_idx = next(src_idxs)
    return src_nodes, dest_nodes
