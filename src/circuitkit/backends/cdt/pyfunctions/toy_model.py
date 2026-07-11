"""
In order to reproduce the results of the docstring experiment, we use this 4-layer attention-only transformer.
This transformer is a standard one that was released by Neel Nanda alongside the TransformerLens library, specifically
for use in mechanistic interpretability experiments.
This file serves as a good example of implementation of CDT for a custom model architecture.
See also the file cdt_basic.py for another helpful example.
"""

# flake8: noqa: F403, F405 - vendored CDT code uses star imports by design

from typing import Optional

import torch
import transformer_lens
from pyfunctions.cdt_basic import *
from pyfunctions.cdt_from_source_nodes import set_rel_at_source_nodes
from pyfunctions.cdt_source_to_target import calculate_contributions, prop_attention_no_output_hh
from pyfunctions.wrappers import (
    AblationSet,
    GPTAttentionWrapper,
    GPTLayerNormWrapper,
    Node,
    OutputDecomposition,
    TargetNodeDecompositionList,
)


def prop_toy_model_4l_layer(
    rel,
    irrel,
    attention_mask,
    head_mask,
    ablation_dict,
    target_nodes,
    level,
    layer_mean_acts,
    layer_module,
    device,
    att_probs=None,
    set_irrel_to_mean=False,
    target_decomp_method="residual",
):
    rel_ln, irrel_ln = prop_layer_norm(rel, irrel, GPTLayerNormWrapper(layer_module.ln1))
    attn_wrapper = GPTAttentionWrapper(
        layer_module.attn
    )  # it's named GPT but really it has to do with the TransformerLens conventions

    rel_summed_values, irrel_summed_values, returned_att_probs, layer_target_decomps = (
        prop_attention_no_output_hh(
            rel_ln,
            irrel_ln,
            attention_mask,
            head_mask,
            attn_wrapper,
            ablation_dict,
            target_nodes,
            level,
            device,
            target_decomp_method=target_decomp_method,
        )
    )

    rel_summed_values, irrel_summed_values = set_rel_at_source_nodes(
        rel_summed_values,
        irrel_summed_values,
        ablation_dict,
        layer_mean_acts,
        level,
        attn_wrapper,
        set_irrel_to_mean,
        device,
    )

    if target_decomp_method == "residual":
        layer_target_decomps = calculate_contributions(
            rel_summed_values,
            irrel_summed_values,
            ablation_dict,
            target_nodes,
            level,
            attn_wrapper,
            device=device,
        )

    rel_attn_residual, irrel_attn_residual = prop_linear(
        rel_summed_values, irrel_summed_values, attn_wrapper.output
    )

    rel_out, irrel_out = rel + rel_attn_residual, irrel + irrel_attn_residual
    # No MLP in this model

    return rel_out, irrel_out, layer_target_decomps, returned_att_probs


def prop_toy_model_4l(
    encoding_idxs: torch.Tensor,
    extended_attention_mask: torch.Tensor,
    model: transformer_lens.HookedTransformer,
    ablation_list: list[AblationSet],
    target_nodes: list[Node],
    device,
    mean_acts: Optional[torch.Tensor] = None,
    att_list: Optional[torch.Tensor] = None,
    set_irrel_to_mean=False,
    cached_pre_layer_acts: Optional[list[torch.Tensor]] = None,
    target_decomp_method="residual",
):
    head_mask = [None] * len(model.blocks)

    # we have to do a "separate" forward pass for each ablation for which we want to perform decomposition
    # so unroll the source nodes along the batch dimension, but keep track of which
    # "examples" belong to which source nodes
    actual_batch_size = encoding_idxs.size()[0]
    ablation_dict = {}
    start_batch_idx = 0
    for ablation in ablation_list:
        ablation_dict[ablation] = list(range(start_batch_idx, start_batch_idx + actual_batch_size))
        start_batch_idx += actual_batch_size

    if cached_pre_layer_acts is None:
        pre_layer_acts = []
        earliest_layer_to_run = 0
        latest_layer_to_run = len(model.blocks) - 1
        embedding_output = model.embed(encoding_idxs) + model.pos_embed(encoding_idxs)
        irrel = embedding_output.repeat(len(ablation_list), 1, 1)
        rel = torch.zeros(irrel.size(), dtype=embedding_output.dtype, device=device)
    else:
        pre_layer_acts = None
        earliest_layer_to_run = len(model.blocks)
        for ablation in ablation_list:
            for source_node in ablation:
                if source_node.layer_idx < earliest_layer_to_run:
                    earliest_layer_to_run = source_node.layer_idx
        if len(target_nodes) == 0:
            # this allows us to calculate contribution of source node to logits with cached values
            latest_layer_to_run = len(model.blocks) - 1
        else:
            latest_layer_to_run = 0
            for target_node in target_nodes:
                if target_node.layer_idx > latest_layer_to_run:
                    latest_layer_to_run = target_node.layer_idx
        irrel = cached_pre_layer_acts[earliest_layer_to_run].repeat(len(ablation_list), 1, 1)
        rel = torch.zeros(irrel.size(), dtype=irrel.dtype, device=device)
    target_decomps = [TargetNodeDecompositionList(x) for x in ablation_list]

    for i in range(earliest_layer_to_run, latest_layer_to_run + 1):
        if cached_pre_layer_acts is None:
            pre_layer_acts.append(rel + irrel)
        layer_module = model.blocks[i]
        layer_head_mask = head_mask[i]
        att_probs = None

        if mean_acts is not None:
            if mean_acts.dim() == 3:
                layer_mean_acts = mean_acts[i]
            else:
                layer_mean_acts = mean_acts[:, i, :, :]
        else:
            layer_mean_acts = None

        rel, irrel, layer_target_decomps, returned_att_probs = prop_toy_model_4l_layer(
            rel,
            irrel,
            extended_attention_mask,
            layer_head_mask,
            ablation_dict,
            target_nodes,
            i,
            layer_mean_acts,
            layer_module,
            device,
            att_probs,
            set_irrel_to_mean=set_irrel_to_mean,
            target_decomp_method=target_decomp_method,
        )
        for idx in range(len(target_decomps)):
            target_decomps[idx] += layer_target_decomps[idx]

        # att_probs_lst.append(returned_att_probs.squeeze(0))
    # return rel + irrel
    rel, irrel = prop_layer_norm(rel, irrel, GPTLayerNormWrapper(model.ln_final))
    rel_out, irrel_out = prop_GPT_unembed(rel, irrel, model.unembed)

    out_decomps = []
    for ablation, batch_indices in ablation_dict.items():
        rel_vec = rel_out[batch_indices, :].detach().cpu().numpy()
        irrel_vec = irrel_out[batch_indices, :].detach().cpu().numpy()
        out_decomps.append(OutputDecomposition(ablation, rel_vec, irrel_vec))
    att_probs_lst = []  # just to adhere to the API of analogous functions
    return out_decomps, target_decomps, att_probs_lst, pre_layer_acts
