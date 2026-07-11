"""
This file contains some functions necessary for our more general method.
Where named similarly, they are analogous to the functions found in cdt_core.py,
but they also contain the logic to patch rel/irrel values in order to calculate decompositions of nodes at an intermediate layer.
"""

# flake8: noqa: F403, F405 - vendored CDT code uses star imports by design

from pyfunctions.cdt_core import *
from pyfunctions.wrappers import TargetNodeDecompositionList


# TODO: automatically check to see if reshape is necessary, because if not, one of the dimensions will already be the right size
import logging

logger = logging.getLogger(__name__)

def reshape_separate_attention_heads(context_layer, sa_module):
    new_shape = context_layer.size()[:-1] + (
        sa_module.num_attention_heads,
        sa_module.attention_head_size,
    )
    context_layer = context_layer.view(new_shape)
    return context_layer


def reshape_concatenate_attention_heads(context_layer, sa_module):
    new_shape = context_layer.size()[:-2] + (sa_module.all_head_size,)
    context_layer = context_layer.view(*new_shape)
    return context_layer


""" This function "patches" by setting the decomposition of a specific node.
It can be used to perform decomposition of a node relative to a target."""


def set_rel_at_source_nodes(
    rel, irrel, ablation_dict, layer_mean_acts, layer_idx, sa_module, set_irrel_to_mean, device
):

    if set_irrel_to_mean and layer_mean_acts is None:
        logger.info(
            "Tried to set decomposition of source node using mean method but no mean activation tensor provided; returning immediately \
               (likely the resulting decomposition will be meaningless)"
        )
    rel = reshape_separate_attention_heads(rel, sa_module)
    irrel = reshape_separate_attention_heads(irrel, sa_module)
    if layer_mean_acts is not None:
        layer_mean_acts = reshape_separate_attention_heads(layer_mean_acts, sa_module)
        if (
            layer_mean_acts.dim() == 3
        ):  # may pass in a 4d tensor to patch with something other than mean
            layer_mean_acts = layer_mean_acts[None, :, :, :]  # add on a batch dimension

    for ablation, batch_indices in ablation_dict.items():
        for source_node in ablation:
            if source_node.layer_idx != layer_idx:
                continue
            sq = source_node.sequence_idx
            head = source_node.attn_head_idx
            if set_irrel_to_mean:
                rel[batch_indices, sq, head, :] = (
                    irrel[batch_indices, sq, head, :]
                    + rel[batch_indices, sq, head, :]
                    - torch.Tensor(layer_mean_acts[:, sq, head, :]).to(device)
                )
                irrel[batch_indices, sq, head, :] = torch.Tensor(
                    layer_mean_acts[:, sq, head, :]
                ).to(device)
            else:
                rel[batch_indices, sq, head, :] = (
                    irrel[batch_indices, sq, head, :] + rel[batch_indices, sq, head, :]
                )
                irrel[batch_indices, sq, head, :] = 0

    rel = reshape_concatenate_attention_heads(rel, sa_module)
    irrel = reshape_concatenate_attention_heads(irrel, sa_module)

    return rel, irrel


"""This function keeps track of the decomposition's value at target nodes, so they can be
processed later."""


def calculate_contributions(rel, irrel, ablation_dict, target_nodes, level, sa_module, device):
    rel = reshape_separate_attention_heads(rel, sa_module)
    irrel = reshape_separate_attention_heads(irrel, sa_module)
    target_nodes_at_level = [node for node in target_nodes if node[0] == level]
    target_decomps = []

    for ablation, batch_indices in ablation_dict.items():
        target_decomps_for_ablation = TargetNodeDecompositionList(ablation)

        for t in target_nodes_at_level:
            target_decomps_for_ablation.append(
                t,
                rel[batch_indices, t.sequence_idx, t.attn_head_idx, :],
                irrel[batch_indices, t.sequence_idx, t.attn_head_idx, :],
            )
        target_decomps.append(target_decomps_for_ablation)
    return target_decomps


def prop_attention_probs(
    rel,
    irrel,
    attention_mask,
    sa_module,
    ablation_dict,
    target_nodes,
    level,
    device,
    target_decomp_method="residual",
    tol=1e-8,
):
    # NOTE: This will run into type errors for the BERT model, but anything which is fundamentally related to a TransformerLens model should work fine.
    target_decomps = None

    # this is the same as the linear_core logic, but i duplicated it here so that i could use einsum instead of having to keep track of dimension transposing
    # I believe this results in more readable code too, despite the duplication
    rel_query_t = einsum(
        "batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head",
        rel,
        sa_module.attn_module.W_Q,
    )
    irrel_query_t = einsum(
        "batch query_pos d_model, n_heads d_model d_head -> batch query_pos n_heads d_head",
        irrel,
        sa_module.attn_module.W_Q,
    )
    exp_query_bias = sa_module.attn_module.b_Q.expand_as(rel_query_t)
    q_tot = torch.abs(rel_query_t) + torch.abs(irrel_query_t) + tol
    rel_query_bias = exp_query_bias * (torch.abs(rel_query_t) / q_tot)
    irrel_query_bias = exp_query_bias * (torch.abs(irrel_query_t) / q_tot)

    rel_queries = rel_query_t + rel_query_bias
    irrel_queries = irrel_query_t + irrel_query_bias
    if target_decomp_method == "query":
        target_decomps = calculate_contributions(
            rel_queries, irrel_queries, ablation_dict, target_nodes, level, sa_module, device
        )

    rel_key_t = einsum(
        "batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
        rel,
        sa_module.attn_module.W_K,
    )
    irrel_key_t = einsum(
        "batch key_pos d_model, n_heads d_model d_head -> batch key_pos n_heads d_head",
        irrel,
        sa_module.attn_module.W_K,
    )
    exp_key_bias = sa_module.attn_module.b_K.expand_as(rel_key_t)
    k_tot = torch.abs(rel_key_t) + torch.abs(irrel_key_t) + tol
    rel_key_bias = exp_key_bias * (torch.abs(rel_key_t) / k_tot)
    irrel_key_bias = exp_key_bias * (torch.abs(irrel_key_t) / k_tot)

    rel_keys = rel_key_t + rel_key_bias
    irrel_keys = irrel_key_t + irrel_key_bias
    if target_decomp_method == "key":
        target_decomps = calculate_contributions(
            rel_keys, irrel_keys, ablation_dict, target_nodes, level, sa_module, device
        )

    # here rel_keys + irrel_keys should equal keys, of course
    total_attention_scores = einsum(
        "batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos",
        (rel_queries + irrel_queries),
        (rel_keys + irrel_keys),
    ) / math.sqrt(rel_keys.shape[-1])
    total_attention_scores += attention_mask

    rel_attention_scores = einsum(
        "batch query_pos n_heads d_head, batch key_pos n_heads d_head -> batch n_heads query_pos key_pos",
        rel_queries,
        rel_keys,
    ) / math.sqrt(rel_keys.shape[-1])
    rel_attention_scores += attention_mask

    # it may be more principled to do a "linearization" of softmax like in the original CD paper to get this,
    # but this decomposition produces decent results.
    total_attention_probs = nn.functional.softmax(total_attention_scores, dim=-1)
    rel_attention_probs = nn.functional.softmax(rel_attention_scores, dim=-1)
    irrel_attention_probs = total_attention_probs - rel_attention_probs

    return rel_attention_probs, irrel_attention_probs, target_decomps


"""
This is actual activation patching.
It is not conceptually the same as other instances of the term "patching" or "ablation" in this repo.
Useful for various mechanistic interpretability experiments/reproductions.
"""


def patch_values(rel, irrel, patch_set, patch_values, layer_idx, sa_module, device):

    rel = reshape_separate_attention_heads(rel, sa_module)
    irrel = reshape_separate_attention_heads(irrel, sa_module)
    patch_values = reshape_separate_attention_heads(patch_values, sa_module)
    patch_values = patch_values[None, :, :, :]  # add on a batch dimension
    for node in patch_set:
        if node.layer_idx != layer_idx:
            continue
        sq = node.sequence_idx
        head = node.attn_head_idx
        # this messes up the decomposition here
        rel[:, sq, head, :] = 0
        irrel[:, sq, head, :] = torch.Tensor(patch_values[:, sq, head, :]).to(device)

    rel = reshape_concatenate_attention_heads(rel, sa_module)
    irrel = reshape_concatenate_attention_heads(irrel, sa_module)

    return rel, irrel
