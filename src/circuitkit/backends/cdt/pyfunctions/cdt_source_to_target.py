# flake8: noqa: F403, F405 - vendored CDT code uses star imports by design
from typing import Optional

import tqdm
import transformer_lens
from fancy_einsum import einsum
from pyfunctions.cdt_basic import *
from pyfunctions.cdt_from_source_nodes import *
from pyfunctions.wrappers import (
    AblationSet,
    GPTAttentionWrapper,
    GPTLayerNormWrapper,
    Node,
    OutputDecomposition,
    TargetNodeDecompositionList,
)
from transformers.activations import NewGELUActivation


import logging

logger = logging.getLogger(__name__)

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


def calculate_contributions_at_query_key(
    rel, irrel, ablation_dict, target_nodes, level, sa_module, device
):

    # rel = reshape_separate_attention_heads(rel, sa_module)
    # irrel = reshape_separate_attention_heads(irrel, sa_module)
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


"""
This is actual activation patching.
Not the same as every other instance of patching or ablation in this repo.
Use it to identify backup name mover heads.
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


def prop_attention_probs_tmp(
    rel,
    irrel,
    attention_mask,
    head_mask,
    sa_module,
    ablation_dict,
    target_nodes,
    level,
    device,
    target_decomp_method="residual",
    tol=1e-8,
):
    target_decomps = None
    # Ensure rel/irrel match the weight dtype for cross-architecture compatibility
    # (e.g. BERT bf16, GPT-2 float32).  W_Q is the authoritative dtype reference.
    w_dtype = sa_module.attn_module.W_Q.dtype
    rel = rel.to(dtype=w_dtype)
    irrel = irrel.to(dtype=w_dtype)

    # this is the linear_core logic, but i duplicated it here so that i could use einsum
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
        target_decomps = calculate_contributions_at_query_key(
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
        target_decomps = calculate_contributions_at_query_key(
            rel_keys, irrel_keys, ablation_dict, target_nodes, level, sa_module, device
        )

    # here rel_keys + irrel_keys should equal keys, ofc

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

    # it may be more principled to do a "linearization" of softmax like in the original CD paper to get this
    total_attention_probs = nn.functional.softmax(total_attention_scores, dim=-1)
    rel_attention_probs = nn.functional.softmax(rel_attention_scores, dim=-1)
    irrel_attention_probs = total_attention_probs - rel_attention_probs

    return rel_attention_probs, irrel_attention_probs, target_decomps


# This function handles what is usually called the attention mechanism, up to the point
# where the softmax'd attention pattern is multiplied by the value vectors.
# It does not include the output matrix.
# The reason for this arrangement is due to the way that HF BERT organizes its internal modules.
# BERT's attention mechanism is split into modules 'self' and 'output', with this corresponding to the former.
# Additionally, BERT and GPT differ in the placement of their LayerNorms, with BERT doing one before addition to the residual,
# and GPT doing one after, so it's messy to write a single "attention" function which handles both.
# However, the contents of this function are common to both BERT and GPT.
def prop_attention_no_output_hh(
    rel,
    irrel,
    attention_mask,
    head_mask,
    sa_module,
    ablation_dict,
    target_nodes,
    level,
    device,
    att_probs=None,
    target_decomp_method="residual",
):
    if att_probs is not None:
        # Precomputed-att_probs path: never exercised by the active LLM
        # discovery path. It cannot proceed — `rel_att_probs` (used below) is
        # the rel/irrel decomposition of att_probs and is not recoverable from
        # a combined att_probs tensor. Fail loudly here instead of dying later
        # with an UnboundLocalError on `rel_att_probs` / `target_decomps`.
        raise NotImplementedError(
            "prop_attention_no_output_hh: a precomputed `att_probs` is not "
            "supported — it carries no rel/irrel decomposition."
        )
    else:
        # att_probs = get_attention_probs(rel[0].unsqueeze(0) + irrel[0].unsqueeze(0), attention_mask, head_mask, sa_module)
        rel_att_probs, irrel_att_probs, target_decomps = prop_attention_probs_tmp(
            rel,
            irrel,
            attention_mask,
            head_mask,
            sa_module,
            ablation_dict,
            target_nodes,
            level,
            device,
            target_decomp_method=target_decomp_method,
        )
        att_probs = rel_att_probs + irrel_att_probs

    rel_value, irrel_value = prop_linear(rel, irrel, sa_module.value)
    if target_decomp_method == "value":
        target_decomps = calculate_contributions(
            rel_value, irrel_value, ablation_dict, target_nodes, level, sa_module, device=device
        )
    total_context = mul_att(att_probs, (rel_value + irrel_value), sa_module)
    rel_context = mul_att(rel_att_probs, rel_value, sa_module)
    irrel_context = total_context - rel_context

    # rel_context = mul_att(att_probs, rel_value, sa_module)
    # irrel_context = mul_att(att_probs, irrel_value, sa_module)

    return rel_context, irrel_context, att_probs, target_decomps


def prop_BERT_attention_hh(
    rel,
    irrel,
    attention_mask,
    head_mask,
    ablation_list,
    target_nodes,
    level,
    layer_mean_acts,
    a_module,
    device,
    att_probs=None,
    set_irrel_to_mean=False,
):

    (
        rel_context,
        irrel_context,
        returned_att_probs,
        _target_decomps,
    ) = prop_attention_no_output_hh(
        rel,
        irrel,
        attention_mask,
        head_mask,
        a_module.self,
        ablation_list,
        target_nodes,
        level,
        device,
        att_probs,
    )

    # now that we've calculated the output of the attention mechanism, set desired inputs to "relevant"
    rel_context, irrel_context = set_rel_at_source_nodes(
        rel_context,
        irrel_context,
        ablation_list,
        layer_mean_acts,
        level,
        a_module.self,
        set_irrel_to_mean,
        device,
    )

    output_module = a_module.output
    rel_dense, irrel_dense = prop_linear(rel_context, irrel_context, output_module.dense)
    # normalize_rel_irrel(rel_dense, irrel_dense)

    rel_tot = rel_dense + rel
    irrel_tot = irrel_dense + irrel
    # normalize_rel_irrel(rel_tot, irrel_tot)

    target_decomps = calculate_contributions(
        rel_tot, irrel_tot, ablation_list, target_nodes, level, a_module.self, device=device
    )
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, output_module.LayerNorm)

    # normalize_rel_irrel(rel_out, irrel_out)

    return rel_out, irrel_out, target_decomps, returned_att_probs


def prop_BERT_layer_hh(
    rel,
    irrel,
    attention_mask,
    head_mask,
    ablation_list,
    target_nodes,
    level,
    layer_mean_acts,
    layer_module,
    device,
    att_probs=None,
    output_att_prob=False,
    set_irrel_to_mean=False,
):

    rel_a, irrel_a, target_decomps, returned_att_probs = prop_BERT_attention_hh(
        rel,
        irrel,
        attention_mask,
        head_mask,
        ablation_list,
        target_nodes,
        level,
        layer_mean_acts,
        layer_module.attention,
        device,
        att_probs,
        set_irrel_to_mean=set_irrel_to_mean,
    )

    i_module = layer_module.intermediate
    rel_id, irrel_id = prop_linear(rel_a, irrel_a, i_module.dense)
    normalize_rel_irrel(rel_id, irrel_id)

    rel_iact, irrel_iact = prop_act(rel_id, irrel_id, i_module.intermediate_act_fn)

    o_module = layer_module.output
    rel_od, irrel_od = prop_linear(rel_iact, irrel_iact, o_module.dense)
    normalize_rel_irrel(rel_od, irrel_od)

    rel_tot = rel_od + rel_a
    irrel_tot = irrel_od + irrel_a
    normalize_rel_irrel(rel_tot, irrel_tot)

    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, o_module.LayerNorm)

    return rel_out, irrel_out, target_decomps, returned_att_probs


def prop_GPT_layer(
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
    # Folded layernorm is handled by cd_propagate_tl (backends/cdt/propagation.py)
    # when the model is a HookedTransformer. This legacy path runs when the model
    # is a raw HuggingFace GPT-2 module (fold_ln=False). No casework needed here.
    rel_ln, irrel_ln = prop_layer_norm(rel, irrel, GPTLayerNormWrapper(layer_module.ln1))
    attn_wrapper = GPTAttentionWrapper(layer_module.attn)

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
    # rel_summed_values, irrel_summed_values = patch_values(rel, irrel, (Node(9, 14, 9), Node(9, 14, 6), Node(10, 14, 0)), layer_mean_acts, level, attn_wrapper, device)

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

    rel_mid, irrel_mid = rel + rel_attn_residual, irrel + irrel_attn_residual
    # rel_mid, irrel_mid = set_rel_at_source_nodes(rel_mid, irrel_mid, ablation_dict, layer_mean_acts, level, attn_wrapper, set_irrel_to_mean, device)

    rel_mid_norm, irrel_mid_norm = prop_layer_norm(
        rel_mid, irrel_mid, GPTLayerNormWrapper(layer_module.ln2)
    )

    # MLP

    rel_after_w_in, irrel_after_w_in = prop_linear_core(
        rel_mid_norm, irrel_mid_norm, layer_module.mlp.W_in, layer_module.mlp.b_in
    )
    normalize_rel_irrel(rel_after_w_in, irrel_after_w_in)

    # since GELU activation is stateless, it's not an attribute of the layer module
    rel_act, irrel_act = prop_act(rel_after_w_in, irrel_after_w_in, NewGELUActivation())
    rel_mlp_residual, irrel_mlp_residual = prop_linear_core(
        rel_act, irrel_act, layer_module.mlp.W_out, layer_module.mlp.b_out
    )
    normalize_rel_irrel(rel_mlp_residual, irrel_mlp_residual)
    rel_out, irrel_out = rel_mid + rel_mlp_residual, irrel_mid + irrel_mlp_residual
    normalize_rel_irrel(rel_out, irrel_out)

    # there is not a layernorm at the end of this block, unlike in BERT
    # print('irrel_norm after adding MLP residual', np.linalg.norm(irrel_out.cpu().numpy()))

    return rel_out, irrel_out, layer_target_decomps, returned_att_probs


# In order to get the contribution of a set of source nodes to a set of target nodes, pass in both, and look at return val target_decomps.
# In order to get the contribute of a set of source nodes to the logits, pass in source nodes, and look at return val out_decomps.
# In order to optimize calculation speed by using cached values for the points before the first source node, pass in cached_pre_layer_acts.
# Note that this will also end the calculation after the last target node is reached, which likely makes return val out_decomps meaningless.
# To avoid this behavior, pass in empty target nodes (e.g, if you want to calculate contribution of source node to logits).


def prop_BERT_hh(
    encoding,
    model,
    ablation_list: list[AblationSet],
    target_nodes: list[Node],
    device,
    mean_acts: Optional[torch.Tensor] = None,
    set_irrel_to_mean=False,
    cached_pre_layer_acts: Optional[torch.Tensor] = None,
):
    input_shape = encoding.input_ids.size()
    extended_attention_mask = get_extended_attention_mask(
        encoding.attention_mask, input_shape, model, device
    )

    head_mask = [None] * model.bert.config.num_hidden_layers
    encoder_module = model.bert.encoder

    # we have to do a "separate" forward pass for each ablation for which we want to perform decomposition
    # so unroll the source nodes along the batch dimension, but keep track of which
    # "examples" belong to which source nodes
    actual_batch_size = encoding.input_ids.size()[0]
    ablation_dict = {}
    start_batch_idx = 0
    for ablation in ablation_list:
        ablation_dict[ablation] = list(range(start_batch_idx, start_batch_idx + actual_batch_size))
        start_batch_idx += actual_batch_size

    target_decomps = [TargetNodeDecompositionList(x) for x in ablation_list]
    att_probs_lst = []

    if cached_pre_layer_acts is None:
        pre_layer_acts = []
        earliest_layer_to_run = 0
        latest_layer_to_run = len(encoder_module.layer) - 1
        irrel = get_embeddings_bert(encoding, model).repeat(len(ablation_list), 1, 1)
        rel = torch.zeros(irrel.size(), dtype=irrel.dtype, device=device)
    else:
        pre_layer_acts = None
        earliest_layer_to_run = len(encoder_module.layer)
        for ablation in ablation_list:
            for source_node in ablation:
                if source_node.layer_idx < earliest_layer_to_run:
                    earliest_layer_to_run = source_node.layer_idx
        if len(target_nodes) == 0:
            # this allows us to calculate contribution of source node to logits with cached values
            latest_layer_to_run = len(encoder_module.layer) - 1
        else:
            latest_layer_to_run = 0
            for target_node in target_nodes:
                if target_node.layer_idx > latest_layer_to_run:
                    latest_layer_to_run = target_node.layer_idx
        irrel = cached_pre_layer_acts[earliest_layer_to_run].repeat(len(ablation_list), 1, 1)
        rel = torch.zeros(irrel.size(), dtype=irrel.dtype, device=device)

    for i in range(earliest_layer_to_run, latest_layer_to_run + 1):
        if cached_pre_layer_acts is None:
            pre_layer_acts.append(rel + irrel)
        layer_module = encoder_module.layer[i]

        layer_head_mask = head_mask[i]
        att_probs = None

        if mean_acts is not None:
            if layer_mean_acts.dim() == 3:
                layer_mean_acts = mean_acts[i]
            else:
                layer_mean_acts = mean_acts[:, i, :, :]
        else:
            layer_mean_acts = None
        rel_n, irrel_n, layer_target_decomps, returned_att_probs = prop_BERT_layer_hh(
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
        )
        for idx in range(len(target_decomps)):
            target_decomps[idx] += layer_target_decomps[idx]

        normalize_rel_irrel(rel_n, irrel_n)
        rel, irrel = rel_n, irrel_n

        if output_att_prob:
            att_probs_lst.append(returned_att_probs.squeeze(0))

    rel_pool, irrel_pool = prop_pooler(rel, irrel, model.bert.pooler)
    rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)

    out_decomps = []
    for ablation, batch_indices in ablation_dict.items():
        rel_vec = rel_out[batch_indices, :].detach().cpu().numpy()
        irrel_vec = irrel_out[batch_indices, :].detach().cpu().numpy()
        out_decomps.append(OutputDecomposition(ablation, rel_vec, irrel_vec))

    return out_decomps, target_decomps, att_probs_lst, pre_layer_acts


# In order to get the contribution of a set of source nodes to a set of target nodes, pass in both, and look at return val target_decomps.
# In order to get the contribute of a set of source nodes to the logits, pass in source nodes, and look at return val out_decomps.
# In order to optimize calculation speed by using cached values for the points before the first source node, pass in cached_pre_layer_acts.
# Note that this will also end the calculation after the last target node is reached, which likely makes return val out_decomps meaningless.
# To avoid this behavior, pass in empty target nodes (e.g, if you want to calculate contribution of source node to logits).
def prop_GPT(
    encoding_idxs: torch.Tensor,
    extended_attention_mask: torch.Tensor,
    model: transformer_lens.HookedTransformer,
    ablation_list: list[AblationSet],
    target_nodes: list[Node],
    device,
    mean_acts: Optional[torch.Tensor] = None,
    att_list: Optional[torch.Tensor] = None,
    set_irrel_to_mean=False,
    cached_pre_layer_acts: Optional[torch.Tensor] = None,
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
    att_probs_lst = []
    for i in range(earliest_layer_to_run, latest_layer_to_run + 1):
        if cached_pre_layer_acts is None:
            pre_layer_acts.append(rel + irrel)
        layer_module = model.blocks[i]
        layer_head_mask = head_mask[i]
        att_probs = None

        if mean_acts is not None:
            layer_mean_acts = mean_acts[i]
        else:
            layer_mean_acts = None

        rel, irrel, layer_target_decomps, returned_att_probs = prop_GPT_layer(
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

        att_probs_lst.append(returned_att_probs.squeeze(0))

    rel, irrel = prop_layer_norm(rel, irrel, GPTLayerNormWrapper(model.ln_final))
    rel_out, irrel_out = prop_GPT_unembed(rel, irrel, model.unembed)
    out_decomps = []
    for ablation, batch_indices in ablation_dict.items():
        rel_vec = rel_out[batch_indices, :].detach().cpu().numpy()
        irrel_vec = irrel_out[batch_indices, :].detach().cpu().numpy()
        out_decomps.append(OutputDecomposition(ablation, rel_vec, irrel_vec))

    return out_decomps, target_decomps, att_probs_lst, pre_layer_acts


"""
This is different from running a model on a batch of input data.
Instead it calculates the decomposition relative to many source nodes at the same time.
"""


def batch_run(prop_model_fn, ablation_list, num_at_time=64, print_progress=False):

    out_decomps = []
    target_decomps = []

    n_ablations = len(ablation_list)
    n_batches = int((n_ablations + (num_at_time - 1)) / num_at_time)

    for b_no in tqdm.tqdm(range(n_batches), desc="Running decomposition in batches..."):
        b_st = b_no * num_at_time
        b_end = min(b_st + num_at_time, n_ablations)
        if print_progress:
            logger.info("Running inputs %d to %d (of %d)" % (b_st, b_end, n_ablations))
        batch_out_decomps, batch_target_decomps, _, _ = prop_model_fn(ablation_list[b_st:b_end])

        out_decomps += batch_out_decomps
        target_decomps += batch_target_decomps

    return out_decomps, target_decomps
