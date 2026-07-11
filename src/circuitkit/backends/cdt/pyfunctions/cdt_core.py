import math
import warnings

import numpy as np
import torch
import transformer_lens
from fancy_einsum import einsum
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin

"""
This file contains some core operations for contextual decomposition, including the decomposition functions for various small models.
It also contains some basic utility functions for dealing with models, especially HF BERT.
"""


# This function's primary purpose is to ensure numerical stability of the decomposition.
# Problems may arise when rel and irrel have different signs in the same position.
# The important invariants to preserve are that the relative magnitudes of rel and irrel are maintained,
# and that rel + irrel = tot.
def normalize_rel_irrel(rel, irrel):
    tot = rel + irrel
    tot_mask = (rel * irrel) < 0
    rel_mask = tot_mask & (rel.abs() >= irrel.abs())
    irrel_mask = tot_mask & (~rel_mask)

    rel[rel_mask] = tot[rel_mask]
    rel[irrel_mask] = 0

    irrel[irrel_mask] = tot[irrel_mask]
    irrel[rel_mask] = 0


def get_encoding(text, tokenizer, device, max_seq_len=512):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        pad_to_max_length=True,
        return_tensors="pt",
    ).to(device)
    return encoding


def get_embeddings_bert(encoding, model):
    embedding_output = model.bert.embeddings(
        input_ids=encoding["input_ids"],
        position_ids=None,
        token_type_ids=encoding["token_type_ids"],
        inputs_embeds=None,
    )
    return embedding_output


def get_att_list(embedding_output, rel_pos, extended_attention_mask, encoder_model):
    att_scores = ()
    act = embedding_output

    for i, layer_module in enumerate(encoder_model.layer):
        key = layer_module.attention.self.key(act)
        query = layer_module.attention.self.query(act)

        att_probs = get_attention_scores(  # noqa: F821 - vendored CDT star-import
            query, key, extended_attention_mask, rel_pos, layer_module.attention.self
        )

        att_scores = att_scores + (att_probs,)

        act = layer_module(act, attention_mask=extended_attention_mask, rel_pos=rel_pos)[0]

    return att_scores


# This is the decomposition for ReLU chosen by the Agglomerative Contextual Decomposition paper.
# It is possible that a decomposition for GeLU, or other activations, would be better.
def prop_act(r, ir, act_mod):
    ir_act = act_mod(ir)
    r_act = act_mod(r + ir) - ir_act
    return r_act, ir_act


def prop_linear_core(rel, irrel, W, b, tol=1e-8):
    rel_t = torch.matmul(rel, W)
    irrel_t = torch.matmul(irrel, W)

    exp_bias = b.expand_as(rel_t)
    tot_wt = torch.abs(rel_t) + torch.abs(irrel_t) + tol

    rel_bias = exp_bias * (torch.abs(rel_t) / tot_wt)
    irrel_bias = exp_bias * (torch.abs(irrel_t) / tot_wt)

    # tot_pred = rel_bias + rel_t + irrel_bias + irrel_t

    return (rel_t + rel_bias), (irrel_t + irrel_bias)


def prop_linear(rel, irrel, linear_module):
    return prop_linear_core(rel, irrel, linear_module.weight.T, linear_module.bias)


def prop_GPT_unembed(rel, irrel, unembed_module):
    return prop_linear_core(rel, irrel, unembed_module.W_U, unembed_module.b_U)


def prop_layer_norm(rel, irrel, layer_norm_module, tol=1e-8):
    tot = rel + irrel
    rel_mn = torch.mean(rel, dim=2).unsqueeze(-1).expand_as(rel)
    irrel_mn = torch.mean(irrel, dim=2).unsqueeze(-1).expand_as(irrel)
    vr = (torch.mean(tot**2, dim=2) - torch.mean(tot, dim=2) ** 2).unsqueeze(-1).expand_as(tot)

    rel_wt = torch.abs(rel)
    irrel_wt = torch.abs(irrel)
    tot_wt = rel_wt + irrel_wt + tol
    """
    # huge hack; instead can refactor function signature but i don't have the tools to do this without editing in at least 30 places
    if hasattr(layer_norm_module, "eps"):
        epsilon = layer_norm_module.eps
        weight = layer_norm_module.weight
        bias = layer_norm_module.bias
    else:
        epsilon = layer_norm_module.cfg.layer_norm_eps
        weight = layer_norm_module.w
        bias = layer_norm_module.b
    """

    rel_t = ((rel - rel_mn) / torch.sqrt(vr + layer_norm_module.eps)) * layer_norm_module.weight
    irrel_t = (
        (irrel - irrel_mn) / torch.sqrt(vr + layer_norm_module.eps)
    ) * layer_norm_module.weight

    rel_bias = layer_norm_module.bias * (rel_wt / tot_wt)
    irrel_bias = layer_norm_module.bias * (irrel_wt / tot_wt)

    return rel_t + rel_bias, irrel_t + irrel_bias


def prop_pooler(rel, irrel, pooler_module):
    rel_first = rel[:, 0]
    irrel_first = irrel[:, 0]

    rel_lin, irrel_lin = prop_linear(rel_first, irrel_first, pooler_module.dense)
    rel_out, irrel_out = prop_act(rel_lin, irrel_lin, pooler_module.activation)

    return rel_out, irrel_out


def prop_classifier_model(encoding, rel_ind_list, model, device, max_seq_len, att_list=None):
    embedding_output = get_embeddings_bert(encoding, model)
    input_shape = encoding["input_ids"].size()
    extended_attention_mask = get_extended_attention_mask(
        attention_mask=encoding["attention_mask"],
        input_shape=input_shape,
        model=model.bert,
        device=device,
    )

    tot_rel = len(rel_ind_list)
    sh = list(embedding_output.shape)
    sh[0] = tot_rel

    rel = torch.zeros(sh, dtype=embedding_output.dtype, device=device)
    irrel = torch.zeros(sh, dtype=embedding_output.dtype, device=device)

    for i in range(tot_rel):
        rel_inds = rel_ind_list[i]
        mask = np.isin(np.arange(max_seq_len), rel_inds)

        rel[i, mask, :] = embedding_output[0, mask, :]
        irrel[i, ~mask, :] = embedding_output[0, ~mask, :]

    head_mask = [None] * model.bert.config.num_hidden_layers
    rel_enc, irrel_enc = prop_encoder(  # noqa: F821 - vendored CDT star-import
        rel, irrel, extended_attention_mask, head_mask, model.bert.encoder, att_list
    )
    rel_pool, irrel_pool = prop_pooler(rel_enc, irrel_enc, model.bert.pooler)
    rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)

    return rel_out, irrel_out


def transpose_for_scores(x, sa_module):
    # handle different attention calculation conventions:
    # if it's the "Standard" attention calculation, all the key and query matrices are concatenated,
    # so the current dimension is [batch, sequence_idx, attention_heads * attn_dim]
    # and we need to unroll it.
    # however, some models do this automatically
    if len(x.size()) == 3:
        new_x_shape = x.size()[:-1] + (sa_module.num_attention_heads, sa_module.attention_head_size)
        x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)


def mul_att(att_probs, value, sa_module):
    context_layer = torch.matmul(att_probs, transpose_for_scores(value, sa_module))
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (sa_module.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return context_layer


"""
Don't read too much into this code; it's taken from transformers.modeling_utils.py
with hacky alterations to make it work with our code.
This function may not actually be necessary, depending on what the shapes
of the inputs are.
TODO: determine whether this function is necessary or vestigial and then update this comment"""


def get_extended_attention_mask(attention_mask, input_shape, model, device):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    dtype = next(model.parameters()).dtype

    is_decoder = False
    if hasattr(model, "config") and model.config.is_decoder:
        is_decoder = True
    if isinstance(model, transformer_lens.HookedTransformer):
        is_decoder = True  # hack; just for GPT2 model
    if not (attention_mask.dim() == 2 and is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.",
                FutureWarning,
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


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
        target_decomps = (
            calculate_contributions_at_query_key(  # noqa: F821 - vendored CDT star-import
                rel_queries, irrel_queries, ablation_dict, target_nodes, level, sa_module, device
            )
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
        target_decomps = (
            calculate_contributions_at_query_key(  # noqa: F821 - vendored CDT star-import
                rel_keys, irrel_keys, ablation_dict, target_nodes, level, sa_module, device
            )
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


def get_attention_probs(tot_embed, attention_mask, head_mask, sa_module):
    mixed_query_layer = sa_module.query(
        tot_embed
    )  # these parentheses are the call to forward(), i think it's easiest to implement another wrapper class

    key_layer = transpose_for_scores(sa_module.key(tot_embed), sa_module)

    query_layer = transpose_for_scores(mixed_query_layer, sa_module)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = attention_scores / math.sqrt(sa_module.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # [1, 12, 512, 512]

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    return attention_probs


"""
This is actual activation patching.
It is not conceptually the same as other instances of the term "patching" or "ablation" in this repo.
Useful for various mechanistic interpretability experiments/reproductions.
"""


def patch_values(rel, irrel, patch_set, patch_values, layer_idx, sa_module, device):

    rel = reshape_separate_attention_heads(rel, sa_module)  # noqa: F821 - vendored CDT star-import
    irrel = reshape_separate_attention_heads(  # noqa: F821 - vendored CDT star-import
        irrel, sa_module
    )
    patch_values = reshape_separate_attention_heads(  # noqa: F821 - vendored CDT star-import
        patch_values, sa_module
    )
    patch_values = patch_values[None, :, :, :]  # add on a batch dimension
    for node in patch_set:
        if node.layer_idx != layer_idx:
            continue
        sq = node.sequence_idx
        head = node.attn_head_idx
        # this messes up the decomposition here
        rel[:, sq, head, :] = 0
        irrel[:, sq, head, :] = torch.Tensor(patch_values[:, sq, head, :]).to(device)

    rel = reshape_concatenate_attention_heads(  # noqa: F821 - vendored CDT star-import
        rel, sa_module
    )
    irrel = reshape_concatenate_attention_heads(  # noqa: F821 - vendored CDT star-import
        irrel, sa_module
    )

    return rel, irrel
