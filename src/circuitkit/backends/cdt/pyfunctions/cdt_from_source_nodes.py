"""
This file contains an example implementation of some functions for the BERT model, analogous to cdt_basic.py.
The difference for the functions found here are that they "patch" the decomposition of source nodes, which you can
use to find the decomposition of a model's task objective relative to any source nodes.
This file may be helpful for understanding the full method, found in cdt_source_to_target.py.
"""

# flake8: noqa: F403, F405 - vendored CDT code uses star imports by design

from pyfunctions.cdt_ablations import *
from pyfunctions.cdt_basic import *
from pyfunctions.cdt_core import *


def prop_self_attention_patched(
    rel, irrel, attention_mask, head_mask, sa_module, att_probs=None, output_att_prob=False
):

    if att_probs is not None:
        att_probs = att_probs
    else:
        att_probs = get_attention_probs(rel + irrel, attention_mask, head_mask, sa_module)

    rel_value, irrel_value = prop_linear(rel, irrel, sa_module.value)

    rel_context = mul_att(att_probs, rel_value, sa_module)
    irrel_context = mul_att(att_probs, irrel_value, sa_module)

    if output_att_prob:
        return rel_context, irrel_context, att_probs
    else:
        return rel_context, irrel_context, None


def prop_attention_patched(
    rel,
    irrel,
    attention_mask,
    head_mask,
    source_nodes,
    layer_mean_acts,
    a_module,
    device,
    att_probs=None,
    output_att_prob=False,
    output_context=False,
    set_irrel_to_mean=False,
):

    rel_context, irrel_context, returned_att_probs = prop_self_attention_patched(
        rel, irrel, attention_mask, head_mask, a_module.self, att_probs, output_att_prob
    )

    if output_context:
        # for head output variance analysis
        context = rel_context + irrel_context
        context = reshape_separate_attention_heads(context, a_module.self)
        ##
    else:
        context = None

    normalize_rel_irrel(rel_context, irrel_context)

    output_module = a_module.output

    rel_dense, irrel_dense = prop_linear(rel_context, irrel_context, output_module.dense)

    normalize_rel_irrel(rel_dense, irrel_dense)

    rel_tot = rel_dense + rel
    irrel_tot = irrel_dense + irrel

    normalize_rel_irrel(rel_tot, irrel_tot)

    # now that we've calculated the output of the attention mechanism, set desired inputs to "relevant"
    rel_tot, irrel_tot = set_rel_at_source_nodes(
        rel_tot, irrel_tot, source_nodes, layer_mean_acts, a_module.self, set_irrel_to_mean, device
    )

    normalize_rel_irrel(rel_tot, irrel_tot)

    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, output_module.LayerNorm)

    normalize_rel_irrel(rel_out, irrel_out)

    return rel_out, irrel_out, returned_att_probs, context


def prop_layer_patched(
    rel,
    irrel,
    attention_mask,
    head_mask,
    source_nodes,
    layer_mean_acts,
    layer_module,
    device,
    att_probs=None,
    output_att_prob=False,
    output_context=False,
    set_irrel_to_mean=False,
):

    # attn module
    rel_a, irrel_a, returned_att_probs, context = prop_attention_patched(
        rel,
        irrel,
        attention_mask,
        head_mask,
        source_nodes,
        layer_mean_acts,
        layer_module.attention,
        device,
        att_probs,
        output_att_prob,
        output_context,
        set_irrel_to_mean=set_irrel_to_mean,
    )
    # linear (dense)
    i_module = layer_module.intermediate
    rel_id, irrel_id = prop_linear(rel_a, irrel_a, i_module.dense)
    normalize_rel_irrel(rel_id, irrel_id)

    # activation
    rel_iact, irrel_iact = prop_act(rel_id, irrel_id, i_module.intermediate_act_fn)

    # linear (dense)
    o_module = layer_module.output
    rel_od, irrel_od = prop_linear(rel_iact, irrel_iact, o_module.dense)
    normalize_rel_irrel(rel_od, irrel_od)

    rel_tot = rel_od + rel_a
    irrel_tot = irrel_od + irrel_a

    normalize_rel_irrel(rel_tot, irrel_tot)

    # layer norm
    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, o_module.LayerNorm)

    return rel_out, irrel_out, returned_att_probs, context


def prop_encoder_patched(
    rel, irrel, attention_mask, head_mask, encoder_module, level=0, att_list=None
):
    rel_enc, irrel_enc = rel, irrel
    for i, layer_module in enumerate(encoder_module.layer):
        if i < level:
            continue
        att_probs = att_list[i] if att_list is not None else None
        layer_head_mask = head_mask[i]

        rel_enc_n, irrel_enc_n = prop_layer(
            rel_enc, irrel_enc, attention_mask, layer_head_mask, layer_module, att_probs
        )

        normalize_rel_irrel(rel_enc_n, irrel_enc_n)
        rel_enc, irrel_enc = rel_enc_n, irrel_enc_n

    return rel_enc, irrel_enc


def prop_classifier_model_patched(
    encoding,
    model,
    device,
    source_nodes=[],
    mean_acts=None,
    att_list=None,
    output_att_prob=False,
    output_context=False,
    set_irrel_to_mean=False,
):
    # source_nodes: attention heads to patch. format: [(level, sequence_position, head)]
    # level: 0-11, sequence_position: 0-511, head: 0-11
    # rel_out: the contribution of the source_nodes
    # irrel_out: the contribution of everything else``

    embedding_output = get_embeddings_bert(encoding, model)
    input_shape = encoding["input_ids"].size()
    extended_attention_mask = get_extended_attention_mask(
        attention_mask=encoding["attention_mask"],
        input_shape=input_shape,
        model=model.bert,
        device=device,
    )

    head_mask = [None] * model.bert.config.num_hidden_layers
    encoder_module = model.bert.encoder

    att_probs_lst = []

    sh = list(embedding_output.shape)

    rel = torch.zeros(sh, dtype=embedding_output.dtype, device=device)
    irrel = torch.zeros(sh, dtype=embedding_output.dtype, device=device)

    irrel[:] = embedding_output[:]

    att_probs_lst = []
    context_lst = []
    for i, layer_module in enumerate(encoder_module.layer):
        layer_source_nodes = [p_entry for p_entry in source_nodes if p_entry[0] == i]
        layer_head_mask = head_mask[i]
        att_probs = None

        if mean_acts is not None:
            layer_mean_acts = mean_acts[i]
        else:
            layer_mean_acts = None

        rel_n, irrel_n, returned_att_probs, context = prop_layer_patched(
            rel,
            irrel,
            extended_attention_mask,
            layer_head_mask,
            layer_source_nodes,
            layer_mean_acts,
            layer_module,
            device,
            att_probs,
            output_att_prob,
            output_context,
            set_irrel_to_mean=set_irrel_to_mean,
        )
        normalize_rel_irrel(rel_n, irrel_n)
        rel, irrel = rel_n, irrel_n

        if output_att_prob:
            att_probs_lst.append(returned_att_probs.squeeze(0))
        if output_context:
            context_lst.append(context.cpu().detach().numpy())

    rel_pool, irrel_pool = prop_pooler(rel, irrel, model.bert.pooler)
    rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)

    return rel_out, irrel_out, att_probs_lst, context_lst
