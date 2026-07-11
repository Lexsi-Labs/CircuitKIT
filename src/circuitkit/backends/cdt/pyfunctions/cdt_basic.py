"""
This file contains an example implementation of vanilla contextual decomposition in the specific case of an HF BERT model.
It may be helpful to understand how the general method of contextual decomposition works, but in general more complex versions of these methods are what is used in our experiments.
It contains the operations necessary to transform a decomposition of the input into a decomposition of the output.
It does not contain any patching/ablation code, which we use for the circuit discovery algorithm and for more fine-grained mechanistic interpretability analysis.

This file also contains some utility functions for evaluating BERT models, since they are similar to the functions necessary to implement CD on BERT.
"""

# flake8: noqa: F403, F405 - vendored CDT code uses star imports by design

from pyfunctions.cdt_core import *


def prop_self_attention(rel, irrel, attention_mask, head_mask, sa_module, att_probs=None):
    if att_probs is not None:
        att_probs = att_probs
    else:
        att_probs = get_attention_probs(rel + irrel, attention_mask, head_mask, sa_module)

    rel_value, irrel_value = prop_linear(rel, irrel, sa_module.value)

    rel_context = mul_att(att_probs, rel_value, sa_module)
    irrel_context = mul_att(att_probs, irrel_value, sa_module)

    return rel_context, irrel_context


def prop_attention(rel, irrel, attention_mask, head_mask, a_module, att_probs=None):
    rel_context, irrel_context = prop_self_attention(
        rel, irrel, attention_mask, head_mask, a_module.self, att_probs
    )
    normalize_rel_irrel(rel_context, irrel_context)  # add

    output_module = a_module.output

    rel_dense, irrel_dense = prop_linear(rel_context, irrel_context, output_module.dense)
    normalize_rel_irrel(rel_dense, irrel_dense)  # add

    rel_tot = rel_dense + rel
    irrel_tot = irrel_dense + irrel
    normalize_rel_irrel(rel_tot, irrel_tot)  # add

    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, output_module.LayerNorm)
    normalize_rel_irrel(rel_out, irrel_out)  # add

    return rel_out, irrel_out


def prop_layer(rel, irrel, attention_mask, head_mask, layer_module, att_probs=None):
    rel_a, irrel_a = prop_attention(
        rel, irrel, attention_mask, head_mask, layer_module.attention, att_probs
    )

    i_module = layer_module.intermediate
    rel_id, irrel_id = prop_linear(rel_a, irrel_a, i_module.dense)
    normalize_rel_irrel(rel_id, irrel_id)  # add
    rel_iact, irrel_iact = prop_act(rel_id, irrel_id, i_module.intermediate_act_fn)

    o_module = layer_module.output
    rel_od, irrel_od = prop_linear(rel_iact, irrel_iact, o_module.dense)
    normalize_rel_irrel(rel_od, irrel_od)  # add

    rel_tot = rel_od + rel_a
    irrel_tot = irrel_od + irrel_a
    normalize_rel_irrel(rel_tot, irrel_tot)  # add

    rel_out, irrel_out = prop_layer_norm(rel_tot, irrel_tot, o_module.LayerNorm)

    return rel_out, irrel_out


def prop_pooler(rel, irrel, pooler_module):
    rel_first = rel[:, 0]
    irrel_first = irrel[:, 0]

    rel_lin, irrel_lin = prop_linear(rel_first, irrel_first, pooler_module.dense)
    rel_out, irrel_out = prop_act(rel_lin, irrel_lin, pooler_module.activation)

    return rel_out, irrel_out


def prop_encoder(rel, irrel, attention_mask, head_mask, encoder_module, att_list=None):
    rel_enc, irrel_enc = rel, irrel
    for i, layer_module in enumerate(encoder_module.layer):
        att_probs = att_list[i] if att_list is not None else None
        layer_head_mask = head_mask[i]

        rel_enc_n, irrel_enc_n = prop_layer(
            rel_enc, irrel_enc, attention_mask, layer_head_mask, layer_module, att_probs
        )

        normalize_rel_irrel(rel_enc_n, irrel_enc_n)
        rel_enc, irrel_enc = rel_enc_n, irrel_enc_n

    return rel_enc, irrel_enc


def prop_encoder_from_level(
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


def prop_classifier_model_from_level(
    encoding, rel_ind_list, model, device, max_seq_len, level=0, att_list=None
):
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

    for i, layer_module in enumerate(encoder_module.layer):
        if i == level:
            break
        embedding_output = layer_module(embedding_output, extended_attention_mask, head_mask[i])[0]

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

    rel_enc, irrel_enc = prop_encoder_from_level(
        rel, irrel, extended_attention_mask, head_mask, encoder_module, level
    )
    rel_pool, irrel_pool = prop_pooler(rel_enc, irrel_enc, model.bert.pooler)
    rel_out, irrel_out = prop_linear(rel_pool, irrel_pool, model.classifier)

    return rel_out, irrel_out


def comp_cd_scores_level_skip(
    model, encoding, label, le_dict, device, max_seq_len, level=0, skip=1, num_at_time=64
):

    closest_competitor, lab_index = get_closest_competitor(model, encoding, label, le_dict)

    L = int((encoding["input_ids"] != 0).long().sum())
    tot_rel, tot_irrel = prop_classifier_model_from_level(
        encoding, [get_rel_inds(0, L - 1)], model, device, max_seq_len=max_seq_len, level=level
    )
    tot_score = proc_score(tot_rel[0, :], lab_index, closest_competitor)
    tot_irrel_score = proc_score(tot_irrel[0, :], lab_index, closest_competitor)

    # get scores
    unit_rel_ind_list = [get_rel_inds(i, min(L - 1, i + skip - 1)) for i in range(0, L, skip)]

    def proc_num_at_time(ind_list):
        scores = np.empty(0)
        irrel_scores = np.empty(0)  # for ablation purposes
        L = len(ind_list)
        for i in range(int(L / num_at_time) + 1):
            cur_scores, cur_irrel = prop_classifier_model_from_level(
                encoding,
                ind_list[i * num_at_time : min(L, (i + 1) * num_at_time)],
                model,
                device,
                max_seq_len=max_seq_len,
                level=level,
            )
            # cur_scores = np.array([proc_score(cur_scores[i, :], lab_index, closest_competitor) - tot_score
            #                       for i in range(cur_scores.shape[0])])
            cur_scores = np.array(
                [
                    proc_score(cur_scores[i, :], lab_index, closest_competitor)
                    for i in range(cur_scores.shape[0])
                ]
            )
            scores = np.append(scores, cur_scores)

            cur_irrel = np.array(
                [
                    proc_score(cur_irrel[i, :], lab_index, closest_competitor)
                    for i in range(cur_irrel.shape[0])
                ]
            )
            irrel_scores = np.append(irrel_scores, cur_irrel)
        return scores, irrel_scores

    scores, irrel_scores = proc_num_at_time(unit_rel_ind_list)

    return scores, irrel_scores


def get_closest_competitor(model, encoding, label, le_dict):

    model_output = model(**encoding)
    lab_index = le_dict[label]

    output = model_output[0].clone().cpu().detach().numpy().squeeze()
    sort_inds = np.argsort(output)

    if sort_inds[-1] != lab_index:
        return sort_inds[-1], lab_index
    else:
        return sort_inds[-2], lab_index


# Custom Score Processing function
def proc_score(tot_score, lab_index, closest_competitor):
    return float(tot_score[lab_index] - tot_score[closest_competitor])


def get_rel_inds(start, stop):
    return list(range(start, stop + 1))
