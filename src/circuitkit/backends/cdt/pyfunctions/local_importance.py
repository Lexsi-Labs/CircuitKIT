import collections
import os
import logging

logger = logging.getLogger(__name__)

import matplotlib
import numpy as np
import scipy as sp
import shap
import torch
import torch.nn.functional as F
import tqdm
from captum.attr import LayerIntegratedGradients, LLMGradientAttribution, TextTokenInput
from datasets import load_dataset
from IPython.core.display import HTML, display
from lime.lime_text import IndexedString, LimeTextExplainer, TextDomainMapper
from matplotlib.colors import LinearSegmentedColormap
from methods.bag_of_ngrams.processing import cleanReports, cleanSplit, stripChars
from pyfunctions.cdt_basic import get_extended_attention_mask
from pyfunctions.cdt_source_to_target import batch_run, prop_BERT_hh, prop_GPT
from pyfunctions.config import BASE_DIR
from pyfunctions.general import readJson
from pyfunctions.ioi_dataset import IOIDataset
from pyfunctions.pathology import exclude_labels, extract_synoptic, fixLabel
from pyfunctions.wrappers import Node
from sklearn import preprocessing
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)

"""
Functions to call different interpretation methods for input feature attribution.
We implemented a wrapper that supprts LIME, SHAP, Layer-integrated-gradients, and CD-T.
See examples in `notebooks/Local_importance.ipynb` to know how to use it.
"""


# ===== MAIN FUNCTIONS TO CALL =====
def load_data_and_model(data_name, model_type, device):
    identifier = f"{data_name}_{model_type}"
    if identifier == "pathology_bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data_path = os.path.join(BASE_DIR, "data/prostate.json")
        data, le_dict = load_path_data(data_path, tokenizer)
        # load model
        model_path = os.path.join(BASE_DIR, "models/path/bert_PrimaryGleason")
        model_checkpoint_file = os.path.join(model_path, "save_output")
        model = load_path_model(model_checkpoint_file, le_dict)
    elif identifier == "pathology_pubmed_bert":
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        )
        data_path = os.path.join(BASE_DIR, "data/prostate.json")
        data, le_dict = load_path_data(data_path, tokenizer)
        # load model
        model_path = os.path.join(BASE_DIR, "models/path/pubmed_bert_PrimaryGleason")
        model_checkpoint_file = os.path.join(model_path, "save_output")
        model = load_path_model(model_checkpoint_file, le_dict)
    elif identifier == "sst2_bert":
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
        data, le_dict = load_sst2_data()
        # load model
        model = AutoModelForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-SST-2"
        )
    elif (identifier == "agnews_bert") or (identifier == "agnews_rand_bert"):
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
        data, le_dict = load_agnews_data()
        # load model
        model = AutoModelForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-ag-news"
        )
    elif identifier == "ioi_gpt2":
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=False,
            refactor_factored_attn_matrices=True,
        )
        data = IOIDataset(
            prompt_type="mixed", N=50, tokenizer=model.tokenizer, prepend_bos=False, nb_templates=2
        )
        tokenizer = model.tokenizer
        le_dict = None

    model = model.eval()
    model.to(device)
    return data, le_dict, tokenizer, model


def run_local_importance(
    text,
    tokenized_prompt,
    io_seq_idx,
    s_seq_idx,
    label_idx,
    max_seq_len,
    model,
    tokenizer,
    device,
    method,
    model_type,
    class_names,
):

    if model_type == "bert":
        pad_token_id = tokenizer.pad_token_id
        if not pad_token_id:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    with torch.no_grad():
        if method == "CDT":
            scores, irrel_scores, intervals, words = run_cdt(
                text,
                model,
                tokenized_prompt,
                tokenizer,
                io_seq_idx,
                s_seq_idx,
                label_idx,
                device,
                model_type,
            )
            visualize_cdt(scores, irrel_scores, intervals, words)
        elif method == "lime":
            # need max_seq_len for padding
            scores, words = run_lime(
                text,
                class_names,
                model,
                tokenizer,
                device,
                io_seq_idx,
                s_seq_idx,
                label_idx,
                max_seq_len,
                model_type,
            )
            if model_type == "gpt2":
                scores = scores[:-1]
                words = words[:-1]
            visualize_common(scores, words, method)
        elif method == "shap":
            # need max_seq_len for padding
            scores, words = run_shap(
                text,
                model,
                tokenizer,
                device,
                io_seq_idx,
                s_seq_idx,
                label_idx,
                max_seq_len,
                model_type,
            )
            visualize_common(scores, words, method)
        elif method == "LIG":
            scores, words = run_lig(
                text, model, tokenized_prompt, tokenizer, device, label_idx, model_type
            )
            visualize_common(scores, words, method)
    return scores


#################################################
def get_ioi_word_scores(out_decomps, io_seq_idx, s_seq_idx):
    logits = out_decomps[0].rel + out_decomps[0].irrel  # 1, seq_len, 50257=d_vocab]

    io_logit = logits[0, -2, io_seq_idx]
    s_logit = logits[0, -2, s_seq_idx]
    full_score = np.abs(io_logit - s_logit)

    # for each source node determine the contribution of rel to the actual score
    rel_word_scores = []
    irrel_word_scores = []
    collections.namedtuple("Result", ("ablation_set", "score"))
    for idx, decomp in enumerate(out_decomps):
        assert idx == decomp.ablation_set[0].sequence_idx
        rel_io_logit = decomp.rel[0, -2, io_seq_idx]
        rel_s_logit = decomp.rel[0, -2, s_seq_idx]
        rel_score = rel_io_logit - rel_s_logit
        rel_norm_score = rel_score / full_score
        rel_word_scores.append(rel_norm_score)
        irrel_io_logit = decomp.irrel[0, -2, io_seq_idx]
        irrel_s_logit = decomp.irrel[0, -2, s_seq_idx]
        irrel_score = irrel_io_logit - irrel_s_logit
        irrel_norm_score = irrel_score / full_score
        irrel_word_scores.append(irrel_norm_score)
    return rel_word_scores[:-1], irrel_word_scores[:-1]


def run_cdt(
    text, model, tokenized_prompt, tokenizer, io_seq_idx, s_seq_idx, label_idx, device, model_type
):
    encoding = tokenizer(text, padding=False, return_tensors="pt").to(device)
    for k in encoding:
        if encoding[k].shape[1] > 512:
            actual = encoding[k][:, 1:-1]
            encoding[k] = torch.cat(
                (encoding[k][:, 0].unsqueeze(0), actual[:, :510], encoding[k][:, -1].unsqueeze(0)),
                axis=-1,
            )

    if model_type == "gpt2":
        encoding_idxs, attention_mask = encoding.input_ids, encoding.attention_mask
        input_shape = encoding_idxs.size()
        extended_attention_mask = get_extended_attention_mask(
            attention_mask, input_shape, model, device
        )

        n_heads = model.cfg.n_heads if hasattr(model, "cfg") else model.config.num_attention_heads
        ablation_sets = [
            tuple(Node(0, pos, head) for head in range(n_heads)) for pos in range(input_shape[1])
        ]
        target_nodes = []

        out_decomp, _, _, pre_layer_activations = prop_GPT(
            encoding_idxs[0:1, :],
            extended_attention_mask,
            model,
            [ablation_sets[0]],
            target_nodes=target_nodes,
            device=device,
            mean_acts=None,
            set_irrel_to_mean=False,
        )

        def prop_fn(ablation_list):
            return prop_GPT(
                encoding_idxs[0:1, :],
                extended_attention_mask,
                model,
                ablation_list,
                target_nodes=target_nodes,
                device=device,
                mean_acts=None,
                set_irrel_to_mean=False,
                cached_pre_layer_acts=pre_layer_activations,
            )

        out_decomps, target_decomps = batch_run(prop_fn, ablation_sets)
        # compute contributions based on task-specific metrics (currently only ioi is defined)
        rel_word_scores, irrel_word_scores = get_ioi_word_scores(out_decomps, io_seq_idx, s_seq_idx)
        words = tokenized_prompt.split("|")[:-1]
        intervals = None

    elif model_type == "bert":
        encoding_idxs, attention_mask = encoding.input_ids, encoding.attention_mask
        input_shape = encoding_idxs.size()
        # extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, model, device)

        n_heads_bert = (
            model.cfg.n_heads if hasattr(model, "cfg") else model.config.num_attention_heads
        )
        ablation_sets = [
            tuple(Node(0, pos, head) for head in range(n_heads_bert))
            for pos in range(input_shape[1])
        ]
        target_nodes = []

        out_decomp, _, _, pre_layer_activations = prop_BERT_hh(
            encoding,
            model,
            [ablation_sets[0]],
            target_nodes,
            device,
            mean_acts=None,
            output_att_prob=False,
            set_irrel_to_mean=False,
        )

        def prop_fn(ablation_list):
            return prop_BERT_hh(
                encoding,
                model,
                ablation_list,
                target_nodes,
                device,
                mean_acts=None,
                output_att_prob=False,
                set_irrel_to_mean=False,
                cached_pre_layer_acts=pre_layer_activations,
            )

        out_decomps, target_decomps = batch_run(prop_fn, ablation_sets)

        rel_word_scores, irrel_word_scores = [], []
        for i in range(input_shape[1]):
            rel_word_scores.append(out_decomps[i].rel[0][label_idx])
            irrel_word_scores.append(out_decomps[i].irrel[0][label_idx])

        toks = tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
        intervals, words = compute_word_intervals(toks)

    return rel_word_scores, irrel_word_scores, intervals, words


# LIME
def run_lime(
    text,
    class_names,
    model,
    tokenizer,
    device,
    io_seq_idx,
    s_seq_idx,
    label_idx,
    max_seq_len,
    model_type,
):
    def lime_predictor(texts):
        batch_size = 128
        if len(texts) % batch_size == 0:
            max_epochs = len(texts) // batch_size
        else:
            max_epochs = len(texts) // batch_size + 1

        total_probas = []
        for L in tqdm.tqdm(range(max_epochs), desc="LIME Batch Processing..."):
            start = batch_size * L
            end = batch_size * (L + 1) if len(texts) > batch_size * (L + 1) else len(texts)
            if model_type == "bert":
                outputs = model(
                    **tokenizer(
                        texts[start:end],
                        max_length=max_seq_len,
                        truncation=True,
                        padding="max_length",
                        return_attention_mask=True,
                        return_tensors="pt",
                    ).to(device)
                )
                tensor_logits = outputs[0]
            elif model_type == "gpt2":
                encoding = tokenizer(
                    texts[start:end],
                    max_length=max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).to(device)
                outputs = model(encoding.input_ids)  # [batch_size, max_seq_len, class_size]
                tensor_logits = outputs[:, -2, :]
            probas = F.softmax(tensor_logits).detach().cpu().numpy()
            total_probas.extend(probas)
        total_probas = np.stack(total_probas)  # [num_samples, num_classes]
        return total_probas

    explainer = LimeTextExplainer(class_names=class_names, bow=False, split_expression=" ")
    indexed_text = IndexedString(text, bow=False, split_expression=" ")
    vocab_size = indexed_text.num_words()

    if model_type == "bert":
        exp = explainer.explain_instance(
            text, lime_predictor, num_features=vocab_size, labels=[label_idx]
        )
        scores = exp.local_exp[label_idx]
        mapper = TextDomainMapper(indexed_text)
        combine_to_weight = mapper.map_exp_ids(scores, positions=True)  # [(word_pos, weight)]
    elif model_type == "gpt2":
        exp = explainer.explain_instance(
            text, lime_predictor, num_features=vocab_size, labels=[io_seq_idx, s_seq_idx]
        )
        io_scores = exp.local_exp[io_seq_idx]
        s_scores = exp.local_exp[s_seq_idx]
        mapper = TextDomainMapper(indexed_text)
        io_combine_to_weight = mapper.map_exp_ids(io_scores, positions=True)  # [(word_pos, weight)]
        s_combine_to_weight = mapper.map_exp_ids(s_scores, positions=True)  # [(word_pos, weight)]
        io_dict = {k: v for k, v in io_combine_to_weight}
        s_dict = {k: v for k, v in s_combine_to_weight}
        combine_to_weight = [(k, io_dict[k] - s_dict[k]) for k in io_dict]

    pos2word, pos2weight = {}, {}
    for combine, w in combine_to_weight:
        tag = combine.find("_")
        word = combine[:tag]
        pos = int(combine[tag + 1 :])
        pos2word[pos] = word
        pos2weight[pos] = w

    od = collections.OrderedDict(sorted(pos2word.items()))
    " ".join([item[1] for item in od.items()])
            
    scores = [pos2weight[pos] for pos in od.keys()]
    words = [pos2word[pos] for pos in od.keys()]

    return scores, words


# SHAP
def run_shap(
    text, model, tokenizer, device, io_seq_idx, s_seq_idx, label_idx, max_seq_len, model_type
):
    def shap_predictor_bert(texts):
        tv, masks = [], []
        for v in texts:
            enc = tokenizer(
                v,
                padding="max_length",
                max_length=max_seq_len,
                truncation=True,
                return_attention_mask=True,
            )
            tv.append(enc.input_ids)
            masks.append(enc.attention_mask)
        tv = torch.tensor(tv).to(device)
        masks = torch.tensor(masks).to(device)
        outputs = model(tv, attention_mask=masks)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:, label_idx])
        return val

    def shap_predictor_gpt_io(texts):
        tv = torch.tensor(
            [
                tokenizer.encode(v, padding="max_length", max_length=max_seq_len, truncation=True)
                for v in texts
            ]
        ).to(device)
        outputs = model(tv)[:, -2, :].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:, io_seq_idx])
        return val

    def shap_predictor_gpt_s(texts):
        tv = torch.tensor(
            [
                tokenizer.encode(v, padding="max_length", max_length=max_seq_len, truncation=True)
                for v in texts
            ]
        ).to(device)
        outputs = model(tv)[:, -2, :].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:, s_seq_idx])
        return val

    if model_type == "bert":
        explainer = shap.Explainer(shap_predictor_bert, tokenizer)
        scores = explainer([text], fixed_context=1)
        word_scores = scores.values[0]
        words = scores.data[0]
    elif model_type == "gpt2":
        io_explainer = shap.Explainer(shap_predictor_gpt_io, tokenizer)
        io_scores = io_explainer([text], fixed_context=1)
        s_explainer = shap.Explainer(shap_predictor_gpt_s, tokenizer)
        s_scores = s_explainer([text], fixed_context=1)
        word_scores = (io_scores.values[0] - s_scores.values[0])[:-1]
        words = io_scores.data[0][:-1]
    return word_scores, words


# LIG
def run_lig(text, model, tokenized_prompt, tokenizer, device, label_idx, model_type):
    def predict_forward_func_bert(
        input_ids, token_type_ids=None, position_ids=None, attention_mask=None
    ):
        """Function passed to ig constructors"""
        return model(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )[0]

    def construct_input_ref_pair(text, ref_token_id, sep_token_id, cls_token_id):
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(text_ids) > 510:
            text_ids = text_ids[:510]
        # construct input token ids
        input_ids = [cls_token_id] + text_ids + [sep_token_id]

        # construct reference token ids
        ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

        return (
            torch.tensor([input_ids], device=device),
            torch.tensor([ref_input_ids], device=device),
            len(text_ids),
        )

    def construct_input_ref_token_type_pair(input_ids, sep_ind=0):
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor(
            [[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device
        )
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1
        return token_type_ids, ref_token_type_ids

    def construct_input_ref_pos_id_pair(input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
        ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids, ref_position_ids

    def construct_attention_mask(input_ids):
        return torch.ones_like(input_ids)

    def construct_whole_bert_embeddings(
        input_ids,
        ref_input_ids,
        token_type_ids=None,
        ref_token_type_ids=None,
        position_ids=None,
        ref_position_ids=None,
    ):
        input_embeddings = model.bert.embeddings(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )
        ref_input_embeddings = model.bert.embeddings(
            ref_input_ids, token_type_ids=ref_token_type_ids, position_ids=ref_position_ids
        )

        return input_embeddings, ref_input_embeddings

    if model_type == "bert":
        ref_token_id = tokenizer.pad_token_id
        sep_token_id = tokenizer.sep_token_id
        cls_token_id = tokenizer.cls_token_id
        if not ref_token_id:
            ref_token_id = tokenizer.sep_token_id

        # text = ' '.join(words)
        input_ids, ref_input_ids, sep_id = construct_input_ref_pair(
            text, ref_token_id, sep_token_id, cls_token_id
        )

        token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, sep_id)
        position_ids, ref_position_ids = construct_input_ref_pos_id_pair(input_ids)
        attention_mask = construct_attention_mask(input_ids)

        lig = LayerIntegratedGradients(predict_forward_func_bert, model.bert.embeddings)

        attributions, delta = lig.attribute(
            inputs=(input_ids, token_type_ids, position_ids),
            baselines=(ref_input_ids, ref_token_type_ids, ref_position_ids),
            additional_forward_args=(attention_mask),
            target=int(label_idx),
            return_convergence_delta=True,
            n_steps=25,
        )
        tok_scores = attributions[0].detach().cpu().numpy().squeeze().sum(1)
        # input_tokens = words
        toks = tokenizer.convert_ids_to_tokens(input_ids[0])
        intervals, words = compute_word_intervals(toks)
        word_scores = combine_token_scores(intervals, tok_scores)

    elif model_type == "gpt2":
        words = tokenized_prompt.split("|")
        lig = LayerIntegratedGradients(model, model.embed)
        llm_attr = LLMGradientAttribution(lig, tokenizer)

        inp = TextTokenInput(
            "".join(words[:-1]),
            tokenizer,
            # skip_tokens=[1],  # skip the special token for the start of the text <s>
        )
        target = " " + words[-1]
        attr_res = llm_attr.attribute(inp, target=target, n_steps=25)
        word_scores = attr_res.seq_attr.cpu().numpy()
        words = [x.replace("Ġ", "") for x in attr_res.input_tokens]

    return word_scores, words


def visualize_common(word_scores, words, method):
    assert len(word_scores) == len(words)
    normalized = normalize_word_scores(word_scores)
    logger.info(f"Viz {method}: ")
    if method != "LIG":
        display_colored_html(words, normalized)
    else:
        display_colored_html(words[1:-1], normalized[1:-1])


# visualization helper
def visualize_cdt(scores, irrel_scores, intervals, words):
    if intervals:
        scores = combine_token_scores(intervals, scores)
        irrel_scores = combine_token_scores(intervals, irrel_scores)  # for ablation purpose

    normalized = normalize_word_scores(scores)
    irrel_normalized = normalize_word_scores(irrel_scores)

    logger.info("Viz rel: ")
    display_colored_html(words[1:-1], normalized[1:-1])
    logger.info("Viz irrel: ")
    display_colored_html(words[1:-1], irrel_normalized[1:-1])
    logger.info("Viz rel-irrel: ")
    display_colored_html(words[1:-1], (normalized - irrel_normalized)[1:-1])


def display_colored_html(words, scores):
    s = colorize(words, scores)
    display(HTML(s))


def normalize_word_scores(word_scores):
    neg_pos_lst = [i for i, x in enumerate(word_scores) if x < 0]
    abs_word_scores = np.abs(word_scores)
    normalized = (abs_word_scores - min(abs_word_scores)) / (
        max(abs_word_scores) - min(abs_word_scores)
    )  # in [0, 1] range
    for i, x in enumerate(normalized):
        if i in neg_pos_lst:
            normalized[i] = -normalized[i]
    return normalized


def chop_cmap_frac(cmap: LinearSegmentedColormap, frac: float) -> LinearSegmentedColormap:
    """Chops off the ending 1- `frac` fraction of a colormap."""
    cmap_as_array = cmap(np.arange(256))
    cmap_as_array = cmap_as_array[: int(frac * len(cmap_as_array))]
    return LinearSegmentedColormap.from_list(cmap.name + f"_frac{frac}", cmap_as_array)


def colorize(words, color_array, mid=0):
    cmap_pos = LinearSegmentedColormap.from_list("", ["white", "#48b6df"])
    cmap_neg = LinearSegmentedColormap.from_list("", ["white", "#dd735b"])
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ""
    for word, color in zip(words, color_array):
        if color > mid:
            color = matplotlib.colors.rgb2hex(cmap_pos(color)[:3])
        elif color < mid:
            color = matplotlib.colors.rgb2hex(cmap_neg(abs(color))[:3])
        colored_string += template.format(color, "&nbsp" + word + "&nbsp")
    return colored_string


def compute_word_intervals(token_lst):
    word_cnt = 0
    interval_dict = collections.defaultdict(list)

    pretok_sent = ""

    tokens_len = len(token_lst)
    for i in range(tokens_len):
        tok = token_lst[i]
        if tok.startswith("##"):
            interval_dict[word_cnt].append(i)
            pretok_sent += tok[2:]
        else:
            word_cnt += 1
            interval_dict[word_cnt].append(i)
            pretok_sent += " " + tok
    pretok_sent = pretok_sent[1:]
    word_lst = pretok_sent.split(" ")

    assert len(interval_dict) == len(word_lst)

    return interval_dict, word_lst


def combine_token_scores(interval_dict, scores):
    word_cnt = len(interval_dict)
    new_scores = np.zeros(word_cnt)
    for i in range(word_cnt):
        t_idx_lst = interval_dict[i + 1]
        if len(t_idx_lst) == 1:
            new_scores[i] = scores[t_idx_lst[0]]
        else:
            new_scores[i] = np.sum(scores[t_idx_lst[0] : t_idx_lst[-1] + 1])
    return new_scores


# data helper
def load_agnews_data():
    raw_agnews = load_dataset("ag_news", split="test")
    label_classes = np.unique(raw_agnews["label"])
    le = preprocessing.LabelEncoder()
    le.fit(label_classes)

    # Map raw label to processed label
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    le_dict = {key: le_dict[key] for key in le_dict}

    data_dict = {"docs": raw_agnews["text"], "labels": raw_agnews["label"]}
    return data_dict, le_dict


def load_sst2_data():
    raw_sst2 = load_dataset("glue", "sst2", split="validation")
    label_classes = np.unique(raw_sst2["label"])
    le = preprocessing.LabelEncoder()
    le.fit(label_classes)
    # Map raw label to processed label
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    le_dict = {key: le_dict[key] for key in le_dict}

    data_dict = {"docs": raw_sst2["sentence"], "labels": raw_sst2["label"]}
    return data_dict, le_dict


def load_path_data(data_path, tokenizer):
    data = readJson(data_path)
    # Clean reports
    data = cleanSplit(data, stripChars)
    data["dev_test"] = cleanReports(data["dev_test"], stripChars)
    data = fixLabel(data)
    # print("Processing train data...")
    # train_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['train']]
    # print("Processing val data...")
    # val_documents = [extract_synoptic(patient['document'].lower(), tokenizer) for patient in data['val']]
    logger.info("Processing test data...")
    test_documents = [
        extract_synoptic(patient["document"].lower(), tokenizer) for patient in data["test"]
    ]

    # Create datasets
    # train_labels = [patient['labels']['PrimaryGleason'] for patient in data['train']]
    # val_labels = [patient['labels']['PrimaryGleason'] for patient in data['val']]
    test_labels = [patient["labels"]["PrimaryGleason"] for patient in data["test"]]

    # train_documents, train_labels = exclude_labels(train_documents, train_labels)
    # val_documents, val_labels = exclude_labels(val_documents, val_labels)
    test_documents, test_labels = exclude_labels(test_documents, test_labels)

    le_dict = {"3": 0, "4": 1, "5": 2}
    # le = preprocessing.LabelEncoder()
    # le.fit(train_labels)

    # Map raw label to processed label
    # le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    # le_dict = {str(key):le_dict[key] for key in le_dict}

    # for label in val_labels + test_labels:
    #    if str(label) not in le_dict:
    #        le_dict[str(label)] = len(le_dict)

    # Map processed label back to raw label
    # inv_le_dict = {v: k for k, v in le_dict.items()}

    # docs_dict = {'train': train_documents, 'val': val_documents, 'test': test_documents}
    # labels_dict = {'train': train_labels, 'val': val_labels, 'test': test_labels}
    data_dict = {"docs": test_documents, "labels": test_labels}

    return data_dict, le_dict


def load_path_model(model_checkpoint_file, le_dict):
    logger.info("Loading in model...")
    model = BertForSequenceClassification.from_pretrained(
        model_checkpoint_file, num_labels=len(le_dict), output_hidden_states=True
    )
    return model


'''# IG
def run_ig(text, model, tokenizer, intervals, device, io_seq_idx, s_seq_idx, label_idx, IG_interpretable_embeds, max_seq_len, model_type):
    def predict_forward_func_bert(input_ids, token_type_ids=None,
                         position_ids=None, attention_mask=None):
        """Function passed to ig constructors"""
        return model(inputs_embeds=input_ids,
                     token_type_ids=token_type_ids,
                     position_ids=position_ids,
                     attention_mask=attention_mask)[0]

    def predict_forward_func_gpt(input_ids):
        """Function passed to ig constructors"""
        return model(inputs_embeds=input_ids)[:, -2, :]

    if model_type == "bert":
        ig = IntegratedGradients(predict_forward_func_bert)
        interpretable_embedding1, interpretable_embedding2, interpretable_embedding3 = IG_interpretable_embeds
        input_data, input_data_embed = get_input_data(interpretable_embedding1, interpretable_embedding2, interpretable_embedding3,
                                                      text, tokenizer, max_seq_len, device, model_type)

        attributions, approximation_error = ig_attribute(ig, int(label_idx), input_data_embed)
        scores = attributions[0].detach().cpu().numpy().squeeze().sum(1)

        word_scores = combine_token_scores(intervals, scores)
    elif model_type == "gpt2":
        ig = IntegratedGradients(predict_forward_func_gpt)
        interpretable_embedding1, interpretable_embedding2, interpretable_embedding3 = IG_interpretable_embeds
        input_data, input_data_embed = get_input_data(interpretable_embedding1, interpretable_embedding2, interpretable_embedding3,
                                                      text, tokenizer, max_seq_len, device, model_type)

    return word_scores

def run_local_importance_bert(text, label, model, tokenizer, le_dict, device, max_seq_len, method, class_names, IG_interpretable_embeds, level=0, skip=1, num_at_time=64):
    encoding = get_encoding(text, tokenizer, device, max_seq_len=max_seq_len)
    toks = tokenizer.convert_ids_to_tokens([x for x in encoding['input_ids'][0] if x !=0 ])
    intervals, words = compute_word_intervals(toks)
    with torch.no_grad():
        if method == "CDT":
            scores, irrel_scores = comp_cd_scores_level_skip(model, encoding, label, le_dict, device, max_seq_len=max_seq_len, level=level, skip=skip, num_at_time=num_at_time)
            visualize_cdt(scores, irrel_scores, intervals, words)
        elif method == "lime":
            # LIME has its own tokenizing scheme - split by spaces
            scores, words = run_lime(text, class_names, model, tokenizer, device, label, le_dict, max_seq_len=max_seq_len)
            visualize_common(scores, words, method)
        elif method == "shap":
            scores = run_shap(text, model, tokenizer, intervals, device, label, le_dict, max_seq_len=max_seq_len)
            visualize_common(scores, words, method)
        elif method == "IG":
            scores = run_ig(text, model, tokenizer, intervals, device, label, le_dict, IG_interpretable_embeds, max_seq_len=max_seq_len)
            visualize_common(scores, words, method)
    return scores
'''
