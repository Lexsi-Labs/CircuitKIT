from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from .....utils.device import get_device
from .....utils.logging import get_logger
from ...core.acdc_utils import (
    MatchNLLMetric,
    frac_correct_metric,
    kl_divergence,
    logit_diff_metric,
    negative_log_probs,
)
from ...core.TLACDCEdge import EdgeType, TorchIndex
from ...core.TLACDCInterpNode import TLACDCInterpNode

logger = get_logger("data.task_data.ioi_utils")

# Simple AllDataThings class for CircuitKit
@dataclass
class AllDataThings:
    """Simple data container for ACDC task data."""

    validation_metric: Any = None
    validation_data: Any = None
    validation_labels: Any = None
    validation_wrong_labels: Any = None
    validation_mask: Any = None
    validation_patch_data: Any = None
    test_metrics: Any = None
    test_data: Any = None
    test_labels: Any = None
    test_wrong_labels: Any = None
    test_mask: Any = None
    test_patch_data: Any = None

from transformer_lens.HookedTransformer import (  # noqa: E402 - import after intentional pre-import setup
    HookedTransformer,
)

from .ioi_dataset import (  # NOTE: we now import this LOCALLY so it is deterministic  # noqa: E402 - import after intentional pre-import setup
    IOIDataset,
)


def get_gpt2_small(device=None) -> HookedTransformer:
    device = device if device is not None else get_device()
    tl_model = HookedTransformer.from_pretrained("gpt2")
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model


def get_ioi_gpt2_small(device=None):
    """For backwards compat"""
    return get_gpt2_small(device=device)

def get_all_ioi_things(num_examples, device, metric_name, model=None, kl_return_one_element=True):
    if model is None:
        tl_model = get_gpt2_small(device=device)
    else:
        tl_model = model

    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=num_examples * 2,
        nb_templates=1,
        model=tl_model,  # Pass the model to IOIDataset
        seed=0,
    )

    abc_dataset = (
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )

    seq_len = ioi_dataset.toks.shape[1]
    assert seq_len == 16, f"Well, I thought ABBA #1 was 16 not {seq_len} tokens long..."

    default_data = ioi_dataset.toks.long()[: num_examples * 2, : seq_len - 1].to(device)
    patch_data = abc_dataset.toks.long()[: num_examples * 2, : seq_len - 1].to(device)
    labels = ioi_dataset.toks.long()[: num_examples * 2, seq_len - 1]
    wrong_labels = torch.as_tensor(
        ioi_dataset.s_tokenIDs[: num_examples * 2], dtype=torch.long, device=device
    )

    assert torch.equal(labels, torch.as_tensor(ioi_dataset.io_tokenIDs, dtype=torch.long))
    labels = labels.to(device)

    validation_data = default_data[:num_examples, :]
    validation_patch_data = patch_data[:num_examples, :]
    validation_labels = labels[:num_examples]
    validation_wrong_labels = wrong_labels[:num_examples]

    test_data = default_data[num_examples:, :]
    test_patch_data = patch_data[num_examples:, :]
    test_labels = labels[num_examples:]
    test_wrong_labels = wrong_labels[num_examples:]

    with torch.no_grad():
        base_model_logits = tl_model(default_data)[:, -1, :]
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)

    base_validation_logprobs = base_model_logprobs[:num_examples, :]
    base_test_logprobs = base_model_logprobs[num_examples:, :]

    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            return_one_element=kl_return_one_element,
        )
    elif metric_name == "logit_diff":
        validation_metric = partial(
            logit_diff_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
        )
    elif metric_name == "frac_correct":
        validation_metric = partial(
            frac_correct_metric,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
        )
    elif metric_name == "nll":
        validation_metric = partial(
            negative_log_probs,
            labels=validation_labels,
            last_seq_element_only=True,
        )
    elif metric_name == "match_nll":
        validation_metric = MatchNLLMetric(
            labels=validation_labels,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
        )
    else:
        raise ValueError(f"metric_name {metric_name} not recognized")

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
        ),
        "logit_diff": partial(
            logit_diff_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        ),
        "frac_correct": partial(
            frac_correct_metric,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
        ),
        "nll": partial(
            negative_log_probs,
            labels=test_labels,
            last_seq_element_only=True,
        ),
        "match_nll": MatchNLLMetric(
            labels=test_labels,
            base_model_logprobs=base_test_logprobs,
            last_seq_element_only=True,
        ),
    }

    return AllDataThings(
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_wrong_labels=validation_wrong_labels,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_wrong_labels=test_wrong_labels,
        test_mask=None,
        test_patch_data=test_patch_data,
    )

def get_ioi_data_only(num_examples, device, model, seed=42):
    """
    Get IOI data for EAP/EAP-IG without calculating metrics.

    This function generates IOI data without running model forward passes
    or creating metric functions. It's optimized for EAP/EAP-IG data loading
    where only the data (tokens and labels) is needed.

    Args:
        num_examples: Number of examples to generate
        device: Device to place tensors on
        model: HookedTransformer model (required for tokenization)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary containing:
        - validation_data: Clean token sequences [num_examples, seq_len-1]
        - validation_patch_data: Corrupted token sequences [num_examples, seq_len-1]
        - validation_labels: Correct answer token IDs [num_examples]
        - validation_wrong_labels: Incorrect answer token IDs [num_examples]
        - end_idxs: Answer positions [num_examples]
    """

    if model is None:
        raise ValueError("Model required for IOI data generation. No default model.")

    try:
        # Try standard IOIDataset approach (works for GPT-2, DistilGPT-2)

        ioi_dataset = IOIDataset(
            prompt_type="ABBA",
            N=num_examples,
            nb_templates=1,
            model=model,
            seed=seed,
        )

        # Use S2 -> RAND flipping to match standard ioi.csv format
        # This preserves the context (IO, S1) and only corrupts the indirect object reference
        abc_dataset = (
            ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=seed + 1)
            .gen_flipped_prompts(("S", "RAND"), seed=seed + 2)
            .gen_flipped_prompts(("S1", "RAND"), seed=seed + 3)
        )

        # Previous ABC corruption (completely different names) - kept for reference if needed
        # abc_dataset = (
        #     ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=seed + 1)
        #     .gen_flipped_prompts(("S", "RAND"), seed=seed + 2)
        #     .gen_flipped_prompts(("S1", "RAND"), seed=seed + 3)
        # )

        seq_len = ioi_dataset.toks.shape[1]
        if seq_len < 10 or seq_len > 50:
            raise ValueError(f"Unexpected sequence length: {seq_len}")

        # Extract texts exactly as EAP does — from ioi_prompts, not from toks.
        # Then re-tokenize via model.to_tokens() so the token sequence is
        # identical to what EAP/EAP-IG produces and to what model.forward()
        # internally expects.
        clean_texts = [p["text"] for p in ioi_dataset.ioi_prompts[:num_examples]]
        corrupted_texts = [p["text"] for p in abc_dataset.ioi_prompts[:num_examples]]

        clean_toks = model.to_tokens(clean_texts).to(device)  # [N, seq_len]
        corrupt_toks = model.to_tokens(corrupted_texts).to(device)

        # Labels: first subword of IO/S name, same as EAP uses via correct_idx
        labels = torch.as_tensor(
            ioi_dataset.io_tokenIDs[:num_examples], dtype=torch.long, device=device
        )
        wrong_labels = torch.as_tensor(
            ioi_dataset.s_tokenIDs[:num_examples], dtype=torch.long, device=device
        )

        # Answer position: last occurrence of the IO token in each clean row,
        # minus one (the "to" position — the last token before the IO name).
        # Scanning backwards handles the ABBA structure where the name appears
        # twice and we want the final occurrence.
        input_len = clean_toks.shape[1]
        end_idxs = torch.full((num_examples,), input_len - 1, dtype=torch.long, device=device)
        for i in range(num_examples):
            io_id = labels[i].item()
            for j in range(input_len - 1, -1, -1):
                if clean_toks[i, j].item() == io_id:
                    end_idxs[i] = max(j - 1, 0)
                    break

        logger.debug(
            # f"IOIDataset succeeded: trunc_len={trunc_len}, "
            f"end_idxs range=[{end_idxs.min().item()}, {end_idxs.max().item()}]"
        )

        return {
            "validation_data": clean_toks,
            "validation_patch_data": corrupt_toks,
            "validation_labels": labels,
            "validation_wrong_labels": wrong_labels,
            "end_idxs": end_idxs,
        }

    except (AssertionError, ValueError, KeyError, IndexError) as e:
        # IOIDataset failed - use model-agnostic fallback
        logger.error(
            f"IOIDataset failed ({type(e).__name__}), using model-agnostic fallback generation"
        )
        return _generate_ioi_data_fallback(num_examples, device, model, seed)

def _generate_ioi_data_fallback(num_examples, device, model, seed=42):
    """
    Model-agnostic IOI data generation fallback.

    Generates IOI prompts directly without relying on IOIDataset's
    GPT-2-specific tokenization assumptions.

    Args:
        num_examples: Number of examples
        device: Target device
        model: HookedTransformer model
        seed: Random seed

    Returns:
        Dictionary with IOI data (same format as get_ioi_data_only)
    """
    import random

    random.seed(seed)
    torch.manual_seed(seed)

    # Name pools
    names = [
        "John",
        "Mary",
        "Michael",
        "Sarah",
        "David",
        "Emma",
        "James",
        "Lisa",
        "Robert",
        "Jennifer",
        "William",
        "Linda",
        "Daniel",
        "Patricia",
        "Thomas",
        "Nancy",
        "Charles",
        "Karen",
        "Christopher",
        "Jessica",
        "Matthew",
        "Ashley",
    ]

    places = ["park", "store", "school", "office", "library", "cafe"]
    objects = ["present", "coffee", "book", "letter", "package", "message"]

    clean_tokens_list = []
    corrupted_tokens_list = []
    io_tokens_list = []
    s_tokens_list = []

    for i in range(num_examples):
        # Sample names: IO (correct answer), S (incorrect answer)
        io_name, s_name = random.sample(names, 2)
        place = random.choice(places)
        obj = random.choice(objects)

        # ABBA template: "When [IO] and [S] went to the [place], [S] gave a [object] to [IO]"
        clean_prompt = (
            f"When {io_name} and {s_name} went to the {place}, {s_name} gave a {obj} to {io_name}"
        )

        # Corrupted version: replace second occurrence of S with random name
        random_name = random.choice([n for n in names if n not in [io_name, s_name]])
        corrupted_prompt = f"When {io_name} and {s_name} went to the {place}, {random_name} gave a {obj} to {io_name}"

        # Tokenize (no BOS - we'll handle BOS consistently)
        clean_toks = model.to_tokens(clean_prompt).squeeze(0)
        corrupted_toks = model.to_tokens(corrupted_prompt).squeeze(0)

        # Get IO and S token IDs (first token of each name)
        io_tok = model.to_tokens(f" {io_name}", prepend_bos=False).squeeze(0)[0]
        s_tok = model.to_tokens(f" {s_name}", prepend_bos=False).squeeze(0)[0]

        clean_tokens_list.append(clean_toks)
        corrupted_tokens_list.append(corrupted_toks)
        io_tokens_list.append(io_tok)
        s_tokens_list.append(s_tok)

    # Pad all sequences to same length
    max_len = max(t.shape[0] for t in clean_tokens_list)
    pad_token_id = (
        model.tokenizer.pad_token_id
        if hasattr(model.tokenizer, "pad_token_id") and model.tokenizer.pad_token_id is not None
        else 0
    )

    padded_clean = []
    padded_corrupted = []
    end_positions = []

    for clean_toks, corrupted_toks, io_tok in zip(
        clean_tokens_list, corrupted_tokens_list, io_tokens_list
    ):
        clean_len = clean_toks.shape[0]
        corrupted_len = corrupted_toks.shape[0]

        # Find answer position: last occurrence of IO first-subword, step back 1
        io_tok_id = io_tok.item()
        answer_pos = clean_len - 2  # fallback
        for j in range(clean_len - 1, -1, -1):
            if clean_toks[j].item() == io_tok_id:
                answer_pos = max(j - 1, 0)
                break
        end_positions.append(answer_pos)

        # Pad clean
        if clean_len < max_len:
            clean_padding = torch.full(
                (max_len - clean_len,),
                pad_token_id,
                dtype=clean_toks.dtype,
                device=clean_toks.device,
            )
            clean_padded = torch.cat([clean_toks, clean_padding])
        else:
            clean_padded = clean_toks

        # Pad corrupted
        if corrupted_len < max_len:
            corrupted_padding = torch.full(
                (max_len - corrupted_len,),
                pad_token_id,
                dtype=corrupted_toks.dtype,
                device=corrupted_toks.device,
            )
            corrupted_padded = torch.cat([corrupted_toks, corrupted_padding])
        else:
            corrupted_padded = corrupted_toks

        padded_clean.append(clean_padded)
        padded_corrupted.append(corrupted_padded)

    # Stack into tensors
    clean_batch = torch.stack(padded_clean).to(device)
    corrupted_batch = torch.stack(padded_corrupted).to(device)
    io_batch = torch.tensor(io_tokens_list, dtype=torch.long, device=device)
    s_batch = torch.tensor(s_tokens_list, dtype=torch.long, device=device)
    end_idxs = torch.tensor(end_positions, dtype=torch.long, device=device)

    # Truncate input to (max answer_pos + 1)
    input_len = int(end_idxs.max().item()) + 1
    validation_data = clean_batch[:, :input_len]
    validation_patch_data = corrupted_batch[:, :input_len]

    # Labels: first subword of IO name (already correct from io_tokens_list)
    validation_labels = io_batch

    logger.debug(
        f"Fallback generation succeeded: input_len={input_len}, "
        f"end_idxs range=[{end_idxs.min().item()}, {end_idxs.max().item()}]"
    )

    return {
        "validation_data": validation_data,
        "validation_patch_data": validation_patch_data,
        "validation_labels": validation_labels,
        "validation_wrong_labels": s_batch,
        "end_idxs": end_idxs,
    }

IOI_CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
    ],
    "backup name mover": [
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
        # (7, 1),
    ],  # unclear exactly what (7,1) does
    "previous token": [
        (2, 2),
        # (2, 9),
        (4, 11),
        # (4, 3),
        # (4, 7),
        # (5, 6),
        # (3, 3),
        # (3, 7),
        # (3, 6),
    ],
}

@dataclass(frozen=True)
class Conn:
    inp: str
    out: str
    qkv: tuple[str, ...]

def get_ioi_true_edges(model):
    nodes_to_mask = []

    all_groups_of_nodes = [group for _, group in IOI_CIRCUIT.items()]
    all_nodes = [node for group in all_groups_of_nodes for node in group]
    assert len(all_nodes) == 26, len(all_nodes)

    nodes_to_mask = []

    for layer_idx in range(12):
        for head_idx in range(12):
            if (layer_idx, head_idx) not in all_nodes:
                for letter in ["q", "k", "v"]:
                    nodes_to_mask.append(
                        TLACDCInterpNode(
                            name=f"blocks.{layer_idx}.attn.hook_{letter}",
                            index=TorchIndex([None, None, head_idx]),
                            incoming_edge_type=EdgeType.DIRECT_COMPUTATION,
                        ),
                    )

    from subnetwork_probing.train import iterative_correspondence_from_mask

    corr, _ = iterative_correspondence_from_mask(
        nodes_to_mask=nodes_to_mask,
        model=model,
    )

    # For all heads...
    for layer_idx, head_idx in all_nodes:
        for letter in "qkv":
            # remove input -> head connection
            edge_to = corr.edges[f"blocks.{layer_idx}.hook_{letter}_input"][
                TorchIndex([None, None, head_idx])
            ]
            edge_to["blocks.0.hook_resid_pre"][TorchIndex([None])].present = False

            # Remove all other_head->this_head connections in the circuit
            for layer_from in range(layer_idx):
                for head_from in range(12):
                    edge_to[f"blocks.{layer_from}.attn.hook_result"][
                        TorchIndex([None, None, head_from])
                    ].present = False

            # Remove connection from this head to the output
            corr.edges["blocks.11.hook_resid_post"][TorchIndex([None])][
                f"blocks.{layer_idx}.attn.hook_result"
            ][TorchIndex([None, None, head_idx])].present = False

    special_connections: set[Conn] = {
        Conn("INPUT", "previous token", ("q", "k", "v")),
        Conn("INPUT", "duplicate token", ("q", "k", "v")),
        Conn("INPUT", "s2 inhibition", ("q",)),
        Conn("INPUT", "negative", ("k", "v")),
        Conn("INPUT", "name mover", ("k", "v")),
        Conn("INPUT", "backup name mover", ("k", "v")),
        Conn("previous token", "induction", ("k", "v")),
        Conn("induction", "s2 inhibition", ("k", "v")),
        Conn("duplicate token", "s2 inhibition", ("k", "v")),
        Conn("s2 inhibition", "negative", ("q",)),
        Conn("s2 inhibition", "name mover", ("q",)),
        Conn("s2 inhibition", "backup name mover", ("q",)),
        Conn("negative", "OUTPUT", ()),
        Conn("name mover", "OUTPUT", ()),
        Conn("backup name mover", "OUTPUT", ()),
    }

    for conn in special_connections:
        if conn.inp == "INPUT":
            idx_from = [(-1, "blocks.0.hook_resid_pre", TorchIndex([None]))]
            for mlp_layer_idx in range(12):
                idx_from.append(
                    (mlp_layer_idx, f"blocks.{mlp_layer_idx}.hook_mlp_out", TorchIndex([None]))
                )
        else:
            idx_from = [
                (
                    layer_idx,
                    f"blocks.{layer_idx}.attn.hook_result",
                    TorchIndex([None, None, head_idx]),
                )
                for layer_idx, head_idx in IOI_CIRCUIT[conn.inp]
            ]

        if conn.out == "OUTPUT":
            idx_to = [(13, "blocks.11.hook_resid_post", TorchIndex([None]))]
            for mlp_layer_idx in range(12):
                idx_to.append(
                    (mlp_layer_idx, f"blocks.{mlp_layer_idx}.hook_mlp_in", TorchIndex([None]))
                )
        else:
            idx_to = [
                (
                    layer_idx,
                    f"blocks.{layer_idx}.hook_{letter}_input",
                    TorchIndex([None, None, head_idx]),
                )
                for layer_idx, head_idx in IOI_CIRCUIT[conn.out]
                for letter in conn.qkv
            ]

        for layer_from, layer_name_from, which_idx_from in idx_from:
            for layer_to, layer_name_to, which_idx_to in idx_to:
                if layer_to > layer_from:
                    corr.edges[layer_name_to][which_idx_to][layer_name_from][
                        which_idx_from
                    ].present = True

    ret = OrderedDict(
        {
            (t[0], t[1].hashable_tuple, t[2], t[3].hashable_tuple): e.present
            for t, e in corr.all_edges().items()
            if e.present
        }
    )
    return ret

GROUP_COLORS = {
    "name mover": "#d7f8ee",
    "backup name mover": "#e7f2da",
    "negative": "#fee7d5",
    "s2 inhibition": "#ececf5",
    "induction": "#fff6db",
    "duplicate token": "#fad6e9",
    "previous token": "#f9ecd7",
}
MLP_COLOR = "#f0f0f0"

def ioi_group_colorscheme():
    assert set(GROUP_COLORS.keys()) == set(IOI_CIRCUIT.keys())

    scheme = {
        "embed": "#cbd5e8",
        "<resid_post>": "#fff2ae",
    }

    for i in range(12):
        scheme[f"<m{i}>"] = MLP_COLOR

    for k, heads in IOI_CIRCUIT.items():
        for layer, head in heads:
            for qkv in ["", "_q", "_k", "_v"]:
                scheme[f"<a{layer}.{head}{qkv}>"] = GROUP_COLORS[k]

    for layer in range(12):
        scheme[f"<m{layer}>"] = "#f0f0f0"
    return scheme
