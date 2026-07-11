"""
Taken from the ARENA 3.0 notebook on the IOI task, written by Callum McDougall.
We are not aware of any formal attribution requirements for this code, including any licenses.
Please contact us if you understand otherwise.
"""

from functools import partial
from typing import Dict, List, Optional, Tuple

import einops
import torch
import transformer_lens
from jaxtyping import Bool, Float
from pyfunctions.ioi_dataset import IOIDataset
from pyfunctions.wrappers import Node
from torch import Tensor
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

CIRCUIT = {
    "name mover": [(9, 9), (10, 0), (9, 6)],
    "backup name mover": [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)],
    "negative name mover": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [(0, 1), (0, 10), (3, 0)],
    "previous token": [(2, 2), (4, 11)],
}

SEQ_POS_TO_KEEP = {
    "name mover": "end",
    "backup name mover": "end",
    "negative name mover": "end",
    "s2 inhibition": "end",
    "induction": "S2",
    "duplicate token": "S2",
    "previous token": "S+1",
}


def get_heads_and_posns_to_keep(
    means_dataset: IOIDataset,
    model: HookedTransformer,
    circuit: Dict[str, List[Tuple[int, int]]],
    seq_pos_to_keep: Dict[str, str],
) -> Dict[int, Bool[Tensor, "batch seq head"]]:  # noqa: F722 - jaxtyping shape annotation
    """
    Returns a dictionary mapping layers to a boolean mask giving the indices of the
    z output which *shouldn't* be mean-ablated.

    The output of this function will be used for the hook function that does ablation.
    """
    heads_and_posns_to_keep = {}
    batch, seq, n_heads = len(means_dataset), means_dataset.max_len, model.cfg.n_heads

    for layer in range(model.cfg.n_layers):

        mask = torch.zeros(size=(batch, seq, n_heads))

        for head_type, head_list in circuit.items():
            seq_pos = seq_pos_to_keep[head_type]
            indices = means_dataset.word_idx[seq_pos]
            for layer_idx, head_idx in head_list:
                if layer_idx == layer:
                    mask[:, indices, head_idx] = 1

        heads_and_posns_to_keep[layer] = mask.bool()

    return heads_and_posns_to_keep


"""
We (the CD-T guys) wrote this one.
It's necessary because the circuits we find don't always have
categorizable heads and we're not always interested in positions
that are labeled by the IOI dataset (as e.g, "IO", "S+1", etc.).
The downside of this is that it can only be run on prompts adhering to one template,
since the indices are explicit.
"""


def get_heads_and_posns_to_keep_explicit(
    batch: int,
    seq: int,
    model: HookedTransformer,
    circuit: List[Node],
) -> Dict[int, Bool[Tensor, "batch seq head"]]:  # noqa: F722 - jaxtyping shape annotation
    """
    Returns a dictionary mapping layers to a boolean mask giving the indices of the
    z output which *shouldn't* be mean-ablated.

    The output of this function will be used for the hook function that does ablation.
    """
    heads_and_posns_to_keep = {}
    n_heads = model.cfg.n_heads

    for layer in range(model.cfg.n_layers):

        mask = torch.zeros(size=(batch, seq, n_heads))

        for node in circuit:
            if node.layer_idx == layer:
                mask[:, node.sequence_idx, node.attn_head_idx] = 1
        heads_and_posns_to_keep[layer] = mask.bool()

    return heads_and_posns_to_keep


def hook_fn_mask_z(
    z: Float[Tensor, "batch seq head d_head"],  # noqa: F722 - jaxtyping shape annotation
    hook: HookPoint,
    heads_and_posns_to_keep: Dict[
        int, Bool[Tensor, "batch seq head"]  # noqa: F722 - jaxtyping shape annotation
    ],
    means: Float[Tensor, "layer batch seq head d_head"],  # noqa: F722 - jaxtyping shape annotation
) -> Float[Tensor, "batch seq head d_head"]:  # noqa: F722 - jaxtyping shape annotation
    """
    Hook function which masks the z output of a transformer head.

    heads_and_posns_to_keep
        Dict created with the get_heads_and_posns_to_keep function. This tells
        us where to mask.

    means
        Tensor of mean z values of the means_dataset over each group of prompts
        with the same template. This tells us what values to mask with.
    """
    # Get the mask for this layer, and add d_head=1 dimension so it broadcasts correctly
    mask_for_this_layer = heads_and_posns_to_keep[hook.layer()].unsqueeze(-1).to(z.device)

    # Set z values to the mean
    z = torch.where(mask_for_this_layer, z, means[hook.layer()])

    return z


def compute_means_by_template(
    means_dataset: IOIDataset, model: HookedTransformer
) -> Float[Tensor, "layer batch seq head_idx d_head"]:  # noqa: F722 - jaxtyping shape annotation
    """
    Returns the mean of each head's output over the means dataset. This mean is
    computed separately for each group of prompts with the same template (these
    are given by means_dataset.groups).
    """
    # Cache the outputs of every head
    _, means_cache = model.run_with_cache(
        means_dataset.toks.long(),
        return_type=None,
        names_filter=lambda name: name.endswith("z"),
    )
    # Create tensor to store means
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    batch, seq_len = len(means_dataset), means_dataset.max_len
    means = torch.zeros(size=(n_layers, batch, seq_len, n_heads, d_head), device=model.cfg.device)

    # Get set of different templates for this data
    for layer in range(model.cfg.n_layers):
        _act_name = transformer_lens.utils.get_act_name("z", layer)
        z_for_this_layer: Float[Tensor, "batch seq head d_head"] = means_cache[  # noqa: F722
            _act_name
        ]
        for template_group in means_dataset.groups:
            z_for_this_template = z_for_this_layer[template_group]
            z_means_for_this_template = einops.reduce(
                z_for_this_template, "batch seq head d_head -> seq head d_head", "mean"
            )
            means[layer, template_group] = z_means_for_this_template

    return means


def add_mean_ablation_hook(
    model: HookedTransformer,
    means_dataset: Optional[IOIDataset] = None,
    patch_values: Optional[
        Float[Tensor, "layer batch seq head_idx d_head"]  # noqa: F722 - jaxtyping shape annotation
    ] = None,
    circuit=CIRCUIT,  # Union(Dict[str, List[Tuple[int, int]]], List[Node]) = CIRCUIT,
    seq_pos_to_keep: Dict[str, str] = SEQ_POS_TO_KEEP,
    is_permanent: bool = True,
) -> HookedTransformer:
    """
    Adds a permanent hook to the model, which ablates according to the circuit and
    seq_pos_to_keep dictionaries.

    In other words, when the model is run on ioi_dataset, every head's output will
    be replaced with the mean over means_dataset for sequences with the same template,
    except for a subset of heads and sequence positions as specified by the circuit
    and seq_pos_to_keep dicts.
    """

    model.reset_hooks(including_permanent=True)

    # Compute the mean of each head's output on the ABC dataset, grouped by template
    if means_dataset is not None:
        means = compute_means_by_template(means_dataset, model)
        batch_size, seq_len = len(means_dataset), means_dataset.max_len
    else:
        means = (
            patch_values  # just patch in these values, shape is "layer batch seq head_idx d_head"
        )
        # or else "layer seq head_idx d_head"
        if patch_values.dim() == 5:
            batch_size = patch_values.size()[1]
            seq_len = patch_values.size()[2]
        else:
            batch_size = 1
            seq_len = patch_values.size()[1]

    # Convert this into a boolean map
    if isinstance(circuit, dict):
        heads_and_posns_to_keep = get_heads_and_posns_to_keep(
            means, model, circuit, seq_pos_to_keep
        )
    else:
        heads_and_posns_to_keep = get_heads_and_posns_to_keep_explicit(
            batch_size, seq_len, model, circuit
        )

    # Get a hook function which will patch in the mean z values for each head, at
    # all positions which aren't important for the circuit
    hook_fn = partial(hook_fn_mask_z, heads_and_posns_to_keep=heads_and_posns_to_keep, means=means)

    # Apply hook
    model.add_hook(lambda name: name.endswith("z"), hook_fn, is_permanent=is_permanent)

    return model


def logits_to_ave_logit_diff_2(
    logits: Float[Tensor, "batch seq d_vocab"],  # noqa: F722 - jaxtyping shape annotation
    ioi_dataset,
    per_prompt=False,
):
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[  # noqa: F821 - jaxtyping shape annotation
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs
    ]
    s_logits: Float[Tensor, "batch"] = logits[  # noqa: F821 - jaxtyping shape annotation
        range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs
    ]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()
