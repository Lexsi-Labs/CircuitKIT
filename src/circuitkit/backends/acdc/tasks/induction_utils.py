from functools import partial
from typing import Optional

import huggingface_hub
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from ..utils.task_utils import AllDataThings, shuffle_tensor

# The following metric functions are copied from the ACDC repository, acdc/acdc_utils.py
# (https://github.com/ArthurConmy/Automatic-Circuit-Discovery, MIT; see THIRD_PARTY_LICENSES.md)
# as they are required to construct the AllDataThings object for this task.
# Note that CircuitKit's own circuit discovery algorithms do not use these directly.


def kl_divergence(
    logits: torch.Tensor,
    base_model_logprobs: torch.Tensor,
    mask_repeat_candidates: Optional[torch.Tensor] = None,
    last_seq_element_only: bool = True,
) -> torch.Tensor:
    if last_seq_element_only:
        logits = logits[:, -1, :]
    logprobs = F.log_softmax(logits, dim=-1)
    kl_div = F.kl_div(logprobs, base_model_logprobs, log_target=True, reduction="none").sum(dim=-1)

    if mask_repeat_candidates is not None:
        answer = kl_div[mask_repeat_candidates]
    elif not last_seq_element_only:
        answer = kl_div.view(-1)
    else:
        answer = kl_div
    return answer.mean()


# --- End of copied metric functions ---


def get_validation_data(device=None, num_examples=None):
    validation_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="validation_data.pt"
    )
    validation_data = torch.load(validation_fname, map_location=device, weights_only=True)
    if num_examples is not None:
        validation_data = validation_data[:num_examples]
    return validation_data.long()


def get_mask_repeat_candidates(device=None, num_examples=None):
    mask_repeat_candidates_fname = huggingface_hub.hf_hub_download(
        repo_id="ArthurConmy/redwood_attn_2l", filename="mask_repeat_candidates.pkl"
    )
    mask_repeat_candidates = torch.load(mask_repeat_candidates_fname, map_location=device, weights_only=True)
    mask_repeat_candidates.requires_grad = False
    if num_examples is not None:
        mask_repeat_candidates = mask_repeat_candidates[:num_examples]
    return mask_repeat_candidates


def get_all_induction_things(
    model: HookedTransformer,
    device: torch.device,
    num_examples: int = 50,
    seq_len: int = 300,
    metric_name: str = "kl_div",
) -> AllDataThings:

    validation_data_orig = get_validation_data(device=device)
    mask_orig = get_mask_repeat_candidates(device=device)
    assert validation_data_orig.shape == mask_orig.shape

    assert seq_len <= validation_data_orig.shape[1] - 1

    # Split and truncate
    train_slice = slice(0, num_examples)
    test_slice = slice(num_examples, num_examples * 2)

    train_data = validation_data_orig[train_slice, :seq_len].contiguous()
    train_labels = validation_data_orig[train_slice, 1 : seq_len + 1].contiguous()
    train_mask = mask_orig[train_slice, :seq_len].contiguous()

    test_data = validation_data_orig[test_slice, :seq_len].contiguous()
    test_labels = validation_data_orig[test_slice, 1 : seq_len + 1].contiguous()
    test_mask = mask_orig[test_slice, :seq_len].contiguous()

    # Create patch data by shuffling the original data
    train_patch_data = shuffle_tensor(train_data, seed=42).contiguous()
    test_patch_data = shuffle_tensor(test_data, seed=43).contiguous()

    with torch.no_grad():
        base_train_logprobs = F.log_softmax(model(train_data), dim=-1).detach()
        base_test_logprobs = F.log_softmax(model(test_data), dim=-1).detach()

    validation_metric = partial(
        kl_divergence,
        base_model_logprobs=base_train_logprobs,
        mask_repeat_candidates=train_mask,
        last_seq_element_only=False,
    )

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            mask_repeat_candidates=test_mask,
            last_seq_element_only=False,
        )
    }

    return AllDataThings(
        validation_metric=validation_metric,
        validation_data=train_data,
        validation_labels=train_labels,
        validation_mask=train_mask,
        validation_patch_data=train_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=test_mask,
        test_patch_data=test_patch_data,
    )
