from functools import partial

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from ..utils.task_utils import AllDataThings
from ..utils.tensor_ops import batch_avg_answer_diff, multibatch_kl_div
from .ioi_dataset import IOIDataset


def get_all_ioi_things(
    model: HookedTransformer,
    device: torch.device,
    metric_name: str = "kl_div",
    num_examples: int = 128,
) -> AllDataThings:
    """
    This function is adapted from the ACDC repository (see THIRD_PARTY_LICENSES.md)
    to work with a pre-existing model.
    It generates the data for the IOI task and returns it in an AllDataThings object.
    """
    # IOI dataset
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=num_examples * 2,  # We generate more data to have held-out test set
        tokenizer=model.tokenizer,
        seed=0,
    )

    # Corrupted dataset by flipping the second name to a random one
    abc_dataset = ioi_dataset.gen_flipped_prompts("S", seed=1)

    clean_tokens = ioi_dataset.toks.long().to(device)
    corrupted_tokens = abc_dataset.toks.long().to(device)

    answers = torch.tensor(ioi_dataset.io_tokenIDs).long().to(device).unsqueeze(1)
    wrong_answers = torch.tensor(ioi_dataset.s_tokenIDs).long().to(device).unsqueeze(1)

    # Split into train and test
    train_slice = slice(0, num_examples)
    test_slice = slice(num_examples, num_examples * 2)

    # Clean data
    train_data = clean_tokens[train_slice]
    test_data = clean_tokens[test_slice]
    train_answers = answers[train_slice]
    test_answers = answers[test_slice]
    train_wrong_answers = wrong_answers[train_slice]
    test_wrong_answers = wrong_answers[test_slice]

    # Corrupted data
    train_patch_data = corrupted_tokens[train_slice]
    test_patch_data = corrupted_tokens[test_slice]

    with torch.no_grad():
        base_model_logits = model(train_data)
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)

    def kl_div_metric(logits: torch.Tensor, base_model_logprobs=base_model_logprobs):
        logprobs = F.log_softmax(logits, dim=-1)
        return multibatch_kl_div(logprobs, base_model_logprobs)

    def logit_diff_metric(logits: torch.Tensor, **kwargs):
        return -batch_avg_answer_diff(logits, kwargs["batch"]).item()

    if metric_name == "kl_div":
        validation_metric = kl_div_metric
    elif metric_name == "logit_diff":

        class DummyBatch:
            def __init__(self, answers, wrong_answers):
                self.answers = answers
                self.wrong_answers = wrong_answers

        dummy_batch = DummyBatch(train_answers, train_wrong_answers)
        validation_metric = partial(logit_diff_metric, batch=dummy_batch)
    else:
        raise ValueError(f"Unknown metric {metric_name}")

    test_metrics = {"kl_div": kl_div_metric, "logit_diff": logit_diff_metric}

    return AllDataThings(
        validation_metric=validation_metric,
        validation_data=train_data,
        validation_labels=train_answers,
        validation_wrong_labels=train_wrong_answers,
        validation_mask=None,
        validation_patch_data=train_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_answers,
        test_wrong_labels=test_wrong_answers,
        test_mask=None,
        test_patch_data=test_patch_data,
    )
