from functools import partial

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from ..utils.task_utils import AllDataThings
from ..utils.tensor_ops import batch_avg_answer_diff, multibatch_kl_div
from . import docstring_prompts


def get_all_docstring_things(
    model: HookedTransformer,
    device: torch.device,
    num_examples: int = 50,
    metric_name: str = "kl_div",
) -> AllDataThings:

    raw_prompts = [
        docstring_prompts.docstring_prompt_gen("rest", n_args=4, seed=i)
        for i in range(num_examples * 2)
    ]
    batched_prompts = docstring_prompts.BatchedPrompts(prompts=raw_prompts, model=model)

    clean_tokens = batched_prompts.clean_tokens
    corrupted_tokens = batched_prompts.corrupt_tokens["random_doc"]

    answers = batched_prompts.correct_tokens
    wrong_answers = batched_prompts.wrong_tokens

    train_slice = slice(0, num_examples)
    test_slice = slice(num_examples, num_examples * 2)

    train_data = clean_tokens[train_slice]
    test_data = clean_tokens[test_slice]
    train_answers = answers[train_slice]
    test_answers = answers[test_slice]
    train_wrong_answers = wrong_answers[train_slice]
    test_wrong_answers = wrong_answers[test_slice]

    train_patch_data = corrupted_tokens[train_slice]
    test_patch_data = corrupted_tokens[test_slice]

    with torch.no_grad():
        base_train_logprobs = F.log_softmax(model(train_data)[:, -1], dim=-1).detach()
        base_test_logprobs = F.log_softmax(model(test_data)[:, -1], dim=-1).detach()

    def kl_div_metric(logits: torch.Tensor, base_model_logprobs=base_train_logprobs):
        logprobs = F.log_softmax(logits[:, -1], dim=-1)
        return multibatch_kl_div(logprobs, base_model_logprobs)

    class DummyBatch:
        def __init__(self, answers, wrong_answers):
            self.answers = answers
            self.wrong_answers = wrong_answers

    train_dummy_batch = DummyBatch(train_answers, train_wrong_answers)
    test_dummy_batch = DummyBatch(test_answers, test_wrong_answers)

    def docstring_metric(logits: torch.Tensor, batch=train_dummy_batch):
        return -batch_avg_answer_diff(logits[:, -1, :], batch)

    if metric_name == "kl_div":
        validation_metric = kl_div_metric
    elif metric_name == "docstring_metric":
        validation_metric = docstring_metric
    else:
        raise ValueError(f"Unknown metric {metric_name}")

    test_metrics = {
        "kl_div": partial(kl_div_metric, base_model_logprobs=base_test_logprobs),
        "docstring_metric": partial(docstring_metric, batch=test_dummy_batch),
    }

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
