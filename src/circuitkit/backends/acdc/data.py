# Adapted from the ACDC repository (Automatic Circuit DisCovery), acdc/data.py:
# https://github.com/ArthurConmy/Automatic-Circuit-Discovery
# (c) 2023 Arthur Conmy, Adria Garriga-Alonso, MIT License. See THIRD_PARTY_LICENSES.md.

from typing import Any, List, Tuple

import torch as t
from torch.utils.data import DataLoader, Dataset
from transformer_lens import HookedTransformer

from .tasks import docstring_utils, induction_utils, ioi_utils
from .types import BatchKey, PromptPair, PromptPairBatch  # noqa: F401 - BatchKey re-exported
from .utils.task_utils import AllDataThings


# <--- CHANGE START: This function factory creates a collate_fn that knows the target device --->
import logging

logger = logging.getLogger(__name__)

def collate_fn_factory(device: t.device):
    """Creates a collate function that moves a batch of data to the specified device."""

    def collate_fn(batch: List[PromptPair]) -> PromptPairBatch:
        clean = t.stack([p.clean for p in batch]).to(device)
        corrupt = t.stack([p.corrupt for p in batch]).to(device)
        answers = t.stack([p.answers for p in batch]).to(device)
        wrong_answers = t.stack([p.wrong_answers for p in batch]).to(device)
        key = hash((str(clean.tolist()), str(corrupt.tolist())))
        return PromptPairBatch(key, clean, corrupt, answers, wrong_answers)

    return collate_fn


# <--- CHANGE END --->


class PromptDataset(Dataset):
    def __init__(
        self,
        clean_prompts: t.Tensor,
        corrupt_prompts: t.Tensor,
        answers: t.Tensor,
        wrong_answers: t.Tensor,
    ):
        assert len(clean_prompts) == len(corrupt_prompts) == len(answers) == len(wrong_answers)
        # Data is now expected to be on the CPU
        self.clean_prompts = clean_prompts
        self.corrupt_prompts = corrupt_prompts
        self.answers = answers
        self.wrong_answers = wrong_answers

    def __len__(self) -> int:
        return len(self.clean_prompts)

    def __getitem__(self, idx: int) -> PromptPair:
        return PromptPair(
            self.clean_prompts[idx],
            self.corrupt_prompts[idx],
            self.answers[idx],
            self.wrong_answers[idx],
        )


class PromptDataLoader(DataLoader[PromptPairBatch]):
    def __init__(self, prompt_dataset: Any, device: t.device, **kwargs: Any):
        collate_fn = collate_fn_factory(device)
        super().__init__(prompt_dataset, **kwargs, collate_fn=collate_fn)


def load_task_data(
    task_name: str,
    model: HookedTransformer,
    device: t.device,
    batch_size: int = 16,
    train_test_size: Tuple[int, int] = (128, 0),
    num_examples: int = None,
    qa_max_truncate_tokens: int = 16,
    qa_task_type: str = "open_qa",
    **_ignored,
) -> Tuple[PromptDataLoader, PromptDataLoader]:
    """
    Loads data for a specified task using the new modular system.
    This version loads data to CPU and transfers to the target device by batch.

    Accepts either ``train_test_size`` (an explicit (n_train, n_test) tuple) or
    ``num_examples`` (the discovery-config style scalar, used for the train
    split with no test split). Unknown extra kwargs are ignored so that the
    full ``discovery.data_params`` dict can be forwarded verbatim by api.py.
    """
    if num_examples is not None:
        # discovery.data_params style: a single scalar => all train, no test.
        n_train, n_test = int(num_examples), 0
    else:
        n_train, n_test = train_test_size
    num_examples = n_train + n_test

    cpu_device = t.device("cpu")

    if task_name == "ioi":
        things: AllDataThings = ioi_utils.get_all_ioi_things(
            model, cpu_device, num_examples=num_examples
        )
    elif task_name == "induction":
        things: AllDataThings = induction_utils.get_all_induction_things(
            model, cpu_device, num_examples=num_examples, seq_len=300
        )
    elif task_name == "docstring":
        things: AllDataThings = docstring_utils.get_all_docstring_things(
            model, cpu_device, num_examples=num_examples
        )
    elif task_name == "qa":
        raise ValueError(
            "QA task functionality has been removed. Please use other available tasks."
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")

    # Process train data
    train_clean = things.validation_data
    train_corrupt = things.validation_patch_data
    train_answers = things.validation_labels
    train_wrong_answers = things.validation_wrong_labels

    if train_answers is None:
        logger.warning(f"Warning: Task '{task_name}' does not provide correct answers. Using dummy labels.")
        train_answers = t.zeros(len(train_clean), 1, dtype=t.long, device=cpu_device)
    if train_wrong_answers is None:
        logger.info(
            f"Warning: Task '{task_name}' does not provide wrong answers. EAP may not be meaningful. Using dummy labels."
        )
        train_wrong_answers = t.zeros(len(train_clean), 1, dtype=t.long, device=cpu_device)

    # Ensure answers have a second dimension
    if train_answers.ndim == 1:
        train_answers = train_answers.unsqueeze(1)
    if train_wrong_answers.ndim == 1:
        train_wrong_answers = train_wrong_answers.unsqueeze(1)

    train_dataset = PromptDataset(
        train_clean,
        train_corrupt,
        train_answers,
        train_wrong_answers,
    )
    # <--- CHANGE START: Pass the target GPU device to our new PromptDataLoader --->
    train_loader = PromptDataLoader(
        train_dataset, device=device, batch_size=batch_size, shuffle=False
    )

    # Process test data (all tensors are on CPU here)
    test_clean = things.test_data
    test_corrupt = things.test_patch_data
    test_answers = things.test_labels
    test_wrong_answers = things.test_wrong_labels

    if len(test_clean) > 0:
        if test_answers is None:
            test_answers = t.zeros(len(test_clean), 1, dtype=t.long, device=cpu_device)
        if test_wrong_answers is None:
            test_wrong_answers = t.zeros(len(test_clean), 1, dtype=t.long, device=cpu_device)
        if test_answers.ndim == 1:
            test_answers = test_answers.unsqueeze(1)
        if test_wrong_answers.ndim == 1:
            test_wrong_answers = test_wrong_answers.unsqueeze(1)

        test_dataset = PromptDataset(test_clean, test_corrupt, test_answers, test_wrong_answers)
        # <--- CHANGE START: Pass the target GPU device to our new PromptDataLoader --->
        test_loader = PromptDataLoader(
            test_dataset, device=device, batch_size=batch_size, shuffle=False
        )
    else:
        # Create a loader with an empty dataset if there is no test data
        class EmptyDataset(Dataset):
            def __len__(self):
                return 0

            def __getitem__(self, idx):
                raise IndexError

        test_loader = PromptDataLoader(EmptyDataset(), device=device, batch_size=batch_size)

    return train_loader, test_loader
