"""CSV-backed dataset for EAP-style circuit discovery.

Moved out of ``circuitkit.api`` (a thin front-door facade should not define a
``torch.utils.data.Dataset``). This is the canonical home; ``circuitkit.api``
re-exports :class:`EAPDiscoveryDataset` for backward compatibility.
"""

from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from ..backends.eap.eap_utils import collate_EAP as collate_eap_data


class EAPDiscoveryDataset(Dataset):
    """
    CSV-backed dataset for EAP-style circuit discovery.

    Expects a CSV with columns: 'clean', 'corrupted', 'correct_idx', 'incorrect_idx'.
    Supports both two-token tasks (e.g. IOI) and four-token tasks (e.g. MMLU).
    """

    def __init__(self, filepath, templated: bool = False):
        """
        Args:
            filepath (str): Path to a CSV file with columns
                'clean', 'corrupted', 'correct_idx', 'incorrect_idx'.
            templated (bool): True iff the CSV's 'clean'/'corrupted' strings
                were written with the model's chat template already applied
                (which renders its own BOS into the text). The flag is carried
                through to the DataLoader so the EAP backend tokenizes with
                ``prepend_bos = not templated`` and avoids a double BOS.
                Defaults to False (raw text) — byte-identical to the legacy
                behavior for every non-templated task and base-model run.
        """
        self.df = pd.read_csv(filepath)
        self.templated = templated

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Returns:
            Tuple[str, str, List[int]]: (clean_text, corrupted_text, labels) where
                labels is [correct, incorrect] for two-token tasks, or
                [correct, inc1, inc2, inc3] for four-token tasks (MMLU).
        """
        row = self.df.iloc[index]
        # Handle both IOI format (2 tokens) and MMLU format (4 tokens)
        correct_idx = row["correct_idx"]
        incorrect_idx = row["incorrect_idx"]

        # If incorrect_idx is a string representation of a list, parse it
        if isinstance(incorrect_idx, str) and incorrect_idx.startswith("["):
            import ast

            incorrect_idx = ast.literal_eval(incorrect_idx)

        # Ensure we have the right format for the metric function
        if isinstance(incorrect_idx, list) and len(incorrect_idx) == 3:
            # MMLU format: [correct, incorrect1, incorrect2, incorrect3]
            labels = [correct_idx] + incorrect_idx
        else:
            # IOI format: [correct, incorrect]
            labels = [correct_idx, incorrect_idx]

        return row["clean"], row["corrupted"], labels

    def to_dataloader(
        self,
        batch_size: int,
        pair_padding_side: str = "right",
        templated: Optional[bool] = None,
    ):
        """
        Wrap this dataset in a DataLoader with the EAP collate function.

        Args:
            batch_size (int): Number of examples per batch.
            pair_padding_side (str): Padding side for clean/corrupted alignment
                ('left' or 'right'). Defaults to 'right'.
            templated (Optional[bool]): True iff the CSV's text already carries
                the model's chat template (so the EAP backend must tokenize
                with ``prepend_bos=False`` to avoid a double BOS). Defaults to
                None, which falls back to the dataset's own ``templated`` flag
                (itself False unless explicitly set). When False the resulting
                DataLoader is byte-identical to the legacy behavior.

        Returns:
            DataLoader: Ready-to-use DataLoader with ``pair_padding_side`` and
                ``templated`` attributes set.
        """
        dl = DataLoader(self, batch_size=batch_size, collate_fn=collate_eap_data)
        dl.pair_padding_side = pair_padding_side
        dl.templated = self.templated if templated is None else templated
        return dl
