"""
TaskSpec Protocol Definition

Defines the interface that all task specifications must implement.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import pandas as pd
from torch.utils.data import DataLoader


def _load_finetuning_data_from_csv(
    cache_path: Path,
    tokenizer,
    n_examples: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Load (clean_texts, query_strings) from an EAP discovery CSV cache.

    Expects columns 'clean' (prompt text) and 'correct_idx' (answer token ID).
    Rows are shuffled deterministically before truncating to n_examples so
    finetuning sees a different ordering than discovery without re-generating.

    The 'clean' column already reflects the chat-template decision frozen by
    discovery: a CSV from a ``_tmpl``-tagged cache holds chat-templated text and
    a ``_raw`` one holds raw text. Either way the prompt is used verbatim — this
    loader never wraps (which would double-template a ``_tmpl`` row). Selecting
    the variant that matches the resolved finetuning mode is the caller's job
    via :func:`_find_task_cache`.

    Args:
        cache_path:  Path to the cached CSV file.
        tokenizer:   HuggingFace tokenizer for decoding the answer token.
        n_examples:  Maximum number of examples to return.
        seed:        Shuffle seed for reproducibility.

    Returns:
        clean_texts:   prompt + decoded answer token  (full sequence for LM loss).
        query_strings: prompt only                    (masked out in LM loss).

    Raises:
        FileNotFoundError: If the cache file does not exist.
        ValueError:        If the cache file is empty after loading.
    """
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Finetuning data cache not found: {cache_path}\n"
            f"Run discover_circuit on this task and model first to populate "
            f"the cache, then retry finetuning."
        )

    df = pd.read_csv(str(cache_path))
    if df.empty:
        raise ValueError(f"Cache file is empty: {cache_path}")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True).head(n_examples)

    clean_texts: List[str] = []
    query_strings: List[str] = []

    for _, row in df.iterrows():
        query = str(row["clean"])
        try:
            answer_token = tokenizer.decode(
                [int(row["correct_idx"])],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            continue
        clean_texts.append(query + answer_token)
        query_strings.append(query)

    return clean_texts, query_strings


def _find_task_cache(
    cache_dir: Path,
    prefix: str,
    model_name_safe: str,
    templated: Optional[bool] = None,
) -> Optional[Path]:
    """
    Locate a task CSV cache file for the given model using glob.

    Matches on task prefix and model name only, tolerating differences in
    the remaining filename (n_samples, seed, etc.) across runs. When multiple
    files match, returns the largest (assumed to have the most examples).

    Discovery now tags its cache filenames with ``_raw`` / ``_tmpl`` to record
    whether the prompts were chat-templated. When ``templated`` is given, only
    files carrying the matching tag are considered — so finetuning
    deterministically picks the variant whose prompt distribution matches the
    resolved chat-template mode rather than whichever file happens to be larger.
    ``templated=None`` keeps the legacy tag-agnostic behavior.

    Args:
        cache_dir:       Directory to search.
        prefix:          Task name prefix, e.g. 'ioi', 'sva'.
        model_name_safe: Model name with '/' replaced by '_'.
        templated:       Resolved chat-template boolean. ``True`` selects only
                         ``_tmpl``-tagged caches, ``False`` only ``_raw``-tagged
                         ones, ``None`` matches any (legacy behavior).

    Returns:
        Path to the best matching cache file, or None if not found.
    """
    if not cache_dir.exists():
        return None
    matches = list(cache_dir.glob(f"{prefix}_{model_name_safe}_*.csv"))
    if templated is not None:
        tag = "_tmpl" if templated else "_raw"
        tagged = [p for p in matches if p.stem.endswith(tag)]
        # Only narrow to the tagged variant when at least one tagged file
        # exists; otherwise fall back to the untagged matches so caches
        # written before tagging was introduced are still found.
        if tagged:
            matches = tagged
    return max(matches, key=lambda p: p.stat().st_size) if matches else None


class TaskSpec(Protocol):
    """
    Protocol defining the interface for task specifications.

    Each task must implement this interface to provide:
    - Configuration validation
    - Data loading and preprocessing
    - Task-specific metrics
    - Artifact metadata
    """

    name: str  # Canonical task name, e.g., "ioi"

    # Per-task chat-template policy: decides whether discovery prompts are
    # wrapped in the model's chat template. One of ``"auto"``, ``"on"``,
    # ``"off"`` (see ``circuitkit.tasks._chat.VALID_MODES``):
    #
    # * ``"auto"`` — wrap iff the model is instruction-tuned (its tokenizer
    #   ships a ``chat_template``). This is the default for custom user tasks
    #   and downstream-behavior tasks (MMLU, BoolQ, GSM8K, ...), which are
    #   presumed to be real behaviors the model is run with its chat template.
    # * ``"on"``  — always wrap prompts in the chat template.
    # * ``"off"`` — never wrap; use raw text. Correct for diagnostic
    #   minimal-pair tasks (IOI, greater-than, SVA, ...).
    #
    # The mode is resolved against a concrete model into a single boolean on
    # every call; it is *not* persisted into artifact metadata, so downstream
    # stages must be handed the same mode (and model type) to stay consistent.
    # Custom tasks default to ``"auto"``.
    chat_template_mode: str  # one of "auto" | "on" | "off"; "auto" is the custom-task default

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """
        Validate task-specific discovery configuration.

        Args:
            discovery_cfg: Discovery configuration dictionary

        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        ...

    def build_dataloader(
        self,
        model,  # HookedTransformer - REQUIRED, no None allowed
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> DataLoader:
        """
        Build and return a DataLoader for the task.

        Args:
            model: HookedTransformer model instance (REQUIRED, no None allowed)
            discovery_cfg: Discovery configuration
            device: Target device for data

        Returns:
            DataLoader configured for the task

        Raises:
            ValueError: If model is None
        """
        ...

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Return (clean_texts, query_strings) for causal LM finetuning.

        Uses only a HuggingFace tokenizer — no HookedTransformer required.
        Implementations should load from the discovery CSV cache when available,
        regenerating only for tasks where tokenizer-only generation is possible
        (e.g. MMLU which loads directly from HuggingFace).

        Args:
            tokenizer:      HuggingFace tokenizer matching the finetuning model.
            model_name:     Full HuggingFace model name (e.g. 'gpt2',
                            'meta-llama/Meta-Llama-3-8B'). Used to locate the
                            discovery cache written by discover_circuit.
            n_examples:     Maximum number of examples to return.
            discovery_cfg:  Optional discovery configuration dict forwarded
                            from the finetuning pipeline. Used for tasks that
                            require config-driven parameters (e.g. MMLU subjects,
                            custom cache_dir).
            seed:           Random seed for shuffling / sampling.

        Returns:
            clean_texts:   Full text strings (prompt + answer token) — the LM
                           training target sequence.
            query_strings: Prompt-only strings (same length as clean_texts) —
                           used to compute query_length so that only answer
                           token(s) contribute to the LM loss.
        """
        ...

    def metric_fn(self) -> Callable:
        """
        Return the metric function used for circuit discovery.

        Returns:
            Callable metric function for EAP/EAP-IG scoring
        """
        ...

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metadata for saved artifacts.

        Args:
            discovery_cfg: Discovery configuration

        Returns:
            Dictionary containing task-specific metadata
        """
        ...
