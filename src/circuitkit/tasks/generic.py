"""
Generic Task Specification (GenericTaskSpec)

Implements TaskSpec for arbitrary datasets (CSV, JSONL, HuggingFace datasets)
with automatic schema mapping and corruption strategy application.

This enables "bring your own dataset" capabilities, allowing users to:
- Load data from multiple sources (CSV, JSONL, HuggingFace)
- Map arbitrary column names to required fields (prompt, answer)
- Apply corruption strategies via CorruptionPipeline
- Generate EAP-compatible dataloaders
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch as t
from torch.utils.data import DataLoader, Dataset

from circuitkit.tasks._chat import (
    VALID_MODES,
    resolve_chat_template,
    resolve_chat_template_from_tokenizer,
    wrap_prompt,
    wrap_prompt_with_tokenizer,
)

try:
    from jinja2 import Template, TemplateError
except ImportError:
    Template = None
    TemplateError = None

logger = logging.getLogger(__name__)


class GenericDataLoader(Dataset):
    """
    In-memory dataset for generic tasks with corruption support.

    Holds extended examples with (clean_text, corrupted_text, answer_idx, answer_span,
    context, choices, metadata) tuples and can be wrapped in a DataLoader with EAP's
    collate function.

    Supports:
    - Single-token and multi-token answers via optional answer_spans
    - Context (for QA, reading comprehension)
    - Multiple choices (MCQ, ranking tasks)
    - Metadata (difficulty, category, ID, etc.)
    """

    def __init__(self, examples: List[Dict[str, Any]]):
        """
        Args:
            examples: List of dicts with keys:
                - 'clean': str, clean prompt text
                - 'corrupted': str, corrupted prompt text
                - 'correct_idx': int, token ID of correct answer (single token)
                - 'incorrect_idx': int or list, token ID(s) of incorrect answer(s)
                - 'answer_start': int, optional start position of answer span in token sequence
                - 'answer_end': int, optional end position of answer span (exclusive)
                - 'context': str, optional context (passage, background info)
                - 'choices': list, optional MCQ choices
                - 'correct_choice_idx': int, optional index of correct choice
                - 'valid_answers': list, optional multiple valid answers (tokens)
                - 'metadata': dict, optional metadata (difficulty, category, id, etc.)
        """
        self.examples = examples
        # Track if any example has answer spans (multi-token support)
        self.has_answer_spans = any("answer_start" in ex or "answer_end" in ex for ex in examples)
        # Track if any example has context
        self.has_context = any("context" in ex for ex in examples)
        # Track if any example has metadata
        self.has_metadata = any("metadata" in ex for ex in examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        clean = ex["clean"]
        corrupted = ex["corrupted"]

        correct_idx = ex["correct_idx"]
        incorrect_idx = ex.get("incorrect_idx", 0)

        # Convert to list format [correct, ...incorrect] for EAP
        if isinstance(incorrect_idx, list):
            labels = [correct_idx] + incorrect_idx
        else:
            labels = [correct_idx, incorrect_idx]

        # Extract answer span if present (for multi-token answers)
        answer_span = None
        if "answer_start" in ex and "answer_end" in ex:
            answer_span = (ex["answer_start"], ex["answer_end"])

        # Build extended output dict
        result = {
            "clean": clean,
            "corrupted": corrupted,
            "labels": labels,
            "answer_span": answer_span,
        }

        # Add optional fields
        if "context" in ex:
            result["context"] = ex["context"]
        if "choices" in ex:
            result["choices"] = ex["choices"]
        if "correct_choice_idx" in ex:
            result["correct_choice_idx"] = ex["correct_choice_idx"]
        if "valid_answers" in ex:
            result["valid_answers"] = ex["valid_answers"]
        if "metadata" in ex:
            result["metadata"] = ex["metadata"]

        return result


class GenericTaskSpec:
    """
    Generic task specification that works with arbitrary datasets.

    Supports CSV, JSONL, and HuggingFace Datasets via schema-based column mapping.
    Applies corruption strategies to generate EAP-compatible training data.

    Example usage:
        task = GenericTaskSpec.from_csv(
            path="questions.csv",
            schema={"prompt": "question", "answer": "answer_token"},
            corruption_strategy=corruption_pipeline,
            metric_fn=accuracy_metric
        )

        task.validate_discovery_config(config)
        loader = task.build_dataloader(model, config, device)
    """

    pair_padding_side = "right"

    #: Chat-template policy for custom tasks. Defaults to ``"auto"`` because a
    #: bring-your-own dataset is presumed to be a real downstream behavior:
    #: wrap prompts in the model's chat template iff the model is
    #: instruction-tuned. Overridable per-task via the ``chat_template_mode``
    #: constructor argument and per-run via ``discovery_cfg``.
    chat_template_mode: str = "auto"

    def __init__(
        self,
        name: str,
        source: Dict[str, Any],
        schema: Dict[str, str],
        corruption_strategy: Optional[Any] = None,
        metric_fn: Optional[Callable] = None,
        prompt_template: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
        chat_template_mode: str = "auto",
    ):
        """
        Args:
            name: Task name (e.g., "my_csv_task")
            source: Data source config with keys:
                - type: "csv" | "jsonl" | "hf"
                - path_or_id: file path or HF dataset ID
                - split: dataset split (for HF only, default "test")
            schema: Column name mapping (extended format)
            corruption_strategy: CorruptionPipeline instance for generating corruptions.
            metric_fn: Metric function. If None, auto-generated from task_type.
            prompt_template: Optional Jinja2 template for rendering prompts.
            metadata_filter: Optional filter dict for metadata.
            task_type: Override auto-detection. One of: "mcq", "classification",
                       "qa", "ranking", "open", "generation", "language_modeling".
            chat_template_mode: Chat-template policy ("auto" | "on" | "off").
                       Defaults to "auto" — custom tasks are presumed real
                       downstream behaviors and are wrapped in the model's
                       chat template iff the model is instruction-tuned.

        Raises:
            ValueError: If chat_template_mode is not a recognized value.
        """
        if chat_template_mode not in VALID_MODES:
            raise ValueError(
                f"chat_template_mode must be one of {VALID_MODES}, got {chat_template_mode!r}"
            )
        self.chat_template_mode = chat_template_mode
        self.name = name
        self.source = source
        self.schema = schema
        self.corruption_strategy = corruption_strategy
        self._metric_fn = metric_fn
        self.prompt_template = prompt_template
        self.metadata_filter = metadata_filter or {}

        # Track whether the caller explicitly requested a task type. An
        # answer-less schema is only valid when a generation-style type was
        # explicitly requested; otherwise it is a misconfiguration that the
        # schema validation must surface rather than silently treat as "open".
        self._task_type_explicit = task_type is not None
        self.task_type = task_type or self._detect_task_type()
        self._validate_schema()

        # Auto-generate metric if not provided
        if self._metric_fn is None:
            self._metric_fn = self._auto_metric()

        # Compile template if provided
        self.template = None
        if prompt_template and Template is not None:
            try:
                self.template = Template(prompt_template)
            except TemplateError as e:
                raise ValueError(
                    f"Invalid Jinja2 'prompt_template': {e}. "
                    f"Fix the template syntax in the task's prompt_template."
                )

    def _validate_schema(self) -> None:
        """
        Validate schema configuration.

        Raises:
            ValueError: If schema is invalid
        """
        if "prompt" not in self.schema:
            raise ValueError(
                "Task schema is missing the required key 'prompt'. Add a "
                "'prompt' entry to the schema mapping it to the dataset column "
                "that holds the input text."
            )

        # Generation-style tasks legitimately have no answer column, but only
        # when the task type was explicitly requested. An answer-less schema
        # that merely auto-detected as "open" is a misconfiguration.
        answer_exempt = self._task_type_explicit and self.task_type in (
            "generation",
            "language_modeling",
            "open",
        )
        if not answer_exempt:
            if not any(k in self.schema for k in ("answer", "answers", "answer_tokens", "choices")):
                raise ValueError(
                    "Task schema must include at least one answer key: one of "
                    "'answer', 'answers', 'answer_tokens', or 'choices'. Add "
                    "one mapping it to the dataset's answer/label column, or "
                    "pass task_type='generation' for an answer-less task."
                )

    def _detect_task_type(self) -> str:
        """
        Auto-detect task type from schema.

        Returns:
            One of: "classification", "qa", "ranking", "mcq", "open"
        """
        schema = self.schema

        if "answers" in schema or ("answer" in schema and "answers" in schema):
            return "ranking"
        if "context" in schema:
            return "qa"
        if "choices" in schema or "correct_choice_idx" in schema:
            return "mcq"
        if "answer" in schema:
            return "classification"
        return "open"

    def _auto_metric(self) -> Callable:
        """
        Auto-generate metric function from task_type.

        Returns:
            Callable metric function suitable for circuit discovery.
        """
        import torch as t
        import torch.nn.functional as F

        if self.task_type in ("mcq", "classification", "qa"):

            def logit_diff(logits, clean_logits, input_length, labels, loss=True, mean=True):
                """Logit difference between correct and incorrect tokens."""
                batch = logits.size(0)
                idx = t.arange(batch, device=logits.device)
                last = logits[idx, input_length - 1]
                correct = labels[:, 0:1].to(logits.device)
                incorrect = labels[:, 1:2].to(logits.device) if labels.size(1) >= 2 else None
                c = t.gather(last, -1, correct).squeeze(-1)
                if incorrect is not None:
                    i = t.gather(last, -1, incorrect).squeeze(-1)
                    result = c - i
                else:
                    result = c
                if loss:
                    result = -result
                if mean:
                    result = result.mean()
                return result

            return logit_diff

        elif self.task_type in ("generation", "language_modeling"):

            def nll_metric(logits, clean_logits, input_length, labels, loss=True, mean=True):
                """Negative log-likelihood of the correct token at the last position."""
                batch = logits.size(0)
                idx = t.arange(batch, device=logits.device)
                last = logits[idx, input_length - 1]
                target = labels[:, 0].to(logits.device)
                nll = F.cross_entropy(last, target, reduction="none")
                if mean:
                    nll = nll.mean()
                return nll if loss else -nll

            return nll_metric

        else:

            def fallback_metric(logits, clean_logits, input_length, labels, loss=True, mean=True):
                batch = logits.size(0)
                idx = t.arange(batch, device=logits.device)
                last = logits[idx, input_length - 1]
                target = labels[:, 0].to(logits.device)
                nll = F.cross_entropy(last, target, reduction="none")
                if mean:
                    nll = nll.mean()
                return nll if loss else -nll

            return fallback_metric

    # ========== Factory Methods ==========

    @classmethod
    def from_csv(
        cls,
        path: str,
        schema: Dict[str, str],
        corruption_strategy: Optional[Any] = None,
        metric_fn: Optional[Callable] = None,
        prompt_template: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> "GenericTaskSpec":
        """
        Create a GenericTaskSpec from a CSV file.

        Args:
            path: Path to CSV file
            schema: Column name mapping (see __init__)
            corruption_strategy: CorruptionPipeline instance
            metric_fn: Metric function
            prompt_template: Optional Jinja2 template
            metadata_filter: Optional filter for metadata fields
            name: Task name (defaults to "csv_<filename>")
            **kwargs: Additional arguments to __init__

        Returns:
            GenericTaskSpec instance
        """
        if name is None:
            name = f"csv_{Path(path).stem}"

        source = {
            "type": "csv",
            "path_or_id": path,
        }

        return cls(
            name=name,
            source=source,
            schema=schema,
            corruption_strategy=corruption_strategy,
            metric_fn=metric_fn,
            prompt_template=prompt_template,
            metadata_filter=metadata_filter,
            **kwargs,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        schema: Dict[str, str],
        corruption_strategy: Optional[Any] = None,
        metric_fn: Optional[Callable] = None,
        prompt_template: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> "GenericTaskSpec":
        """
        Create a GenericTaskSpec from a JSONL file.

        Args:
            path: Path to JSONL file
            schema: Column name mapping
            corruption_strategy: CorruptionPipeline instance
            metric_fn: Metric function
            prompt_template: Optional Jinja2 template
            metadata_filter: Optional filter for metadata fields
            name: Task name (defaults to "jsonl_<filename>")
            **kwargs: Additional arguments to __init__

        Returns:
            GenericTaskSpec instance
        """
        if name is None:
            name = f"jsonl_{Path(path).stem}"

        source = {
            "type": "jsonl",
            "path_or_id": path,
        }

        return cls(
            name=name,
            source=source,
            schema=schema,
            corruption_strategy=corruption_strategy,
            metric_fn=metric_fn,
            prompt_template=prompt_template,
            metadata_filter=metadata_filter,
            **kwargs,
        )

    @classmethod
    def from_huggingface(
        cls,
        dataset_id: str,
        schema: Dict[str, str],
        corruption_strategy: Optional[Any] = None,
        metric_fn: Optional[Callable] = None,
        prompt_template: Optional[str] = None,
        split: str = "test",
        metadata_filter: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs,
    ) -> "GenericTaskSpec":
        """
        Create a GenericTaskSpec from a HuggingFace dataset.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., "squad", "wikitext")
            schema: Column name mapping
            corruption_strategy: CorruptionPipeline instance
            metric_fn: Metric function
            prompt_template: Optional Jinja2 template
            split: Dataset split to load (default "test")
            metadata_filter: Optional filter for metadata fields
            name: Task name (defaults to dataset_id)
            **kwargs: Additional arguments to __init__

        Returns:
            GenericTaskSpec instance
        """
        if name is None:
            name = dataset_id

        source = {
            "type": "hf",
            "path_or_id": dataset_id,
            "split": split,
        }

        return cls(
            name=name,
            source=source,
            schema=schema,
            corruption_strategy=corruption_strategy,
            metric_fn=metric_fn,
            prompt_template=prompt_template,
            metadata_filter=metadata_filter,
            **kwargs,
        )

    # ========== Data Loading ==========

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load raw data from source (CSV, JSONL, or HF).

        Returns:
            List of dictionaries with all columns/fields from source

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If format is unsupported
        """
        source_type = self.source.get("type", "csv")
        path_or_id = self.source.get("path_or_id")

        if source_type == "csv":
            return self._load_csv(path_or_id)
        elif source_type == "jsonl":
            return self._load_jsonl(path_or_id)
        elif source_type == "hf":
            return self._load_huggingface(path_or_id)
        else:
            raise ValueError(
                f"Unsupported source type {source_type!r}. "
                f"Set the task's source 'type' to one of: 'csv', 'jsonl', 'hf'."
            )

    def _load_csv(self, path: str) -> List[Dict[str, Any]]:
        """Load CSV file using pandas."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {path}. Check the task source's "
                f"'path_or_id' points to an existing CSV file."
            )

        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load JSONL file (one JSON object per line)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"JSONL file not found: {path}. Check the task source's "
                f"'path_or_id' points to an existing JSONL file."
            )

        examples = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples

    def _load_huggingface(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Load HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required for HuggingFace support. "
                "Install with: pip install datasets"
            )

        split = self.source.get("split", "test")
        subset = self.source.get("subset")
        if subset:
            dataset = load_dataset(dataset_id, subset, split=split)
        else:
            dataset = load_dataset(dataset_id, split=split)
        return [dict(ex) for ex in dataset]

    def _extract_fields(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and render fields according to schema.

        Handles:
        - Prompt (required, supports templating)
        - Answer (single or multiple)
        - Context (optional, for QA/RC)
        - Choices (optional, for MCQ)
        - Answer positions (optional, for span-based answers)
        - Metadata (optional, for filtering/analysis)

        Args:
            example: Raw example from data source

        Returns:
            Dictionary with keys:
            - 'prompt': rendered prompt text
            - 'answer' or 'answers': answer(s)
            - 'context': context text (if present)
            - 'choices': choice options (if present)
            - 'valid_answers': list of valid answer tokens (for ranking)
            - 'metadata': metadata dict (if present)
            - all original fields (for use in template/corruption)
        """
        result = dict(example)  # Start with all fields

        # Extract and render prompt
        prompt_col = self.schema.get("prompt")
        prompt_text = str(example.get(prompt_col, ""))

        if self.template:
            try:
                prompt_text = self.template.render(**example)
            except Exception:
                # Fall back to raw prompt if template fails
                pass

        result["prompt"] = prompt_text

        # Extract context if present (for QA/RC tasks)
        context_col = self.schema.get("context")
        if context_col and context_col in example:
            context = str(example.get(context_col, ""))
            result["context"] = context
            # Combine context with prompt for unified input
            result["context_boundary"] = len(context)  # Track where context ends

        # Extract an explicit corrupted/counter-factual prompt if the data
        # provides one. The schema may name the column via "corrupted" or
        # "corrupted_prompt"; we also accept those literal column names even
        # when the schema does not declare them, so bring-your-own data with a
        # `corrupted_prompt` column works without extra configuration.
        corrupted_col = self.schema.get("corrupted") or self.schema.get("corrupted_prompt")
        corrupted_value = None
        if corrupted_col and corrupted_col in example:
            corrupted_value = example.get(corrupted_col)
        else:
            for default_col in ("corrupted_prompt", "corrupted"):
                if default_col in example:
                    corrupted_value = example.get(default_col)
                    break
        if corrupted_value is not None and str(corrupted_value).strip():
            result["corrupted_prompt"] = str(corrupted_value)

        # Extract answer(s)
        answer_col = self.schema.get("answer")
        answers_col = self.schema.get("answers")

        if answer_col and answer_col in example:
            result["answer"] = str(example.get(answer_col, ""))
        elif answers_col and answers_col in example:
            result["answers"] = example[answers_col]

        # Extract a corrupted-answer column if present (paired with an explicit
        # corrupted prompt). Used to derive the incorrect-token label so the
        # logit-diff metric is contrastive even without a corruption strategy.
        corrupted_answer_col = self.schema.get("corrupted_answer") or self.schema.get(
            "corrupt_answer"
        )
        if corrupted_answer_col and corrupted_answer_col in example:
            result["corrupted_answer"] = str(example.get(corrupted_answer_col, ""))
        elif "corrupted_answer" in example:
            result["corrupted_answer"] = str(example.get("corrupted_answer", ""))

        # Extract answer position/span if present
        answer_start_col = self.schema.get("answer_start")
        answer_end_col = self.schema.get("answer_end")

        if answer_start_col and answer_start_col in example:
            result["answer_start"] = int(example.get(answer_start_col, 0))
        if answer_end_col and answer_end_col in example:
            result["answer_end"] = int(example.get(answer_end_col, 0))

        # Extract choices if present (for MCQ)
        choices_col = self.schema.get("choices")
        if choices_col and choices_col in example:
            result["choices"] = example[choices_col]

        # Extract correct choice index if present
        choice_idx_col = self.schema.get("correct_choice_idx")
        if choice_idx_col and choice_idx_col in example:
            result["correct_choice_idx"] = int(example.get(choice_idx_col, 0))

        # Extract metadata fields
        metadata = {}

        id_col = self.schema.get("id")
        if id_col and id_col in example:
            metadata["id"] = example[id_col]

        difficulty_col = self.schema.get("difficulty")
        if difficulty_col and difficulty_col in example:
            metadata["difficulty"] = example[difficulty_col]

        category_col = self.schema.get("category")
        if category_col and category_col in example:
            metadata["category"] = example[category_col]

        meta_col = self.schema.get("metadata")
        if meta_col and meta_col in example:
            meta_val = example[meta_col]
            if isinstance(meta_val, str):
                try:
                    meta_val = json.loads(meta_val)
                except (json.JSONDecodeError, TypeError):
                    pass
            metadata.update(meta_val if isinstance(meta_val, dict) else {})

        if metadata:
            result["metadata"] = metadata

        return result

    # ========== TaskSpec Protocol Implementation ==========

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """
        Validate generic task discovery configuration.

        Args:
            discovery_cfg: Discovery configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # Check algorithm — accept any registered discovery algorithm
        from circuitkit.utils.exceptions import DISCOVERY_ALGORITHMS

        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in DISCOVERY_ALGORITHMS:
            if not algorithm:
                raise ValueError(
                    "GenericTaskSpec discovery config is missing the required "
                    "key 'algorithm'. Add 'algorithm' to the discovery config. "
                    "GenericTaskSpec only supports discovery algorithms; valid "
                    f"values: {sorted(DISCOVERY_ALGORITHMS)}."
                )
            raise ValueError(
                f"GenericTaskSpec only supports discovery algorithms, "
                f"got: {algorithm!r}. "
                f"Set discovery config key 'algorithm' to one of: "
                f"{sorted(DISCOVERY_ALGORITHMS)}."
            )

        # Check batch_size if present
        batch_size = discovery_cfg.get("batch_size")
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError(
                    f"GenericTaskSpec has invalid 'batch_size': {batch_size!r}. "
                    f"Set discovery config key 'batch_size' to a positive "
                    f"integer (e.g. 16)."
                )

        # Check num_examples if present
        num_examples = discovery_cfg.get("num_examples")
        if num_examples is not None:
            if not isinstance(num_examples, int) or num_examples <= 0:
                raise ValueError(
                    f"GenericTaskSpec has invalid 'num_examples': "
                    f"{num_examples!r}. Set discovery config key 'num_examples' "
                    f"to a positive integer (e.g. 128)."
                )

    def _matches_metadata_filter(self, example: Dict[str, Any]) -> bool:
        """
        Check if example matches metadata filter.

        Args:
            example: Extracted example with metadata

        Returns:
            True if example matches all filter conditions, False otherwise
        """
        if not self.metadata_filter:
            return True

        metadata = example.get("metadata", {})

        for key, value in self.metadata_filter.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False

        return True

    def build_dataloader(  # noqa: C901 - complex function, refactor out of scope for lint pass
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> DataLoader:
        """
        Build DataLoader for generic task.

        Process:
        1. Load raw data from source
        2. Extract fields using schema (including context, choices, metadata)
        3. Apply metadata filtering if configured
        4. Apply template rendering if configured
        5. Apply corruption strategy to generate corrupted versions
        6. Convert answers to token IDs
        7. Create EAP-compatible dataset and DataLoader

        Args:
            model: HookedTransformer model (required for tokenizer)
            discovery_cfg: Discovery configuration with keys:
                - algorithm: "eap" | "eap-ig"
                - batch_size: int
                - num_examples: optional max examples to use
                - seed: random seed (default 42)
                - metadata_filter: optional override for metadata filtering
            device: Target device

        Returns:
            DataLoader with EAP-compatible format

            Attached attributes:
            - answer_spans: List of (start, end) tuples for multi-token answers
            - has_multi_token: Bool indicating if any examples have multi-token answers
            - has_context: Bool indicating if any examples have context
            - has_metadata: Bool indicating if any examples have metadata
            - task_type: Detected task type

        Raises:
            ValueError: If model is None or data loading fails
        """
        if model is None:
            raise ValueError("GenericTaskSpec requires model for tokenizer")

        # Validate config
        self.validate_discovery_config(discovery_cfg)

        # Load raw data
        raw_examples = self._load_data()

        # Extract fields
        examples = [self._extract_fields(ex) for ex in raw_examples]

        # Apply metadata filter if configured
        filter_override = discovery_cfg.get("metadata_filter")
        if filter_override:
            old_filter = self.metadata_filter
            self.metadata_filter = filter_override

        examples = [ex for ex in examples if self._matches_metadata_filter(ex)]

        if filter_override:
            self.metadata_filter = old_filter

        if not examples:
            raise ValueError(
                "No examples match metadata filter. "
                f"Filter: {self.metadata_filter or discovery_cfg.get('metadata_filter')}"
            )

        # Limit to num_examples if specified
        num_examples = discovery_cfg.get("num_examples")
        if num_examples:
            seed = discovery_cfg.get("seed", 42)
            rng = random.Random(seed)
            examples = rng.sample(examples, min(num_examples, len(examples)))

        # Apply corruption strategy
        corrupted_examples = self._apply_corruptions(examples, discovery_cfg)

        # Resolve the chat-template policy once for this run. A per-run override
        # in discovery_cfg takes precedence over the task's own default; the
        # declared mode collapses against the concrete model into a boolean.
        # When this resolves to False (base models / "off" tasks), wrap_prompt
        # is a no-op so behavior is byte-identical to the legacy raw-text path.
        chat_mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply_chat = resolve_chat_template(chat_mode, model)

        # IBCircuit format: return {tokens, labels, answer_positions}
        algo = discovery_cfg.get("algorithm", "eap").lower()
        if algo == "ibcircuit":
            return self._build_ibcircuit(examples, model, device, discovery_cfg, apply_chat)

        # Create EAP format with extended support
        eap_examples = []
        answer_spans = []
        valid_answers_list = []
        n_degenerate_answer = 0

        for clean_ex, corr_ex in zip(examples, corrupted_examples):
            try:
                # Handle context: prepend to prompt if present
                clean_prompt = clean_ex["prompt"]
                corr_prompt = corr_ex["prompt"]

                if "context" in clean_ex:
                    context = clean_ex["context"]
                    clean_prompt = context + " " + clean_prompt
                    if "context" in corr_ex:
                        context_corr = corr_ex["context"]
                        corr_prompt = context_corr + " " + corr_prompt

                # Get answer and valid answers
                answer = clean_ex.get("answer", "")
                answers = clean_ex.get("answers", [])
                valid_answers = [answer] if answer else (answers if answers else [])

                if not valid_answers:
                    if self.task_type in ("generation", "language_modeling"):
                        # For generation/LM: use first token of prompt as placeholder label
                        first_tok = model.to_tokens(str(clean_prompt), prepend_bos=False)
                        if first_tok.shape[1] > 0:
                            correct_idx = first_tok[0, 0].item()
                            valid_answers = [correct_idx]
                        else:
                            continue
                    else:
                        continue

                # Tokenize answers to get token IDs
                valid_answer_ids = []
                correct_idx = None

                for ans in valid_answers:
                    ans_tokens = model.to_tokens(str(ans), prepend_bos=False, padding_side="right")
                    if ans_tokens.shape[1] > 0:
                        ans_id = ans_tokens[0, 0].item()
                        valid_answer_ids.append(ans_id)
                        if correct_idx is None:
                            correct_idx = ans_id

                if correct_idx is None:
                    continue

                # Wrap clean and corrupted prompts identically in the model's
                # chat template (no-op when apply_chat is False). Applying the
                # same template to both keeps the contrastive pair token-aligned.
                clean_prompt = wrap_prompt(model, clean_prompt, apply=apply_chat)
                corr_prompt = wrap_prompt(model, corr_prompt, apply=apply_chat)

                # Build EAP example with extended fields
                eap_ex = {
                    "clean": clean_prompt,
                    "corrupted": corr_prompt,
                    "correct_idx": correct_idx,
                    "labels": [correct_idx],
                }

                # For contrastive tasks (classification, MCQ, QA): try to find wrong token
                if self.task_type in ("mcq", "classification", "qa"):
                    # Try to get wrong answer from corrupted example
                    wrong_answer = corr_ex.get("answer", "")
                    wrong_answers = corr_ex.get("answers", [])
                    wrong_candidates = (
                        [wrong_answer] if wrong_answer else (wrong_answers if wrong_answers else [])
                    )
                    if wrong_candidates:
                        wrong_toks = model.to_tokens(str(wrong_candidates[0]), prepend_bos=False)
                        if wrong_toks.shape[1] > 0:
                            eap_ex["incorrect_idx"] = wrong_toks[0, 0].item()
                    # Fallback: use choices if available
                    if "incorrect_idx" not in eap_ex and "choices" in clean_ex:
                        choices = clean_ex.get("choices", [])
                        correct_choice_idx = clean_ex.get("correct_choice_idx")
                        if correct_choice_idx is not None and len(choices) > 1:
                            wrong = choices[(correct_choice_idx + 1) % len(choices)]
                            wrong_toks = model.to_tokens(str(wrong), prepend_bos=False)
                            if wrong_toks.shape[1] > 0:
                                eap_ex["incorrect_idx"] = wrong_toks[0, 0].item()
                    # Last resort: skip non-contrastive (metric will use NLL)
                # For non-contrastive tasks (generation, LM): no incorrect token needed

                # Update labels with incorrect_idx if available
                if "incorrect_idx" in eap_ex:
                    if eap_ex["incorrect_idx"] == eap_ex["correct_idx"]:
                        # Correct and incorrect answers share a first sub-word token,
                        # so logit_diff would compute (correct - incorrect) on the same
                        # index — an identically-zero, degenerate metric. Drop the pair,
                        # matching NormalizedTaskSpec's discriminative-answer filter.
                        n_degenerate_answer += 1
                        continue
                    eap_ex["labels"] = [eap_ex["correct_idx"], eap_ex["incorrect_idx"]]

                # Add context boundary tracking if present
                if "context" in clean_ex:
                    eap_ex["context"] = clean_ex["context"]

                # Add choices if present
                if "choices" in clean_ex:
                    eap_ex["choices"] = clean_ex["choices"]
                if "correct_choice_idx" in clean_ex:
                    eap_ex["correct_choice_idx"] = clean_ex["correct_choice_idx"]

                # Add valid answers for multi-answer tasks
                if len(valid_answer_ids) > 1:
                    eap_ex["valid_answers"] = valid_answer_ids
                    valid_answers_list.append(valid_answer_ids)
                else:
                    valid_answers_list.append([correct_idx])

                # Add metadata if present
                if "metadata" in clean_ex:
                    eap_ex["metadata"] = clean_ex["metadata"]

                # Handle answer spans
                if "answer_start" in clean_ex and "answer_end" in clean_ex:
                    eap_ex["answer_start"] = clean_ex["answer_start"]
                    eap_ex["answer_end"] = clean_ex["answer_end"]
                    answer_spans.append((clean_ex["answer_start"], clean_ex["answer_end"]))
                else:
                    answer_spans.append(None)

                eap_examples.append(eap_ex)
            except Exception:
                # Skip examples that fail tokenization
                continue

        if not eap_examples:
            degenerate_note = (
                f" All {n_degenerate_answer} contrastive pair(s) were dropped because their "
                "correct and incorrect answers share a first sub-word token (the logit_diff "
                "metric would be identically zero) — use answers whose first tokens differ."
                if n_degenerate_answer
                else ""
            )
            raise ValueError(
                "No valid examples after filtering and tokenization. "
                "Check schema mapping and data format." + degenerate_note
            )
        if n_degenerate_answer:
            logger.warning(
                "Dropped %d contrastive pair(s) whose correct and incorrect answers share a "
                "first sub-word token (logit_diff would be identically zero for them). Use "
                "answers whose first tokens differ if you expected these pairs to count.",
                n_degenerate_answer,
            )

        # Create dataset and dataloader (always use fallback collate which handles dicts)
        dataset = GenericDataLoader(eap_examples)
        batch_size = discovery_cfg.get("batch_size", 32)

        def collate_EAP_with_spans(xs):
            """Collate function for extended EAP format with context, choices, metadata."""
            batch = {
                "clean": [],
                "corrupted": [],
                "labels": [],
                "answer_spans": [],
            }

            # Optional fields
            has_context = any("context" in item for item in xs)
            has_choices = any("choices" in item for item in xs)
            has_valid_answers = any("valid_answers" in item for item in xs)
            has_metadata = any("metadata" in item for item in xs)

            if has_context:
                batch["context"] = []
            if has_choices:
                batch["choices"] = []
            if has_valid_answers:
                batch["valid_answers"] = []
            if has_metadata:
                batch["metadata"] = []

            for item in xs:
                batch["clean"].append(item["clean"])
                batch["corrupted"].append(item["corrupted"])
                batch["labels"].append(item["labels"])
                batch["answer_spans"].append(item.get("answer_span"))

                if has_context:
                    batch["context"].append(item.get("context"))
                if has_choices:
                    batch["choices"].append(item.get("choices"))
                if has_valid_answers:
                    batch["valid_answers"].append(item.get("valid_answers"))
                if has_metadata:
                    batch["metadata"].append(item.get("metadata"))

            # Convert labels to tensor
            batch["labels"] = t.tensor(batch["labels"])

            return batch["clean"], batch["corrupted"], batch["labels"]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_EAP_with_spans,
            shuffle=False,
        )

        # Set pair padding side for EAP
        dataloader.pair_padding_side = "right"

        # Attach metadata
        dataloader.answer_spans = answer_spans
        dataloader.has_multi_token = dataset.has_answer_spans
        dataloader.has_context = dataset.has_context
        dataloader.has_metadata = dataset.has_metadata
        dataloader.task_type = self.task_type
        dataloader.valid_answers = valid_answers_list
        # EAP backend reads `.templated` to choose prepend_bos; name it that.
        dataloader.templated = apply_chat

        return dataloader

    def _extract_field(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for _extract_fields (singular method name)."""
        return self._extract_fields(example)

    def _build_ibcircuit(self, examples, model, device, discovery_cfg, apply_chat: bool = False):
        """Build IBCircuit format: {tokens, labels, answer_positions}.

        Args:
            apply_chat: Resolved chat-template boolean. When False the prompt is
                tokenized as raw text (legacy behavior, byte-identical).
        """
        import torch as t

        from circuitkit.tasks._chat import to_tokens

        token_ids = []
        answer_labels = []
        answer_positions = []

        for ex in examples:
            prompt = ex.get("prompt", "")
            if "context" in ex:
                prompt = ex["context"] + " " + prompt
            answer = ex.get("answer", "")
            if not answer and "answers" in ex:
                answers = ex["answers"]
                answer = str(answers[0]) if answers else ""
            if not answer:
                continue

            # Wrap prompt (no-op when apply_chat is False) and tokenize with BOS
            # handled by to_tokens so a chat template's own BOS is not doubled.
            wrapped_prompt = wrap_prompt(model, prompt, apply=apply_chat)
            full_text = wrapped_prompt + " " + str(answer)
            tokens = to_tokens(model, full_text, templated=apply_chat)
            prompt_len = to_tokens(model, wrapped_prompt, templated=apply_chat).size(1)
            answer_len = tokens.size(1) - prompt_len

            token_ids.append(tokens[0])
            answer_tok = model.to_tokens(str(answer), prepend_bos=False)
            if answer_tok.size(1) > 0:
                answer_labels.append(answer_tok[0, 0].item())
            else:
                continue
            answer_positions.append(prompt_len + answer_len - 1)

        if not token_ids:
            raise ValueError("No valid examples for IBCircuit")

        # Pad to uniform length
        max_len = max(t.size(0) for t in token_ids)
        pad_id = model.tokenizer.pad_token_id or 0
        padded = t.full((len(token_ids), max_len), pad_id, dtype=t.long)
        for i, tok in enumerate(token_ids):
            padded[i, : tok.size(0)] = tok

        batch = {
            "tokens": padded.to(device),
            "labels": t.tensor(answer_labels, device=device, dtype=t.long),
            "answer_positions": t.tensor(answer_positions, device=device, dtype=t.long),
        }

        class SingleBatchLoader:
            def __init__(self, b):
                self.batch = b

            def __iter__(self):
                yield self.batch

        return SingleBatchLoader(batch)

    @staticmethod
    def _check_corruption_effectiveness(
        n_identical: int,
        n_total: int,
        discovery_cfg: Dict[str, Any],
        context: str,
    ) -> None:
        """Fail loud (or warn) when corruption produced identical clean/corrupt pairs.

        Discovering on ``clean == corrupt`` pairs gives EAP-family attribution no
        contrastive signal, so a *fully* degenerate corruption is an ERROR by
        default — otherwise a meaningless circuit is returned as if it succeeded
        (the exact silent-failure mode that bites auto-corrupted custom datasets
        whose prompts have no named entities / swappable tokens). A *partial*
        no-op only dilutes the signal, so it is surfaced as a loud warning with
        the exact fraction. Set ``discovery_cfg['allow_degenerate_corruption']``
        to True to force a fully-degenerate run through anyway.
        """
        if n_total == 0 or n_identical == 0:
            return
        frac = n_identical / n_total
        msg = (
            f"{n_identical}/{n_total} ({frac:.0%}) corrupted prompts are identical to "
            f"their clean prompt — {context}. Circuit discovery has no contrastive "
            f"signal for these examples."
        )
        if n_identical == n_total and not discovery_cfg.get("allow_degenerate_corruption", False):
            raise ValueError(
                msg + " Every pair is identical, so the discovered circuit would be "
                "meaningless. Provide an explicit corrupted-prompt column, choose a "
                "corruption strategy that applies to your data, or set discovery config "
                "'allow_degenerate_corruption'=True to run anyway."
            )
        logger.warning(msg)

    def _apply_corruptions(
        self, examples: List[Dict[str, Any]], discovery_cfg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply corruption strategy to examples.

        Args:
            examples: List of extracted examples
            discovery_cfg: Discovery configuration

        Returns:
            List of corrupted examples (same length as input)
        """
        # Path 1: data carries explicit corrupted prompts. Use them verbatim —
        # an explicit counter-factual always overrides a corruption strategy,
        # and means a working contrastive pair is available even when the
        # configured strategy needs an optional dependency (e.g. spaCy).
        if any("corrupted_prompt" in ex for ex in examples):
            corrupted = []
            for ex in examples:
                corr_ex = dict(ex)
                if "corrupted_prompt" in ex:
                    corr_ex["prompt"] = ex["corrupted_prompt"]
                if "corrupted_answer" in ex:
                    corr_ex["answer"] = ex["corrupted_answer"]
                corrupted.append(corr_ex)
            n_same = sum(
                1 for c, x in zip(examples, corrupted) if c.get("prompt") == x.get("prompt")
            )
            self._check_corruption_effectiveness(
                n_same, len(corrupted), discovery_cfg,
                context="explicit corrupted prompts are identical to clean prompts",
            )
            return corrupted

        # Path 2: no corruption strategy and no explicit corrupted column.
        if self.corruption_strategy is None:
            logger.warning(
                "No corruption strategy and no explicit 'corrupted'/'corrupted_prompt' "
                "column: corrupted prompts equal clean prompts, so circuit "
                "discovery has no contrastive signal. Provide a corruption "
                "strategy or a corrupted-prompt column."
            )
            return [dict(ex) for ex in examples]

        # Path 3: apply the configured corruption strategy.
        seed = discovery_cfg.get("seed", 42)
        rng = random.Random(seed)

        corrupted = []
        n_failed = 0
        for ex in examples:
            try:
                corr_ex = self.corruption_strategy.corrupt_example(ex, rng)
                corrupted.append(corr_ex)
                # A strategy can report a real ``strategy_used`` yet still leave the
                # prompt unchanged when it finds nothing to corrupt (no entity/token/
                # role to swap). Count by prompt equality, not the strategy label, so
                # those silent no-ops are caught by the effectiveness guard below.
                if (
                    corr_ex.get("strategy_used", None) == "none"
                    or corr_ex.get("prompt") == ex.get("prompt")
                ):
                    n_failed += 1
            except Exception:
                # If corruption fails, use clean example
                corrupted.append(dict(ex))
                n_failed += 1

        self._check_corruption_effectiveness(
            n_failed, len(examples), discovery_cfg,
            context=(
                "the corruption strategy left them unchanged (it may require an "
                "optional dependency such as spaCy, or may not apply to this data — "
                "e.g. no named entities / swappable tokens in the prompts)"
            ),
        )

        return corrupted

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

        Supports extended schema:
        - Context: prepended to prompt
        - Multiple answers: first answer used
        - Metadata: can be filtered via discovery_cfg

        When the finetuning model is instruction-tuned (and the task's resolved
        ``chat_template_mode`` is not ``"off"``) each prompt is wrapped in the
        model's chat template. ``build_dataloader`` wraps discovery prompts with
        an empty assistant prefix, so the same empty prefix is used here — the
        finetuning text then matches the discovery text. For base models /
        ``"off"`` the prompt text is byte-identical to the legacy raw-text
        behavior.

        Args:
            tokenizer: HuggingFace tokenizer
            model_name: Model name (unused for generic tasks)
            n_examples: Maximum number of examples
            discovery_cfg: Optional discovery configuration (can include
                           metadata_filter and a chat_template_mode override)
            seed: Random seed

        Returns:
            Tuple of (clean_texts, query_strings) for LM finetuning
        """
        # Load and extract data
        raw_examples = self._load_data()
        examples = [self._extract_fields(ex) for ex in raw_examples]

        # Apply metadata filter if provided
        if discovery_cfg:
            filter_override = discovery_cfg.get("metadata_filter")
            if filter_override:
                old_filter = self.metadata_filter
                self.metadata_filter = filter_override
                examples = [ex for ex in examples if self._matches_metadata_filter(ex)]
                self.metadata_filter = old_filter

        # Shuffle and limit
        rng = random.Random(seed)
        examples = rng.sample(examples, min(n_examples, len(examples)))

        # Resolve the chat-template decision from the tokenizer (a tokenizer
        # carrying a chat_template ⇒ chat model). A discovery_cfg override takes
        # precedence over the task default, mirroring build_dataloader.
        cfg = discovery_cfg or {}
        mode = cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template_from_tokenizer(mode, tokenizer)

        clean_texts = []
        query_strings = []

        for ex in examples:
            # Build prompt (may include context)
            prompt = ex.get("prompt", "")
            if "context" in ex:
                prompt = ex["context"] + " " + prompt

            # Wrap with an empty assistant prefix — matches build_dataloader's
            # wrap_prompt call (no-op when apply is False — byte-identical to
            # the legacy raw-text behavior).
            prompt = wrap_prompt_with_tokenizer(tokenizer, prompt, apply=apply)

            # Get answer (prefer single, fall back to multiple)
            answer = ex.get("answer", "")
            if not answer and "answers" in ex:
                answers = ex["answers"]
                answer = answers[0] if answers else ""

            clean_texts.append(prompt + answer)
            query_strings.append(prompt)

        return clean_texts, query_strings

    def metric_fn(self) -> Callable:
        """
        Return the metric function for circuit discovery.

        Returns:
            Callable metric function

        Raises:
            ValueError: If no metric_fn was provided
        """
        if self._metric_fn is None:
            raise ValueError(
                "No metric function provided for task. " "Pass metric_fn argument to __init__."
            )
        return self._metric_fn

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metadata for saved artifacts.

        Args:
            discovery_cfg: Discovery configuration

        Returns:
            Dictionary with task metadata including:
            - task_name: Task name
            - task_type: Auto-detected task type (classification, qa, mcq, etc.)
            - source_type: CSV, JSONL, or HuggingFace
            - source_path: Path to data source
            - schema: Full schema mapping
            - algorithm: Discovery algorithm (eap, eap-ig)
            - corruption_strategy: Corruption strategy class name
            - has_context: Whether task has context field
            - has_choices: Whether task has MCQ choices
            - has_multiple_answers: Whether task supports multiple valid answers
            - metadata_filter: Any metadata filters applied
            - chat_template_mode: Declared chat-template policy for this run
              (honors a discovery_cfg override). Frozen here so downstream
              stages resolve it identically against the same model.
        """
        return {
            "task_name": self.name,
            "task_type": self.task_type,
            "source_type": self.source.get("type"),
            "source_path": self.source.get("path_or_id"),
            "schema": self.schema,
            "algorithm": discovery_cfg.get("algorithm"),
            "corruption_strategy": (
                self.corruption_strategy.__class__.__name__ if self.corruption_strategy else "none"
            ),
            "has_context": "context" in self.schema,
            "has_choices": "choices" in self.schema,
            "has_multiple_answers": "answers" in self.schema,
            "metadata_filter": self.metadata_filter if self.metadata_filter else None,
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }
