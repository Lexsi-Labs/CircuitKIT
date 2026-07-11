"""
HuggingFace Dataset Factory for Auto-Schema Detection

Provides high-level factory functions for creating TaskSpecs from any
HuggingFace dataset with automatic schema detection.

Key Functions:
- auto_task_from_hf: Create GenericTaskSpec from HF dataset (auto-detect)
- list_compatible_datasets: Find known compatible datasets
- preview_schema: Show detected schema without loading full dataset
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
GenericTaskSpec = None
SchemaAnalyzer = None
TaskType = None


def _ensure_imports():
    """Lazy import to avoid circular dependencies."""
    global GenericTaskSpec, SchemaAnalyzer, TaskType
    if GenericTaskSpec is None:
        from .auto_schema import SchemaAnalyzer as SA
        from .auto_schema import TaskType as TT
        from .generic import GenericTaskSpec as GTS

        GenericTaskSpec = GTS
        SchemaAnalyzer = SA
        TaskType = TT


@dataclass
class SchemaPreview:
    """Preview of detected dataset schema."""

    dataset_name: str
    subset: Optional[str]
    split: str
    columns: List[str]
    num_examples: int
    detected_task_type: str
    confidence: float
    suggested_mapping: Dict[str, str]
    sample: Dict[str, Any]
    reasoning: str


def auto_task_from_hf(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    max_samples: int = 10000,
    force_task_type: Optional[str] = None,
    custom_mapping: Optional[Dict[str, str]] = None,
    corruption_strategy: Optional[Any] = None,
    metric_fn: Optional[Any] = None,
    prompt_template: Optional[str] = None,
    name: Optional[str] = None,
) -> Any:  # Returns GenericTaskSpec
    """
    Automatically create GenericTaskSpec from any HuggingFace dataset.

    Detects task type from column names and data patterns, then maps
    columns to GenericTaskSpec schema format.

    Args:
        dataset_name: HuggingFace dataset ID (e.g., "squad", "glue")
        subset: Dataset subset/config name (e.g., "sst2" for GLUE)
        split: Dataset split to load (default "train")
        max_samples: Max examples to analyze for schema detection
        force_task_type: Override auto-detected task type
            ("qa", "mcq", "classification", "ranking", "paraphrase")
        custom_mapping: Override auto-detected column mapping
            Maps GenericTaskSpec schema keys to column names
        corruption_strategy: CorruptionPipeline instance (optional)
        metric_fn: Metric function (optional)
        prompt_template: Jinja2 template for prompt rendering (optional)
        name: Task name (defaults to dataset_name)

    Returns:
        GenericTaskSpec instance configured for the dataset

    Raises:
        ImportError: If datasets library not available
        ValueError: If dataset not found or schema detection fails
        RuntimeError: If task type cannot be reliably detected

    Examples:
        >>> task = auto_task_from_hf("squad")
        >>> task = auto_task_from_hf("glue", subset="sst2")
        >>> task = auto_task_from_hf("mmlu", force_task_type="mcq")
        >>> task = auto_task_from_hf(
        ...     "custom_dataset",
        ...     custom_mapping={"prompt": "text", "answer": "label"}
        ... )
    """
    _ensure_imports()

    try:
        import datasets
    except ImportError:
        raise ImportError(
            "datasets library required for auto_task_from_hf. " "Install with: pip install datasets"
        )

    # Load dataset
    logger.info(f"Loading {dataset_name} (subset={subset}, split={split})...")
    try:
        if subset:
            dataset = datasets.load_dataset(dataset_name, subset, split=split)
        else:
            dataset = datasets.load_dataset(dataset_name, split=split)
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset {dataset_name}: {e}\n"
            f"Check dataset name and subset at huggingface.co/datasets"
        )

    # Convert to list for analysis
    samples = list(dataset.select(range(min(len(dataset), max_samples))))

    # Detect schema if not overridden
    if not custom_mapping:
        logger.info("Auto-detecting schema...")
        detection = SchemaAnalyzer.analyze(samples, max_samples=max_samples)

        # Log confidence
        logger.info(
            f"Detected task type: {detection.task_type.value} "
            f"(confidence: {detection.confidence:.2f})"
        )
        logger.info(f"Reasoning: {detection.reasoning}")

        # Check confidence threshold
        if detection.task_type == TaskType.UNKNOWN and force_task_type is None:
            logger.warning(
                f"Could not confidently detect task type. "
                f"Detected features: {detection.detected_features}"
            )
            # Still use the suggested mapping with minimal confidence
            mapping = detection.suggested_mapping
        else:
            mapping = SchemaAnalyzer.suggest_mapping(samples, detection.task_type)

        # Allow force override
        if force_task_type:
            logger.info(f"Forcing task type: {force_task_type}")
            # Re-suggest mapping for forced type
            try:
                forced_type = TaskType(force_task_type)
                mapping = SchemaAnalyzer.suggest_mapping(samples, forced_type)
            except ValueError:
                raise ValueError(
                    f"Invalid task type: {force_task_type}. "
                    f"Must be one of: {[t.value for t in TaskType]}"
                )
    else:
        logger.info(f"Using custom mapping: {custom_mapping}")
        mapping = custom_mapping

    # Validate mapping has required fields
    if "prompt" not in mapping:
        # Fallback: if only 'text' column exists, use it as prompt
        columns = list(dataset.features.keys())
        if "text" in columns:
            mapping["prompt"] = "text"
            logger.info(f"No 'prompt' column, using 'text' as prompt. Full mapping: {mapping}")
        else:
            raise ValueError(
                "Schema mapping must include 'prompt' key. " f"Available columns: {columns}"
            )

    # Resolve task type (before answer validation)
    resolved_task_type = force_task_type
    if not resolved_task_type:
        try:
            resolved_task_type = detection.task_type.value
        except NameError:
            pass  # not detected

    if "answer" not in mapping and "answers" not in mapping and "choices" not in mapping:
        if resolved_task_type not in ("generation", "language_modeling"):
            logger.warning(
                "No 'answer', 'answers', or 'choices' columns detected. "
                f"Defaulting prompt-only to generation/LM task type. "
                f"Mapped columns: {list(mapping.keys())}"
            )
            resolved_task_type = "generation"

    # Pick the right type-spec class based on task type + schema
    spec_cls, base_task_type = _select_type_spec(resolved_task_type, mapping)

    if name is None:
        name = dataset_name
        if subset:
            name = f"{dataset_name}_{subset}"

    logger.info(f"Creating {spec_cls.__name__} '{name}' with mapping: {mapping}")

    task = spec_cls(
        name=name,
        source={"type": "hf", "path_or_id": dataset_name, "split": split, "subset": subset},
        schema=mapping,
        corruption_strategy=corruption_strategy,
        metric_fn=metric_fn,
        prompt_template=prompt_template,
    )

    return task


def _auto_corruption_strategy(task_type: Optional[str] = None):
    """Pick default corruption strategy per task type (legacy, kept for backward compat)."""
    return None  # type specs now set their own corruption


def _select_type_spec(task_type: str, schema: Dict[str, str]):
    """Pick the right type-spec class based on task type + schema features.

    Returns a (SpecClass, base_task_type) tuple.
    """
    from .generic import GenericTaskSpec
    from .type_specs.classification_spec import ClassificationSpec
    from .type_specs.generation_spec import GenerationSpec
    from .type_specs.mcq_spec import MCQSpec
    from .type_specs.qa_spec import ExtractiveQASpec, QASpec, SimpleQASpec
    from .type_specs.summarization_spec import SummarizationSpec
    from .type_specs.translation_spec import TranslationSpec

    mapping = {
        "mcq": MCQSpec,
        "classification": ClassificationSpec,
        "ranking": GenericTaskSpec,
    }

    if task_type in mapping:
        return mapping[task_type], task_type

    # QA: probe answer values to distinguish binary vs extractive
    if task_type == "qa":
        if "context" in schema and ("answer" in schema or "answers" in schema):
            return QASpec, "qa"
        if "context" in schema:
            return ExtractiveQASpec, "qa"
        if "answer" in schema or "answers" in schema:
            return QASpec, "qa"
        return SimpleQASpec, "qa"

    # Generation: check original column names to detect summarization/translation
    if task_type == "generation":
        schema.get("prompt", "").lower()
        answer_col = schema.get("answer", "").lower()
        # Summarization: needs BOTH article-like prompt AND summary-like answer
        if answer_col in ("summary", "highlights", "abstract"):
            return SummarizationSpec, "generation"
        # Translation: needs BOTH source prompt AND target answer
        if answer_col in ("target", "target_text", "tgt", "translation"):
            return TranslationSpec, "generation"
        return GenerationSpec, "generation"

    # Fallback
    return GenericTaskSpec, task_type


def preview_schema(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
    max_samples: int = 100,
) -> SchemaPreview:
    """
    Preview detected schema without loading full dataset.

    Useful for understanding what schema will be auto-detected
    before creating the full task spec.

    Args:
        dataset_name: HuggingFace dataset ID
        subset: Dataset subset/config name
        split: Dataset split to load
        max_samples: Max examples to analyze

    Returns:
        SchemaPreview with detected columns, mapping, and sample

    Raises:
        ImportError: If datasets library not available
        ValueError: If dataset not found
    """
    _ensure_imports()

    try:
        import datasets
    except ImportError:
        raise ImportError(
            "datasets library required for preview_schema. " "Install with: pip install datasets"
        )

    # Load small sample
    try:
        if subset:
            dataset = datasets.load_dataset(dataset_name, subset, split=split)
        else:
            dataset = datasets.load_dataset(dataset_name, split=split)
    except Exception as e:
        raise ValueError(f"Failed to load dataset {dataset_name}: {e}")

    samples = list(dataset.select(range(min(len(dataset), max_samples))))

    # Detect schema
    detection = SchemaAnalyzer.analyze(samples, max_samples=max_samples)

    # Get sample
    sample = samples[0] if samples else {}

    return SchemaPreview(
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        columns=list(samples[0].keys()) if samples else [],
        num_examples=len(dataset),
        detected_task_type=detection.task_type.value,
        confidence=detection.confidence,
        suggested_mapping=detection.suggested_mapping,
        sample=sample,
        reasoning=detection.reasoning,
    )


def list_compatible_datasets(
    task_type: Optional[str] = None,
    max_results: int = 100,
) -> List[Dict[str, str]]:
    """
    List HuggingFace datasets known to work with auto-detection.

    Returns a curated list of popular datasets that auto-detection
    handles well, optionally filtered by task type.

    Args:
        task_type: Filter by task type
            ("qa", "mcq", "classification", "ranking", "paraphrase")
        max_results: Max datasets to return

    Returns:
        List of dicts with 'name', 'subset', 'task_type', 'description'
    """
    _ensure_imports()

    # Curated list of well-known datasets
    compatible = [
        # QA Tasks
        {
            "name": "squad",
            "subset": None,
            "task_type": "qa",
            "description": "Stanford Question Answering Dataset",
        },
        {
            "name": "squad_v2",
            "subset": None,
            "task_type": "qa",
            "description": "SQuAD v2.0 with unanswerable questions",
        },
        {
            "name": "wikiqa",
            "subset": None,
            "task_type": "qa",
            "description": "Wikipedia-based QA dataset",
        },
        # Classification Tasks
        {
            "name": "glue",
            "subset": "sst2",
            "task_type": "classification",
            "description": "GLUE: SST-2 sentiment classification",
        },
        {
            "name": "glue",
            "subset": "mrpc",
            "task_type": "paraphrase",
            "description": "GLUE: MRPC paraphrase detection",
        },
        {
            "name": "glue",
            "subset": "qqp",
            "task_type": "paraphrase",
            "description": "GLUE: QQP question paraphrase",
        },
        # MCQ Tasks
        {
            "name": "mmlu",
            "subset": None,
            "task_type": "mcq",
            "description": "Massive Multitask Language Understanding",
        },
        # Ranking Tasks
        {
            "name": "ms_marco",
            "subset": "v1.1",
            "task_type": "ranking",
            "description": "MS MARCO dataset for ranking",
        },
        # Additional classification
        {
            "name": "ag_news",
            "subset": None,
            "task_type": "classification",
            "description": "AG News topic classification",
        },
        {
            "name": "dbpedia_14",
            "subset": None,
            "task_type": "classification",
            "description": "DBpedia ontology classification",
        },
    ]

    # Filter by task type if specified
    if task_type:
        compatible = [d for d in compatible if d["task_type"] == task_type]

    return compatible[:max_results]


def validate_hf_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "train",
) -> Dict[str, Any]:
    """
    Validate that a HuggingFace dataset can be loaded and analyzed.

    Args:
        dataset_name: HuggingFace dataset ID
        subset: Dataset subset
        split: Dataset split

    Returns:
        Dict with validation results:
        - is_valid: bool
        - num_examples: int
        - columns: List[str]
        - error: Optional[str]
    """
    try:
        import datasets
    except ImportError:
        return {
            "is_valid": False,
            "num_examples": 0,
            "columns": [],
            "error": "datasets library not installed",
        }

    try:
        if subset:
            dataset = datasets.load_dataset(dataset_name, subset, split=split)
        else:
            dataset = datasets.load_dataset(dataset_name, split=split)

        return {
            "is_valid": True,
            "num_examples": len(dataset),
            "columns": dataset.column_names,
            "error": None,
        }
    except Exception as e:
        return {
            "is_valid": False,
            "num_examples": 0,
            "columns": [],
            "error": str(e),
        }
