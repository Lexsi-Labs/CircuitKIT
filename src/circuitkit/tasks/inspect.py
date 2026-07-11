"""
CLI tool: inspect a HuggingFace dataset and report auto-selected type spec.

Usage:
    python -m circuitkit.tasks.inspect super_glue/boolq
    python -m circuitkit.tasks.inspect wikitext --subset wikitext-2-v1
"""

from __future__ import annotations

import argparse
import logging

# Static reference (only for quick-lookup; actual selection lives in _select_type_spec)
TYPE_SPEC_MAP = {
    "mcq": "MCQSpec",
    "classification": "ClassificationSpec",
    "qa+context": "ExtractiveQASpec",
    "qa+answer": "QASpec",
    "qa only": "SimpleQASpec",
    "generation (summarization)": "SummarizationSpec",
    "generation (translation)": "TranslationSpec",
    "generation (generic)": "GenerationSpec",
}

CONTRASTIVE_TYPES = {"mcq", "classification", "qa"}
NON_CONTRASTIVE_TYPES = {"generation", "language_modeling", "instruction", "open", "ranking"}

logger = logging.getLogger(__name__)


def inspect_dataset(dataset_name: str, subset: str = None, split: str = "train"):
    """Inspect a dataset and print contrastive analysis."""
    from .hf_factory import _ensure_imports, _select_type_spec

    _ensure_imports()

    try:
        import datasets
    except ImportError:
        logger.warning("ERROR: datasets library not installed. Run: pip install datasets")
        return

    from .auto_schema import SchemaAnalyzer

    # Load a sample
    msg = f"Inspecting: {dataset_name}"
    if subset:
        msg += f" (subset={subset})"
    msg += f" split={split}"
    logger.info(msg)

    try:
        if subset:
            ds = datasets.load_dataset(dataset_name, subset, split=split)
        else:
            ds = datasets.load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.warning(f"  ERROR: Could not load dataset: {e}")
        return

    # Analyze schema
    samples = list(ds.select(range(min(len(ds), 200))))
    detection = SchemaAnalyzer.analyze(samples, max_samples=200)

    task_type = detection.task_type.value
    mapping = detection.suggested_mapping
    spec_cls, _ = _select_type_spec(task_type, mapping)
    is_contrastive = task_type in CONTRASTIVE_TYPES

    logger.info(f"  Task type:    {task_type} (confidence: {detection.confidence:.2f})")
    logger.info(f"  Type spec:    {spec_cls.__name__}")
    logger.info(f"  Contrastive:  {'YES' if is_contrastive else 'NO'}")
    logger.info(f"  Reasoning:    {detection.reasoning}")
    logger.info(f"  Columns:      {list(ds.features.keys())}")
    logger.info(f"  Mapping:      {mapping}")
    logger.info(f"  Rows:         {len(ds)}")

    if not is_contrastive:
        logger.info("  NOTE: Non-contrastive task — uses NLL metric instead of logit diff.")
        logger.info("  The type spec auto-selects the appropriate corruption + metric.")
        logger.info("  Use: task = auto_task_from_hf('...')")


def main():
    p = argparse.ArgumentParser(description="Inspect HF dataset for EAP compatibility")
    p.add_argument("dataset", help="HuggingFace dataset name")
    p.add_argument("--subset", default=None, help="Dataset subset/config")
    p.add_argument("--split", default="train", help="Dataset split")
    args = p.parse_args()
    inspect_dataset(args.dataset, args.subset, args.split)


if __name__ == "__main__":
    main()
