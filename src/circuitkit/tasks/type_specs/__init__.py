"""Type-specific task specs that inherit from GenericTaskSpec.

Each spec encapsulates ONE contrastive pair strategy + ONE metric.
No if/else branches on task_type.

Usage:
    from circuitkit.tasks.type_specs import QASpec, GenerationSpec

    spec = QASpec(
        name="boolq",
        source={"type": "hf", "path_or_id": "super_glue/boolq"},
        schema={"prompt": "question", "answer": "label"},
    )
"""

from .classification_spec import ClassificationSpec
from .generation_spec import GenerationSpec
from .mcq_spec import MCQSpec
from .qa_spec import QASpec
from .summarization_spec import SummarizationSpec
from .translation_spec import TranslationSpec

__all__ = [
    "QASpec",
    "MCQSpec",
    "ClassificationSpec",
    "GenerationSpec",
    "SummarizationSpec",
    "TranslationSpec",
]
