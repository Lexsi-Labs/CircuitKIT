"""Classification task spec: label-flip corruption for text classification."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ...data.corruption.final_answer_swap import FinalAnswerSwap
from ..generic import GenericTaskSpec


class ClassificationSpec(GenericTaskSpec):
    """Task spec for text classification (GLUE style).

    Contrastive pair: flip the label (e.g., positive↔negative).
    Metric: logit difference.
    Schema requires: 'prompt' + 'answer'.
    """

    def __init__(
        self,
        name: str,
        source: Dict[str, Any],
        schema: Dict[str, str],
        corruption_strategy: Optional[Any] = None,
        metric_fn: Optional[Callable] = None,
        prompt_template: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ):
        corruption_strategy = corruption_strategy or FinalAnswerSwap()
        super().__init__(
            name=name,
            source=source,
            schema=schema,
            corruption_strategy=corruption_strategy,
            metric_fn=metric_fn,
            prompt_template=prompt_template,
            metadata_filter=metadata_filter,
            task_type="classification",
        )

    def _validate_schema(self) -> None:
        if "prompt" not in self.schema:
            raise ValueError(
                "Classification task schema is missing the required key 'prompt'. Add a 'prompt' entry to the schema mapping it to the input text column."
            )
        if "answer" not in self.schema:
            raise ValueError(
                "Classification task schema is missing the required key 'answer'. Add an 'answer' entry to the schema mapping it to the label column."
            )
