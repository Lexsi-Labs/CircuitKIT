"""MCQ task spec: multiple-choice with choice-swap corruption."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ...data.corruption.mcq_choice_swap import MCQChoiceSwap
from ..generic import GenericTaskSpec


class MCQSpec(GenericTaskSpec):
    """Task spec for multiple-choice datasets (MMLU, TruthfulQA MC).

    Contrastive pair: swap correct/incorrect choice.
    Metric: logit difference between correct and incorrect choice tokens.
    Schema requires: 'prompt' + 'choices' + 'correct_choice_idx'.
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
        corruption_strategy = corruption_strategy or MCQChoiceSwap()
        super().__init__(
            name=name,
            source=source,
            schema=schema,
            corruption_strategy=corruption_strategy,
            metric_fn=metric_fn,
            prompt_template=prompt_template,
            metadata_filter=metadata_filter,
            task_type="mcq",
        )

    def _validate_schema(self) -> None:
        if "prompt" not in self.schema:
            raise ValueError(
                "MCQ task schema is missing the required key 'prompt'. Add a 'prompt' entry to the schema mapping it to the question column."
            )
        if "choices" not in self.schema:
            raise ValueError(
                "MCQ task schema is missing the required key 'choices'. Add a 'choices' entry to the schema mapping it to the answer-options column."
            )
