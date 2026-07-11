"""QA task specs: binary, extractive, and simple QA — each with appropriate corruption."""

from __future__ import annotations

from ...data.corruption.entity_swap import EntitySwap
from ...data.corruption.final_answer_swap import FinalAnswerSwap
from ..generic import GenericTaskSpec


class QASpec(GenericTaskSpec):
    """Binary QA (BoolQ): passage + question → yes/no answer.

    Corruption: flip the answer token (yes↔no, True↔False).
    Metric: logit difference.
    """

    def __init__(
        self,
        name,
        source,
        schema,
        corruption_strategy=None,
        metric_fn=None,
        prompt_template=None,
        metadata_filter=None,
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
            task_type="qa",
        )

    def _validate_schema(self):
        if "prompt" not in self.schema:
            raise ValueError(
                "QA task schema is missing the required key 'prompt'. Add a 'prompt' entry to the schema mapping it to the question column."
            )
        if not any(
            k in self.schema for k in ("answer", "answers", "choices", "correct_choice_idx")
        ):
            raise ValueError(
                "QA task schema is missing an answer key. Add one of 'answer', 'answers', or 'choices' to the schema."
            )


class ExtractiveQASpec(GenericTaskSpec):
    """Extractive QA (SQuAD): passage + question → text span.

    Corruption: swap entities in the context passage.
    Metric: logit difference over the answer span.
    """

    def __init__(
        self,
        name,
        source,
        schema,
        corruption_strategy=None,
        metric_fn=None,
        prompt_template=None,
        metadata_filter=None,
    ):
        corruption_strategy = corruption_strategy or EntitySwap()
        super().__init__(
            name=name,
            source=source,
            schema=schema,
            corruption_strategy=corruption_strategy,
            metric_fn=metric_fn,
            prompt_template=prompt_template,
            metadata_filter=metadata_filter,
            task_type="qa",
        )

    def _validate_schema(self):
        if "prompt" not in self.schema:
            raise ValueError(
                "ExtractiveQA task schema is missing the required key 'prompt'. Add a 'prompt' entry to the schema mapping it to the question column."
            )
        if "context" not in self.schema:
            raise ValueError(
                "ExtractiveQA task schema is missing the required key 'context'. Add a 'context' entry to the schema mapping it to the passage column."
            )


class SimpleQASpec(GenericTaskSpec):
    """Simple QA (TriviaQA, Natural Questions): question → answer.

    Corruption: swap entities in the question text.
    Metric: logit difference over the answer token.
    """

    def __init__(
        self,
        name,
        source,
        schema,
        corruption_strategy=None,
        metric_fn=None,
        prompt_template=None,
        metadata_filter=None,
    ):
        corruption_strategy = corruption_strategy or EntitySwap()
        super().__init__(
            name=name,
            source=source,
            schema=schema,
            corruption_strategy=corruption_strategy,
            metric_fn=metric_fn,
            prompt_template=prompt_template,
            metadata_filter=metadata_filter,
            task_type="qa",
        )

    def _validate_schema(self):
        if "prompt" not in self.schema:
            raise ValueError(
                "SimpleQA task schema is missing the required key 'prompt'. Add a 'prompt' entry to the schema mapping it to the question column."
            )
        if "answer" not in self.schema:
            raise ValueError(
                "SimpleQA task schema is missing the required key 'answer'. Add an 'answer' entry to the schema mapping it to the answer column."
            )
