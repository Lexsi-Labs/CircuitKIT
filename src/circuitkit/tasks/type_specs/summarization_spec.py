"""Summarization task spec: article-corruption with span NLL."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ...data.corruption.entity_swap import EntitySwap
from ..generic import GenericTaskSpec


class SummarizationSpec(GenericTaskSpec):
    """Task spec for summarization (CNN/DailyMail style).

    Contrastive pair: swap named entities in the article text.
    Metric: NLL over the summary token span.
    Schema requires: 'prompt' (article). 'answer' (summary) optional.
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
        corruption_strategy = corruption_strategy or EntitySwap()
        super().__init__(
            name=name,
            source=source,
            schema=schema,
            corruption_strategy=corruption_strategy,
            metric_fn=metric_fn,
            prompt_template=prompt_template,
            metadata_filter=metadata_filter,
            task_type="generation",
        )

    def _validate_schema(self) -> None:
        if "prompt" not in self.schema:
            raise ValueError(
                "Summarization task schema is missing the required key 'prompt'. Add a 'prompt' entry to the schema mapping it to the article/source-text column."
            )
