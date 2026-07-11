"""Generation task spec: context-corruption with NLL metric."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ...data.corruption.instruction_swap import InstructionSwap
from ..generic import GenericTaskSpec


class GenerationSpec(GenericTaskSpec):
    """Task spec for generation / open QA / instruction following.

    No contrastive pair — uses NLL metric instead of logit difference.
    Default corruption: InstructionSwap (swaps directive verb).
    Fallback: Resample (pairs with random example).
    Schema requires only 'prompt'. No 'answer' needed.
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
        corruption_strategy = corruption_strategy or InstructionSwap()
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
                "Generation task schema is missing the required key 'prompt'. Add a 'prompt' entry to the schema mapping it to the input text column."
            )
        # No answer required — uses NLL metric
