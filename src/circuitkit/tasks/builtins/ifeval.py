"""
IFEval Task Specification

Thin wrapper for the instruction-following evaluation dataset
(HuggingFaceH4/instruction-following-eval).

This task is primarily for COLLATERAL evaluation (not circuit discovery).
It provides a simple interface for loading instruction prompts.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from ...utils.logging import get_logger

logger = get_logger("task.ifeval")


class IFEvalTaskSpec:
    """IFEval task for instruction-following evaluation."""

    name = "ifeval"
    pair_padding_side = "left"
    # Downstream-behavior task: wrap prompts in the model's chat template iff
    # the model is instruction-tuned ("auto"). Frozen into artifact metadata.
    # IFEval is collateral-evaluation only (no circuit discovery), so this is
    # carried for metadata parity with the other downstream-behavior tasks.
    chat_template_mode: str = "auto"

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """IFEval does not support circuit discovery."""
        raise ValueError(
            "IFEval does not support circuit discovery. It is a collateral-"
            "evaluation-only task. Choose a discovery-capable task "
            "(e.g. ioi, sva, boolq, gsm8k) for discover_circuit, or use "
            "IFEval only as a downstream/collateral evaluation benchmark."
        )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> "DataLoader":
        """IFEval does not support circuit discovery dataloaders."""
        raise ValueError(
            "IFEval does not support circuit discovery. It is a collateral-"
            "evaluation-only task. Choose a discovery-capable task "
            "(e.g. ioi, sva, boolq, gsm8k) for discover_circuit, or use "
            "IFEval only as a downstream/collateral evaluation benchmark."
        )

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Load IFEval prompts from the HuggingFace dataset.

        Returns (clean_texts, query_strings) where each element is the
        instruction prompt.
        """
        try:
            import random

            from datasets import load_dataset

            ds = load_dataset("HuggingFaceH4/instruction-following-eval", split="train")
            indices = list(range(len(ds)))
            rng = random.Random(seed)
            rng.shuffle(indices)
            indices = indices[:n_examples]

            prompts: List[str] = []
            for idx in indices:
                ex = ds[int(idx)]
                prompt = ex.get("prompt", "")
                if prompt:
                    prompts.append(prompt)

            if not prompts:
                raise ValueError("No IFEval prompts could be loaded.")

            return prompts, prompts

        except Exception as e:
            raise RuntimeError(f"Failed to load IFEval dataset: {e}")

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """IFEval does not support circuit discovery metrics."""
        raise ValueError(
            "IFEval does not support circuit discovery metrics. It is a "
            "collateral-evaluation-only task and has no discovery metric. "
            "Use IFEval only as a downstream/collateral evaluation benchmark."
        )

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for IFEval artifacts.

        The declared ``chat_template_mode`` (honoring a ``discovery_cfg``
        override) is frozen here so downstream stages read back an identical
        chat-template policy.
        """
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        return {
            "task": "ifeval",
            "data_source": "HuggingFaceH4/instruction-following-eval",
            "purpose": "collateral_evaluation",
            "chat_template_mode": mode,
        }
