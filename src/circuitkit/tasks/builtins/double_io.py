"""
DoubleIO (Double Indirect Object) Task Specification

A generalization test companion to the IOI task. In DoubleIO, both the
subject AND the indirect object names appear twice in the prompt, which
breaks the standard IOI algorithm's "remove duplicates" step.

Standard IOI:
  "When John and Mary went to the store, John gave a drink to ___"
  S appears 2x, IO appears 1x → remove duplicate (John) → predict Mary ✓

DoubleIO:
  "When John and Mary went to the store, Mary was happy. John gave a drink to ___"
  S appears 2x, IO appears 2x → remove duplicates → BOTH removed → algorithm fails
  Yet GPT-2 still predicts Mary correctly (~89% of the time)

This task shares the SAME metric (logit_diff) and the SAME dataloader format
as IOI, making it a valid generalization target for Pillar 6 evaluation.

Usage as generalization target:
  python benchmark_circuit_discovery.py \\
      --model gpt2 --task ioi --target-task double_io --algo eap --level node

Reference: "Adaptive Circuit Behavior and Generalization in Mechanistic
           Interpretability" (arxiv 2411.16105)
"""

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import torch

from ...data.eap_dataset import EAPDiscoveryDataset
from ...utils.logging import get_logger
from .._algorithm_families import (
    ACDC_FAMILY,
    CDT_FAMILY,
    EAP_FAMILY,
    IB_FAMILY,
    is_eap_family,
    unsupported_algorithm_message,
)
from .._chat import resolve_chat_template
from ..specs import _find_task_cache, _load_finetuning_data_from_csv

logger = get_logger("task.double_io")


class DoubleIOTaskSpec:
    """
    DoubleIO task specification for CircuitKit.

    Structurally identical to IOITaskSpec in interface — same metric,
    same dataloader format, same label schema. The only difference is
    the data generator: prompts contain a filler clause that makes IO
    appear twice.

    This makes it the ideal first generalization target for IOI circuits:
    - Same algorithmic "shape" (entity tracking + copying)
    - Same metric (logit difference)
    - Same label format ([correct_idx, incorrect_idx])
    - Different data distribution (IO appears 2x instead of 1x)
    """

    name = "double_io"
    pair_padding_side = "right"
    # diagnostic minimal-pair task -- discovered raw, per the circuit-discovery literature
    chat_template_mode: str = "off"

    def __init__(self):
        pass

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate DoubleIO-specific discovery configuration."""
        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in (EAP_FAMILY | ACDC_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "DoubleIO task", algorithm, EAP_FAMILY | ACDC_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"DoubleIO task discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        batch_size = discovery_cfg.get("batch_size")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError(
                f"DoubleIO task has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 16)."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> "DataLoader":
        """
        Build DataLoader for DoubleIO task.

        Routes to the appropriate format based on algorithm:
        - eap / eap-ig: EAP text-pair format (clean, corrupted, labels)
        - ibcircuit: dict batch format {tokens, labels, answer_positions}

        Args:
            model: HookedTransformer model
            discovery_cfg: Discovery configuration dict
            device: Target device

        Returns:
            DataLoader for the requested algorithm
        """
        if model is None:
            raise ValueError("DoubleIO task requires a model for tokenization.")

        # DoubleIO contrastive pairs are built directly from generated token
        # tensors (get_double_io_data_only) -- there is no single prompt-string
        # finalization point to wrap. Default "off" keeps everything raw; an
        # explicit override fails loudly rather than being silently ignored.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)
        if apply:
            raise NotImplementedError(
                f"{self.name}: this diagnostic task is discovered on raw "
                f"(non-chat-templated) prompts and does not support a "
                f"chat_template_mode override that enables templating. "
                f"Remove the 'chat_template_mode' key from the discovery "
                f"config or set it to 'off'."
            )

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if is_eap_family(algorithm) or algorithm == "cdt":
            return self._build_eap_dataloader(model, discovery_cfg, device)
        elif algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(model, discovery_cfg, device)
        else:
            # Fallback for evaluate_circuit's internal dl_cfg, which hardcodes
            # algorithm='eap'. Not reachable from normal discover_circuit calls.
            return self._build_eap_dataloader(model, discovery_cfg, device)

    def _build_eap_dataloader(self, model, discovery_cfg: Dict[str, Any], device: str):
        """Build EAP-format DataLoader with CSV caching."""
        import pandas as pd

        from ...data.task_data.tasks.double_io.double_io_dataset import get_double_io_data_only

        data_params = discovery_cfg.get("data_params", {})
        num_examples = data_params.get("num_examples", 128)
        seed = data_params.get("seed", 42)
        prompt_type = data_params.get("prompt_type", "ABBA")
        cache_dir = Path(data_params.get("cache_dir", "./cache/double_io"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = getattr(model.cfg, "model_name", "unknown_model").replace("/", "_")
        cache_file = cache_dir / f"double_io_{model_name}_{num_examples}_seed{seed}.csv"

        if cache_file.exists():
            logger.info(f"Loading cached DoubleIO data from {cache_file}")
            dataset = EAPDiscoveryDataset(str(cache_file))
        else:
            logger.info(
                f"Generating DoubleIO data for {model_name} "
                f"(N={num_examples}, seed={seed}, type={prompt_type})"
            )

            dio_data = get_double_io_data_only(
                num_examples=num_examples,
                device=device,
                model=model,
                seed=seed,
                prompt_type=prompt_type,
            )

            eap_rows = []
            for i in range(len(dio_data["validation_data"])):
                clean_text = model.to_string(dio_data["validation_data"][i])
                corrupted_text = model.to_string(dio_data["validation_patch_data"][i])
                correct_idx = dio_data["validation_labels"][i].item()
                incorrect_idx = dio_data["validation_wrong_labels"][i].item()

                eap_rows.append(
                    {
                        "clean": clean_text,
                        "corrupted": corrupted_text,
                        "correct_idx": correct_idx,
                        "incorrect_idx": incorrect_idx,
                    }
                )

            df = pd.DataFrame(eap_rows)
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached DoubleIO data → {cache_file}")

            dataset = EAPDiscoveryDataset(str(cache_file))

        from torch.utils.data import DataLoader

        from ...backends.eap.eap_utils import collate_EAP

        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        dl = DataLoader(
            dataset,
            batch_size=discovery_cfg.get("batch_size", 16),
            shuffle=True,
            collate_fn=collate_EAP,
        )
        dl.pair_padding_side = side
        return dl

    def _build_ibcircuit_dataloader(self, model, discovery_cfg: Dict[str, Any], device: str):
        """
        Build IBCircuit-format DataLoader (single fixed batch).

        IBCircuit trains on a fixed batch for all epochs. Returns a simple
        iterator yielding {tokens, labels, answer_positions}.
        No corrupted prompts needed — IBCircuit uses IB noise instead.
        """
        from ...data.task_data.tasks.double_io.double_io_dataset import get_double_io_data_only

        data_params = discovery_cfg.get("data_params", {})
        num_examples = data_params.get("num_examples", 100)
        seed = data_params.get("seed", 42)
        prompt_type = data_params.get("prompt_type", "ABBA")

        logger.info(f"Generating DoubleIO data for IBCircuit (N={num_examples}, seed={seed})")

        dio_data = get_double_io_data_only(
            num_examples=num_examples,
            device=device,
            model=model,
            seed=seed,
            prompt_type=prompt_type,
        )

        tokens = dio_data["validation_data"]
        labels = dio_data["validation_labels"]
        answer_positions = dio_data["end_idxs"]

        # Validate answer_positions
        if (answer_positions < 0).any() or (answer_positions >= tokens.shape[1]).any():
            logger.warning("end_idxs out of bounds, computing from padding")
            answer_positions = self._compute_answer_positions(tokens, model)

        logger.info(
            f"DoubleIO IBCircuit: {num_examples} examples, seq_len={tokens.shape[1]}, "
            f"answer_pos=[{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )

        batch = {
            "tokens": tokens,
            "labels": labels,
            "answer_positions": answer_positions,
        }

        class SingleBatchDataLoader:
            """DataLoader that yields a single fixed batch repeatedly."""

            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

    @staticmethod
    def _compute_answer_positions(tokens: "torch.Tensor", model) -> "torch.Tensor":
        """Compute answer positions as last non-padding token per sequence."""
        import torch

        batch_size, seq_len = tokens.shape
        pad_id = getattr(model.tokenizer, "pad_token_id", None) or 0

        positions = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
        for i in range(batch_size):
            non_pad = (tokens[i] != pad_id).nonzero(as_tuple=True)[0]
            positions[i] = non_pad[-1] if len(non_pad) > 0 else seq_len - 1
        return positions

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """
        Return the metric function for DoubleIO.

        Uses the SAME metric as IOI (logit_diff). This is critical —
        generalization evaluation requires source and target to share
        the same metric so transfer_ratio is meaningful.
        """
        if metric_type == "logit_diff":
            return partial(self._logit_diff, loss=True, mean=True)
        elif metric_type == "kl":
            return partial(self._kl_divergence, loss=True, mean=True)
        else:
            raise ValueError(
                f"DoubleIO does not support metric_type {metric_type!r}. "
                f"Use one of: 'logit_diff', 'kl'."
            )

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task": "double_io",
            "data_source": "built_in_generation",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "num_examples": discovery_cfg.get("data_params", {}).get("num_examples", 128),
            "prompt_type": discovery_cfg.get("data_params", {}).get("prompt_type", "ABBA"),
            "description": "DoubleIO variant — IO name appears twice",
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """Load DoubleIO finetuning data from the discovery cache."""
        cfg = discovery_cfg or {}
        model_name_safe = model_name.replace("/", "_")
        cache_dir = Path(cfg.get("cache_dir", "./cache/double_io"))

        cache_path = _find_task_cache(cache_dir, self.name, model_name_safe)
        if cache_path is None:
            raise FileNotFoundError(
                f"No '{self.name}' cache found for model '{model_name}' "
                f"in {cache_dir}.\n"
                f"Run discover_circuit with task='{self.name}' first."
            )
        return _load_finetuning_data_from_csv(cache_path, tokenizer, n_examples, seed)

    # ── Metrics (identical to IOI) ───────────────────────────────────────────

    @staticmethod
    def _logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """
        Logit difference: logit(correct) - logit(incorrect).

        Identical to IOI's metric. labels[:, 0] = IO (correct),
        labels[:, 1] = S (incorrect).
        """
        batch_size = logits.size(0)
        idx = torch.arange(batch_size, device=logits.device)
        logits = logits[idx, input_length - 1]
        good_bad = torch.gather(logits, -1, labels.to(logits.device))
        results = good_bad[:, 0] - good_bad[:, 1]
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    @staticmethod
    def _kl_divergence(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """KL divergence between patched and clean distributions."""
        import torch.nn.functional as F

        results = (
            F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(clean_logits, dim=-1),
                reduction="none",
            )
            .sum(-1)
            .mean(-1)
        )

        if mean:
            results = results.mean()
        return results
