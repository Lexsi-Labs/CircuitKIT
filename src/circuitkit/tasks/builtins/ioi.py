"""
IOI (Indirect Object Identification) Task Specification

This is a thin wrapper on GenericTaskSpec that demonstrates how all
hardcoded task logic can be expressed generically.

The IOI task uses:
- Builtin data generator: get_ioi_data_only()
- Corruption: EntitySwapCorruption (swaps person names)
- Metric: logit difference between correct and incorrect token
"""

import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import torch as t

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

logger = get_logger("task.ioi")


class IOITaskSpec:
    """
    IOI task using generic framework.

    This replaces ~392 lines of hardcoded logic with a thin configuration-driven approach.
    Unlike most generic tasks, IOI works directly with token data rather than text prompts.

    The task:
    - Generates IOI data with correct and "corrupted" (subject-swapped) versions
    - Uses EntitySwapCorruption to corrupt person names in text representations
    - Computes logit difference between correct and incorrect tokens
    """

    name = "ioi"
    pair_padding_side = "right"
    # diagnostic minimal-pair task -- discovered raw, per the circuit-discovery literature
    chat_template_mode: str = "off"

    def __init__(self):
        """Initialize IOI task spec."""

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate IOI-specific discovery configuration."""
        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in (EAP_FAMILY | ACDC_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "IOI task", algorithm, EAP_FAMILY | ACDC_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"IOI task discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        batch_size = discovery_cfg.get("batch_size")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError(
                f"IOI task has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 16)."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> "DataLoader":
        """
        Build DataLoader for IOI task using builtin data generation.

        Unlike GenericTaskSpec, IOI works directly with tokens:
        1. Loads IOI data (tokens + labels)
        2. Converts tokens to text for caching
        3. Returns EAP-format dataloader

        Args:
            model: HookedTransformer model
            discovery_cfg: Discovery configuration
            device: Target device

        Returns:
            DataLoader configured for IOI task
        """

        if model is None:
            raise ValueError("IOI task requires model for tokenizer. No default model.")

        # IOI contrastive pairs are built directly from generated token
        # tensors (get_ioi_data_only) -- there is no single prompt-string
        # finalization point to wrap. Default "off" keeps everything raw;
        # an explicit override fails loudly rather than being ignored.
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

        if algorithm == "acdc":
            from ...backends.acdc.data import load_task_data

            train_loader, _ = load_task_data(
                task_name="ioi", model=model, device=device, **discovery_cfg.get("data_params", {})
            )
            return train_loader

        elif is_eap_family(algorithm) or algorithm == "cdt":
            # CD-T also consumes the paired text dataloader; only the
            # downstream attribution differs.
            return self._build_eap_dataloader(discovery_cfg, device, model)

        elif algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)

        else:
            raise ValueError(
                unsupported_algorithm_message(
                    "IOI task", algorithm, EAP_FAMILY | ACDC_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

    def _build_eap_dataloader(
        self, discovery_cfg: Dict[str, Any], device: str, model
    ) -> "DataLoader":
        """Build DataLoader for EAP/EAP-IG using data-only generation with caching."""
        import pandas as pd

        from ...data.task_data.tasks.ioi.utils import get_ioi_data_only

        # Get configuration parameters
        data_params = discovery_cfg.get("data_params", {})
        num_examples = data_params.get("num_examples", 128)
        seed = data_params.get("seed", 42)
        cache_dir = Path(data_params.get("cache_dir", "./cache/ioi"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = getattr(model.cfg, "model_name", "unknown_model").replace("/", "_")
        cache_file = cache_dir / f"ioi_{model_name}_{num_examples}_seed{seed}.csv"

        if cache_file.exists():
            logger.info(f"Loading cached IOI data for {model_name} from {cache_file}")
            dataset = EAPDiscoveryDataset(str(cache_file))
        else:
            logger.info(
                f"Generating new IOI data for {model_name} (samples={num_examples}, seed={seed})"
            )
            ioi_data = get_ioi_data_only(
                num_examples=num_examples, device=device, model=model, seed=seed
            )

            eap_data = []
            for i in range(len(ioi_data["validation_data"])):
                clean_tokens = ioi_data["validation_data"][i]
                corrupted_tokens = ioi_data["validation_patch_data"][i]
                correct_idx = ioi_data["validation_labels"][i].item()
                incorrect_idx = ioi_data["validation_wrong_labels"][i].item()

                clean_text = model.to_string(clean_tokens)
                corrupted_text = model.to_string(corrupted_tokens)

                eap_data.append(
                    {
                        "clean": clean_text,
                        "corrupted": corrupted_text,
                        "correct_idx": correct_idx,
                        "incorrect_idx": incorrect_idx,
                    }
                )

            df = pd.DataFrame(eap_data)
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved generated data to cache: {cache_file}")
            dataset = EAPDiscoveryDataset(str(cache_file))

        from torch.utils.data import DataLoader

        from ...backends.eap.eap_utils import collate_EAP

        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        g = t.Generator()
        g.manual_seed(seed)
        dl = DataLoader(
            dataset,
            batch_size=discovery_cfg.get("batch_size", 16),
            shuffle=True,
            collate_fn=collate_EAP,
            generator=g,
        )
        dl.pair_padding_side = side

        logger.debug(
            f"[DEBUG PADDING] ioi EAP dataloader pair_padding_side='{side}' batch_size={discovery_cfg.get('batch_size', 16)}"
        )
        return dl

    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """
        Build DataLoader for IBCircuit using IOI data.

        IBCircuit trains on a fixed batch for all epochs (by design), so this
        returns a simple iterator that yields the same batch repeatedly.
        """
        from ...data.task_data.tasks.ioi.utils import get_ioi_data_only

        data_params = discovery_cfg.get("data_params", {})
        num_examples = data_params.get("num_examples", 100)
        seed = data_params.get("seed", 42)

        logger.info(f"Generating IOI data for IBCircuit (samples={num_examples}, seed={seed})")

        ioi_data = get_ioi_data_only(
            num_examples=num_examples, device=device, model=model, seed=seed
        )
        tokens = ioi_data["validation_data"]
        labels = ioi_data["validation_labels"]

        if "end_idxs" in ioi_data:
            answer_positions = ioi_data["end_idxs"]
            if not ((answer_positions >= 0).all() and (answer_positions < tokens.shape[1]).all()):
                logger.warning("IOIDataset end_idxs invalid, computing model-agnostic positions")
                answer_positions = None
            else:
                logger.info(
                    f"Using IOIDataset end_idxs: min={answer_positions.min().item()}, max={answer_positions.max().item()}"
                )
        else:
            answer_positions = None

        if answer_positions is None:
            logger.info("Computing model-agnostic answer positions (last non-padding token)")
            answer_positions = self._compute_answer_positions(tokens, model)

        seq_len = tokens.shape[1]
        logger.info(f"IOI data prepared: {num_examples} examples, seq_len={seq_len}")
        logger.debug(
            f"[DEBUG PADDING] ioi IBCircuit within-batch=right-padded max_len={seq_len} answer_pos range=[{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )

        batch = {"tokens": tokens, "labels": labels, "answer_positions": answer_positions}

        class SingleBatchDataLoader:
            """DataLoader that yields a single fixed batch repeatedly."""

            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

    def _compute_answer_positions(self, tokens: t.Tensor, model) -> t.Tensor:
        """Compute answer positions as the last non-padding token for each sequence."""
        batch_size, seq_len = tokens.shape

        pad_token_id = (
            model.tokenizer.pad_token_id if hasattr(model.tokenizer, "pad_token_id") else 0
        )
        if pad_token_id is None:
            pad_token_id = 0

        answer_positions = t.zeros(batch_size, dtype=t.long, device=tokens.device)

        for i in range(batch_size):
            non_pad_mask = tokens[i] != pad_token_id
            non_pad_positions = t.where(non_pad_mask)[0]
            answer_positions[i] = (
                non_pad_positions[-1] if len(non_pad_positions) > 0 else seq_len - 1
            )

        return answer_positions

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """Load IOI finetuning data from the discovery cache."""
        cfg = discovery_cfg or {}
        model_name_safe = model_name.replace("/", "_")
        cache_dir = Path(cfg.get("cache_dir", "./cache/ioi"))

        cache_path = _find_task_cache(cache_dir, self.name, model_name_safe)
        if cache_path is None:
            raise FileNotFoundError(
                f"No '{self.name}' cache found for model '{model_name}' "
                f"in {cache_dir}.\n"
                f"Run discover_circuit with task='{self.name}' first."
            )
        return _load_finetuning_data_from_csv(cache_path, tokenizer, n_examples, seed)

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return the EAP/EAP-IG compatible metric for IOI."""
        if metric_type == "logit_diff":
            return partial(self._ioi_logit_diff, loss=True, mean=True)
        elif metric_type == "kl":
            return partial(self._ioi_kl_divergence, loss=True, mean=True)
        else:
            raise ValueError(
                f"IOI does not support metric_type {metric_type!r}. "
                f"Use one of: 'logit_diff', 'kl'."
            )

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for IOI artifacts."""
        return {
            "task": "ioi",
            "data_source": "built_in_generation",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "num_examples": discovery_cfg.get("data_params", {}).get("num_examples", 128),
            "template_type": "ABBA",
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }

    @staticmethod
    def _ioi_logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """
        IOI metric: logit difference between correct and incorrect tokens.

        Computes: logit(correct_token) - logit(incorrect_token)

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            clean_logits: Clean run logits (for reference)
            input_length: Length of input sequence
            labels: Label tensor [batch_size, 2] where:
                labels[:, 0] = correct token idx (IO - indirect object)
                labels[:, 1] = incorrect token idx (S - subject)
            mean: Whether to return mean across batch
            loss: Whether to negate (for minimization)

        Returns:
            Tensor of logit differences, optionally meaned
        """
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)

        # Get logits at answer position (input_length - 1)
        logits = logits[idx, input_length - 1]

        # Gather logits for correct and incorrect tokens
        # labels[:, 0] = IO (correct), labels[:, 1] = S (incorrect)
        good_bad = t.gather(logits, -1, labels.to(logits.device))
        results = good_bad[:, 0] - good_bad[:, 1]

        if loss:
            results = -results  # Negate for minimization

        if mean:
            results = results.mean()

        return results

    @staticmethod
    def _ioi_kl_divergence(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """
        IOI metric: KL divergence between patched and clean distributions.

        Args:
            logits: Patched model output logits [batch_size, seq_len, vocab_size]
            clean_logits: Clean run logits [batch_size, seq_len, vocab_size]
            input_length: Length of input sequence
            labels: Label tensor (unused for KL)
            mean: Whether to return mean across batch
            loss: Whether to return as loss (default True)

        Returns:
            Tensor of KL divergences, optionally meaned
        """
        import torch.nn.functional as F

        # KL between patched and clean distributions at answer position
        results = (
            F.kl_div(
                F.log_softmax(logits, dim=-1), F.softmax(clean_logits, dim=-1), reduction="none"
            )
            .sum(-1)
            .mean(-1)
        )

        if mean:
            results = results.mean()

        return results


class IOITaskSpecLegacy:
    """
    Legacy IOI task implementation (deprecated).

    This is the original ~392-line hardcoded implementation.
    Use IOITaskSpec instead (generic version).

    Warning:
        IOITaskSpecLegacy is deprecated and will be removed in v0.3.
        Please use IOITaskSpec (the new generic version) instead.
    """

    def __init__(self):
        warnings.warn(
            "IOITaskSpecLegacy is deprecated. Use IOITaskSpec instead.\n"
            "IOITaskSpec is a thin wrapper on GenericTaskSpec that provides the same "
            "functionality with less code and better maintainability.\n"
            "IOITaskSpecLegacy will be removed in CircuitKit v0.3.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import and initialize the legacy implementation
        from .ioi_legacy import IOITaskSpec as _LegacyIOITaskSpec

        self._impl = _LegacyIOITaskSpec()

    def __getattr__(self, name):
        """Delegate all attribute access to legacy implementation."""
        return getattr(self._impl, name)
