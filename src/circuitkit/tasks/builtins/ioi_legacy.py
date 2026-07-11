"""
IOI (Indirect Object Identification) Task Specification - LEGACY

DEPRECATED: this module is the historical IOI task implementation kept only
for backwards compatibility. The current canonical implementation lives in
``circuitkit.tasks.builtins.ioi`` (registered automatically via
``tasks.bootstrap``). New code MUST import ``IOITaskSpec`` from there.

This module will be removed in a future release.
"""

import warnings as _warnings

_warnings.warn(
    "circuitkit.tasks.builtins.ioi_legacy is deprecated; use "
    "circuitkit.tasks.builtins.ioi (IOITaskSpec) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from functools import partial  # noqa: E402 - import after intentional pre-import setup
from pathlib import Path  # noqa: E402 - import after intentional pre-import setup
from typing import (  # noqa: E402 - import after intentional pre-import setup
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import pandas as pd  # noqa: E402 - import after intentional pre-import setup
import torch as t  # noqa: E402 - import after intentional pre-import setup
from torch.utils.data import DataLoader  # noqa: E402 - import after intentional pre-import setup

from ...data.eap_dataset import EAPDiscoveryDataset  # noqa: E402 - import after intentional pre-import setup
from ...utils.logging import get_logger  # noqa: E402 - import after intentional pre-import setup
from .._chat import resolve_chat_template  # noqa: E402 - import after intentional pre-import setup
from ..specs import (  # noqa: E402 - import after intentional pre-import setup
    _find_task_cache,
    _load_finetuning_data_from_csv,
)

logger = get_logger("task.ioi_legacy")


class IOITaskSpec:
    """TaskSpec implementation for IOI (Indirect Object Identification)."""

    name = "ioi"
    pair_padding_side = "right"
    # diagnostic minimal-pair task -- discovered raw, per the circuit-discovery literature
    chat_template_mode: str = "off"

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """
        Validate IOI-specific discovery configuration.

        Args:
            discovery_cfg: Discovery configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        algorithm = discovery_cfg.get("algorithm", "").lower()

        valid_algorithms = ["eap", "eap-ig", "acdc", "ibcircuit"]
        if algorithm not in valid_algorithms:
            if not algorithm:
                raise ValueError(
                    "IOI task discovery config is missing the required key "
                    "'algorithm'. Add 'algorithm' to the discovery config. "
                    f"IOI supports: {', '.join(valid_algorithms)}."
                )
            raise ValueError(
                f"IOI task does not support algorithm '{algorithm}'. "
                f"Set discovery config key 'algorithm' to one of: "
                f"{', '.join(valid_algorithms)}."
            )

        # Validate level
        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"IOI task discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        # Validate batch_size if present
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
    ) -> DataLoader:
        """
        Build DataLoader for IOI task using built-in data generation.

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

        if algorithm == "acdc":
            # Use ACDC data loading
            from ...backends.acdc.data import load_task_data

            train_loader, _ = load_task_data(
                task_name="ioi", model=model, device=device, **discovery_cfg.get("data_params", {})
            )
            return train_loader

        elif algorithm in ["eap", "eap-ig"]:
            # Use data-only generation for EAP/EAP-IG
            return self._build_eap_dataloader(discovery_cfg, device, model)

        elif algorithm == "ibcircuit":
            # Use IBCircuit training
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)

        else:
            raise ValueError(
                f"IOI task does not support algorithm '{algorithm}'. "
                f"Set discovery config key 'algorithm' to one of: "
                f"eap, eap-ig, acdc, ibcircuit."
            )

    def _build_eap_dataloader(
        self, discovery_cfg: Dict[str, Any], device: str, model
    ) -> DataLoader:
        """Build DataLoader for EAP/EAP-IG using data-only generation with caching."""
        from pathlib import Path

        from ...data.task_data.tasks.ioi.utils import get_ioi_data_only

        if model is None:
            raise ValueError("Model required for IOI data generation. No default model.")

        # Get configuration parameters
        data_params = discovery_cfg.get("data_params", {})
        num_examples = data_params.get("num_examples", 128)
        seed = data_params.get("seed", 42)
        cache_dir = Path(data_params.get("cache_dir", "./cache/ioi"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path includes model_name along with num_examples and seed for uniqueness
        # This ensures that cached data is only reused for the exact same model
        model_name = getattr(model.cfg, "model_name", "unknown_model").replace("/", "_")
        cache_file = cache_dir / f"ioi_{model_name}_{num_examples}_seed{seed}.csv"

        # Check if cached data exists
        if cache_file.exists():
            logger.info(f"Loading cached IOI data for {model_name} from {cache_file}")
            # Load from cache
            dataset = EAPDiscoveryDataset(str(cache_file))
        else:
            logger.info(
                f"Generating new IOI data for {model_name} (samples={num_examples}, seed={seed})"
            )
            # Generate IOI data without metrics
            ioi_data = get_ioi_data_only(
                num_examples=num_examples, device=device, model=model, seed=seed
            )

            # Convert to EAP format
            eap_data = []
            for i in range(len(ioi_data["validation_data"])):
                clean_tokens = ioi_data["validation_data"][i]
                corrupted_tokens = ioi_data["validation_patch_data"][i]
                correct_idx = ioi_data["validation_labels"][i].item()
                incorrect_idx = ioi_data["validation_wrong_labels"][i].item()

                # Convert tokens back to text
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

            # Save to cache
            df = pd.DataFrame(eap_data)
            df.to_csv(cache_file, index=False)
            logger.info(f"Saved generated data to cache: {cache_file}")

            # Create dataset
            dataset = EAPDiscoveryDataset(str(cache_file))

        # Create dataloader
        from ...backends.eap.eap_utils import collate_EAP

        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)  # 'right'
        dl = DataLoader(
            dataset,
            batch_size=discovery_cfg.get("batch_size", 16),
            shuffle=True,
            collate_fn=collate_EAP,
        )
        dl.pair_padding_side = side

        logger.debug(
            f"[DEBUG PADDING] ioi EAP dataloader  pair_padding_side='{side}'  batch_size={discovery_cfg.get('batch_size', 16)}"
        )

        return dl

    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """
        Build DataLoader for IBCircuit using IOI data.

        IBCircuit trains on a fixed batch for all epochs (by design), so this
        returns a simple iterator that yields the same batch repeatedly.

        IOI Sequence Structure:
            - Full IOI prompt: 16 tokens (e.g., "When John and Mary ... to [IO]")
            - Input data: 15 tokens (excludes final IO token)
            - Label: Token 16 (the IO name, e.g., "Mary")
            - Answer position: Last non-padding token in the sequence

        Args:
            discovery_cfg: Discovery configuration
            device: Target device ('cuda' or 'cpu')
            model: HookedTransformer model (required for data generation)

        Returns:
            DataLoader-like object that yields fixed batch
        """

        from ...data.task_data.tasks.ioi.utils import get_ioi_data_only

        if model is None:
            raise ValueError("Model required for IOI data generation. No default model.")

        # Get configuration parameters
        data_params = discovery_cfg.get("data_params", {})
        num_examples = data_params.get("num_examples", 100)
        seed = data_params.get("seed", 42)

        logger.info(f"Generating IOI data for IBCircuit (samples={num_examples}, seed={seed})")

        # Try to generate IOI data with built-in end_idxs
        try:
            ioi_data = get_ioi_data_only(
                num_examples=num_examples, device=device, model=model, seed=seed
            )
            tokens = ioi_data["validation_data"]
            labels = ioi_data["validation_labels"]

            # Try to use provided end_idxs if they're valid
            if "end_idxs" in ioi_data:
                answer_positions = ioi_data["end_idxs"]
                # Validate end_idxs are reasonable
                if (answer_positions >= 0).all() and (answer_positions < tokens.shape[1]).all():
                    logger.info(
                        f"Using IOIDataset end_idxs: min={answer_positions.min().item()}, "
                        f"max={answer_positions.max().item()}"
                    )
                else:
                    logger.warning(
                        "IOIDataset end_idxs invalid, computing model-agnostic positions"
                    )
                    answer_positions = None
            else:
                answer_positions = None

        except (AssertionError, KeyError, ValueError) as e:
            logger.warning(
                f"IOIDataset generation failed ({type(e).__name__}: {str(e)}), "
                f"falling back to model-agnostic approach"
            )
            # Fallback: generate data without end_idxs computation
            # We'll compute positions ourselves below
            ioi_data = get_ioi_data_only(
                num_examples=num_examples, device=device, model=model, seed=seed
            )
            tokens = ioi_data["validation_data"]
            labels = ioi_data["validation_labels"]
            answer_positions = None

        # If answer_positions not available or invalid, compute model-agnostic version
        if answer_positions is None:
            logger.info("Computing model-agnostic answer positions (last non-padding token)")
            answer_positions = self._compute_answer_positions(tokens, model)

        seq_len = tokens.shape[1]
        logger.info(f"IOI data prepared: {num_examples} examples, seq_len={seq_len}")
        logger.info(
            f"Answer positions: min={answer_positions.min().item()}, "
            f"max={answer_positions.max().item()}, "
            f"mode={answer_positions.mode().values.item() if hasattr(answer_positions.mode(), 'values') else 'N/A'}"
        )
        logger.info("Predicting IO tokens (e.g., 'Mary' in 'When John and Mary ... to [Mary]')")

        # Create batch dictionary
        batch = {"tokens": tokens, "labels": labels, "answer_positions": answer_positions}

        logger.debug(
            f"[DEBUG PADDING] ioi IBCircuit  within-batch=right-padded  max_len={seq_len}  answer_pos range=[{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )
        logger.debug(f"[DEBUG PADDING]   tokens[0] tail: {model.to_str_tokens(tokens[0, -5:])}")

        # Create single-batch dataloader
        class SingleBatchDataLoader:
            """DataLoader that yields a single fixed batch repeatedly."""

            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

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

    def _compute_answer_positions(self, tokens: t.Tensor, model) -> t.Tensor:
        """
        Compute answer positions in a model-agnostic way.

        For IOI, we need to predict the indirect object at the end of the sequence.
        The answer position is the last non-padding token.

        Args:
            tokens: Token tensor [batch_size, seq_len]
            model: HookedTransformer model

        Returns:
            Answer positions [batch_size] - indices of last non-padding tokens
        """
        batch_size, seq_len = tokens.shape

        # Get pad token ID (0 for most models)
        pad_token_id = (
            model.tokenizer.pad_token_id if hasattr(model.tokenizer, "pad_token_id") else 0
        )
        if pad_token_id is None:
            pad_token_id = 0

        # Find last non-padding token for each sequence
        answer_positions = t.zeros(batch_size, dtype=t.long, device=tokens.device)

        for i in range(batch_size):
            # Find positions of non-padding tokens
            non_pad_mask = tokens[i] != pad_token_id
            non_pad_positions = t.where(non_pad_mask)[0]

            if len(non_pad_positions) > 0:
                # Last non-padding position
                answer_positions[i] = non_pad_positions[-1]
            else:
                # Fallback: use last position if all tokens are padding (shouldn't happen)
                answer_positions[i] = seq_len - 1

        return answer_positions

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return the EAP/EAP-IG compatible metric for IOI."""
        if metric_type == "logit_diff":
            return partial(self._eap_ioi_logit_diff, loss=True, mean=True)
        elif metric_type == "kl":
            return partial(self._eap_ioi_kl_divergence, loss=True, mean=True)
        else:
            raise ValueError(
                f"IOI does not support metric_type {metric_type!r}. "
                f"Use one of: 'logit_diff', 'kl'."
            )

    @staticmethod
    def _eap_ioi_logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)
        logits = logits[idx, input_length - 1]
        # labels[:, 0] = IO (correct), labels[:, 1] = S (incorrect)
        good_bad = t.gather(logits, -1, labels.to(logits.device))
        results = good_bad[:, 0] - good_bad[:, 1]
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    @staticmethod
    def _eap_ioi_kl_divergence(logits, clean_logits, input_length, labels, mean=True, loss=False):
        import torch.nn.functional as F

        # KL between patched and clean distributions
        results = (
            F.kl_div(
                F.log_softmax(logits, dim=-1), F.softmax(clean_logits, dim=-1), reduction="none"
            )
            .sum(-1)
            .mean(-1)
        )
        if loss:
            results = results
        if mean:
            results = results.mean()
        return results

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metadata for IOI artifacts.

        Args:
            discovery_cfg: Discovery configuration

        Returns:
            Dictionary containing IOI-specific metadata
        """
        return {
            "task": "ioi",
            "data_source": "built_in_generation",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "num_examples": discovery_cfg.get("data_params", {}).get("num_examples", 128),
            "template_type": "ABBA",
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }
