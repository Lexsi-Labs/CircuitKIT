"""
IOI (Indirect Object Identification) Task Specification - ACDC Version

Implements the TaskSpec interface for the IOI task using ACDC data generation
with intelligent caching and file management.
"""

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Union

import torch as t
from torch.utils.data import DataLoader

from ...api import _eap_logit_diff
from ...data.task_data.generation import ACDCDataManager
from .._chat import resolve_chat_template


import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .ioi import IOITaskSpec


class IOITaskSpecACDC:
    """TaskSpec implementation for IOI using ACDC data generation."""

    name = "ioi"
    # diagnostic minimal-pair task -- discovered raw, per the circuit-discovery literature
    chat_template_mode: str = "off"

    def __init__(self, enable_cache: bool = True, storage_dir: str = None):
        """
        Initialize IOI task specification with ACDC data generation.

        Args:
            enable_cache: Whether to enable intelligent caching
            storage_dir: Custom storage directory for generated data
        """
        self.data_manager = ACDCDataManager(base_storage_dir=storage_dir, enable_cache=enable_cache)

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """
        Validate IOI-specific discovery configuration.

        Args:
            discovery_cfg: Discovery configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        algorithm = discovery_cfg.get("algorithm", "").lower()

        if algorithm in ["eap", "eap-ig", "acdc"]:
            # For predefined tasks, we use ACDC data generation
            # data_path is optional and will be ignored in favor of ACDC generation

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

            # Validate num_examples if present
            num_examples = discovery_cfg.get("num_examples", 128)
            if not isinstance(num_examples, int) or num_examples <= 0:
                raise ValueError(
                    f"IOI task has invalid 'num_examples': {num_examples!r}. "
                    f"Set discovery config key 'num_examples' to a positive integer (e.g. 128)."
                )

            # Validate prompt_type if present
            prompt_type = discovery_cfg.get("prompt_type", "ABBA")
            valid_prompt_types = ["ABBA", "BABA", "mixed", "ABC", "BAC", "ABC mixed"]
            if prompt_type not in valid_prompt_types:
                raise ValueError(
                    f"IOI task has invalid 'prompt_type': {prompt_type!r}. "
                    f"Set discovery config key 'prompt_type' to one of: "
                    f"{', '.join(valid_prompt_types)}."
                )

            # Validate seed if present
            seed = discovery_cfg.get("seed", 42)
            if not isinstance(seed, int):
                raise ValueError(
                    f"IOI task has invalid 'seed': {seed!r}. "
                    f"Set discovery config key 'seed' to an integer (e.g. 42)."
                )

        else:
            if not algorithm:
                raise ValueError(
                    "IOI task discovery config is missing the required key "
                    "'algorithm'. Add 'algorithm' to the discovery config. "
                    "IOI (ACDC backend) supports: eap, eap-ig, acdc."
                )
            raise ValueError(
                f"IOI task does not support algorithm '{algorithm}'. "
                f"Set discovery config key 'algorithm' to one of: eap, eap-ig, acdc."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> DataLoader:
        """
        Build DataLoader for IOI task using ACDC data generation.

        Args:
            model: HookedTransformer model
            discovery_cfg: Discovery configuration
            device: Target device

        Returns:
            DataLoader configured for IOI task
        """
        # IOI/ACDC contrastive pairs are built directly from generated token
        # tensors (ACDCDataManager) -- there is no single prompt-string
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

        if algorithm in ["eap", "eap-ig", "acdc"]:
            return self._build_acdc_dataloader(model, discovery_cfg, device)
        else:
            if not algorithm:
                raise ValueError(
                    "IOI task discovery config is missing the required key "
                    "'algorithm'. Add 'algorithm' to the discovery config. "
                    "IOI (ACDC backend) supports: eap, eap-ig, acdc."
                )
            raise ValueError(
                f"IOI task does not support algorithm '{algorithm}'. "
                f"Set discovery config key 'algorithm' to one of: eap, eap-ig, acdc."
            )

    def _build_acdc_dataloader(
        self, model, discovery_cfg: Dict[str, Any], device: str
    ) -> DataLoader:
        """Build DataLoader using ACDC data generation."""

        # Extract configuration parameters
        num_examples = discovery_cfg.get("num_examples", 128)
        prompt_type = discovery_cfg.get("prompt_type", "ABBA")
        seed = discovery_cfg.get("seed", 42)
        metric_name = discovery_cfg.get("metric_name", "logit_diff")

        # Check if CSV export is requested
        if discovery_cfg.get("export_csv", False):
            csv_path = self.data_manager.export_to_csv(
                task_name="ioi", num_examples=num_examples, prompt_type=prompt_type, seed=seed
            )
            logger.info(f"📄 IOI CSV exported to: {csv_path}")

        # Generate data using ACDC
        data = self.data_manager.get_task_data(
            task_name="ioi",
            num_examples=num_examples,
            model=model,
            device=device,
            prompt_type=prompt_type,
            seed=seed,
            metric_name=metric_name,
        )

        # Convert to DataLoader format
        if "validation_data" in data:
            # Full ACDC format with model
            return self._convert_acdc_to_dataloader(data, discovery_cfg, device)
        else:
            # Basic IOI dataset format
            return self._convert_ioi_dataset_to_dataloader(data, discovery_cfg, device)

    def _convert_acdc_to_dataloader(
        self, data: Dict[str, Any], discovery_cfg: Dict[str, Any], device: str
    ) -> DataLoader:
        """Convert ACDC data format to DataLoader."""
        from ...backends.acdc.data import PromptDataLoader, PromptDataset

        # Extract data components
        validation_data = data["validation_data"]
        validation_patch_data = data["validation_patch_data"]
        validation_labels = data["validation_labels"]
        validation_wrong_labels = data["validation_wrong_labels"]

        # Create dataset
        dataset = PromptDataset(
            clean_prompts=validation_data,
            corrupt_prompts=validation_patch_data,
            answers=validation_labels,
            wrong_answers=validation_wrong_labels,
        )

        # Create DataLoader
        batch_size = discovery_cfg.get("batch_size", 4)
        return PromptDataLoader(dataset, device=device, batch_size=batch_size, shuffle=False)

    def _convert_ioi_dataset_to_dataloader(
        self, data: Dict[str, Any], discovery_cfg: Dict[str, Any], device: str
    ) -> DataLoader:
        """Convert IOI dataset format to DataLoader."""
        # For basic IOI dataset, we need to create a simple DataLoader
        # This is a simplified version - in practice you'd want more sophisticated conversion

        tokens = data.get("tokens")
        if tokens is None:
            raise ValueError("IOI dataset missing tokens data")

        # Create a simple dataset wrapper
        class SimpleIOIDataset:
            def __init__(self, tokens):
                self.tokens = tokens

            def __len__(self):
                return len(self.tokens)

            def __getitem__(self, idx):
                return self.tokens[idx]

        dataset = SimpleIOIDataset(tokens)
        batch_size = discovery_cfg.get("batch_size", 4)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: t.stack(x).to(device),
        )

    def metric_fn(self) -> Callable:
        """
        Return the metric function for IOI task.

        Returns:
            Metric function for EAP/EAP-IG scoring
        """
        return partial(_eap_logit_diff, loss=True, mean=True)

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
            "data_source": "acdc_generation",  # Indicates ACDC generation
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "num_examples": discovery_cfg.get("num_examples", 128),
            "prompt_type": discovery_cfg.get("prompt_type", "ABBA"),
            "seed": discovery_cfg.get("seed", 42),
            "metric_name": discovery_cfg.get("metric_name", "logit_diff"),
            "generation_stats": self.data_manager.get_generation_stats(),
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get data generation statistics."""
        return self.data_manager.get_generation_stats()

    def cleanup_old_data(self, max_files: int = 10) -> int:
        """Clean up old generated data files."""
        return self.data_manager.cleanup_old_data("ioi", max_files)

    def clear_all_data(self) -> int:
        """Clear all generated IOI data files."""
        return self.data_manager.clear_all_data("ioi")

    def list_available_data(self) -> Dict[str, Any]:
        """List available generated IOI data files."""
        return self.data_manager.list_available_data("ioi")


# Create a factory function for backward compatibility
def create_ioi_task_spec(use_acdc: bool = True, **kwargs) -> Union[IOITaskSpecACDC, "IOITaskSpec"]:
    """
    Factory function to create IOI task specification.

    Args:
        use_acdc: Whether to use ACDC data generation (default: True)
        **kwargs: Additional arguments for task specification

    Returns:
        IOI task specification instance
    """
    if use_acdc:
        return IOITaskSpecACDC(**kwargs)
    else:
        # Import the original IOI task spec for backward compatibility
        from .ioi import IOITaskSpec


        return IOITaskSpec(**kwargs)
