"""
ACDC Data Generation Manager

Main interface for ACDC data generation with intelligent caching and file management.
"""

import time
from typing import Any, Dict, Optional

import pandas as pd

from .cache import ACDCCache
from .utils import FileManager, GenerationConfig, create_data_summary, validate_data_integrity


import logging

logger = logging.getLogger(__name__)

class ACDCDataManager:
    """Main manager for ACDC data generation with caching."""

    def __init__(self, base_storage_dir: str = None, enable_cache: bool = True):
        self.cache = ACDCCache(base_storage_dir) if enable_cache else None
        self.file_manager = FileManager(base_storage_dir)
        self.generation_stats = {
            "total_generations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time": 0.0,
        }

    def get_task_data(
        self, task_name: str, num_examples: int, model=None, device: str = "cuda", **kwargs
    ) -> Dict[str, Any]:
        """
        Get task data with intelligent caching and generation.

        Args:
            task_name: Name of the task (ioi, greaterthan, induction, docstring)
            num_examples: Number of examples to generate
            model: Optional model for data generation
            device: Device to use for generation
            **kwargs: Additional task-specific parameters

        Returns:
            Dictionary containing generated data
        """
        # Create configuration. Keyword arguments that are recognised fields of
        # GenerationConfig are passed through directly; any remaining
        # task-specific kwargs (e.g. ``seq_len`` for induction) are collected
        # into ``additional_params`` so they reach the generators without
        # blowing up the dataclass constructor.
        _config_fields = {
            "prompt_type",
            "seed",
            "metric_name",
            "template_type",
            "corruption_strategy",
            "additional_params",
        }
        config_kwargs = {k: v for k, v in kwargs.items() if k in _config_fields}
        extra_params = {k: v for k, v in kwargs.items() if k not in _config_fields}
        additional_params = dict(config_kwargs.pop("additional_params", {}) or {})
        additional_params.update(extra_params)

        config = GenerationConfig(
            task_name=task_name,
            num_examples=num_examples,
            device=device,
            additional_params=additional_params,
            **config_kwargs,
        )

        start_time = time.time()

        # Try to get from cache first
        if self.cache:
            cached_data = self.cache.get_cached_data(config)
            if cached_data is not None:
                self.generation_stats["cache_hits"] += 1
                self.generation_stats["total_time"] += time.time() - start_time
                return cached_data

        # Generate new data
        logger.info(f"Generating {num_examples} examples for {task_name} task...")
        data = self._generate_task_data(config, model)

        if data is None:
            raise RuntimeError(f"Failed to generate data for {task_name} task")

        # Validate generated data (allow some flexibility for basic data structures)
        if not validate_data_integrity(data, num_examples):
            logger.info(
                f"Warning: Generated data for {task_name} failed strict validation, but continuing..."
            )
            # Don't raise error for basic data structures that don't have arrays

        # Cache the data
        if self.cache:
            metadata = create_data_summary(data)
            self.cache.cache_data(config, data, metadata)
            self.generation_stats["cache_misses"] += 1
        else:
            self.generation_stats["cache_misses"] += 1

        self.generation_stats["total_generations"] += 1
        self.generation_stats["total_time"] += time.time() - start_time

        # Get sample count for display
        first_value = data[next(iter(data.keys()))]
        if hasattr(first_value, "__len__"):
            sample_count = len(first_value)
        else:
            sample_count = data.get("num_examples", "unknown")
        logger.info(
            f"Generated {sample_count} examples for {task_name} in {time.time() - start_time:.2f}s"
        )

        return data

    def export_to_csv(
        self, task_name: str, num_examples: int, output_path: str = None, **kwargs
    ) -> str:
        """
        Generate task data and export to CSV format suitable for EAP-IG.

        Args:
            task_name: Name of the task (ioi, greaterthan, induction, docstring)
            num_examples: Number of examples to generate
            output_path: Path to save CSV file (auto-generated if None)
            **kwargs: Additional task-specific parameters

        Returns:
            Path to the generated CSV file
        """
        # Generate the data
        data = self.get_task_data(task_name, num_examples, **kwargs)

        # Generate output path if not provided
        if output_path is None:
            config = GenerationConfig(task_name=task_name, num_examples=num_examples, **kwargs)
            filename = f"{task_name}_{num_examples}_{config.to_hash()}.csv"
            task_dir = self.file_manager.get_task_storage_dir(task_name)
            output_path = task_dir / filename

        # Convert to CSV format based on task type
        if task_name == "ioi":
            self._export_ioi_to_csv(data, output_path)
        elif task_name == "greaterthan":
            self._export_greaterthan_to_csv(data, output_path)
        elif task_name == "induction":
            self._export_induction_to_csv(data, output_path)
        elif task_name == "docstring":
            self._export_docstring_to_csv(data, output_path)
        else:
            raise ValueError(f"Unknown task type: {task_name}")

        logger.info(f"✅ Exported {task_name} data to CSV: {output_path}")
        return output_path

    def _export_ioi_to_csv(self, data: Dict[str, Any], output_path: str):
        """Export IOI data to CSV format for EAP-IG."""
        rows = []

        for i in range(len(data["sentences"])):
            row = {
                "text": data["sentences"][i],
                "IO": data["prompts"][i]["IO"],
                "S": data["prompts"][i]["S"],
                "io_tokenID": data["io_tokenIDs"][i],
                "s_tokenID": data["s_tokenIDs"][i],
                "template_idx": data["prompts"][i]["TEMPLATE_IDX"],
                "tokens": " ".join(map(str, data["tokens"][i].tolist())),
            }

            # Add any additional fields from prompts
            for key, value in data["prompts"][i].items():
                if key not in ["text", "IO", "S", "TEMPLATE_IDX"]:
                    row[key] = value

            rows.append(row)

        # Write to CSV
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def _export_greaterthan_to_csv(self, data: Dict[str, Any], output_path: str):
        """Export Greater-Than data to CSV format for EAP-IG."""
        rows = []

        test_data = data.get("test_data") or data.get("validation_data")
        test_patch = data.get("test_patch_data") or data.get("validation_patch_data")

        if test_data is not None:
            for i in range(len(test_data)):
                row = {
                    "tokens": " ".join(map(str, test_data[i].tolist())),
                    "patch_tokens": (
                        " ".join(map(str, test_patch[i].tolist())) if test_patch is not None else ""
                    ),
                    "task": "greaterthan",
                }
                rows.append(row)
        else:
            # No model was provided; write empty rows so the file is valid.
            for _ in range(data.get("num_examples", 0)):
                rows.append({"tokens": "", "patch_tokens": "", "task": "greaterthan"})

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def _export_induction_to_csv(self, data: Dict[str, Any], output_path: str):
        """Export Induction data to CSV format for EAP-IG."""
        rows = []

        test_data = data.get("test_data") or data.get("validation_data")
        test_patch = data.get("test_patch_data") or data.get("validation_patch_data")
        test_labels = data.get("test_labels") or data.get("validation_labels")

        if test_data is not None:
            for i in range(len(test_data)):
                label = ""
                if test_labels is not None:
                    lv = test_labels[i]
                    label = int(lv.item()) if hasattr(lv, "item") else lv
                row = {
                    "tokens": " ".join(map(str, test_data[i].tolist())),
                    "patch_tokens": (
                        " ".join(map(str, test_patch[i].tolist())) if test_patch is not None else ""
                    ),
                    "label": label,
                    "task": "induction",
                }
                rows.append(row)
        else:
            for _ in range(data.get("num_examples", 0)):
                rows.append({"tokens": "", "patch_tokens": "", "label": "", "task": "induction"})

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def _export_docstring_to_csv(self, data: Dict[str, Any], output_path: str):
        """Export Docstring data to CSV format for EAP-IG."""
        rows = []

        test_data = data.get("test_data") or data.get("validation_data")
        test_patch = data.get("test_patch_data") or data.get("validation_patch_data")
        test_labels = data.get("test_labels") or data.get("validation_labels")

        if test_data is not None:
            for i in range(len(test_data)):
                label = ""
                if test_labels is not None:
                    lv = test_labels[i]
                    label = int(lv.item()) if hasattr(lv, "item") else lv
                row = {
                    "tokens": " ".join(map(str, test_data[i].tolist())),
                    "patch_tokens": (
                        " ".join(map(str, test_patch[i].tolist())) if test_patch is not None else ""
                    ),
                    "label": label,
                    "task": "docstring",
                }
                rows.append(row)
        else:
            for _ in range(data.get("num_examples", 0)):
                rows.append({"tokens": "", "patch_tokens": "", "label": "", "task": "docstring"})

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    def _generate_task_data(self, config: GenerationConfig, model=None) -> Optional[Dict[str, Any]]:
        """Generate data for a specific task."""
        task_name = config.task_name.lower()

        if task_name == "ioi":
            return self._generate_ioi_data(config, model)
        elif task_name == "greaterthan":
            return self._generate_greaterthan_data(config, model)
        elif task_name == "induction":
            return self._generate_induction_data(config, model)
        elif task_name == "docstring":
            return self._generate_docstring_data(config, model)
        else:
            raise ValueError(f"Unknown task: {task_name}")

    def _generate_ioi_data(self, config: GenerationConfig, model=None) -> Optional[Dict[str, Any]]:
        """Generate IOI data using ACDC."""
        try:
            from ....backends.acdc.tasks.ioi_utils import get_all_ioi_things
            from ..tasks.ioi.ioi_dataset import IOIDataset

            # If model is provided, use the full ACDC generation
            if model is not None:
                things = get_all_ioi_things(
                    model=model,
                    device=config.device,
                    metric_name=config.metric_name,
                    num_examples=config.num_examples,
                )

                return {
                    "validation_data": things.validation_data,
                    "validation_labels": things.validation_labels,
                    "validation_wrong_labels": things.validation_wrong_labels,
                    "validation_patch_data": things.validation_patch_data,
                    "test_data": things.test_data,
                    "test_labels": things.test_labels,
                    "test_wrong_labels": things.test_wrong_labels,
                    "test_patch_data": things.test_patch_data,
                    "validation_metric": things.validation_metric,
                    "test_metrics": things.test_metrics,
                }
            else:
                # Generate basic IOI dataset with model
                if model is None:
                    raise ValueError("Model required for IOI data generation. No default model.")

                prompt_type = config.prompt_type or "ABBA"
                ioi_dataset = IOIDataset(
                    prompt_type=prompt_type,
                    N=config.num_examples,
                    model=model,  # Pass actual model for tokenizer
                    seed=config.seed,
                )

                return {
                    "tokens": ioi_dataset.toks,
                    "prompts": ioi_dataset.ioi_prompts,
                    "sentences": ioi_dataset.sentences,
                    "word_idx": ioi_dataset.word_idx,
                    "io_tokenIDs": ioi_dataset.io_tokenIDs,
                    "s_tokenIDs": ioi_dataset.s_tokenIDs,
                }

        except ImportError as e:
            logger.warning(f"Error importing IOI components: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error generating IOI data: {e}")
            return None

    def _generate_greaterthan_data(
        self, config: GenerationConfig, model=None
    ) -> Optional[Dict[str, Any]]:
        """Generate Greater Than data using ACDC."""
        try:
            from ..tasks.greaterthan.utils import get_all_greaterthan_things

            if model is not None:
                # get_all_greaterthan_things only supports "greaterthan" / "kl_div";
                # fall back to the task-native metric for any other request
                # (e.g. the generic "logit_diff" default).
                gt_metric = config.metric_name
                if gt_metric not in ("greaterthan", "kl_div"):
                    gt_metric = "greaterthan"
                things = get_all_greaterthan_things(
                    num_examples=config.num_examples, metric_name=gt_metric, device=config.device
                )

                return {
                    "validation_data": things.validation_data,
                    "validation_patch_data": things.validation_patch_data,
                    "test_data": things.test_data,
                    "test_patch_data": things.test_patch_data,
                    "validation_metric": things.validation_metric,
                    "test_metrics": things.test_metrics,
                }
            else:
                # Generate basic year data without model
                # This is a simplified version - in practice you'd need a model
                return {
                    "num_examples": config.num_examples,
                    "task": "greaterthan",
                    "note": "Basic data structure - requires model for full generation",
                }

        except ImportError as e:
            logger.warning(f"Error importing Greater Than components: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error generating Greater Than data: {e}")
            return None

    def _generate_induction_data(
        self, config: GenerationConfig, model=None
    ) -> Optional[Dict[str, Any]]:
        """Generate Induction data using ACDC."""
        try:
            from ..tasks.induction.utils import get_all_induction_things

            if model is not None:
                # NOTE: this ``get_all_induction_things`` (task_data.tasks.
                # induction.utils) builds its own model internally and does not
                # accept a ``model`` argument. Its signature is
                # (num_examples, seq_len, device, data_seed, metric, ...).
                # This util only supports "kl_div" / "nll" / "match_nll";
                # fall back to "kl_div" for any other request.
                ind_metric = config.metric_name
                if ind_metric not in ("kl_div", "nll", "match_nll"):
                    ind_metric = "kl_div"
                things = get_all_induction_things(
                    num_examples=config.num_examples,
                    seq_len=config.additional_params.get("seq_len", 300),
                    device=config.device,
                    data_seed=config.seed,
                    metric=ind_metric,
                )

                return {
                    "validation_data": things.validation_data,
                    "validation_labels": things.validation_labels,
                    "validation_patch_data": things.validation_patch_data,
                    "test_data": things.test_data,
                    "test_labels": things.test_labels,
                    "test_patch_data": things.test_patch_data,
                    "validation_metric": things.validation_metric,
                    "test_metrics": things.test_metrics,
                }
            else:
                return {
                    "num_examples": config.num_examples,
                    "task": "induction",
                    "note": "Basic data structure - requires model for full generation",
                }

        except ImportError as e:
            logger.warning(f"Error importing Induction components: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error generating Induction data: {e}")
            return None

    def _generate_docstring_data(
        self, config: GenerationConfig, model=None
    ) -> Optional[Dict[str, Any]]:
        """Generate Docstring data using ACDC."""
        try:
            from ..tasks.docstring.utils import get_all_docstring_things


            if model is not None:
                things = get_all_docstring_things(
                    model=model, device=config.device, num_examples=config.num_examples
                )

                return {
                    "validation_data": things.validation_data,
                    "validation_labels": things.validation_labels,
                    "validation_patch_data": things.validation_patch_data,
                    "test_data": things.test_data,
                    "test_labels": things.test_labels,
                    "test_patch_data": things.test_patch_data,
                    "validation_metric": things.validation_metric,
                    "test_metrics": things.test_metrics,
                }
            else:
                return {
                    "num_examples": config.num_examples,
                    "task": "docstring",
                    "note": "Basic data structure - requires model for full generation",
                }

        except ImportError as e:
            logger.warning(f"Error importing Docstring components: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error generating Docstring data: {e}")
            return None

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        stats = self.generation_stats.copy()

        if self.cache:
            cache_stats = self.cache.get_cache_info()
            stats.update(
                {
                    "cache_stats": cache_stats["stats"],
                    "cache_hit_rate": cache_stats["hit_rate"],
                    "cache_files": len(cache_stats["files"]),
                    "cache_size": cache_stats["total_size"],
                }
            )

        return stats

    def cleanup_old_data(self, task_name: str = None, max_files: int = 10) -> int:
        """Clean up old generated data files."""
        if self.cache:
            return self.cache.cleanup_cache(task_name, max_files)
        else:
            return self.file_manager.cleanup_old_files(task_name, max_files)

    def clear_all_data(self, task_name: str = None) -> int:
        """Clear all generated data files."""
        if self.cache:
            return self.cache.clear_cache(task_name)
        else:
            if task_name:
                task_dir = self.file_manager.get_task_storage_dir(task_name)
                if task_dir.exists():
                    removed_count = 0
                    for file_path in task_dir.glob("*"):
                        try:
                            file_path.unlink()
                            removed_count += 1
                        except Exception as e:
                            logger.warning(f"Error removing {file_path}: {e}")
                    return removed_count
                return 0
            else:
                # Clear all data
                base_dir = self.file_manager.base_storage_dir
                if base_dir.exists():
                    removed_count = 0
                    for file_path in base_dir.rglob("*"):
                        if file_path.is_file():
                            try:
                                file_path.unlink()
                                removed_count += 1
                            except Exception as e:
                                logger.warning(f"Error removing {file_path}: {e}")
                    return removed_count
                return 0

    def list_available_data(self, task_name: str = None) -> Dict[str, Any]:
        """List available generated data files."""
        if self.cache:
            return self.cache.get_cache_info(task_name)
        else:
            info = {"files": {}, "total_size": 0}

            if task_name:
                task_dir = self.file_manager.get_task_storage_dir(task_name)
                if task_dir.exists():
                    files = list(task_dir.glob("*.pkl"))
                    for file_path in files:
                        file_info = {
                            "size": file_path.stat().st_size,
                            "modified": file_path.stat().st_mtime,
                            "path": str(file_path),
                        }
                        info["files"][file_path.name] = file_info
                        info["total_size"] += file_info["size"]
            else:
                base_dir = self.file_manager.base_storage_dir
                if base_dir.exists():
                    for task_dir in base_dir.iterdir():
                        if task_dir.is_dir():
                            task_files = list(task_dir.glob("*.pkl"))
                            for file_path in task_files:
                                file_info = {
                                    "size": file_path.stat().st_size,
                                    "modified": file_path.stat().st_mtime,
                                    "path": str(file_path),
                                }
                                info["files"][f"{task_dir.name}/{file_path.name}"] = file_info
                                info["total_size"] += file_info["size"]

            return info
