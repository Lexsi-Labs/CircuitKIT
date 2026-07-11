"""
YAML Task Loader

Enables declarative task definition via YAML files, allowing users to define
tasks without writing Python. Supports CSV, JSONL, and HuggingFace datasets
with schema mapping and corruption strategies.

Example YAML format:
    name: my_task
    source:
      type: csv
      path: data.csv
      split: test  # for hf only

    schema:
      prompt: question_col
      answer: label_col
      choices: options_col  # optional

    corruption:
      strategy: entity_swap
      config:
        entity_types: [PERSON, GPE]

    metric: logit_diff
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..corruption import (
    CorruptionPipeline,
    DistractorInjectionCorruption,
    EntitySwapCorruption,
    ParaphraseCorruption,
    RoleSwapCorruption,
    TokenSwapCorruption,
)
from ._chat import VALID_MODES
from .generic import GenericTaskSpec

logger = logging.getLogger(__name__)


class YAMLTaskLoader:
    """Load task specifications from YAML configuration files."""

    # Mapping of strategy names to corruption classes
    CORRUPTION_STRATEGIES = {
        "entity_swap": EntitySwapCorruption,
        "token_swap": TokenSwapCorruption,
        "paraphrase": ParaphraseCorruption,
        "distractor": DistractorInjectionCorruption,
        "role_swap": RoleSwapCorruption,
    }

    # Mapping of metric names to metric functions
    METRICS = {
        "logit_diff": "_eap_logit_diff",
        "accuracy": "_eap_accuracy",
        "kl": "_eap_kl_divergence",
    }

    @staticmethod
    def load(yaml_path: Path) -> GenericTaskSpec:
        """
        Load a GenericTaskSpec from a YAML configuration file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            GenericTaskSpec instance configured according to the YAML.

        Raises:
            FileNotFoundError: If YAML file does not exist.
            ValueError: If YAML format is invalid or required fields are missing.
            KeyError: If unknown strategy or metric is referenced.
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML task file not found: {yaml_path}")

        # Parse YAML
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError(f"YAML file is empty: {yaml_path}")

        # Validate required fields
        required_fields = ["name", "source", "schema"]
        for field in required_fields:
            if field not in config:
                raise ValueError(
                    f"YAML missing required field '{field}'. " f"Required fields: {required_fields}"
                )

        # Extract task name
        task_name = config["name"]
        if not isinstance(task_name, str):
            raise ValueError(
                f"YAML field 'name' must be a string, got {type(task_name).__name__}. "
                f"Set 'name' in the task YAML to a string task name."
            )

        # Parse source configuration
        source = YAMLTaskLoader._parse_source(config["source"])

        # Parse schema
        schema = YAMLTaskLoader._parse_schema(config["schema"])

        # Parse corruption strategy (optional)
        corruption_strategy = None
        if "corruption" in config:
            corruption_strategy = YAMLTaskLoader._parse_corruption(config["corruption"])

        # Parse metric function (optional, defaults to accuracy)
        metric_fn = None
        if "metric" in config:
            metric_fn = YAMLTaskLoader._get_metric_fn(config["metric"])

        # Parse prompt template (optional)
        prompt_template = config.get("prompt_template", None)

        # Parse chat-template policy (optional, defaults to "auto" for custom tasks)
        chat_template_mode = config.get("chat_template_mode", "auto")
        if chat_template_mode not in VALID_MODES:
            raise ValueError(
                f"chat_template_mode must be one of {VALID_MODES}, got: {chat_template_mode!r}"
            )

        # Create GenericTaskSpec based on source type
        source_type = source["type"]

        if source_type == "csv":
            task = GenericTaskSpec.from_csv(
                path=source["path_or_id"],
                schema=schema,
                corruption_strategy=corruption_strategy,
                metric_fn=metric_fn,
                prompt_template=prompt_template,
                name=task_name,
                chat_template_mode=chat_template_mode,
            )
        elif source_type == "jsonl":
            task = GenericTaskSpec.from_jsonl(
                path=source["path_or_id"],
                schema=schema,
                corruption_strategy=corruption_strategy,
                metric_fn=metric_fn,
                prompt_template=prompt_template,
                name=task_name,
                chat_template_mode=chat_template_mode,
            )
        elif source_type == "hf":
            task = GenericTaskSpec.from_huggingface(
                dataset_id=source["path_or_id"],
                schema=schema,
                corruption_strategy=corruption_strategy,
                metric_fn=metric_fn,
                prompt_template=prompt_template,
                split=source.get("split", "test"),
                name=task_name,
                chat_template_mode=chat_template_mode,
            )
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        logger.info(f"Loaded task '{task_name}' from YAML: {yaml_path}")
        return task

    @staticmethod
    def _parse_source(source_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate source configuration.

        Args:
            source_config: Source configuration dictionary.

        Returns:
            Validated source configuration with 'type' and 'path_or_id' keys.

        Raises:
            ValueError: If source configuration is invalid.
        """
        if not isinstance(source_config, dict):
            raise ValueError(
                f"YAML field 'source' must be a mapping, got "
                f"{type(source_config).__name__}. Define 'source' as a block "
                f"with 'type' and 'path'/'dataset_id' keys."
            )

        source_type = source_config.get("type", "").lower()
        if source_type not in ["csv", "jsonl", "hf"]:
            raise ValueError(
                f"YAML field 'source.type' has invalid value {source_type!r}. "
                f"Set 'source.type' to one of: 'csv', 'jsonl', 'hf'."
            )

        # Determine path_or_id key name
        path_key = "path" if source_type in ["csv", "jsonl"] else "dataset_id"

        if path_key not in source_config and "path_or_id" not in source_config:
            raise ValueError(
                f"YAML 'source' block is missing a data location. Add a "
                f"'{path_key}' (or 'path_or_id') key under 'source'."
            )

        path_or_id = source_config.get("path_or_id") or source_config.get(path_key)

        result = {
            "type": source_type,
            "path_or_id": path_or_id,
        }

        # Include split for HuggingFace datasets
        if source_type == "hf" and "split" in source_config:
            result["split"] = source_config["split"]

        return result

    @staticmethod
    def _parse_schema(schema_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Parse and validate schema configuration.

        Args:
            schema_config: Schema configuration dictionary.

        Returns:
            Schema dictionary with at least 'prompt' and 'answer' keys.

        Raises:
            ValueError: If schema is invalid or missing required fields.
        """
        if not isinstance(schema_config, dict):
            raise ValueError(
                f"YAML field 'schema' must be a mapping, got "
                f"{type(schema_config).__name__}. Define 'schema' as a block "
                f"mapping schema keys to dataset column names."
            )

        required_schema_fields = ["prompt", "answer"]
        for field in required_schema_fields:
            if field not in schema_config:
                raise ValueError(
                    f"schema missing required field '{field}'. "
                    f"Required fields: {required_schema_fields}"
                )

        # Validate field values are strings
        for field, value in schema_config.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"YAML field 'schema.{field}' must be a string column name, "
                    f"got {type(value).__name__}."
                )

        return dict(schema_config)

    @staticmethod
    def _parse_corruption(corruption_config: Dict[str, Any]) -> Optional[CorruptionPipeline]:
        """
        Parse and instantiate corruption strategy from YAML configuration.

        Args:
            corruption_config: Corruption configuration with 'strategy' and optional 'config' keys.

        Returns:
            CorruptionPipeline instance, or None if no corruption is configured.

        Raises:
            ValueError: If strategy is unknown.
            KeyError: If strategy configuration is invalid.
        """
        if not isinstance(corruption_config, dict):
            raise ValueError(
                f"YAML field 'corruption' must be a mapping, got "
                f"{type(corruption_config).__name__}. Define 'corruption' as a "
                f"block with a 'strategy' key."
            )

        strategy_name = corruption_config.get("strategy")
        if not strategy_name:
            logger.warning("corruption.strategy not specified; skipping corruption")
            return None

        strategy_name = strategy_name.lower()
        if strategy_name not in YAMLTaskLoader.CORRUPTION_STRATEGIES:
            valid = ", ".join(YAMLTaskLoader.CORRUPTION_STRATEGIES.keys())
            raise ValueError(
                f"Unknown corruption strategy: {strategy_name}. " f"Valid strategies: {valid}"
            )

        # Get strategy class
        strategy_class = YAMLTaskLoader.CORRUPTION_STRATEGIES[strategy_name]

        # Extract strategy-specific config (optional)
        strategy_config = corruption_config.get("config", {})
        if not isinstance(strategy_config, dict):
            raise ValueError(
                f"YAML field 'corruption.config' must be a mapping, got "
                f"{type(strategy_config).__name__}."
            )

        # Instantiate strategy with config
        try:
            if strategy_config:
                strategy = strategy_class(**strategy_config)
            else:
                strategy = strategy_class()
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate {strategy_name} with config {strategy_config}: {e}"
            )

        # paraphrase and distractor preserve the correct answer, so they are not
        # counterfactual and give little contrastive signal for EAP-family discovery.
        if strategy_name in ("paraphrase", "distractor"):
            logger.warning(
                "Corruption strategy '%s' preserves the correct answer, so it is not "
                "counterfactual and produces weak/near-zero contrastive signal for "
                "EAP-family discovery. Prefer entity_swap/token_swap/role_swap on "
                "syntactic tasks, or supply explicit corrupted_prompt/corrupted_answer "
                "columns for non-syntactic data.",
                strategy_name,
            )

        # Wrap in pipeline
        pipeline = CorruptionPipeline([strategy])
        return pipeline

    @staticmethod
    def _get_metric_fn(metric_name: str):
        """
        Get metric function by name.

        Supported metrics:
            - 'logit_diff': differentiable logit-difference metric (default).
            - 'kl': differentiable KL-divergence metric (multi-token answers).
            - 'accuracy': argmax-accuracy metric. NOTE: non-differentiable;
              usable for reporting but NOT for EAP-family circuit discovery
              (the attribution backward pass requires a differentiable metric).

        Args:
            metric_name: Name of the metric.

        Returns:
            Callable metric function.
        """
        metric_name = metric_name.lower().strip()

        # Import metrics from api module
        from ..api import _eap_accuracy, _eap_kl_divergence, _eap_logit_diff

        metrics_map = {
            "logit_diff": _eap_logit_diff,
            "kl": _eap_kl_divergence,
            "accuracy": _eap_accuracy,
        }

        if metric_name not in metrics_map:
            valid = ", ".join(metrics_map.keys())
            logger.warning(
                f"Unknown metric: {metric_name}. Valid metrics: {valid}. "
                f"Defaulting to 'logit_diff'."
            )
            return _eap_logit_diff

        if metric_name == "accuracy":
            logger.warning(
                "metric 'accuracy' is non-differentiable and cannot be used "
                "for EAP-family circuit discovery; use 'logit_diff' or 'kl'."
            )

        return metrics_map[metric_name]
