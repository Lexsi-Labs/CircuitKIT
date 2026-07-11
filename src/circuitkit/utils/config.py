import copy
import logging
from typing import Any, Dict, Union

import yaml

# Define a dictionary of default values. This makes the tool easier to use
# as users only need to specify what they want to change.
DEFAULT_CONFIG = {
    "model": {"precision": "bfloat16"},
    "discovery": {
        "algorithm": "eap-ig",
        # "task" intentionally omitted — must be supplied by user config or inline data section
        "level": "node",  # 'node' or 'neuron'
        "data_params": {"batch_size": 16, "num_examples": 128},
        "batch_size": 4,
        "method": "EAP-IG-inputs",
        "ig_steps": 5,
        "intervention": "patching",  # Discovery intervention mode
        # IBCircuit-specific defaults
        "num_epochs": 1000,
        "learning_rate": 0.01,
        "alpha": 1.0,
        "beta": 0.001,
        "alpha_loss": "kl",
        "log_interval": 100,
        "mask_type": "sigmoid",
        "scope": "heads",  # 'heads', 'mlp', or 'both' — used by IBCircuit
        "mlp_hook": "mlp_out",  # Hook point for MLP analysis
    },
    "pruning": {"target_sparsity": 0.3, "scope": "both"},  # 'heads', 'mlp', or 'both'
    "output_path": "./circuit_discovery_results.pt",
    "eval": {
        "num_examples": 256,
        "seed": 42,
        "full_faithfulness_eval": False,
    },
    "data": None,  # Optional inline data config;
}


def deep_merge(source: Dict, destination: Dict) -> Dict:
    """
    Recursively merges source dict into destination dict.
    Nested dictionaries are merged, other values in destination are overwritten.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
            destination[key] = deep_merge(value, destination[key])
        else:
            destination[key] = value
    return destination


def _validate_ibcircuit_params(discovery_config: Dict[str, Any]):
    """
    Validate IBCircuit-specific training configuration parameters.

    Args:
        discovery_config: Discovery configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """

    # Check required keys
    required_keys = ["num_epochs", "learning_rate", "alpha", "beta"]
    missing_keys = [k for k in required_keys if k not in discovery_config]

    if missing_keys:
        raise ValueError(
            f"IBCircuit config missing required keys: {missing_keys}.\n"
            f"Got keys: {list(discovery_config.keys())}\n"
            f"Required keys: {required_keys}"
        )

    # Validate alpha_loss mode
    if "alpha_loss" in discovery_config and discovery_config["alpha_loss"] not in ["kl", "ce"]:
        raise ValueError(
            f"Invalid alpha_loss mode: {discovery_config['alpha_loss']}.\n"
            f"Must be 'kl' (KL divergence, recommended) or 'ce' (cross-entropy)."
        )

    # Validate numeric parameters
    if discovery_config["num_epochs"] <= 0:
        raise ValueError(f"num_epochs must be positive, got {discovery_config['num_epochs']}")

    if discovery_config["learning_rate"] <= 0:
        raise ValueError(f"learning_rate must be positive, got {discovery_config['learning_rate']}")

    if discovery_config["alpha"] < 0:
        raise ValueError(f"alpha must be non-negative, got {discovery_config['alpha']}")

    if discovery_config["beta"] < 0:
        raise ValueError(f"beta must be non-negative, got {discovery_config['beta']}")

    # Warn about unusual parameter values
    logger = logging.getLogger("circuitkit.config")

    if discovery_config["beta"] > discovery_config["alpha"]:
        logger.warning(
            f"Beta ({discovery_config['beta']}) > Alpha ({discovery_config['alpha']}). "
            f"This means IB regularization is weighted more than task preservation. "
            f"This is unusual and may lead to over-sparsification."
        )

    if discovery_config["num_epochs"] < 100:
        logger.warning(
            f"num_epochs is quite low ({discovery_config['num_epochs']}). "
            f"IBCircuit typically requires 500-1000 epochs for convergence."
        )


def _validate_config(config: Dict[str, Any]):
    """
    Performs validation checks on the merged configuration.
    Raises ValueError for invalid configurations.
    """
    logger = logging.getLogger("circuitkit.config")

    # Top-level keys
    required_top_keys = ["model", "discovery", "pruning"]
    for key in required_top_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: '{key}'")

    # Model config
    if "name" not in config["model"]:
        raise ValueError("Missing required key 'model.name'")

    # Discovery config
    algo = config["discovery"].get("algorithm")
    if not algo:
        raise ValueError("Missing required key 'discovery.algorithm'")

    algo = algo.lower()
    valid_algos = [
        "acdc",
        "eap",
        "eap-ig",
        "ibcircuit",
        # Tier-0 promotions: top-level keys for the EAP-internal methods
        "eap-ig-activations",
        "eap-clean-corrupted",
        "eap-exact",
        # AtP+GradDrop (Kramár et al. 2024)
        "atp-gd",
        # EAP-GP / GradPath (Zhang et al. 2025, NeurIPS)
        "eap-gp",
        # RelP / Relevance Patching (Mohebbi et al. 2025, NeurIPS)
        "relp",
        # CD-T / PEAP / IFR — accepted at config level
        "cdt",
        "peap",
        "eap-ifr",
    ]
    if algo not in valid_algos:
        raise ValueError(f"Invalid algorithm '{algo}'. Must be one of {valid_algos}")

    if algo in valid_algos:
        has_inline_data = bool(config.get("data") and config["data"].get("type"))
        if "task" not in config["discovery"] and not has_inline_data:
            raise ValueError(
                f"{algo.upper()} requires 'discovery.task' (e.g., 'ioi', 'greater_than') "
                f"or a 'data' section with 'type' set."
            )

    # IBCircuit-specific validation
    if algo == "ibcircuit":
        _validate_ibcircuit_params(config["discovery"])

        # Avoid float16 precision for numerical stability, avoid float32 to prevent OOM errors
        if config["model"].get("precision") != "bfloat16":
            logger.warning(
                f"Avoid float16 precision for numerical stability, avoid float32 to prevent OOM errors. "
                f"Current precision: {config['model'].get('precision')}. "
            )

    # Pruning config
    sparsity = config["pruning"].get("target_sparsity")
    if not isinstance(sparsity, (int, float)) or not (0.0 <= sparsity <= 1.0):
        raise ValueError(
            f"'pruning.target_sparsity' must be a float between 0.0 and 1.0, but got {sparsity}"
        )

    scope = config["pruning"].get("scope")
    valid_scopes = ["heads", "mlp", "both"]
    if scope not in valid_scopes:
        raise ValueError(f"Invalid pruning scope '{scope}'. Must be one of {valid_scopes}")

    # Validate inline data section if present
    data_cfg = config.get("data")
    if data_cfg and isinstance(data_cfg, dict):
        data_type = data_cfg.get("type")
        if data_type not in ("template", "auto", "clean_only"):
            raise ValueError(
                f"data.type must be 'template', 'auto', or 'clean_only', got {data_type!r}"
            )
        if not data_cfg.get("path"):
            raise ValueError("data.path is required when data section is present")
        if data_type == "template":
            tpl = data_cfg.get("template")
            if not tpl or not isinstance(tpl, dict):
                raise ValueError("data.template must be a dict when data.type='template'")

            # Determine which template keys are required based on algorithm.
            # IBCircuit and CD-T only need the clean side.
            _algo = algo.lower()
            _clean_only_algos = {
                "ibcircuit", "cdt", "peap", "eap-ifr",
            }
            if _algo in _clean_only_algos:
                required_tpl_keys = {"clean_prompt"}
            else:
                required_tpl_keys = {
                    "clean_prompt", "corrupt_prompt", "clean_answer", "corrupt_answer"
                }
            missing = required_tpl_keys - set(tpl.keys())
            if missing:
                raise ValueError(
                    f"data.template is missing keys: {missing}. "
                    f"Required for algorithm '{_algo}': {sorted(required_tpl_keys)}"
                )

            # Alignment strategy and padding options only apply when corrupt keys
            # are present (i.e. paired data is being built).
            has_corrupt_keys = bool(tpl.get("corrupt_prompt") and tpl.get("corrupt_answer"))
            if has_corrupt_keys:
                align_strategy = data_cfg.get("align_strategy", "filter")
                valid_strategies = ("filter", "pad_question", "none")
                if align_strategy not in valid_strategies:
                    raise ValueError(
                        f"data.align_strategy must be one of {valid_strategies}, "
                        f"got {align_strategy!r}."
                    )
                if align_strategy == "pad_question":
                    pad_region_end = data_cfg.get("pad_region_end")
                    if not pad_region_end or not isinstance(pad_region_end, str):
                        raise ValueError(
                            "data.pad_region_end is required and must be a non-empty string "
                            "when data.align_strategy='pad_question'. "
                            "E.g. pad_region_end: 'Answer:'"
                        )
                pair_padding_side = data_cfg.get("pair_padding_side", "left")
                if pair_padding_side not in ("left", "right"):
                    raise ValueError(
                        f"data.pair_padding_side must be 'left' or 'right', "
                        f"got {pair_padding_side!r}."
                    )
        if data_type == "clean_only":
            prompt_col = data_cfg.get("prompt_column", "prompt")
            if not isinstance(prompt_col, str) or not prompt_col.strip():
                raise ValueError("data.prompt_column must be a non-empty string")
    
    logger.info("Configuration validated successfully.")


def _normalize_example_count_aliases(user_config: Dict[str, Any], logger) -> Dict[str, Any]:
    """Rename ``n_examples`` -> ``num_examples`` in the blocks that read it.

    ``n_examples`` is the flat-API parameter name (``ck.discover``,
    ``ck.faithfulness``); dict-config internally reads ``num_examples``. Accept
    both so a dict-config user can write either. Applied to the blocks that
    actually consume the value: ``discovery.data_params`` and ``eval``. When a
    block sets both, the explicit ``num_examples`` wins and ``n_examples`` is
    dropped with a warning.
    """
    if not isinstance(user_config, dict):
        return user_config

    def _fix(block: Any) -> None:
        if not isinstance(block, dict) or "n_examples" not in block:
            return
        alias = block.pop("n_examples")
        if "num_examples" in block:
            logger.warning(
                "Both 'num_examples' (%r) and 'n_examples' (%r) set; using "
                "'num_examples' and ignoring 'n_examples'.",
                block["num_examples"],
                alias,
            )
        else:
            block["num_examples"] = alias

    discovery = user_config.get("discovery")
    if isinstance(discovery, dict):
        _fix(discovery.get("data_params"))
    _fix(user_config.get("eval"))
    return user_config


def load_and_validate_config(config_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Loads a configuration from a file path or dictionary, merges it with
    defaults, and validates it.

    Args:
        config_input (Union[str, Dict[str, Any]]): Either a path to a YAML
                                                   config file or a config dictionary.

    Returns:
        A validated and complete configuration dictionary.
    """
    logger = logging.getLogger("circuitkit.config")

    if isinstance(config_input, str):
        logger.info(f"Loading configuration from YAML file: {config_input}")
        with open(config_input, "r") as f:
            user_config = yaml.safe_load(f)
    elif isinstance(config_input, dict):
        user_config = config_input
    else:
        raise TypeError("config_input must be a dictionary or a file path (string).")

    # Accept ``n_examples`` (the flat-API spelling) as an alias for
    # ``num_examples`` in dict-config, so one name works across both APIs.
    # Applied to the *user* config before merge — after merge the default
    # ``num_examples`` is always present and would mask the alias.
    user_config = _normalize_example_count_aliases(user_config, logger)

    # Start with a deep copy of the defaults
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Merge the user's config on top of the defaults
    config = deep_merge(user_config, config)

    # Validate the final merged config
    _validate_config(config)

    return config
