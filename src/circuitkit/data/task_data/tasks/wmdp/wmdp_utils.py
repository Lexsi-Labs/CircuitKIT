"""
WMDP Dataset Utilities for CircuitKit

This module provides utilities for loading and formatting WMDP (Weapons of Mass Destruction Proxy)
dataset for circuit discovery experiments.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch as t
from datasets import load_dataset
from transformer_lens import HookedTransformer


import logging

logger = logging.getLogger(__name__)

def _load_dataset_with_cache_fix(dataset_name: str, config: str, split: str = "test"):
    """
    Helper function to load dataset with automatic cache clearing if outdated format is detected.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "cais/wmdp", "cais/mmlu")
        config: Dataset config name
        split: Dataset split (default: "test")

    Returns:
        Loaded dataset
    """
    try:
        return load_dataset(dataset_name, config, split=split)
    except ValueError as e:
        # Check if it's the 'List' feature type error (outdated cache)
        if "Feature type 'List' not found" in str(e):
            logger.info(
                f"Detected outdated cache format for dataset '{dataset_name}' config '{config}'. Clearing cache and retrying..."
            )

            # Find the cache directory - check environment variable first, then default location
            cache_base = os.environ.get("HF_DATASETS_CACHE")
            if cache_base is None:
                cache_base = os.path.join(
                    os.path.expanduser("~"), ".cache", "huggingface", "datasets"
                )
            else:
                cache_base = os.path.expanduser(cache_base)

            cache_base_path = Path(cache_base)
            if cache_base_path.exists():
                # The cache structure is: dataset_name/config_name
                # e.g., cais___wmdp/wmdp-cyber
                dataset_cache_dir = dataset_name.replace("/", "___")
                cache_path1 = cache_base_path / dataset_cache_dir / config
                cache_path2 = cache_base_path / dataset_cache_dir

                # Clear the config-specific subdirectory first
                cleared_any = False
                for cache_path in [cache_path1, cache_path2]:
                    if cache_path.exists():
                        try:
                            shutil.rmtree(cache_path)
                            logger.info(f"Cleared cache at {cache_path}")
                            cleared_any = True
                        except Exception as cache_error:
                            logger.warning(f"Warning: Could not clear cache at {cache_path}: {cache_error}")

                # If we cleared the config subdirectory but the parent still exists,
                # also try to clear any dataset_info.json or other metadata files
                if cleared_any and cache_path2.exists():
                    try:
                        # Look for dataset_info.json or other metadata files
                        for item in cache_path2.iterdir():
                            if item.is_file() and (
                                item.name.endswith(".json") or item.name.endswith(".lock")
                            ):
                                try:
                                    item.unlink()
                                    logger.info(f"Removed metadata file: {item}")
                                except Exception:
                                    pass
                    except Exception:
                        pass

            # Retry loading with fresh download
            return load_dataset(dataset_name, config, split=split, download_mode="force_redownload")
        else:
            raise
    except Exception:
        raise


def load_wmdp_dataset(config: str = "wmdp-bio", split: str = "test"):
    """
    Load WMDP dataset from HuggingFace.

    Args:
        config: Dataset config ('wmdp-bio', 'wmdp-chem', or 'wmdp-cyber')
        split: Dataset split ('test')

    Returns:
        Dataset object
    """
    dataset = _load_dataset_with_cache_fix("cais/wmdp", config, split=split)
    return dataset


def format_wmdp_prompt(question: str, choices: List[str]) -> str:
    """
    Format WMDP question and choices into a prompt.

    Args:
        question: The question text
        choices: List of 4 choice strings

    Returns:
        Formatted prompt string with instruction to output only A, B, C, or D
    """
    letters = ["A", "B", "C", "D"][: len(choices)]
    lines = [
        "Please choose the answer that is most relevant to the question from options A, B, C and D."
    ]
    lines.append("")
    lines.append(f"Q: {question}")
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}) {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def corrupt_wmdp_query(question: str, choices: List[str]) -> str:
    """
    Create a corrupted version of WMDP query using MMLU-style corruption.

    Args:
        question: Original question text
        choices: List of 4 choice strings

    Returns:
        Corrupted prompt with generic question stem
    """
    return format_wmdp_prompt("Which is the most possible answer?", choices)


def generate_wmdp_eap_data(
    example: Dict[str, Any], model: HookedTransformer, seed: int = 42
) -> Optional[Dict[str, Any]]:
    """
    Generate EAP format data from a single WMDP example with corruption strategy.

    Args:
        example: WMDP dataset example with 'question', 'choices', 'answer' fields
        model: HookedTransformer model for tokenization
        seed: Random seed (unused, kept for API consistency)

    Returns:
        Dictionary with EAP data or None if processing fails
    """
    question = example.get("question", "")
    choices = example.get("choices", [])
    answer = example.get("answer")

    if not question or not choices or answer is None:
        return None

    # Ensure choices is a list
    if not isinstance(choices, list):
        choices = list(choices)

    if len(choices) < 4:
        return None

    # Format the clean prompt
    clean_prompt = format_wmdp_prompt(question, choices)
    correct_letter = "ABCD"[answer] if answer < len(choices) else "A"
    clean_text = f"{clean_prompt} {correct_letter}"

    # Create corrupted prompt
    corrupted_prompt = corrupt_wmdp_query(question, choices)
    corrupted_text = f"{corrupted_prompt} {correct_letter}"

    # Get token IDs for all 4 options (A, B, C, D)
    try:
        option_tokens = []
        for letter in "ABCD":
            test_text = f"{clean_prompt} {letter}"
            tokens = model.to_tokens(test_text, prepend_bos=True)
            # Get the last token (the option letter)
            option_tokens.append(tokens[0, -1].item())

        # For WMDP metric (same as MMLU), we need: [correct_token, incorrect1, incorrect2, incorrect3]
        correct_token = option_tokens[answer]
        incorrect_tokens = [option_tokens[i] for i in range(4) if i != answer]

        # Pad with correct token if we don't have 3 incorrect options
        while len(incorrect_tokens) < 3:
            incorrect_tokens.append(correct_token)

        all_option_tokens = [correct_token] + incorrect_tokens[:3]

    except Exception:
        # If tokenization failed, skip
        return None

    return {
        "clean": clean_text,
        "corrupted": corrupted_text,
        "correct_idx": all_option_tokens[0],  # correct option token
        "incorrect_idx": all_option_tokens[1:4],  # 3 incorrect option tokens
    }


def check_model_correctness(
    model: HookedTransformer, example: Dict[str, Any], max_samples: int = 100
) -> bool:
    """
    Check if the model predicts the correct answer for a WMDP example.

    Args:
        model: HookedTransformer model
        example: WMDP dataset example
        max_samples: Maximum number of samples to check (for early exit)

    Returns:
        True if model predicts correct answer, False otherwise
    """
    question = example.get("question", "")
    choices = example.get("choices", [])
    answer = example.get("answer")

    if not question or not choices or answer is None:
        return False

    if not isinstance(choices, list):
        choices = list(choices)

    if len(choices) < 4:
        return False

    try:
        # Format prompt without answer
        clean_prompt = format_wmdp_prompt(question, choices)

        # Get token IDs for all 4 options
        option_tokens = []
        for letter in "ABCD":
            test_text = f"{clean_prompt} {letter}"
            tokens = model.to_tokens(test_text, prepend_bos=True)
            option_tokens.append(tokens[0, -1].item())

        # Get logits for the prompt (without any answer letter)
        tokens = model.to_tokens(clean_prompt, prepend_bos=True)

        with t.no_grad():
            logits = model(tokens)
            # Get logits at the last position (where answer token would be)
            last_logits = logits[0, -1, :]  # [vocab_size]

            # Extract logits for each choice token
            choice_logits = []
            for token_id in option_tokens:
                if token_id < last_logits.size(0):
                    choice_logits.append(last_logits[token_id].item())
                else:
                    choice_logits.append(float("-inf"))

            choice_logits_tensor = t.tensor(choice_logits)

            # Check if correct answer has highest logit
            predicted_idx = choice_logits_tensor.argmax().item()
            is_correct = predicted_idx == answer

            return is_correct

    except Exception:
        return False


def wmdp_logit_diff_metric():
    """
    Return the metric function for WMDP task (same as MMLU).

    For WMDP, the metric is the logit difference between the correct option
    and the average of the incorrect ones.

    Returns:
        Metric function with signature (logits, clean_logits, input_length, labels, mean=True, loss=False)
    """
    from .....api import _eap_logit_diff_mmlu


    return _eap_logit_diff_mmlu
