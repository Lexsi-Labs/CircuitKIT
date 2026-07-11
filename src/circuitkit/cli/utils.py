"""
Utility functions for CircuitKit CLI.
"""

import logging
import re
from pathlib import Path
from typing import List


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_model_name(model_name: str) -> bool:
    """Validate if a model name is supported."""
    # Basic validation - check if it's a valid model identifier
    if not model_name or len(model_name) < 2:
        return False

    # Check for common patterns
    valid_patterns = [
        r"^[a-zA-Z0-9_.-]+$",  # Simple model names like "gpt2"
        r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$",  # HuggingFace format like "meta-llama/Meta-Llama-3-8B" or "Qwen/Qwen3-0.6B"
    ]

    for pattern in valid_patterns:
        if re.match(pattern, model_name):
            return True

    return False


def get_supported_models() -> List[str]:
    """Get list of supported model names from TransformerLens."""
    # Complete list of models supported by TransformerLens as of the latest version
    # Source: https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html
    return sorted(
        [
            # GPT-2 family
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            # GPT-Neo family
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            # GPT-J family
            "EleutherAI/gpt-j-6b",
            # GPT-NeoX family
            "EleutherAI/gpt-neox-20b",
            # Pythia family
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
            "EleutherAI/pythia-12b",
            # OPT family
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "facebook/opt-13b",
            "facebook/opt-30b",
            "facebook/opt-66b",
            # BLOOM family
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            "bigscience/bloom-1b7",
            "bigscience/bloom-3b",
            "bigscience/bloom-7b1",
            # Llama family
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "meta-llama/Meta-Llama-3-70B",
            # Mistral family
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.1",
            # Mixtral family
            "mistralai/Mixtral-8x7B-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            # Gemma family
            "google/gemma-2b",
            "google/gemma-7b",
            "google/gemma-2b-it",
            "google/gemma-7b-it",
            "google/gemma-2-2b",
            "google/gemma-2-2b-it",
            "google/gemma-2-9b",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b",
            "google/gemma-2-27b-it",
            # Yi family
            "01-ai/Yi-6B",
            "01-ai/Yi-34B",
            "01-ai/Yi-6B-Chat",
            "01-ai/Yi-34B-Chat",
            # DialoGPT family
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            # BERT family
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-large-cased",
            # RoBERTa family
            "roberta-base",
            "roberta-large",
            # T5 family
            "google-t5/t5-small",
            "google-t5/t5-base",
            "google-t5/t5-large",
            "google-t5/t5-3b",
            "google-t5/t5-11b",
            # DistilBERT family
            "distilbert-base-uncased",
            "distilbert-base-cased",
            # DistilGPT-2
            "distilgpt2",
            # ALBERT family
            "albert-base-v2",
            "albert-large-v2",
            "albert-xlarge-v2",
            "albert-xxlarge-v2",
            # Qwen family
            "Qwen/Qwen-7B",
            "Qwen/Qwen-14B",
            "Qwen/Qwen-32B",
            "Qwen/Qwen-72B",
            "Qwen/Qwen-7B-Chat",
            "Qwen/Qwen-14B-Chat",
            "Qwen/Qwen-32B-Chat",
            "Qwen/Qwen-72B-Chat",
            "Qwen/Qwen2-0.5B",
            "Qwen/Qwen2-1.5B",
            "Qwen/Qwen2-7B",
            "Qwen/Qwen2-14B",
            "Qwen/Qwen2-72B",
            "Qwen/Qwen2-0.5B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-14B-Instruct",
            "Qwen/Qwen2-72B-Instruct",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-14B",
            "Qwen/Qwen2.5-32B",
            "Qwen/Qwen2.5-72B",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            # Phi family
            "microsoft/phi-1",
            "microsoft/phi-1_5",
            "microsoft/phi-2",
            "microsoft/Phi-3-mini-4k-instruct",
            "microsoft/Phi-3-mini-128k-instruct",
            "microsoft/Phi-3-small-8k-instruct",
            "microsoft/Phi-3-small-128k-instruct",
            "microsoft/Phi-3-medium-4k-instruct",
            "microsoft/Phi-3-medium-128k-instruct",
            # CodeLlama family
            "codellama/CodeLlama-7b-hf",
            "codellama/CodeLlama-13b-hf",
            "codellama/CodeLlama-34b-hf",
            "codellama/CodeLlama-7b-Python-hf",
            "codellama/CodeLlama-13b-Python-hf",
            "codellama/CodeLlama-34b-Python-hf",
            "codellama/CodeLlama-7b-Instruct-hf",
            "codellama/CodeLlama-13b-Instruct-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            # mGPT
            "ai-forever/mGPT",
            # Additional models that might be supported
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
        ]
    )


def get_model_info(model_name: str) -> dict:
    """Get information about a model without loading model weights."""
    return {
        "name": model_name,
        "type": _get_model_type(model_name),
        "size": _get_model_size_from_name(model_name),
        "layers": "Unknown",
        "d_model": "Unknown",
        "n_heads": "Unknown",
        "d_vocab": "Unknown",
    }


def _get_model_type(model_name: str) -> str:
    """Determine model type from name."""
    model_name_lower = model_name.lower()
    if "gpt" in model_name_lower:
        return "GPT"
    elif "llama" in model_name_lower or "codellama" in model_name_lower:
        return "Llama"
    elif "opt" in model_name_lower:
        return "OPT"
    elif "dialo" in model_name_lower:
        return "DialoGPT"
    elif "bert" in model_name_lower:
        return "BERT"
    elif "roberta" in model_name_lower:
        return "RoBERTa"
    elif "t5" in model_name_lower:
        return "T5"
    elif "bloom" in model_name_lower:
        return "BLOOM"
    elif "pythia" in model_name_lower:
        return "Pythia"
    elif "neox" in model_name_lower:
        return "GPT-NeoX"
    elif "mistral" in model_name_lower:
        return "Mistral"
    elif "mixtral" in model_name_lower:
        return "Mixtral"
    elif "gemma" in model_name_lower:
        return "Gemma"
    elif "yi" in model_name_lower:
        return "Yi"
    elif "qwen" in model_name_lower:
        return "Qwen"
    elif "phi" in model_name_lower:
        return "Phi"
    elif "albert" in model_name_lower:
        return "ALBERT"
    elif "distil" in model_name_lower:
        return "Distil"
    else:
        return "Other"


def _get_model_size_from_config(model_cfg) -> str:
    """Estimate model size from config."""
    try:
        # Try to estimate parameters
        n_layers = (
            getattr(model_cfg, "n_layers", 0)
            or getattr(model_cfg, "num_hidden_layers", 0)
            or getattr(model_cfg, "n_layer", 0)
        )
        d_model = (
            getattr(model_cfg, "d_model", 0)
            or getattr(model_cfg, "hidden_size", 0)
            or getattr(model_cfg, "n_embd", 0)
        )

        if n_layers and d_model:
            # Rough parameter estimation
            approx_params = n_layers * d_model * d_model * 4  # Very rough estimate
            if approx_params < 100_000_000:
                return "Small"
            elif approx_params < 1_000_000_000:
                return "Medium"
            elif approx_params < 10_000_000_000:
                return "Large"
            elif approx_params < 100_000_000_000:
                return "XL"
            else:
                return "XXL"
        else:
            return "Unknown"
    except Exception:
        return "Unknown"


def _get_model_size_from_name(model_name: str) -> str:
    """Estimate model size from model name."""
    model_name_lower = model_name.lower()

    # Size indicators in names
    if any(size in model_name_lower for size in ["125m", "160m", "350m", "560m", "70m"]):
        return "Small"
    elif any(size in model_name_lower for size in ["1.3b", "1b1", "1b7", "1.4b", "2.7b", "2.8b"]):
        return "Medium"
    elif any(size in model_name_lower for size in ["6b", "6.7b", "6.9b", "7b", "8b"]):
        return "Large"
    elif any(size in model_name_lower for size in ["13b", "20b", "30b"]):
        return "XL"
    elif any(size in model_name_lower for size in ["66b", "70b"]):
        return "XXL"
    elif "small" in model_name_lower:
        return "Small"
    elif "medium" in model_name_lower:
        return "Medium"
    elif "large" in model_name_lower:
        return "Large"
    elif "xl" in model_name_lower:
        return "XL"
    else:
        return "Unknown"


def ensure_output_dir(file_path: str) -> None:
    """Ensure the output directory exists."""
    output_dir = Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"


def get_file_info(file_path: str) -> dict:
    """Get file information."""
    path = Path(file_path)
    if not path.exists():
        return {"exists": False}

    stat = path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "modified": stat.st_mtime,
    }
