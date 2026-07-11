"""
Model Architecture Registry (Family-Based)

Maps HF model types to their layer structure, module names, and special handling.
Organized by architecture family to support all TransformerLens models (237+).

Each family entry defines:
  - name: Human-readable description
  - models: List of HF model_type values that use this architecture
  - layers_path, attn, mlp: Layer structure definitions
  - status: PRODUCTION, READY, NOT_STARTED
  - notes: Special handling required

Usage:
    from circuitkit.applications.arch_registry import MODEL_ARCH_REGISTRY
    arch = MODEL_ARCH_REGISTRY["llama"]
    layers = get_layers(hf_model, arch)

    # New: auto-detect from model.config.model_type
    from circuitkit.applications import detect_model_architecture
    arch_family = detect_model_architecture(hf_model)
    arch = MODEL_ARCH_REGISTRY[arch_family]
"""

MODEL_ARCH_REGISTRY = {
    # TIER 1: FULLY TESTED & OPTIMIZED (Production-Ready)
    "llama": {
        "name": "LLaMA / Llama-2 / Llama-3 / Llama-3.1 / CodeLlama",
        "models": ["llama", "llama2", "llama3", "codellama"],  # HF model_type values
        "layers_path": ["model.layers"],
        "attn": {
            "module": "self_attn",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "q_proj": "q_proj",
            "o_proj": "o_proj",
            "head_dim": "head_dim",
        },
        "mlp": {
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        },
        "gqa_capable": True,
        "transformer_lens_support": "full",
        "status": "PRODUCTION",
        "priority": 1,
        "notes": "Meta's flagship model, well-tested",
    },
    "qwen": {
        "name": "Qwen / Qwen2 / Qwen2.5 / Qwen3 (all versions)",
        "models": ["qwen", "qwen2", "qwen2.5", "qwen3", "qwen1.5"],  # HF model_type values
        "layers_path": ["model.layers"],
        "attn": {
            "module": "self_attn",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "q_proj": "q_proj",
            "o_proj": "o_proj",
            "head_dim": "head_dim",
        },
        "mlp": {
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        },
        "special_layers": ["self_attn.q_norm", "self_attn.k_norm"],  # Qwen3-specific RMSNorm
        "gqa_capable": True,
        "transformer_lens_support": "full",
        "status": "PRODUCTION",
        "priority": 2,
        "notes": "Alibaba's model, handles GQA + layer norms",
    },
    "gemma": {
        "name": "Google Gemma / Gemma-2 (all sizes)",
        "models": ["gemma", "gemma2"],  # HF model_type values
        "layers_path": ["model.layers"],
        "attn": {
            "module": "self_attn",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "q_proj": "q_proj",
            "o_proj": "o_proj",
            "head_dim": "head_dim",
        },
        "mlp": {
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        },
        "gqa_capable": True,
        "transformer_lens_support": "full",
        "status": "PRODUCTION",
        "priority": 3,
        "notes": "Google's open model, LLaMA-compatible",
    },
    "gemma3": {
        "name": "Google Gemma-3 (text; gemma-3-270m / 1b / 4b / 12b / 27b)",
        "models": ["gemma3", "gemma3_text"],  # HF model_type values
        "layers_path": ["model.layers"],
        "attn": {
            "module": "self_attn",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "q_proj": "q_proj",
            "o_proj": "o_proj",
            "head_dim": "head_dim",
        },
        "mlp": {
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        },
        "gqa_capable": True,
        "transformer_lens_support": "full",
        "status": "PRODUCTION",
        "priority": 3,
        "notes": "Gemma-3 text decoder; LLaMA-compatible nn.Linear layout, GQA",
    },
    # TIER 2: READY FOR SUPPORT (High confidence, minimal testing needed)
    "mistral": {
        "name": "Mistral-7B / Mistral 8x7B / Mistral Nemo (all versions)",
        "models": ["mistral"],  # HF model_type value
        "layers_path": ["model.layers"],
        "attn": {
            "module": "self_attn",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "q_proj": "q_proj",
            "o_proj": "o_proj",
            "head_dim": "head_dim",
        },
        "mlp": {
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        },
        "gqa_capable": True,
        "transformer_lens_support": "full",
        "status": "READY",
        "priority": 4,
        "notes": "Identical to LLaMA structure, MoE variant supported",
    },
    "phi": {
        "name": "Phi-3 / Phi-3.5 / Phi-4",
        "models": ["phi"],  # HF model_type value
        "layers_path": ["model.layers"],
        "attn": {
            "module": "self_attn",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "q_proj": "q_proj",
            "o_proj": "o_proj",
            "head_dim": "head_dim",
        },
        "mlp": {
            "gate_proj": "gate_proj",
            "up_proj": "up_proj",
            "down_proj": "down_proj",
        },
        "gqa_capable": False,
        "transformer_lens_support": "full",
        "status": "READY",
        "priority": 5,
        "notes": "Microsoft's efficient model",
    },
    # TIER 3: MEDIUM EFFORT (Different naming, but clear structure)
    "gpt2": {
        "name": "GPT-2 (all sizes)",
        "models": ["gpt2"],  # HF model_type value
        "layers_path": ["transformer.h"],
        "attn": {
            "module": "attn",
            "c_attn": "c_attn",  # Contains Q, K, V stacked
            "c_proj": "c_proj",  # Output projection
            "head_dim": None,  # Computed from config.hidden_size / config.num_attention_heads
        },
        "mlp": {
            "c_fc": "c_fc",  # Feed-forward up projection
            "c_proj": "c_proj",  # Feed-forward down projection
        },
        "gqa_capable": False,
        "transformer_lens_support": "full",
        "status": "READY",
        "priority": 6,
        "notes": "Needs custom head_dim calculation, c_attn is stacked Q,K,V",
    },
    "falcon": {
        "name": "Falcon-7B / Falcon-40B / Falcon-180B",
        "models": ["falcon"],  # HF model_type value
        "layers_path": ["transformer.h"],
        "attn": {
            "module": "self_attention",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "q_proj": "q_proj",
            "dense": "dense",  # Output projection
            "head_dim": "head_dim",
        },
        "mlp": {
            "dense_h_to_4h": "dense_h_to_4h",
            "dense_4h_to_h": "dense_4h_to_h",
        },
        "gqa_capable": True,
        "transformer_lens_support": "partial",
        "status": "READY",
        "priority": 7,
        "notes": "Different naming for MLP projections",
    },
    # TIER 4: HIGH EFFORT (Significant structural differences)
    "bloom": {
        "name": "BLOOM / BLOOMZ",
        "models": ["bloom"],  # HF model_type value
        "layers_path": ["h"],
        "attn": {
            "module": "self_attention",
            "dense": "dense",  # Q, K, V, O
            "head_dim": "head_dim",
        },
        "mlp": {
            "dense_h_to_4h": "dense_h_to_4h",
            "dense_4h_to_h": "dense_4h_to_h",
        },
        "gqa_capable": False,
        "transformer_lens_support": "partial",
        "status": "NOT_STARTED",
        "priority": 8,
        "notes": "Monolithic dense modules, not separate projections",
    },
    "bert": {
        "name": "BERT / RoBERTa / DistilBERT",
        "models": ["bert", "roberta", "distilbert"],  # HF model_type values
        "layers_path": ["bert.encoder.layer"],
        "attn": {
            "module": "attention.self",
            "query": "query",
            "key": "key",
            "value": "value",
            "dense": None,  # Output in attention.output
            "head_dim": "head_dim",
        },
        "mlp": {
            "dense": "dense",  # In intermediate
            "output_dense": "dense",  # In output
        },
        "gqa_capable": False,
        "transformer_lens_support": "limited",
        "status": "NOT_STARTED",
        "priority": 9,
        "notes": "Bidirectional, different projection structure",
    },
    "t5": {
        "name": "T5 / FLAN-T5 / mT5",
        "models": ["t5", "mt5"],  # HF model_type values
        "layers_path": ["encoder.block", "decoder.block"],  # Encoder + decoder
        "attn": {
            "module": "layer",  # Contains both self-attn and cross-attn
            "query": "query",
            "key": "key",
            "value": "value",
            "head_dim": "head_dim",
        },
        "mlp": {
            "DenseReluDense": "DenseReluDense",
        },
        "gqa_capable": False,
        "transformer_lens_support": "limited",
        "status": "NOT_STARTED",
        "priority": 10,
        "notes": "Encoder-decoder, separate encoder/decoder processing",
    },
}

# Build model_type → family mapping for fast lookup
# This allows detect_model_architecture to map config.model_type to registry key
_MODEL_TO_FAMILY = {}
for family_key, family_cfg in MODEL_ARCH_REGISTRY.items():
    for model_type in family_cfg.get("models", []):
        _MODEL_TO_FAMILY[model_type] = family_key


def get_model_family(model_type: str) -> str:
    """
    Map HF model_type to architecture family.

    Parameters
    ----------
    model_type : str
        The model_type from model.config.model_type

    Returns
    -------
    str : The family key in MODEL_ARCH_REGISTRY (e.g., "llama", "qwen")

    Raises
    ------
    KeyError : If model_type is not recognized
    """
    if model_type not in _MODEL_TO_FAMILY:
        raise KeyError(
            f"Model type '{model_type}' not found in architecture registry.\n"
            f"Supported families and their model_types:\n"
            + "\n".join(
                f"  {family}: {cfg.get('models', [])}"
                for family, cfg in MODEL_ARCH_REGISTRY.items()
            )
        )
    return _MODEL_TO_FAMILY[model_type]


# Utility constants
SUPPORTED_FAMILIES = list(MODEL_ARCH_REGISTRY.keys())
SUPPORTED_MODELS = list(_MODEL_TO_FAMILY.keys())  # All HF model_type values
PRODUCTION_FAMILIES = [f for f, cfg in MODEL_ARCH_REGISTRY.items() if cfg["status"] == "PRODUCTION"]
READY_FAMILIES = [f for f, cfg in MODEL_ARCH_REGISTRY.items() if cfg["status"] == "READY"]
