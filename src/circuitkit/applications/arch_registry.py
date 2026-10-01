"""
Model Architecture Registry (Family-Based)

Maps HF model types to their layer structure, module names, and special handling.
Organized by architecture family to support all TransformerLens models (237+).

Each family entry defines:
  - name: Human-readable description
  - models: List of HF model_type values that use this architecture
  - layers_path, attn, mlp: Layer structure definitions
  - status: PRODUCTION, READY, EXPERIMENTAL, NOT_STARTED
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
    # TIER: EXPERIMENTAL — discovery supported via the circuitkit TransformerLens
    # port (circuitkit.backends._tl_compat), not upstream TL. Intervention
    # (pruning/quantization) path not yet validated.
    "cohere": {
        "name": "Cohere Command R / Aya Expanse (cohere), Command R7B / Tiny Aya (cohere2)",
        "models": ["cohere", "cohere2"],  # HF model_type values
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
        "transformer_lens_support": "full",  # via circuitkit._tl_compat port (not upstream TL)
        "status": "EXPERIMENTAL",
        "priority": 11,
        "notes": (
            "cohere2 (tiny-aya, Command R7B): parallel attn+MLP block with a "
            "single input_layernorm, GQA, and per-layer sliding-window vs full "
            "attention (NoPE on full-attention layers). cohere (Aya Expanse): "
            "same shape but rotary applies to every layer (no NoPE, no sliding "
            "window). Discovery via the circuitkit _tl_compat TransformerLens "
            "port; gated weights (Command R7B, Aya Expanse, tiny-aya) need "
            "HF_TOKEN. See docs/advanced/tiny-aya.md.\n"
            "Stage 5 intervention validation (real weights, all three models: "
            "tiny-aya/CohereLabs/tiny-aya-base, Command R7B/CohereLabs/"
            "c4ai-command-r7b-12-2024, Aya Expanse/CohereLabs/aya-expanse-8b): "
            "evaluation (circuitkit.api.evaluate_circuit faithfulness, "
            "patching+ablation pillars) confirmed finite on greater_than for "
            "all three. Pruning (applications/pruning/score_extractor."
            "build_importance_dict) confirmed resolving real k_proj/gate_proj "
            "nn.Linear modules via this family's layers_path/attn/mlp config "
            "on a real layer subset for all three. Quantization "
            "(applications/quantization/quant_utils.build_patterns) confirmed "
            "its fnmatch patterns match real nn.Linear submodules on all three "
            "real loaded models; optimum-quanto's actual quantize()/freeze() "
            "call itself was NOT run (optional dependency not installed in the "
            "validation environment) -- only target-module resolution was "
            "exercised, which is what this family config is responsible for. "
            "Weight steering (applications/steering/weight_steering."
            "CircuitWeightSteering) confirmed resolving real per-head W_Q/W_K/"
            "W_V/W_O weight slices (including the GQA query-head -> kv-head "
            "floor-div mapping) and applying a steering vector on real weights "
            "for all three, producing a still-finite forward pass; note this "
            "module's circuit format ('A{layer}.{head}') is unrelated to and "
            "does not match the discovery Graph's node-name convention "
            "('a{layer}.h{head}') -- a pre-existing property of the module, "
            "not something this family's registry entry controls. See "
            "tests/regression/test_{cohere,command_r7b,aya_expanse}_"
            "{evaluation,interventions}.py."
        ),
    },
    "smollm3": {
        "name": "SmolLM3-3B",
        "models": ["smollm3"],  # HF model_type value
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
        "transformer_lens_support": "full",  # via circuitkit._tl_compat port (not upstream TL)
        "status": "EXPERIMENTAL",
        "priority": 12,
        "notes": (
            "SmolLM3-3B: Llama-family (sequential attn/MLP block, RMSNorm, "
            "gated SiLU MLP), GQA, standard (non-interleaved) Llama rotary with "
            "per-layer NoPE driven by the HF config's no_rope_layers list (every "
            "4th layer skips rotary on the real checkpoint), tied embeddings, no "
            "logit scale. Discovery via the circuitkit _tl_compat TransformerLens "
            "port; public weights, no HF_TOKEN required. See docs/advanced/ "
            "(SmolLM3 page, once added in Stage 6).\n"
            "Stage 5 intervention validation (real weights, HuggingFaceTB/"
            "SmolLM3-3B): evaluation (circuitkit.api.evaluate_circuit "
            "faithfulness, patching+ablation pillars) confirmed finite on "
            "greater_than. Pruning (applications/pruning/score_extractor."
            "build_importance_dict) confirmed resolving real k_proj/gate_proj "
            "nn.Linear modules via this family's layers_path/attn/mlp config on "
            "a real layer subset. Quantization (applications/quantization/"
            "quant_utils.build_patterns) confirmed its fnmatch patterns match "
            "real nn.Linear submodules on the real loaded model; "
            "optimum-quanto's actual quantize()/freeze() call itself was NOT "
            "run (optional dependency not installed in the validation "
            "environment) -- only target-module resolution was exercised, "
            "which is what this family config is responsible for. Weight "
            "steering (applications/steering/weight_steering."
            "CircuitWeightSteering) confirmed resolving real per-head W_Q/W_K/"
            "W_V/W_O weight slices (including the 16:4 GQA query-head -> "
            "kv-head floor-div mapping) and applying a steering vector on real "
            "weights, producing a still-finite forward pass. This model was "
            "small enough (~6GB bf16) that no CPU fallback was needed for any "
            "Stage 5 gate, unlike the two ~7-8B cohere-family models. See "
            "tests/regression/test_smollm3_{evaluation,interventions}.py."
        ),
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
EXPERIMENTAL_FAMILIES = [
    f for f, cfg in MODEL_ARCH_REGISTRY.items() if cfg["status"] == "EXPERIMENTAL"
]
