"""
prune.py — Universal circuit-guided pruning pipeline for transformer models.

Supports any HuggingFace CausalLM model. Architecture is auto-detected.
Reference implementations for model-specific optimization (e.g., Qwen's
q_norm/k_norm handling) are available in prune_qwen.py and prune_llama.py.

Pipeline
--------
1. Run EAP-IG circuit discovery (or load pre-saved scores).
2. Load model via AutoModelForCausalLM (auto-detects architecture).
3. Evaluate BASE model on task data (accuracy, loss, latency).
4. Prune with CIRCUIT scores → evaluate (pre-finetune).
5. Prune with RANDOM scores → evaluate (pre-finetune).
6. Fine-tune each pruned model → evaluate (post-finetune).

Tested architectures
--------------------
PRODUCTION (fully tested):
  - LLaMA / Llama-2 / Llama-3
  - Qwen2 / Qwen2.5 / Qwen3
  - Gemma / Gemma-2

READY (high confidence):
  - Mistral-7B
  - Phi-3 / Phi-3.5

For Qwen models with special handling (q_norm/k_norm normalization):
  Use prune_qwen.py instead for optimized layer configuration.

Examples
--------
# Discover + prune + finetune in one shot (LLaMA):
python prune.py \\
    --base-model meta-llama/Meta-Llama-3-8B \\
    --task ioi --num-examples 200 --ig-steps 5 \\
    --attn-sparsity 0.2 --mlp-sparsity 0.3 \\
    --ft-examples 1000 --ft-epochs 2 \\
    --output-dir output/llama3_8b_ioi

# With Gemma:
python prune.py \\
    --base-model google/gemma-2-9b \\
    --task ioi --num-examples 200 \\
    --attn-sparsity 0.2 --mlp-sparsity 0.3 \\
    --output-dir output/gemma2_ioi

# Re-run pruning from saved scores (skip expensive discovery):
python prune.py \\
    --base-model meta-llama/Meta-Llama-3-8B \\
    --scores-path scores/llama3_ioi.pt \\
    --eval-data-path scores/llama3_ioi_eval.pt \\
    --attn-sparsity 0.2 --mlp-sparsity 0.3
"""

import argparse
import copy
import gc
import logging
import os
import random
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: make both circuitkit and LLM-Pruner importable
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import circuitkit as _ck  # anchor on the installed package: these scripts
# were relocated under examples/<area>/, so counting "../../.." overshot
# the repo root; the package dir is the stable location of applications/.
_APP_ROOT = os.path.dirname(os.path.abspath(_ck.__file__))  # .../src/circuitkit
_LLM_PRUNER = os.path.abspath(os.path.join(_APP_ROOT, "..", "LLM-Pruner"))

for _p in [_LLM_PRUNER]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# LLM-Pruner
import LLMPruner.torch_pruning as tp  # noqa: E402 - import after intentional pre-import setup

# circuitkit architecture support
from applications import (  # noqa: E402 - import after intentional pre-import setup
    UnsupportedArchitectureError,
    detect_model_architecture,
    get_arch_config,
)
from circuitkit.applications.pruning.eval_utils import (  # noqa: E402 - import after intentional pre-import setup
    full_eval,
    print_results_table,
)
from circuitkit.applications.pruning.finetune_utils import (  # noqa: E402 - import after intentional pre-import setup
    finetune_on_alpaca,
    finetune_on_task_data,
)
from circuitkit.applications.pruning.importance import (  # noqa: E402 - import after intentional pre-import setup
    CircuitKitImportance,
)

# circuitkit application helpers
from circuitkit.applications.pruning.score_extractor import (  # noqa: E402 - import after intentional pre-import setup
    aggregate_to_kv_heads,
    build_importance_dict,
    collect_eval_data,
    extract_mlp_neuron_scores,
    extract_q_head_scores,
    load_eval_data,
    load_scores,
    run_discovery,
    save_eval_data,
    save_scores,
)
from LLMPruner.pruner import (  # noqa: E402 - import after intentional pre-import setup
    hf_llama_pruner as llama_pruner,
)

# HuggingFace
from transformers import (  # noqa: E402 - import after intentional pre-import setup
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _layer_range(start: int, end: int, excluded: list) -> list:
    return [i for i in range(start, end) if i not in excluded]


def _get_rmsnorm_type(model):
    """
    Auto-detect the RMSNorm type used by the model.

    Returns (rmsnorm_class, module_path) tuple for use with customized_pruners.
    """
    # Try to find RMSNorm in the model's modules
    for module_type in type(model).__mro__:
        if hasattr(module_type, "__module__"):
            mod_name = module_type.__module__
            # Check common RMSNorm implementations
            if "llama" in mod_name.lower():
                try:
                    from transformers.models.llama.modeling_llama import LlamaRMSNorm

                    return LlamaRMSNorm
                except ImportError:
                    pass
            elif "qwen" in mod_name.lower():
                try:
                    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

                    return Qwen3RMSNorm
                except ImportError:
                    pass

    # Fallback: search for RMSNorm in model's actual modules
    for module in model.modules():
        module_type_name = type(module).__name__
        if "RMSNorm" in module_type_name:
            return type(module)

    # If no RMSNorm found, return None (some architectures may not use it)
    return None


def _prune_model(
    model,
    importance,
    attn_layers: list,
    mlp_layers: list,
    attn_sparsity: float,
    mlp_sparsity: float,
    global_pruning: bool,
    device: str,
    forward_prompts: torch.Tensor = None,
):
    """
    Apply one-shot block-wise pruning to a HuggingFace model (in-place).

    This is a universal implementation that works with LLaMA-compatible
    architectures (LLaMA, Gemma, Mistral, etc.). For specialized handling
    of specific architectures (e.g., Qwen's q_norm/k_norm), refer to
    the model-specific pruner scripts.

    Parameters
    ----------
    model          : HuggingFace CausalLM model
    importance     : tp.importance.Importance instance
    attn_layers    : List of attention layer indices to prune
    mlp_layers     : List of MLP layer indices to prune
    attn_sparsity  : Target sparsity for attention (0-1)
    mlp_sparsity   : Target sparsity for MLP (0-1)
    global_pruning : Whether to use global pruning
    device         : Device to run on
    """
    from circuitkit.applications import get_attn_proj, get_layers, get_mlp_proj

    # Detect architecture to get correct layer access patterns
    try:
        arch_type = detect_model_architecture(model)
        arch_cfg = get_arch_config(arch_type)
        layers = get_layers(model, arch_cfg)
    except UnsupportedArchitectureError as e:
        logger.warning(
            f"Architecture not in registry, attempting standard LLaMA-style access: {e}\n"
            f"If this fails, use model-specific pruner script (e.g., prune_qwen.py)"
        )
        # Fallback: assume LLaMA-style structure
        layers = model.model.layers
        arch_cfg = None

    # Build sparsity dict using architecture-aware layer access
    ch_sparsity_dict = {}
    for i in attn_layers:
        layer = layers[i]
        if arch_cfg:
            k_proj = get_attn_proj(layer, arch_cfg, "k_proj")
        else:
            k_proj = layer.self_attn.k_proj
        ch_sparsity_dict[k_proj] = attn_sparsity

    for i in mlp_layers:
        layer = layers[i]
        if arch_cfg:
            gate_proj = get_mlp_proj(layer, arch_cfg, "gate_proj")
        else:
            gate_proj = layer.mlp.gate_proj
        if gate_proj is not None:
            ch_sparsity_dict[gate_proj] = mlp_sparsity

    if forward_prompts is None:
        forward_prompts = torch.tensor(
            [
                [1, 306, 4658, 278, 6593, 310, 2834, 338],
                [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
            ]
        ).to(device)

    # Build consecutive_groups for GQA-aware pruning
    consecutive_groups = {}
    for i in attn_layers:
        layer = layers[i]
        if arch_cfg:
            k_proj = get_attn_proj(layer, arch_cfg, "k_proj")
            head_dim = layer.self_attn.head_dim
        else:
            k_proj = layer.self_attn.k_proj
            head_dim = layer.self_attn.head_dim
        consecutive_groups[k_proj] = head_dim

    # Get RMSNorm type for customized_pruners
    rmsnorm_type = _get_rmsnorm_type(model)
    customized_pruners = {}
    if rmsnorm_type is not None:
        customized_pruners[rmsnorm_type] = llama_pruner.hf_rmsnorm_pruner

    # Build root instances
    root_instances = []
    for i in attn_layers:
        layer = layers[i]
        if arch_cfg:
            k_proj = get_attn_proj(layer, arch_cfg, "k_proj")
        else:
            k_proj = layer.self_attn.k_proj
        root_instances.append(k_proj)

    for i in mlp_layers:
        layer = layers[i]
        if arch_cfg:
            gate_proj = get_mlp_proj(layer, arch_cfg, "gate_proj")
        else:
            gate_proj = layer.mlp.gate_proj
        if gate_proj is not None:
            root_instances.append(gate_proj)

    kwargs = {
        "importance": importance,
        "global_pruning": global_pruning,
        "iterative_steps": 1,
        "ch_sparsity": max(attn_sparsity, mlp_sparsity),
        "ch_sparsity_dict": ch_sparsity_dict,
        "ignored_layers": [],
        "consecutive_groups": consecutive_groups,
        "customized_pruners": customized_pruners,
        "root_module_types": None,
        "root_instances": root_instances,
    }

    # Disable KV cache for dependency tracing
    model.config.use_cache = False
    pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
    model.config.use_cache = True
    model.zero_grad()
    pruner.step()

    # Update head-count attributes post-pruning (LLaMA-compatible)
    for layer in layers:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, "q_proj") and hasattr(attn, "head_dim"):
                attn.num_heads = attn.q_proj.weight.data.shape[0] // attn.head_dim
                attn.num_key_value_heads = attn.k_proj.weight.data.shape[0] // attn.head_dim

    model.zero_grad()
    del pruner
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(args):
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: circuit discovery  +  eval data collection
    # ------------------------------------------------------------------
    eval_data = None

    if args.scores_path and args.eval_data_path:
        print(f"[pipeline] Loading pre-saved scores from {args.scores_path}")
        q_head_scores, mlp_scores = load_scores(args.scores_path)
        print(f"[pipeline] Loading pre-saved eval data from {args.eval_data_path}")
        eval_data = load_eval_data(args.eval_data_path)
    else:
        print("[pipeline] Running circuit discovery …")
        data_params = {}
        if args.task == "mmlu" and args.mmlu_subjects:
            data_params["subjects"] = args.mmlu_subjects
        if args.samples_per_subject:
            data_params["samples_per_subject"] = args.samples_per_subject

        discovery_cfg = {
            "algorithm": "eap-ig",
            "task": args.task,
            "level": "neuron",
            "mlp_hook": "post_act",
            "batch_size": args.batch_size,
            "ig_steps": args.ig_steps,
            "model_name": args.base_model,
            "data_params": {"num_examples": args.num_examples, **data_params},
            **data_params,
        }

        graph, tl_model = run_discovery(
            model_name=args.base_model,
            task=args.task,
            ig_steps=args.ig_steps,
            num_examples=args.num_examples,
            batch_size=args.batch_size,
            device=args.device,
            mlp_hook="post_act",
            precision=args.precision,
            data_params=data_params,
        )
        q_head_scores = extract_q_head_scores(graph)
        mlp_scores = extract_mlp_neuron_scores(graph)

        # Collect task eval data while TL model is still loaded
        print("[pipeline] Collecting task evaluation data …")
        eval_data = collect_eval_data(
            tl_model,
            args.task,
            discovery_cfg,
            args.device,
            max_examples=args.eval_examples,
        )

        if args.save_scores:
            save_scores(q_head_scores, mlp_scores, args.save_scores)
            eval_data_path = args.save_scores.replace(".pt", "_eval.pt")
            save_eval_data(eval_data, eval_data_path)

        del tl_model, graph
        gc.collect()
        torch.cuda.empty_cache()

    if not eval_data:
        raise RuntimeError("No eval data available. Run discovery or provide --eval-data-path.")

    if args.attn_sparsity == 0.0 and args.mlp_sparsity == 0.0:
        print("[pipeline] Both sparsities are 0.0 — scores saved, no pruning.")
        return

    # ------------------------------------------------------------------
    # Phase 2: load HuggingFace model (auto-detects architecture)
    # ------------------------------------------------------------------
    print(f"\n[pipeline] Loading {args.base_model} via HuggingFace …")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto" if args.device != "cpu" else None,
        torch_dtype=torch.float16,
    )

    # Auto-detect architecture
    try:
        arch_type = detect_model_architecture(base_model)
        print(f"[pipeline] Detected architecture: {arch_type}")
    except UnsupportedArchitectureError as e:
        print(f"[pipeline] WARNING: {e}")
        arch_type = None

    if args.device != "cpu":
        base_model.half()
    base_model.to(args.device)

    n_layers = base_model.config.num_hidden_layers
    n_q_heads = base_model.config.num_attention_heads
    n_kv_heads = getattr(base_model.config, "num_key_value_heads", n_q_heads)

    attn_layers = _layer_range(
        args.block_attention_layer_start,
        args.block_attention_layer_end or n_layers,
        args.excluded_attn_layers,
    )
    mlp_layers = _layer_range(
        args.block_mlp_layer_start,
        args.block_mlp_layer_end or n_layers,
        args.excluded_mlp_layers,
    )
    print(f"[pipeline] GQA: {n_q_heads} Q → {n_kv_heads} KV heads")
    print(f"[pipeline] Attn layers: {attn_layers}")
    print(f"[pipeline] MLP  layers: {mlp_layers}")

    # Aggregate Q → KV head scores once
    kv_head_scores = aggregate_to_kv_heads(
        q_head_scores, n_q_heads, n_kv_heads, reduction=args.head_aggregation
    )

    results = {}

    # ------------------------------------------------------------------
    # Phase 3: evaluate BASE model
    # ------------------------------------------------------------------
    print("\n[pipeline] Evaluating BASE model …")
    base_model.config.pad_token_id = tokenizer.pad_token_id = 0
    results["Base"] = full_eval(
        base_model, tokenizer, eval_data, args.eval_device, max_eval_examples=args.eval_examples
    )
    print(
        f"  accuracy={results['Base']['accuracy']:.4f}  "
        f"loss={results['Base']['loss']:.4f}  "
        f"lat={results['Base']['latency_ms_per_token']:.2f} ms/tok"
    )

    # ------------------------------------------------------------------
    # Phase 4: CIRCUIT pruning
    # ------------------------------------------------------------------
    print("[pipeline] Pruning with CIRCUIT scores …")
    for p in base_model.parameters():
        p.requires_grad_(True)
    circuit_model = copy.deepcopy(base_model)

    forward_prompts = tokenizer(
        ["The capital of France is", "In recent years, the development of"],
        return_tensors="pt",
        padding=True,
    )["input_ids"].to(args.device)

    print("\n[pipeline] Building circuit importance dict …")
    scores_dict = build_importance_dict(
        circuit_model, kv_head_scores, mlp_scores, attn_layers, mlp_layers
    )
    circuit_imp = CircuitKitImportance(scores_dict)
    _prune_model(
        circuit_model,
        circuit_imp,
        attn_layers,
        mlp_layers,
        args.attn_sparsity,
        args.mlp_sparsity,
        args.global_pruning,
        args.device,
        forward_prompts=forward_prompts,
    )
    circuit_model.config.pad_token_id = tokenizer.pad_token_id = 0
    circuit_model.to(args.eval_device)

    print("[pipeline] Evaluating CIRCUIT pruned (pre-FT) …")
    results["Circuit (pre-FT)"] = full_eval(
        circuit_model,
        tokenizer,
        eval_data,
        args.eval_device,
        max_eval_examples=args.eval_examples,
    )
    print(
        f"  accuracy={results['Circuit (pre-FT)']['accuracy']:.4f}  "
        f"loss={results['Circuit (pre-FT)']['loss']:.4f}  "
        f"lat={results['Circuit (pre-FT)']['latency_ms_per_token']:.2f} ms/tok"
    )

    # ------------------------------------------------------------------
    # Phase 5: RANDOM pruning baseline (if requested)
    # ------------------------------------------------------------------
    if args.compare_random:
        print("\n[pipeline] Pruning with RANDOM scores …")
        random_imp = tp.importance.RandomImportance()

        random_model = copy.deepcopy(base_model)
        _prune_model(
            random_model,
            random_imp,
            attn_layers,
            mlp_layers,
            args.attn_sparsity,
            args.mlp_sparsity,
            args.global_pruning,
            args.device,
            forward_prompts=forward_prompts,
        )
        random_model.config.pad_token_id = tokenizer.pad_token_id = 0
        random_model.to(args.eval_device)

        print("[pipeline] Evaluating RANDOM pruned (pre-FT) …")
        results["Random (pre-FT)"] = full_eval(
            random_model,
            tokenizer,
            eval_data,
            args.eval_device,
            max_eval_examples=args.eval_examples,
        )
        print(
            f"  accuracy={results['Random (pre-FT)']['accuracy']:.4f}  "
            f"loss={results['Random (pre-FT)']['loss']:.4f}  "
            f"lat={results['Random (pre-FT)']['latency_ms_per_token']:.2f} ms/tok"
        )

    # ------------------------------------------------------------------
    # Phase 6: Fine-tuning (optional)
    # ------------------------------------------------------------------
    if args.ft_epochs > 0 and args.ft_examples > 0:
        print("\n[pipeline] Fine-tuning models …")

        if args.ft_data == "alpaca":
            print("[finetune] Using Alpaca instruction-following data")
            ft_kwargs = {
                "n_examples": args.ft_examples,
                "n_epochs": args.ft_epochs,
                "device": args.device,
            }
            finetune_fn = finetune_on_alpaca
        else:
            print("[finetune] Using task discovery data")
            ft_kwargs = {
                "eval_data": eval_data,
                "n_epochs": args.ft_epochs,
                "device": args.device,
            }
            finetune_fn = finetune_on_task_data

        # Fine-tune circuit model
        print("[finetune] Fine-tuning CIRCUIT pruned model …")
        circuit_model = finetune_fn(circuit_model, tokenizer, **ft_kwargs)
        circuit_model.to(args.eval_device)

        print("[pipeline] Evaluating CIRCUIT pruned (post-FT) …")
        results["Circuit (post-FT)"] = full_eval(
            circuit_model,
            tokenizer,
            eval_data,
            args.eval_device,
            max_eval_examples=args.eval_examples,
        )
        print(
            f"  accuracy={results['Circuit (post-FT)']['accuracy']:.4f}  "
            f"loss={results['Circuit (post-FT)']['loss']:.4f}  "
            f"lat={results['Circuit (post-FT)']['latency_ms_per_token']:.2f} ms/tok"
        )

        if args.compare_random:
            print("[finetune] Fine-tuning RANDOM pruned model …")
            random_model = finetune_fn(random_model, tokenizer, **ft_kwargs)
            random_model.to(args.eval_device)

            print("[pipeline] Evaluating RANDOM pruned (post-FT) …")
            results["Random (post-FT)"] = full_eval(
                random_model,
                tokenizer,
                eval_data,
                args.eval_device,
                max_eval_examples=args.eval_examples,
            )
            print(
                f"  accuracy={results['Random (post-FT)']['accuracy']:.4f}  "
                f"loss={results['Random (post-FT)']['loss']:.4f}  "
                f"lat={results['Random (post-FT)']['latency_ms_per_token']:.2f} ms/tok"
            )

    # ------------------------------------------------------------------
    # Phase 7: Print results
    # ------------------------------------------------------------------
    print_results_table(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Circuit-guided pruning for any HuggingFace CausalLM (universal)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        "--base-model", type=str, required=True, help="HuggingFace model ID or local path"
    )

    # Discovery
    p.add_argument(
        "--task",
        type=str,
        default="ioi",
        choices=[
            "ioi",
            "mmlu",
            "sva",
            "greater_than",
            "hypernymy",
            "gender_bias",
            "capital_country",
        ],
        help="circuitkit task for attribution and evaluation",
    )
    p.add_argument(
        "--ig-steps",
        type=int,
        default=5,
        help="Integrated Gradients steps (higher = slower, more accurate)",
    )
    p.add_argument(
        "--num-examples", type=int, default=200, help="Task examples used for attribution scoring"
    )
    p.add_argument(
        "--batch-size", type=int, default=4, help="Dataloader batch size during discovery"
    )
    p.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="TransformerLens model precision for discovery",
    )
    p.add_argument(
        "--mmlu-subjects",
        type=str,
        nargs="+",
        default=None,
        help="MMLU subjects to use (default: all)",
    )
    p.add_argument(
        "--samples-per-subject", type=int, default=None, help="MMLU examples per subject"
    )

    # Score / eval data I/O
    p.add_argument(
        "--scores-path",
        type=str,
        default=None,
        help="Pre-saved scores file; skips discovery if --eval-data-path also given",
    )
    p.add_argument(
        "--eval-data-path",
        type=str,
        default=None,
        help="Pre-saved eval data file (required when --scores-path is set)",
    )
    p.add_argument(
        "--save-scores",
        type=str,
        default=None,
        help="Where to save discovery scores (eval data saved alongside)",
    )

    # Pruning
    p.add_argument(
        "--attn-sparsity",
        type=float,
        default=0.0,
        help="Target sparsity for attention layers [0, 1]",
    )
    p.add_argument(
        "--mlp-sparsity", type=float, default=0.0, help="Target sparsity for MLP layers [0, 1]"
    )
    p.add_argument(
        "--block-attention-layer-start", type=int, default=0, help="First attention layer to prune"
    )
    p.add_argument(
        "--block-attention-layer-end",
        type=int,
        default=None,
        help="Last attention layer to prune (default: num_layers)",
    )
    p.add_argument("--block-mlp-layer-start", type=int, default=0, help="First MLP layer to prune")
    p.add_argument(
        "--block-mlp-layer-end",
        type=int,
        default=None,
        help="Last MLP layer to prune (default: num_layers)",
    )
    p.add_argument(
        "--excluded-attn-layers", type=int, nargs="+", default=[], help="Attention layers to skip"
    )
    p.add_argument(
        "--excluded-mlp-layers", type=int, nargs="+", default=[], help="MLP layers to skip"
    )
    p.add_argument(
        "--head-aggregation",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="How to aggregate Q-head scores to KV-head level",
    )
    p.add_argument(
        "--global-pruning",
        action="store_true",
        help="Use global pruning (compare across all channels)",
    )
    p.add_argument(
        "--compare-random", action="store_true", help="Also run random-sparsity pruning as baseline"
    )

    # Fine-tuning
    p.add_argument(
        "--ft-epochs",
        type=int,
        default=0,
        help="Number of fine-tuning epochs (0 = skip fine-tuning)",
    )
    p.add_argument(
        "--ft-examples", type=int, default=500, help="Number of examples for fine-tuning"
    )
    p.add_argument("--ft-batch-size", type=int, default=4, help="Batch size for fine-tuning")
    p.add_argument(
        "--ft-data",
        type=str,
        default="alpaca",
        choices=["alpaca", "task"],
        help="Fine-tuning data: Alpaca instructions or circuit discovery task",
    )

    # Evaluation
    p.add_argument(
        "--eval-examples",
        type=int,
        default=None,
        help="Max examples for task evaluation (None = all)",
    )

    # Devices
    p.add_argument("--device", type=str, default="cuda", help="Device for discovery and pruning")
    p.add_argument(
        "--eval-device", type=str, default="cuda", help="Device for evaluation forward passes"
    )

    # Output
    p.add_argument(
        "--output-dir", type=str, default="output/pruned", help="Directory for saving outputs"
    )

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
