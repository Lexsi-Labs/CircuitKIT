"""
prune_llama.py — Full circuit-guided + random-baseline pruning pipeline for
                 LLaMA / Llama-2 / Llama-3 models.

Pipeline
--------
1. Run EAP-IG circuit discovery (or load pre-saved scores).
   Also collects task evaluation data while the TL model is loaded.
2. Load HuggingFace LlamaForCausalLM.
3. Evaluate BASE model on task data (accuracy, loss, latency).
4. Prune with CIRCUIT scores → evaluate (pre-finetune).
5. Prune with RANDOM scores   → evaluate (pre-finetune).
6. Fine-tune each pruned model → evaluate (post-finetune).
   --ft-data=alpaca  (default) trains on Alpaca instruction-following data.
   --ft-data=task    trains on the circuit-discovery task data itself.
7. Print comparison table:
       Base | Circuit (pre-FT) | Random (pre-FT) | Circuit (post-FT) | Random (post-FT)

GQA note: For Llama-3 / Llama-3.1 the pruner anchors on k_proj with
consecutive_groups=head_dim, ensuring whole KV heads are removed.
Q-head circuit scores are aggregated to KV-head level first.

Examples
--------
# Discover + prune + finetune in one shot:
python prune_llama.py \\
    --base-model meta-llama/Meta-Llama-3-8B \\
    --task ioi --num-examples 200 --ig-steps 5 \\
    --attn-sparsity 0.2 --mlp-sparsity 0.3 \\
    --ft-examples 1000 --ft-epochs 2 \\
    --save-scores scores/llama3_8b_ioi.pt \\
    --output-dir output/llama3_8b_ioi

# Re-run pruning from saved scores (skip expensive discovery):
python prune_llama.py \\
    --base-model meta-llama/Meta-Llama-3-8B \\
    --scores-path scores/llama3_8b_ioi.pt \\
    --eval-data-path scores/llama3_8b_ioi_eval.pt \\
    --attn-sparsity 0.2 --mlp-sparsity 0.3 \\
    --ft-examples 500 --ft-epochs 1
"""

import argparse
import copy
import gc
import os
import random
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: make both circuitkit and LLM-Pruner importable
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))  # circuitkit/
_LLM_PRUNER = os.path.abspath(os.path.join(_APP_ROOT, "..", "LLM-Pruner"))

for _p in [_APP_ROOT, _LLM_PRUNER]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# LLM-Pruner
import LLMPruner.torch_pruning as tp  # noqa: E402 - import after intentional pre-import setup
from applications.pruning.eval_utils import (  # noqa: E402 - import after intentional pre-import setup
    full_eval,
    print_results_table,
)
from applications.pruning.finetune_utils import (  # noqa: E402 - import after intentional pre-import setup
    finetune_on_alpaca,
    finetune_on_task_data,
)
from applications.pruning.importance import (  # noqa: E402 - import after intentional pre-import setup
    CircuitKitImportance,
)

# circuitkit application helpers
from applications.pruning.score_extractor import (  # noqa: E402 - import after intentional pre-import setup
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
    AutoTokenizer,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import (  # noqa: E402 - import after intentional pre-import setup
    LlamaRMSNorm,
)

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


def _prune_model(
    model,
    importance,
    attn_layers: list,
    mlp_layers: list,
    attn_sparsity: float,
    mlp_sparsity: float,
    global_pruning: bool,
    device: str,
):
    """
    Apply one-shot block-wise pruning to a LlamaForCausalLM.

    The model is modified IN-PLACE.  Call on a deep copy to preserve the
    original.

    importance : a tp.importance.Importance instance (CircuitKitImportance
                 or tp.importance.RandomImportance).
    """
    ch_sparsity_dict = {}
    for i in attn_layers:
        ch_sparsity_dict[model.model.layers[i].self_attn.k_proj] = attn_sparsity
    for i in mlp_layers:
        ch_sparsity_dict[model.model.layers[i].mlp.gate_proj] = mlp_sparsity

    forward_prompts = torch.tensor(
        [
            [1, 306, 4658, 278, 6593, 310, 2834, 338],
            [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
        ]
    ).to(device)

    kwargs = {
        "importance": importance,
        "global_pruning": global_pruning,
        "iterative_steps": 1,
        "ch_sparsity": max(attn_sparsity, mlp_sparsity),
        "ch_sparsity_dict": ch_sparsity_dict,
        "ignored_layers": [],
        "consecutive_groups": {
            model.model.layers[i].self_attn.k_proj: model.model.layers[i].self_attn.head_dim
            for i in attn_layers
        },
        "customized_pruners": {LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner},
        "root_module_types": None,
        "root_instances": (
            [model.model.layers[i].self_attn.k_proj for i in attn_layers]
            + [model.model.layers[i].mlp.gate_proj for i in mlp_layers]
        ),
    }

    # Disable KV cache so the model returns a plain tensor (logits only) during
    # dependency tracing.  Newer transformers return a DynamicCache object for
    # past_key_values which is not handled by LLM-Pruner's flatten_as_list.
    model.config.use_cache = False
    pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
    model.config.use_cache = True
    model.zero_grad()
    pruner.step()

    # Update head-count attributes post-pruning
    for layer in model.model.layers:
        attn = layer.self_attn
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
    # Phase 2: load HuggingFace model once; take deep copies for each variant
    # ------------------------------------------------------------------
    print(f"\n[pipeline] Loading {args.base_model} via HuggingFace …")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto" if args.device != "cpu" else None,
        torch_dtype=torch.float16,
    )
    if args.device != "cpu":
        base_model.half()
    base_model.to(args.device)

    n_layers = base_model.config.num_hidden_layers
    n_q_heads = base_model.config.num_attention_heads
    n_kv_heads = base_model.config.num_key_value_heads

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
        f"loss={results['Circuit (pre-FT)']['loss']:.4f}"
    )

    # ------------------------------------------------------------------
    # Phase 5: RANDOM pruning
    # ------------------------------------------------------------------
    print("\n[pipeline] Pruning with RANDOM scores …")
    random_imp = tp.importance.RandomImportance()
    for p in base_model.parameters():
        p.requires_grad_(True)
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
        f"loss={results['Random (pre-FT)']['loss']:.4f}"
    )

    # ------------------------------------------------------------------
    # Phase 6: post-fine-tuning
    # ------------------------------------------------------------------
    skip_ft = args.ft_data == "alpaca" and args.ft_examples == 0
    if not skip_ft:
        ft_output_circuit = os.path.join(args.output_dir, "circuit_finetuned")
        ft_output_random = os.path.join(args.output_dir, "random_finetuned")

        ft_label = "task data" if args.ft_data == "task" else f"{args.ft_examples} Alpaca examples"

        def _finetune(pruned_model, tag, ft_output):
            print(f"\n[pipeline] Fine-tuning {tag} pruned model on {ft_label} …")
            if args.ft_data == "task":
                return finetune_on_task_data(
                    pruned_model,
                    tokenizer,
                    eval_data=eval_data,
                    n_epochs=args.ft_epochs,
                    learning_rate=args.ft_lr,
                    lora_r=args.lora_r,
                    device=args.device,
                    output_dir=ft_output,
                )
            else:
                return finetune_on_alpaca(
                    pruned_model,
                    tokenizer,
                    n_examples=args.ft_examples,
                    n_epochs=args.ft_epochs,
                    learning_rate=args.ft_lr,
                    lora_r=args.lora_r,
                    device=args.device,
                    output_dir=ft_output,
                )

        circuit_ft_model = _finetune(circuit_model, "CIRCUIT", ft_output_circuit)
        circuit_ft_model.config.pad_token_id = 0
        circuit_ft_model.to(args.eval_device)

        print("[pipeline] Evaluating CIRCUIT pruned (post-FT) …")
        results["Circuit (post-FT)"] = full_eval(
            circuit_ft_model,
            tokenizer,
            eval_data,
            args.eval_device,
            max_eval_examples=args.eval_examples,
        )
        print(
            f"  accuracy={results['Circuit (post-FT)']['accuracy']:.4f}  "
            f"loss={results['Circuit (post-FT)']['loss']:.4f}"
        )

        random_ft_model = _finetune(random_model, "RANDOM", ft_output_random)
        random_ft_model.config.pad_token_id = 0
        random_ft_model.to(args.eval_device)

        print("[pipeline] Evaluating RANDOM pruned (post-FT) …")
        results["Random (post-FT)"] = full_eval(
            random_ft_model,
            tokenizer,
            eval_data,
            args.eval_device,
            max_eval_examples=args.eval_examples,
        )
        print(
            f"  accuracy={results['Random (post-FT)']['accuracy']:.4f}  "
            f"loss={results['Random (post-FT)']['loss']:.4f}"
        )

        # Save pruned + fine-tuned models
        if args.save_model:
            ckpt_dir = os.path.join(args.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {"model": circuit_ft_model, "tokenizer": tokenizer},
                os.path.join(ckpt_dir, "circuit_finetuned.pt"),
            )
            torch.save(
                {"model": random_ft_model, "tokenizer": tokenizer},
                os.path.join(ckpt_dir, "random_finetuned.pt"),
            )
            print(f"[pipeline] Saved fine-tuned checkpoints to {ckpt_dir}")

    elif args.save_model:
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {"model": circuit_model, "tokenizer": tokenizer},
            os.path.join(ckpt_dir, "circuit_pruned.pt"),
        )
        torch.save(
            {"model": random_model, "tokenizer": tokenizer},
            os.path.join(ckpt_dir, "random_pruned.pt"),
        )
        print(f"[pipeline] Saved pruned checkpoints to {ckpt_dir}")

    # ------------------------------------------------------------------
    # Phase 7: print comparison table
    # ------------------------------------------------------------------
    print_results_table(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Circuit-guided + random-baseline structured pruning for LLaMA",
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
        "--precision", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    p.add_argument("--mmlu-subjects", type=str, nargs="+", default=None)
    p.add_argument("--samples-per-subject", type=int, default=None)

    # Score / eval data I/O
    p.add_argument(
        "--scores-path",
        type=str,
        default=None,
        help="Pre-saved scores file; skips discovery if set",
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
        help="Where to save discovery scores (auto-saves eval data alongside)",
    )

    # Pruning
    p.add_argument(
        "--attn-sparsity",
        type=float,
        default=0.2,
        help="Fraction of KV heads to prune per attention layer (0–1)",
    )
    p.add_argument(
        "--mlp-sparsity",
        type=float,
        default=0.3,
        help="Fraction of MLP gate_proj channels to prune per layer (0–1)",
    )
    p.add_argument(
        "--global-pruning",
        action="store_true",
        help="Apply a single global sparsity threshold across all layers",
    )
    p.add_argument(
        "--head-aggregation",
        type=str,
        default="sum",
        choices=["sum", "mean", "max"],
        help="How to combine Q-head scores into KV-head scores (GQA)",
    )

    # Layer selection
    p.add_argument("--block-attention-layer-start", type=int, default=0)
    p.add_argument("--block-attention-layer-end", type=int, default=None)
    p.add_argument("--block-mlp-layer-start", type=int, default=0)
    p.add_argument("--block-mlp-layer-end", type=int, default=None)
    p.add_argument(
        "--excluded-attn-layers",
        type=int,
        nargs="*",
        default=[],
        help="Attention layers to skip (e.g. first/last layers)",
    )
    p.add_argument(
        "--excluded-mlp-layers", type=int, nargs="*", default=[], help="MLP layers to skip"
    )

    # Fine-tuning
    p.add_argument(
        "--ft-data",
        type=str,
        default="alpaca",
        choices=["alpaca", "task"],
        help="Fine-tuning data source: 'alpaca' for general instruction "
        "data, 'task' for the circuit-discovery task data itself",
    )
    p.add_argument(
        "--ft-examples",
        type=int,
        default=1000,
        help="Alpaca examples for post-pruning fine-tuning (0 = skip FT). "
        "Ignored when --ft-data=task (uses all collected eval data).",
    )
    p.add_argument("--ft-epochs", type=int, default=2, help="Fine-tuning epochs")
    p.add_argument("--ft-lr", type=float, default=3e-4, help="LoRA learning rate")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank")

    # Evaluation
    p.add_argument(
        "--eval-examples",
        type=int,
        default=None,
        help="Max examples for evaluation (None = all collected)",
    )

    # Devices
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval-device", type=str, default="cuda")

    # Output
    p.add_argument(
        "--output-dir",
        type=str,
        default="output/llama_prune",
        help="Directory for checkpoints and saved scores",
    )
    p.add_argument(
        "--save-model", action="store_true", help="Save pruned (and fine-tuned) model checkpoints"
    )

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
