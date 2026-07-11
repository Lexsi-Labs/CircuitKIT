"""
quantize_llama.py — Full circuit-guided quantization pipeline for
                    LLaMA / Llama-2 / Llama-3 models.

Pipeline
--------
1. Run EAP-IG circuit discovery at NODE level (or load pre-saved scores).
   Also collects task evaluation data while the TL model is loaded.
2. Load HuggingFace LlamaForCausalLM.
3. Evaluate BASE model on task data (accuracy, loss, latency).
4. Quantize with CIRCUIT-guided mixed precision → freeze → evaluate.
5. If --compare-uniform: quantize uniformly (same low precision everywhere)
   as a comparison baseline → freeze → evaluate.
6. Print plan summary and results table.

Mixed-precision strategy
------------------------
Node-level discovery (is_neuron=False) assigns one scalar score per
attention head and one per MLP block.  Head scores are aggregated to a
single per-layer attention score.  Circuit-important layers are protected
at higher precision (--high-weights, default: native float16) while
unimportant layers are quantized aggressively (--low-weights, default:
qint4).  --high-fraction controls what fraction of layers are "important".

Discovery uses mlp_hook="mlp_out" (mlp2: measures at the output of the
complete MLP block in d_model space).

Reuses
------
* score_extractor.py  (this package)          — node-level discovery + eval data
* eval_utils.py       (applications/pruning/) — task accuracy + latency
* quant_utils.py      (this package)          — tier assignment + quantization

Examples
--------
# Discover + circuit-quantize in one shot:
python quantize_llama.py \\
    --base-model meta-llama/Llama-3.2-1B \\
    --task ioi --num-examples 200 --ig-steps 5 \\
    --low-weights qint4 --high-weights qint8 --high-fraction 0.3 \\
    --compare-uniform --eval-ppl \\
    --save-scores scores/llama_ioi.pt

# Re-quantize from saved scores (skip expensive discovery):
python quantize_llama.py \\
    --base-model meta-llama/Llama-3.2-1B \\
    --scores-path scores/llama_ioi.pt \\
    --eval-data-path scores/llama_ioi_eval.pt \\
    --low-weights qint4 --high-weights qint8 --high-fraction 0.3 \\
    --compare-uniform --eval-ppl
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
# CUDA_HOME fix: optimum-quanto's qint4 AWQ kernel JIT-compiles on first use
# and dies if CUDA_HOME is unset.  Try multiple strategies to locate CUDA.
# ---------------------------------------------------------------------------
def _resolve_cuda_home() -> str | None:
    import glob
    import subprocess

    # 1. torch's own detection (works when torch was installed with cuda toolkit)
    try:
        from torch.utils.cpp_extension import CUDA_HOME as _th

        if _th and os.path.isdir(_th):
            return _th
    except Exception:
        pass

    # 2. nvcc in PATH  →  …/bin/nvcc  →  parent = cuda root
    try:
        r = subprocess.run(["which", "nvcc"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            return os.path.dirname(os.path.dirname(r.stdout.strip()))
    except Exception:
        pass

    # 3. common installation paths (latest first)
    for pattern in ["/usr/local/cuda-*", "/usr/local/cuda", "/opt/cuda"]:
        for candidate in sorted(glob.glob(pattern), reverse=True):
            if os.path.isfile(os.path.join(candidate, "bin", "nvcc")):
                return candidate

    return None


if "CUDA_HOME" not in os.environ:
    _cuda_home = _resolve_cuda_home()
    if _cuda_home:
        os.environ["CUDA_HOME"] = _cuda_home
        print(f"[env] Set CUDA_HOME={_cuda_home}")
    else:
        print(
            "[env] WARNING: could not detect CUDA_HOME; qint4 AWQ kernel may fail. "
            "Set CUDA_HOME manually or use --low-weights qint8."
        )


# ---------------------------------------------------------------------------
# TORCH_CUDA_ARCH_LIST fix: optimum-quanto's qint4 kernels (gemm_cuda /
# fp8_marlin) use sm_80+ PTX instructions (cp.async, m16n8k16 MMA).  Without an
# arch list torch builds the extension for *every* known architecture,
# including sm_75 (Turing), and ptxas aborts the whole build on the sm_75
# target -- so qint4 quantization and quanto-checkpoint reload both fail.
# Pin the JIT build to the local GPU's actual compute capability instead.
# ---------------------------------------------------------------------------
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _caps = {
                f"{_maj}.{_min}"
                for _i in range(_torch.cuda.device_count())
                for (_maj, _min) in [_torch.cuda.get_device_capability(_i)]
            }
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(_caps))
            print(f"[env] Set TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Path setup: make circuitkit, pruning-app utilities, and optimum-quanto importable
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))  # circuitkit/
_QUANTO_ROOT = os.path.abspath(os.path.join(_APP_ROOT, "..", "optimum-quanto"))
_QUANT_APP = os.path.join(_APP_ROOT, "applications", "quantization")

for _p in [_APP_ROOT, _QUANTO_ROOT, _QUANT_APP]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from applications.pruning.eval_utils import (  # noqa: E402 - import after intentional pre-import setup
    full_eval,
    print_results_table,
)

# Quantization utilities (this package)
from applications.quantization.quant_utils import (  # noqa: E402 - import after intentional pre-import setup
    calibrate_quantized_model,
    circuit_quantize,
    compute_ppl,
    freeze_model,
    print_quantization_plan,
    random_quantize,
)

# circuitkit application helpers (node-level score extractor for quantization)
from applications.quantization.score_extractor import (  # noqa: E402 - import after intentional pre-import setup
    collect_eval_data,
    extract_node_head_scores,
    extract_node_mlp_scores,
    load_eval_data,
    load_scores,
    run_discovery,
    save_eval_data,
    save_scores,
)

# optimum-quanto
from optimum.quanto import (  # noqa: E402 - import after intentional pre-import setup
    freeze as quanto_freeze,
)
from optimum.quanto import (  # noqa: E402 - import after intentional pre-import setup
    qfloat8,
    qint4,
    qint8,
    quantization_map,
)
from optimum.quanto import (  # noqa: E402 - import after intentional pre-import setup
    quantize as quanto_quantize,
)

# HuggingFace
from transformers import (  # noqa: E402 - import after intentional pre-import setup
    AutoTokenizer,
    LlamaForCausalLM,
)

# ---------------------------------------------------------------------------
# qtype registry
# ---------------------------------------------------------------------------
QTYPES = {
    "qint4": qint4,
    "qint8": qint8,
    "qfloat8": qfloat8,
    "none": None,
}


def resolve_qtype(name: str):
    if name not in QTYPES:
        raise ValueError(f"Unknown qtype {name!r}. Choose from: {list(QTYPES)}")
    return QTYPES[name]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(args):
    set_random_seed(args.seed)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    low_weights = resolve_qtype(args.low_weights)
    high_weights = resolve_qtype(args.high_weights)
    mid_weights = resolve_qtype(args.mid_weights)
    act_weights = resolve_qtype(args.activations)

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
            "level": "node",
            "mlp_hook": "mlp_out",
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
            mlp_hook="mlp_out",
            precision=args.precision,
            data_params=data_params,
        )
        q_head_scores = extract_node_head_scores(graph)  # {(layer, head): float}
        mlp_scores = extract_node_mlp_scores(graph)  # {layer: float}

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
            print(f"[pipeline] Saved scores → {args.save_scores}")
            print(f"[pipeline] Saved eval data → {eval_data_path}")

        del tl_model, graph
        gc.collect()
        torch.cuda.empty_cache()

    if not eval_data:
        raise RuntimeError("No eval data available. Run discovery or provide --eval-data-path.")

    if low_weights is None:
        print("[pipeline] --low-weights is 'none' — scores saved, no quantization.")
        return

    # ------------------------------------------------------------------
    # Phase 2: load HuggingFace model
    # ------------------------------------------------------------------
    print(f"\n[pipeline] Loading {args.base_model} via HuggingFace …")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or 0

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
    )
    base_model.to(args.device)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    n_layers = base_model.config.num_hidden_layers
    print(f"[pipeline] Model has {n_layers} layers")

    results = {}

    # ------------------------------------------------------------------
    # Phase 3: evaluate BASE model
    # ------------------------------------------------------------------
    print("\n[pipeline] Evaluating BASE model …")
    results["Base"] = full_eval(
        base_model,
        tokenizer,
        eval_data,
        args.eval_device,
        max_eval_examples=args.eval_examples,
    )
    print(
        f"  accuracy={results['Base']['accuracy']:.4f}  "
        f"loss={results['Base']['loss']:.4f}  "
        f"lat={results['Base']['latency_ms_per_token']:.2f} ms/tok"
    )

    if args.eval_ppl:
        print("[pipeline] Computing BASE perplexity …")
        results["Base"]["ppl"] = compute_ppl(
            base_model,
            tokenizer,
            args.eval_device,
            n_samples=args.ppl_samples,
            seq_len=args.ppl_seq_len,
        )
        print(f"  ppl={results['Base']['ppl']:.2f}")

    # ------------------------------------------------------------------
    # Phase 4: CIRCUIT-guided quantization
    # ------------------------------------------------------------------
    print("\n[pipeline] Applying CIRCUIT-guided quantization …")
    circuit_model = copy.deepcopy(base_model)

    plan = circuit_quantize(
        circuit_model,
        q_head_scores=q_head_scores,
        mlp_scores=mlp_scores,
        n_layers=n_layers,
        low_weights=low_weights,
        high_weights=high_weights,
        mid_weights=mid_weights,
        activations=act_weights,
        high_fraction=args.high_fraction,
        mid_fraction=args.mid_fraction,
        score_aggregation=args.score_aggregation,
        exclude_lm_head=True,
        # model_type is now auto-detected from model.config.model_type
    )

    if act_weights is not None:
        calibrate_quantized_model(
            circuit_model,
            tokenizer,
            eval_data,
            args.device,
            n_samples=args.calib_samples,
        )

    freeze_model(circuit_model)
    circuit_model.config.pad_token_id = tokenizer.pad_token_id
    circuit_model.to(args.eval_device)

    print_quantization_plan(plan, low_weights, high_weights, mid_weights)

    print("[pipeline] Evaluating CIRCUIT-quantized model …")
    results["Circuit-Quant"] = full_eval(
        circuit_model,
        tokenizer,
        eval_data,
        args.eval_device,
        max_eval_examples=args.eval_examples,
    )
    print(
        f"  accuracy={results['Circuit-Quant']['accuracy']:.4f}  "
        f"loss={results['Circuit-Quant']['loss']:.4f}  "
        f"lat={results['Circuit-Quant']['latency_ms_per_token']:.2f} ms/tok"
    )

    if args.eval_ppl:
        print("[pipeline] Computing CIRCUIT-quantized perplexity …")
        results["Circuit-Quant"]["ppl"] = compute_ppl(
            circuit_model,
            tokenizer,
            args.eval_device,
            n_samples=args.ppl_samples,
            seq_len=args.ppl_seq_len,
        )
        print(f"  ppl={results['Circuit-Quant']['ppl']:.2f}")

    # ------------------------------------------------------------------
    # Phase 5: RANDOM quantization baseline
    # ------------------------------------------------------------------
    if args.compare_random:
        print("\n[pipeline] Applying RANDOM-tier quantization baseline …")
        random_model = copy.deepcopy(base_model)

        random_plan = random_quantize(
            random_model,
            n_layers=n_layers,
            low_weights=low_weights,
            high_weights=high_weights,
            mid_weights=mid_weights,
            activations=act_weights,
            high_fraction=args.high_fraction,
            mid_fraction=args.mid_fraction,
            exclude_lm_head=True,
            # model_type is now auto-detected from model.config.model_type
            seed=args.seed,
        )

        if act_weights is not None:
            calibrate_quantized_model(
                random_model,
                tokenizer,
                eval_data,
                args.device,
                n_samples=args.calib_samples,
            )

        freeze_model(random_model)
        random_model.config.pad_token_id = tokenizer.pad_token_id
        random_model.to(args.eval_device)

        print_quantization_plan(random_plan, low_weights, high_weights, mid_weights)

        print("[pipeline] Evaluating RANDOM-tier quantized model …")
        results["Random-Quant"] = full_eval(
            random_model,
            tokenizer,
            eval_data,
            args.eval_device,
            max_eval_examples=args.eval_examples,
        )
        print(
            f"  accuracy={results['Random-Quant']['accuracy']:.4f}  "
            f"loss={results['Random-Quant']['loss']:.4f}  "
            f"lat={results['Random-Quant']['latency_ms_per_token']:.2f} ms/tok"
        )

        if args.eval_ppl:
            print("[pipeline] Computing RANDOM-tier perplexity …")
            results["Random-Quant"]["ppl"] = compute_ppl(
                random_model,
                tokenizer,
                args.eval_device,
                n_samples=args.ppl_samples,
                seq_len=args.ppl_seq_len,
            )
            print(f"  ppl={results['Random-Quant']['ppl']:.2f}")

    # ------------------------------------------------------------------
    # Phase 6: UNIFORM quantization baseline
    # ------------------------------------------------------------------
    if args.compare_uniform:
        print("\n[pipeline] Applying UNIFORM quantization baseline …")
        uniform_model = copy.deepcopy(base_model)

        quanto_quantize(
            uniform_model,
            weights=low_weights,
            activations=act_weights,
            exclude=["lm_head"],
        )

        if act_weights is not None:
            calibrate_quantized_model(
                uniform_model,
                tokenizer,
                eval_data,
                args.device,
                n_samples=args.calib_samples,
            )

        quanto_freeze(uniform_model)
        uniform_model.config.pad_token_id = tokenizer.pad_token_id
        uniform_model.to(args.eval_device)

        print("[pipeline] Evaluating UNIFORM-quantized model …")
        results["Uniform-Quant"] = full_eval(
            uniform_model,
            tokenizer,
            eval_data,
            args.eval_device,
            max_eval_examples=args.eval_examples,
        )
        print(
            f"  accuracy={results['Uniform-Quant']['accuracy']:.4f}  "
            f"loss={results['Uniform-Quant']['loss']:.4f}  "
            f"lat={results['Uniform-Quant']['latency_ms_per_token']:.2f} ms/tok"
        )

        if args.eval_ppl:
            print("[pipeline] Computing UNIFORM-quantized perplexity …")
            results["Uniform-Quant"]["ppl"] = compute_ppl(
                uniform_model,
                tokenizer,
                args.eval_device,
                n_samples=args.ppl_samples,
                seq_len=args.ppl_seq_len,
            )
            print(f"  ppl={results['Uniform-Quant']['ppl']:.2f}")

    # ------------------------------------------------------------------
    # Phase 7: save circuit-quantized model
    # ------------------------------------------------------------------
    if args.save_model and args.output_dir:
        ckpt_path = os.path.join(args.output_dir, "circuit_quantized.pt")
        qmap = quantization_map(circuit_model)
        torch.save(
            {"state_dict": circuit_model.state_dict(), "quantization_map": qmap},
            ckpt_path,
        )
        print(f"[pipeline] Saved circuit-quantized model → {ckpt_path}")

    # ------------------------------------------------------------------
    # Phase 8: print comparison table
    # ------------------------------------------------------------------
    print_results_table(results)

    # Also print PPL comparison if collected
    if args.eval_ppl:
        print("\n  PPL comparison:")
        for name, m in results.items():
            if "ppl" in m:
                print(f"    {name:<22}: {m['ppl']:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Circuit-guided mixed-precision quantization for LLaMA",
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

    # Quantization
    p.add_argument(
        "--low-weights",
        type=str,
        default="qint4",
        choices=list(QTYPES),
        help="Weight qtype for low-importance layers",
    )
    p.add_argument(
        "--high-weights",
        type=str,
        default="none",
        choices=list(QTYPES),
        help="Weight qtype for high-importance layers (none = keep native float16)",
    )
    p.add_argument(
        "--mid-weights",
        type=str,
        default="none",
        choices=list(QTYPES),
        help="Weight qtype for mid-importance layers (only used when --mid-fraction > 0)",
    )
    p.add_argument(
        "--activations",
        type=str,
        default="none",
        choices=list(QTYPES),
        help="Activation qtype (none = weights-only quantization)",
    )
    p.add_argument(
        "--high-fraction",
        type=float,
        default=0.3,
        help="Fraction of layers protected as high-importance [0, 1]",
    )
    p.add_argument(
        "--mid-fraction",
        type=float,
        default=0.0,
        help="Fraction of layers assigned to mid tier [0, 1]",
    )
    p.add_argument(
        "--score-aggregation",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="How to reduce per-head/per-neuron scores to per-layer scalars",
    )
    p.add_argument(
        "--compare-random",
        action="store_true",
        help="Also run random-tier quantization with the same fractions as the circuit run",
    )
    p.add_argument(
        "--compare-uniform",
        action="store_true",
        help="Also run uniform quantization (all layers at --low-weights) as baseline",
    )

    # Calibration
    p.add_argument(
        "--calib-samples",
        type=int,
        default=64,
        help="Number of examples for activation calibration (when --activations != none)",
    )

    # Evaluation
    p.add_argument(
        "--eval-examples",
        type=int,
        default=None,
        help="Max examples for task evaluation (None = all collected)",
    )
    p.add_argument(
        "--eval-ppl",
        action="store_true",
        help="Compute perplexity on wikitext-2 for all model variants",
    )
    p.add_argument(
        "--ppl-samples", type=int, default=128, help="Number of windows for PPL evaluation"
    )
    p.add_argument(
        "--ppl-seq-len", type=int, default=512, help="Sequence length for PPL evaluation"
    )

    # Devices
    p.add_argument(
        "--device", type=str, default="cuda", help="Device for discovery and quantization"
    )
    p.add_argument(
        "--eval-device", type=str, default="cuda", help="Device for evaluation forward passes"
    )

    # Output
    p.add_argument("--output-dir", type=str, default=None, help="Directory for saving checkpoints")
    p.add_argument(
        "--save-model",
        action="store_true",
        help="Save circuit-quantized model state_dict + quantization_map",
    )

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
