"""
finetune_qwen.py — Selective finetuning pipeline for Qwen family models.

Structurally identical to finetune_llama.py. Material differences:
  - Model loaded via AutoModelForCausalLM (handles Qwen2, Qwen2.5, Qwen3, ...).
  - _read_model_config adds a head_dim direct-attribute fallback present in
    some Qwen3 configs, and guards against older Qwen1 checkpoints that do
    not expose num_key_value_heads.
  - --trust-remote-code flag for Qwen1 / QwenVL checkpoints.
  - Default --dtype is bfloat16 (Qwen's recommended dtype).

The weight matrix accessors in finetune_utils.py already include hasattr
fallbacks for both LLaMA and Qwen attribute names, so no changes are needed
there — only the model loading call and config reading differ.

Usage
-----
    python finetune_qwen.py \\
        --scores-path outputs/eap-ig_ioi_qwen_scores.pt \\
        --model-name  Qwen/Qwen2.5-7B-Instruct \\
        --task        ioi \\
        --scope       both \\
        --top-frac    0.10 \\
        --n-examples  256 \\
        --n-epochs    3   \\
        --lr          2e-5 \\
        --output-dir  outputs/finetune_qwen_ioi
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 1. Get the directory of the current script (.../selective_finetuning/examples)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the parent directory (.../selective_finetuning)
parent_dir = os.path.dirname(current_dir)

# 3. Insert the parent directory at the front of the Python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import score_loader  # noqa: E402 - import after intentional pre-import setup
import selector  # noqa: E402 - import after intentional pre-import setup

from circuitkit.applications.selective_finetuning.finetune_utils import (  # noqa: E402 - import after intentional pre-import setup
    build_finetune_dataloader,
    run_finetuning,
)

# ---------------------------------------------------------------------------
# Task registry — identical to finetune_llama.py; keep in sync if extended.
# ---------------------------------------------------------------------------

_TASK_ALIASES: Dict[str, str] = {
    "ioi": "circuitkit.tasks.builtins.ioi.IOITaskSpec",
    "greater_than": "circuitkit.tasks.builtins.greater_than.GreaterThanTaskSpec",
    "sva": "circuitkit.tasks.builtins.sva.SVATaskSpec",
    "hypernymy": "circuitkit.tasks.builtins.hypernymy.HypernymyTaskSpec",
    "gender_bias": "circuitkit.tasks.builtins.gender_bias.GenderBiasTaskSpec",
    "capital_country": "circuitkit.tasks.builtins.capital_country.CapitalCountryTaskSpec",
    "mmlu": "circuitkit.tasks.builtins.mmlu.MMLUTaskSpec",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_task(task_arg: str) -> Any:
    """
    Instantiate a CircuitKit task spec by short name or fully-qualified path.

    All built-in task specs have no-argument constructors. Accepts short aliases
    from _TASK_ALIASES or a dotted 'package.module.ClassName' string directly.
    """
    dotted = _TASK_ALIASES.get(task_arg, task_arg)
    parts = dotted.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Cannot parse task path {dotted!r}. "
            "Expected 'package.module.ClassName' or a registered alias."
        )
    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Could not import task module {module_path!r}. "
            f"Is circuitkit installed?\nOriginal error: {exc}"
        ) from exc

    task_cls = getattr(module, class_name, None)
    if task_cls is None:
        raise AttributeError(f"Module {module_path!r} has no class {class_name!r}.")
    return task_cls()


def _read_model_config(model_name: str, trust_remote_code: bool = False) -> Dict[str, int]:
    """
    Pull architecture constants from HuggingFace config without loading weights.

    Handles Qwen-specific attribute naming:
      - num_key_value_heads: present in Qwen2/2.5/3 (GQA). Absent in Qwen1
        (MHA) — falls back to n_q_heads.
      - head_dim: exposed directly in some Qwen3 configs. Derived from
        hidden_size // num_attention_heads otherwise.

    Returns dict with keys: n_layers, n_q_heads, n_kv_heads, head_dim, d_model.
    """
    cfg = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    n_layers = cfg.num_hidden_layers
    n_q_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_q_heads)
    d_model = cfg.hidden_size
    # Qwen3 exposes head_dim directly; derive for all other variants.
    head_dim = getattr(cfg, "head_dim", d_model // n_q_heads)
    return dict(
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        d_model=d_model,
    )


def _build_eval_dataloader(
    task_spec,
    tokenizer,
    model_name: str,
    device: torch.device,
    n_examples: int,
    batch_size: int,
    max_length: int,
    seed: int,
):
    # Generates HF-compatible data without HookedTransformer
    clean_texts, query_strings = task_spec.build_finetuning_dataset(
        tokenizer=tokenizer,
        model_name=model_name,
        n_examples=n_examples,
        discovery_cfg={},
        seed=seed,
    )

    from circuitkit.applications.selective_finetuning.finetune_utils import LanguageModelingDataset

    dataset = LanguageModelingDataset(tokenizer, clean_texts, query_strings, max_length)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _make_eval_fn(device: torch.device):
    """
    Return an eval_fn with signature: (model, eval_dataloader) -> Dict[str, float].

    Compatible with run_finetuning's eval_fn parameter.

    Expects EAP-format batches: (clean_tokens, patch_tokens, answers).
    Accuracy is next-token prediction at the last query position:
      logits[:, -2, :].argmax(-1) vs answers (= clean_tokens[:, -1] in all
      standard circuitkit tasks). Multiple-answer tasks are supported — any
      matching answer token counts as a hit.
    """

    def eval_fn(model: torch.nn.Module, eval_dl) -> Dict[str, float]:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in eval_dl:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                query_lengths = batch["query_length"].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

                batch_idx = torch.arange(input_ids.size(0), device=device)
                # query_length gives the exact index of the last prompt token
                preds = logits[batch_idx, query_lengths - 1, :].argmax(dim=-1)

                # The target answer is the token immediately following the query
                ans = input_ids[batch_idx, query_lengths]

                correct += (preds == ans).sum().item()
                total += preds.shape[0]

        return {"accuracy": correct / max(total, 1)}

    return eval_fn


def _free_model(model: torch.nn.Module) -> None:
    """
    Release a model's GPU memory.

    Deletes the local reference, triggers Python GC, and empties the CUDA
    cache. The caller must also del their own variable immediately after this
    call so the object's refcount drops to zero and tensors are freed.
    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_results_table(results: Dict[str, Any], epoch_logs: Dict[str, List]) -> None:
    """Print final accuracy comparison and per-epoch training loss curves."""
    width = 68
    print(f"\n{'=' * width}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * width}")
    print(f"  {'Condition':<16} {'Accuracy':>10}  {'Delta vs Base':>14}")
    print(f"  {'-'*16} {'-'*10}  {'-'*14}")
    base_acc = results.get("base", {}).get("accuracy", float("nan"))
    for cond in ("base", "circuit", "random", "baseline"):
        if cond not in results:
            continue
        acc = results[cond].get("accuracy", float("nan"))
        delta = acc - base_acc if cond != "base" else 0.0
        delta_str = f"{delta:+.4f}" if cond != "base" else "—"
        print(f"  {cond:<16} {acc:>10.4f}  {delta_str:>14}")
    print(f"{'=' * width}")

    if epoch_logs:
        print("\n  Per-epoch training loss:")
        print(f"  {'Epoch':<8}", end="")
        for cond in ("circuit", "random", "baseline"):
            if cond in epoch_logs:
                print(f"  {cond:>10}", end="")
        print()
        max_epochs = max(len(v) for v in epoch_logs.values())
        for ep in range(max_epochs):
            print(f"  {ep+1:<8}", end="")
            for cond in ("circuit", "random", "baseline"):
                logs = epoch_logs.get(cond, [])
                if ep < len(logs):
                    print(f"  {logs[ep]['loss']:>10.4f}", end="")
                else:
                    print(f"  {'—':>10}", end="")
            print()
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Selective finetuning pipeline for Qwen family models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    req = p.add_argument_group("required")
    req.add_argument(
        "--scores-path", required=True, help="Path to the *_scores.pt file from discover_circuit."
    )
    req.add_argument(
        "--model-name",
        required=True,
        help="HuggingFace model identifier " "(e.g. Qwen/Qwen2.5-7B-Instruct).",
    )
    req.add_argument(
        "--task",
        required=True,
        help="Task short name (ioi, greater_than, ...) or a "
        "fully-qualified circuitkit task class path.",
    )

    # ── Selection ─────────────────────────────────────────────────────────────
    sel = p.add_argument_group("selection")
    sel.add_argument(
        "--scope",
        default="both",
        choices=["attn", "mlp", "both"],
        help="Which component types to selectively fine-tune.",
    )
    sel.add_argument(
        "--top-frac",
        type=float,
        default=0.10,
        help="Fraction of top-scoring components to select (0,1].",
    )
    sel.add_argument(
        "--exclude-first-n",
        type=int,
        default=0,
        help="Exclude this many layers from the start of the model.",
    )
    sel.add_argument(
        "--exclude-last-n",
        type=int,
        default=0,
        help="Exclude this many layers from the end of the model.",
    )
    sel.add_argument(
        "--selection-seed",
        type=int,
        default=42,
        help="Seed for random component selection condition.",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data = p.add_argument_group("data")
    data.add_argument(
        "--n-examples",
        type=int,
        default=256,
        help="Number of finetuning examples drawn from the task.",
    )
    data.add_argument(
        "--n-eval-examples",
        type=int,
        default=256,
        help="Number of examples used for each evaluation pass.",
    )
    data.add_argument(
        "--max-length", type=int, default=128, help="Token sequence length for the LM dataset."
    )
    data.add_argument(
        "--prepend-bos", action="store_true", help="Pass prepend_bos=True to the task dataloader."
    )
    data.add_argument(
        "--strict-split",
        action="store_true",
        help="Ensure finetuning examples do not overlap " "circuit-discovery examples.",
    )
    data.add_argument(
        "--task-batch-size",
        type=int,
        default=8,
        help="Batch size passed to task_spec.build_dataloader " "during data collection.",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    trn = p.add_argument_group("training")
    trn.add_argument("--n-epochs", type=int, default=3)
    trn.add_argument("--lr", type=float, default=2e-5)
    trn.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for the finetuning DataLoader."
    )
    trn.add_argument("--max-grad-norm", type=float, default=1.0)
    trn.add_argument("--log-every", type=int, default=10, help="Print step loss every N steps.")
    trn.add_argument("--seed", type=int, default=0, help="Global RNG seed.")

    # ── Model loading ─────────────────────────────────────────────────────────
    mdl = p.add_argument_group("model loading")
    mdl.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype. Qwen models default to bfloat16.",
    )
    mdl.add_argument(
        "--device", default="cuda", help="PyTorch device string (e.g. 'cuda', 'cuda:1', 'cpu')."
    )
    mdl.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model "
        "and config. Required by some Qwen1 / QwenVL "
        "checkpoints that ship custom modelling code.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        default="outputs/finetune_qwen",
        help="Directory for results JSON and optional saved model.",
    )
    out.add_argument(
        "--save-model", action="store_true", help="Save the circuit-finetuned model state dict."
    )
    out.add_argument(
        "--eval-batch-size", type=int, default=16, help="Batch size for evaluation passes."
    )

    return p


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    # ── Setup ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n[pipeline] Device : {device}  |  dtype: {args.dtype}")
    print(f"[pipeline] Output : {args.output_dir}\n")

    # ── Phase 1: Load scores ──────────────────────────────────────────────────
    print("=" * 60)
    print("Phase 1 — Loading circuit discovery scores")
    print("=" * 60)
    head_scores, mlp_scores, score_meta = score_loader.load_scores(
        scores_path=args.scores_path,
        model_name=args.model_name,  # only used for EAP neuron-level decoding
    )
    print(
        f"[pipeline] level={score_meta['level']}  algo={score_meta['algo']}  "
        f"attn_entries={score_meta['n_heads_loaded']}  "
        f"mlp_entries={score_meta['n_mlp_loaded']}"
    )

    # ── Phase 2: Config → component selection ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 — Reading model config and selecting components")
    print("=" * 60)
    arch = _read_model_config(args.model_name, trust_remote_code=args.trust_remote_code)
    print(
        f"[pipeline] n_layers={arch['n_layers']}  "
        f"n_q_heads={arch['n_q_heads']}  "
        f"n_kv_heads={arch['n_kv_heads']}  "
        f"head_dim={arch['head_dim']}"
    )

    shared_sel = dict(
        head_scores=head_scores,
        mlp_scores=mlp_scores,
        metadata=score_meta,
        n_layers=arch["n_layers"],
        n_q_heads=arch["n_q_heads"],
        n_kv_heads=arch["n_kv_heads"],
        head_dim=arch["head_dim"],
        exclude_first_n=args.exclude_first_n,
        exclude_last_n=args.exclude_last_n,
    )

    circuit_sel = selector.select_components(
        **shared_sel,
        top_frac=args.top_frac,
        scope=args.scope,
    )
    random_sel = selector.random_selection(
        **shared_sel,
        circuit_result=circuit_sel,
        seed=args.selection_seed,
    )
    baseline_sel = selector.build_baseline_selection(
        head_scores=head_scores,
        mlp_scores=mlp_scores,
        metadata=score_meta,
        scope=args.scope,
        n_layers=arch["n_layers"],
        exclude_first_n=args.exclude_first_n,
        exclude_last_n=args.exclude_last_n,
    )

    selector.print_selection_summary(
        circuit_sel, random_sel, baseline_sel, head_dim=arch["head_dim"]
    )

    # ── Phase 3: Load tokeniser + model; evaluate base ─────────────────────────
    print("=" * 60)
    print("Phase 3 — Loading model and evaluating base accuracy")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[pipeline] pad_token set to eos_token.")

    # Load weights to CPU first, then move to the target device.
    # Intentionally avoid device_map: HuggingFace's device_map attaches
    # accelerate dispatch hooks to every module, and copy.deepcopy on a
    # hook-decorated model either fails or clones dangling references.
    # Loading without device_map and calling .to(device) keeps the model as a
    # plain nn.Module that deepcopy handles correctly.
    print(f"[pipeline] Loading AutoModelForCausalLM ({args.dtype}) ...")
    t0 = time.time()
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    base_model.to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    print(f"[pipeline] Model loaded in {time.time() - t0:.1f}s  " f"({type(base_model).__name__}).")

    # ── Phase 4: Build dataloaders ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 4 — Building task spec and dataloaders")
    print("=" * 60)

    # Task specs have no-arg constructors; model/tokenizer not needed at init.
    task_spec = _load_task(args.task)
    print(f"[pipeline] Task: {type(task_spec).__name__}")

    # discovery_cfg is forwarded to build_finetuning_dataset.
    # For MMLU: add 'subjects' / 'samples_per_subject' overrides here if needed.
    # For all tasks: 'cache_dir' overrides the default discovery cache location.
    discovery_cfg: Dict[str, Any] = {}

    stripped_model_name = args.model_name.split("/")[-1]

    finetune_dl = build_finetune_dataloader(
        task_spec=task_spec,
        tokenizer=tokenizer,
        model_name=stripped_model_name,
        discovery_cfg=discovery_cfg,
        device=device,
        n_examples=args.n_examples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        strict_split=args.strict_split,
    )

    eval_dl = _build_eval_dataloader(
        task_spec=task_spec,
        max_length=args.max_length,
        tokenizer=tokenizer,
        model_name=stripped_model_name,
        device=device,
        n_examples=args.n_eval_examples,
        batch_size=args.eval_batch_size,
        seed=args.seed + 100,
    )
    eval_fn = _make_eval_fn(device)

    print("[pipeline] Evaluating base model ...")
    base_model.eval()
    base_metrics = eval_fn(base_model, eval_dl)
    print(f"[pipeline] Base accuracy: {base_metrics['accuracy']:.4f}")

    results: Dict[str, Dict] = {"base": base_metrics}
    epoch_logs: Dict[str, List] = {}

    # ── Phase 5–6: Train each condition sequentially ──────────────────────────
    conditions = [
        ("circuit", circuit_sel),
        ("random", random_sel),
        ("baseline", baseline_sel),
    ]

    for cond_name, selection in conditions:
        print("\n" + "=" * 60)
        print(f"Phase 5/6 — Condition: {cond_name.upper()}")
        print("=" * 60)

        # Initialise to None before the try so the finally block can reference
        # model_copy safely even if copy.deepcopy itself raises.
        model_copy = None
        try:
            # Deepcopy is safe here because base_model has no accelerate hooks.
            model_copy = copy.deepcopy(base_model)
            model_copy.to(device)

            trained_model, logs = run_finetuning(
                model=model_copy,
                selection=selection,
                finetune_dataloader=finetune_dl,
                device=device,
                n_epochs=args.n_epochs,
                lr=args.lr,
                max_grad_norm=args.max_grad_norm,
                eval_dataloader=eval_dl,
                eval_fn=eval_fn,
                log_every=args.log_every,
            )

            # Final evaluation after all epochs complete.
            trained_model.eval()
            metrics = eval_fn(trained_model, eval_dl)
            print(f"[pipeline] {cond_name} final accuracy: {metrics['accuracy']:.4f}")
            results[cond_name] = metrics
            epoch_logs[cond_name] = logs

            if cond_name == "circuit" and args.save_model:
                save_path = os.path.join(args.output_dir, "circuit_model.pt")
                torch.save(trained_model.state_dict(), save_path)
                print(f"[pipeline] Circuit model saved -> {save_path}")

        finally:
            # Always release GPU memory before starting the next condition.
            # Nulling model_copy drops the last reference in this scope so
            # CUDA tensors are freed immediately rather than at loop-end.
            if model_copy is not None:
                _free_model(model_copy)
                model_copy = None

    # ── Phase 7: Results ──────────────────────────────────────────────────────
    # _free_model only deletes its local copy of the reference.
    # The explicit del here drops the binding in this scope so the base model's
    # tensors are freed before the (lightweight) JSON write below.
    _free_model(base_model)
    del base_model

    _print_results_table(results, epoch_logs)

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "arch": arch,
                "score_meta": {
                    k: v for k, v in score_meta.items() if not isinstance(v, torch.Tensor)
                },
                "results": results,
                "epoch_logs": epoch_logs,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[pipeline] Results written -> {results_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if not (0.0 < args.top_frac <= 1.0):
        parser.error(f"--top-frac must be in (0, 1], got {args.top_frac}.")
    if args.n_examples < 1:
        parser.error("--n-examples must be >= 1.")

    main(args)
