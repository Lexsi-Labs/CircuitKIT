"""E5-E7 — Downstream applications: pruning, quantization, selective fine-tuning.

Runs ONE discovery phase on Llama-3.2-3B-Instruct / MMLU (eap-ig, node level),
then evaluates three downstream applications sharing the same circuit scores:

  E5 — Structural pruning:      circuit vs magnitude vs random → 5-shot MMLU
  E6 — Mixed-precision quant:   circuit vs uniform vs random   → MMLU + PPL
  E7 — Selective fine-tuning:   circuit vs random vs baseline  → BoolQ accuracy

All results are saved to a single JSON.  Stability↔reliability scatter can
be assembled from the output.

Output
------
results/e567_downstream/results.json
results/e567_downstream/checkpoints/   — exported HF checkpoints (E5)

Run
---
    python examples/experiments/e567_downstream_applications.py
"""
from __future__ import annotations

import copy
import gc
import importlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import circuitkit as ck
from circuitkit.artifacts.scores import CircuitScores

# ── sys.path for optimum-quanto ───────────────────────────────────────────────
_CK_ROOT = os.path.dirname(os.path.abspath(ck.__file__))
_QUANTO_ROOT = os.path.abspath(os.path.join(_CK_ROOT, "..", "optimum-quanto"))
if _QUANTO_ROOT not in sys.path:
    sys.path.insert(0, _QUANTO_ROOT)

# ── sys.path for selective-finetuning bare modules ────────────────────────────
import circuitkit.applications.selective_finetuning as _sft_pkg

_SFT_DIR = os.path.dirname(os.path.abspath(_sft_pkg.__file__))
if _SFT_DIR not in sys.path:
    sys.path.insert(0, _SFT_DIR)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
SEED = 42

# Shared discovery
TASK_DISC = "mmlu"
ALPHA = 0.15
N_DISC = 256
IG_STEPS = 5
BS_DISC = 4

# E5 — Pruning
MMLU_FEWSHOT = 5
MMLU_LIMIT = 200

# E6 — Quantization
HIGH_FRAC = 0.30

# E7 — Fine-tuning
TASK_FT = "boolq"
TOP_FRAC = 0.15
N_FT = 256
N_EVAL = 256
N_EPOCHS = 3
LR = 2e-5
BS_FT = 8
BS_EVAL = 16
MAX_LEN = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("results/e567_downstream")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def banner(text: str) -> None:
    w = 64
    print(f"\n{'━' * w}")
    print(f"  {text}")
    print(f"{'━' * w}")


def progress(msg: str, t0: float | None = None) -> float:
    elapsed = f"  ({time.time() - t0:.1f}s)" if t0 is not None else ""
    print(f"  → {msg}{elapsed}")
    return time.time()


def done(msg: str, t0: float | None = None) -> None:
    elapsed = f" ({time.time() - t0:.1f}s)" if t0 else ""
    print(f"  ✓ {msg}{elapsed}")


def free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def node_scores_from_quant(
    q_head: dict[tuple, float], mlp: dict[int, float]
) -> dict[str, float]:
    """Convert (layer, head) / layer-keyed dicts to 'A{l}.{h}' / 'MLP {l}'."""
    ns: dict[str, float] = {}
    for (l, h), s in q_head.items():
        ns[f"A{l}.{h}"] = s
    for l, s in mlp.items():
        ns[f"MLP {l}"] = s
    return ns


def quant_from_node_scores(
    ns: dict[str, float],
) -> tuple[dict[tuple, float], dict[int, float]]:
    """Convert 'A{l}.{h}' / 'MLP {l}' to (layer, head) / layer-keyed dicts."""
    head, mlp = {}, {}
    for name, s in ns.items():
        m = re.match(r"A(\d+)\.(\d+)$", name)
        if m:
            head[(int(m.group(1)), int(m.group(2)))] = s
            continue
        m = re.match(r"MLP (\d+)$", name)
        if m:
            mlp[int(m.group(1))] = s
    return head, mlp


def bottom_nodes(scores: dict[str, float], frac: float) -> list[str]:
    """Return the lowest-scoring frac fraction of nodes (to be pruned)."""
    attn = [(k, v) for k, v in scores.items() if k.startswith("A")]
    mlp = [(k, v) for k, v in scores.items() if k.startswith("MLP")]
    out: list[str] = []
    for group in (attn, mlp):
        s = sorted(group, key=lambda x: abs(x[1]))
        n = int(len(s) * frac)
        out.extend(k for k, _ in s[:n])
    return out


def make_circuit(nodes: list[str], scores: dict[str, float], algo: str = "eap-ig"):
    cs = CircuitScores(
        task=TASK_DISC, model=MODEL, algorithm=algo, level="node",
        node_scores=scores, timestamp=CircuitScores.create_timestamp(),
    )
    return ck.Circuit(
        nodes, scores, circuit_scores=cs, level="node",
        task=TASK_DISC, model_name=MODEL, algorithm=algo,
    )


def magnitude_scores(model_name: str, ref_scores: dict[str, float]) -> dict[str, float]:
    """L1 weight-magnitude importance for the same node set."""
    hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
    nh = hf.config.num_attention_heads
    hd = hf.config.hidden_size // nh
    out: dict[str, float] = {}
    for name in ref_scores:
        m = re.match(r"A(\d+)\.(\d+)$", name)
        if m:
            l, h = int(m.group(1)), int(m.group(2))
            out[name] = float(hf.model.layers[l].self_attn.o_proj.weight[:, h*hd:(h+1)*hd].abs().sum())
            continue
        m = re.match(r"MLP (\d+)$", name)
        if m:
            l = int(m.group(1))
            out[name] = float(hf.model.layers[l].mlp.down_proj.weight.abs().sum())
    del hf
    free()
    return out


def random_scores(ref_scores: dict[str, float], seed: int = SEED) -> dict[str, float]:
    rng = random.Random(seed)
    return {k: rng.random() for k in ref_scores}


def read_arch(model_name: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_name)
    nl = cfg.num_hidden_layers
    nqh = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", nqh)
    hd = cfg.hidden_size // nqh
    return dict(n_layers=nl, n_q_heads=nqh, n_kv_heads=nkv, head_dim=hd, d_model=cfg.hidden_size)


def load_task(name: str):
    aliases = {
        "ioi": "circuitkit.tasks.builtins.ioi.IOITaskSpec",
        "boolq": "circuitkit.tasks.builtins.boolq.BoolQTaskSpec",
        "mmlu": "circuitkit.tasks.builtins.mmlu.MMLUTaskSpec",
    }
    mod_path, cls = aliases[name].rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), cls)()


def print_table(title: str, rows: list[dict], cols: list[str]) -> None:
    print(f"\n  {title}")
    ws = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) + 2 for c in cols]
    print("  " + "".join(str(c).ljust(w) for c, w in zip(cols, ws)))
    print("  " + "".join("─" * w for w in ws))
    for r in rows:
        print("  " + "".join(str(r.get(c, "—")).ljust(w) for c, w in zip(cols, ws)))


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — SHARED DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def phase_discovery() -> tuple[dict[str, float], dict[tuple, float], dict[int, float], list, object]:
    """Run EAP-IG discovery once, return (node_scores, q_head_scores, mlp_scores,
    eval_data, tl_model).  Caller is responsible for freeing tl_model."""
    from circuitkit.applications.quantization.score_extractor import (
        collect_eval_data,
        extract_node_head_scores,
        extract_node_mlp_scores,
        run_discovery,
        save_eval_data,
        save_scores,
    )

    banner("PHASE 0 — Shared Discovery (EAP-IG / MMLU / node level)")
    total_t0 = time.time()

    data_params = {"samples_per_subject": 5}
    disc_cfg = {
        "algorithm": "eap-ig", "task": TASK_DISC, "level": "node",
        "mlp_hook": "mlp_out", "batch_size": BS_DISC, "ig_steps": IG_STEPS,
        "model_name": MODEL,
        "data_params": {"num_examples": N_DISC, **data_params},
        **data_params,
    }

    scores_cache = OUTPUT_DIR / "discovery" / "scores.pt"
    eval_cache = OUTPUT_DIR / "discovery" / "eval_data.pt"

    if scores_cache.exists() and eval_cache.exists():
        from circuitkit.applications.quantization.score_extractor import (
            load_eval_data,
            load_scores,
        )
        progress("Loading cached discovery scores + eval data ...")
        q_head, mlp = load_scores(str(scores_cache))
        eval_data = load_eval_data(str(eval_cache))
        node_sc = node_scores_from_quant(q_head, mlp)
        done(f"Loaded {len(node_sc)} node scores from cache", total_t0)
        print(f"    ({len(q_head)} attention heads, {len(mlp)} MLP layers)")
        return node_sc, q_head, mlp, eval_data, None

    (OUTPUT_DIR / "discovery").mkdir(parents=True, exist_ok=True)

    t0 = progress(f"Running EAP-IG discovery ({N_DISC} examples, {IG_STEPS} IG steps) ...")
    graph, tl_model = run_discovery(
        model_name=MODEL, task=TASK_DISC, ig_steps=IG_STEPS,
        num_examples=N_DISC, batch_size=BS_DISC, device=DEVICE,
        mlp_hook="mlp_out", precision="bfloat16", data_params=data_params,
    )
    done("Discovery complete", t0)

    q_head = extract_node_head_scores(graph)
    mlp = extract_node_mlp_scores(graph)
    node_sc = node_scores_from_quant(q_head, mlp)
    print(f"    Scored {len(node_sc)} nodes ({len(q_head)} attn, {len(mlp)} MLP)")

    top5 = sorted(node_sc.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"    Top 5: {[f'{n}={s:.4f}' for n, s in top5]}")

    t0 = progress("Collecting eval data for quantization evaluation ...")
    eval_data = collect_eval_data(tl_model, TASK_DISC, disc_cfg, DEVICE, max_examples=200)
    done(f"Collected {len(eval_data)} eval examples", t0)

    save_scores(q_head, mlp, str(scores_cache))
    save_eval_data(eval_data, str(eval_cache))
    print(f"    Cached to {scores_cache.parent}/")

    done("Phase 0 complete", total_t0)
    return node_sc, q_head, mlp, eval_data, tl_model


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — E5: STRUCTURAL PRUNING
# ═══════════════════════════════════════════════════════════════════════════════

def phase_pruning(node_scores: dict[str, float], tl_model=None) -> dict:
    """E5: circuit vs magnitude vs random pruning, evaluated on 5-shot MMLU."""
    banner("PHASE 1 / E5 — Structural Pruning (5-shot MMLU)")

    total_t0 = time.time()
    ckpt_root = OUTPUT_DIR / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Load TL model if not passed from discovery
    if tl_model is None:
        t0 = progress("Loading TransformerLens model ...")
        tl_model = ck.load_model(MODEL, dtype="bfloat16")
        done("Model loaded", t0)
    else:
        progress("Reusing TransformerLens model from discovery")

    # Build the circuit-guided Circuit
    pruned_nodes = bottom_nodes(node_scores, ALPHA)
    circuit = make_circuit(pruned_nodes, node_scores, algo="eap-ig")
    print(f"    Circuit: {len(node_scores)} scored, {len(pruned_nodes)} pruned at α={ALPHA}")

    def prune_export_bench(circuit_obj, tag: str) -> dict:
        t0 = progress(f"{tag}: prune → export → benchmark ...")
        pruned = ck.prune(tl_model, circuit_obj, sparsity=ALPHA, scope="both", inplace=False)
        ckpt = str(ckpt_root / tag)
        ck.export_checkpoint(pruned, circuit_obj, ckpt)
        del pruned; free()
        scores = ck.benchmark(ckpt, tasks=["mmlu"], fewshot=MMLU_FEWSHOT, limit=MMLU_LIMIT, dtype="float16")
        mmlu_acc = scores.get("mmlu", {}).get("acc", scores.get("mmlu", {}).get("acc,none"))
        done(f"{tag}: MMLU acc = {mmlu_acc}", t0)
        return {"tag": tag, "mmlu_acc": float(mmlu_acc) if mmlu_acc is not None else None,
                "mmlu_scores": scores.get("mmlu", {})}

    results: dict = {}

    # Base model (no pruning)
    t0 = progress("Exporting + benchmarking BASE model (unpruned) ...")
    base_ckpt = str(ckpt_root / "base")
    ck.export_checkpoint(tl_model, circuit, base_ckpt)
    base_sc = ck.benchmark(base_ckpt, tasks=["mmlu"], fewshot=MMLU_FEWSHOT, limit=MMLU_LIMIT, dtype="float16")
    base_acc = base_sc.get("mmlu", {}).get("acc", base_sc.get("mmlu", {}).get("acc,none"))
    results["Base"] = {"mmlu_acc": float(base_acc) if base_acc is not None else None,
                       "mmlu_scores": base_sc.get("mmlu", {})}
    done(f"Base MMLU acc = {base_acc}", t0)

    # Circuit-guided pruning
    results["Circuit"] = prune_export_bench(circuit, "circuit")

    # Magnitude pruning
    t0 = progress("Computing magnitude scores (L1 weight norms) ...")
    mag_sc = magnitude_scores(MODEL, node_scores)
    done("Magnitude scores computed", t0)
    mag_nodes = bottom_nodes(mag_sc, ALPHA)
    mag_circuit = make_circuit(mag_nodes, mag_sc, algo="magnitude")
    results["Magnitude"] = prune_export_bench(mag_circuit, "magnitude")

    # Random pruning
    rand_sc = random_scores(node_scores, seed=SEED)
    rand_nodes = bottom_nodes(rand_sc, ALPHA)
    rand_circuit = make_circuit(rand_nodes, rand_sc, algo="random")
    results["Random"] = prune_export_bench(rand_circuit, "random")

    # Free TL model
    del tl_model; free()

    # Print summary
    base_val = results["Base"]["mmlu_acc"] or 0.0
    rows = []
    for key in ("Base", "Circuit", "Magnitude", "Random"):
        acc = results[key].get("mmlu_acc") or 0.0
        delta = f"{acc - base_val:+.4f}" if key != "Base" else "—"
        rows.append({"Strategy": key, "MMLU Acc": f"{acc:.4f}", "Δ Base": delta})
    print_table("E5 Results — 5-shot MMLU", rows, ["Strategy", "MMLU Acc", "Δ Base"])
    done("Phase 1 / E5 complete", total_t0)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — E6: MIXED-PRECISION QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def phase_quantization(
    q_head_scores: dict[tuple, float],
    mlp_scores: dict[int, float],
    eval_data: list,
) -> dict:
    """E6: circuit-guided vs uniform vs random quantization, MMLU + PPL."""
    from circuitkit.applications.pruning.eval_utils import full_eval
    from circuitkit.applications.quantization.quant_utils import (
        circuit_quantize, compute_ppl, freeze_model, print_quantization_plan,
        random_quantize,
    )
    from optimum.quanto import freeze as quanto_freeze, qint4
    from optimum.quanto import quantize as quanto_quantize

    banner("PHASE 2 / E6 — Mixed-Precision Quantization (MMLU + PPL)")
    total_t0 = time.time()

    t0 = progress(f"Loading HF model ({MODEL}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token_id = tokenizer.pad_token_id or 0
    base_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    base_model.to(DEVICE)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    n_layers = base_model.config.num_hidden_layers
    done(f"Loaded ({n_layers} layers, top {HIGH_FRAC*100:.0f}% kept native)", t0)

    results: dict = {}

    # Base
    t0 = progress("Evaluating BASE model (accuracy + PPL) ...")
    results["Base"] = full_eval(base_model, tokenizer, eval_data, DEVICE, max_eval_examples=200)
    results["Base"]["ppl"] = compute_ppl(base_model, tokenizer, DEVICE, n_samples=128, seq_len=512)
    done(f"Base: accuracy={results['Base']['accuracy']:.4f}  ppl={results['Base']['ppl']:.2f}", t0)

    # Circuit-guided quantization
    t0 = progress("CIRCUIT-guided quantization (high-importance fp16, rest qint4) ...")
    circuit_model = copy.deepcopy(base_model)
    plan = circuit_quantize(
        circuit_model, q_head_scores=q_head_scores, mlp_scores=mlp_scores,
        n_layers=n_layers, low_weights=qint4, high_weights=None, activations=None,
        high_fraction=HIGH_FRAC, score_aggregation="mean", exclude_lm_head=True,
    )
    freeze_model(circuit_model)
    circuit_model.config.pad_token_id = tokenizer.pad_token_id
    circuit_model.to(DEVICE)
    print_quantization_plan(plan, qint4, None, None)
    results["Circuit-Quant"] = full_eval(circuit_model, tokenizer, eval_data, DEVICE, max_eval_examples=200)
    results["Circuit-Quant"]["ppl"] = compute_ppl(circuit_model, tokenizer, DEVICE, n_samples=128, seq_len=512)
    done(f"Circuit-Quant: acc={results['Circuit-Quant']['accuracy']:.4f}  ppl={results['Circuit-Quant']['ppl']:.2f}", t0)
    del circuit_model; free()

    # Uniform quantization
    t0 = progress("UNIFORM quantization (all layers qint4) ...")
    uniform_model = copy.deepcopy(base_model)
    quanto_quantize(uniform_model, weights=qint4, activations=None, exclude=["lm_head"])
    quanto_freeze(uniform_model)
    uniform_model.config.pad_token_id = tokenizer.pad_token_id
    uniform_model.to(DEVICE)
    results["Uniform-Quant"] = full_eval(uniform_model, tokenizer, eval_data, DEVICE, max_eval_examples=200)
    results["Uniform-Quant"]["ppl"] = compute_ppl(uniform_model, tokenizer, DEVICE, n_samples=128, seq_len=512)
    done(f"Uniform-Quant: acc={results['Uniform-Quant']['accuracy']:.4f}  ppl={results['Uniform-Quant']['ppl']:.2f}", t0)
    del uniform_model; free()

    # Random-tier quantization
    t0 = progress("RANDOM-tier quantization (random 30% fp16, rest qint4) ...")
    random_model = copy.deepcopy(base_model)
    random_quantize(
        random_model, n_layers=n_layers, low_weights=qint4, high_weights=None,
        activations=None, high_fraction=HIGH_FRAC, exclude_lm_head=True, seed=SEED,
    )
    freeze_model(random_model)
    random_model.config.pad_token_id = tokenizer.pad_token_id
    random_model.to(DEVICE)
    results["Random-Quant"] = full_eval(random_model, tokenizer, eval_data, DEVICE, max_eval_examples=200)
    results["Random-Quant"]["ppl"] = compute_ppl(random_model, tokenizer, DEVICE, n_samples=128, seq_len=512)
    done(f"Random-Quant: acc={results['Random-Quant']['accuracy']:.4f}  ppl={results['Random-Quant']['ppl']:.2f}", t0)
    del random_model, base_model; free()

    # Summary
    rows = []
    for key in ("Base", "Circuit-Quant", "Uniform-Quant", "Random-Quant"):
        r = results[key]
        rows.append({"Strategy": key, "Accuracy": f"{r['accuracy']:.4f}", "PPL": f"{r['ppl']:.2f}"})
    print_table("E6 Results — MMLU Accuracy + Wikitext-2 PPL", rows, ["Strategy", "Accuracy", "PPL"])
    done("Phase 2 / E6 complete", total_t0)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — E7: SELECTIVE FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════════════

def phase_finetuning(
    q_head_scores: dict[tuple, float],
    mlp_scores: dict[int, float],
) -> dict:
    """E7: circuit vs random vs baseline fine-tuning on BoolQ."""
    import score_loader  # noqa: F811 — loaded from sys.path
    import selector
    from circuitkit.applications.selective_finetuning.finetune_utils import (
        LanguageModelingDataset,
        build_finetune_dataloader,
        run_finetuning,
    )

    banner("PHASE 3 / E7 — Selective Fine-Tuning (BoolQ)")
    total_t0 = time.time()

    device = torch.device(DEVICE)
    arch = read_arch(MODEL)
    print(f"    n_layers={arch['n_layers']}  n_q_heads={arch['n_q_heads']}  "
          f"n_kv_heads={arch['n_kv_heads']}  head_dim={arch['head_dim']}")

    # Construct metadata for selector (same format as score_loader returns)
    score_meta = {
        "level": "node",
        "algo": "eap-ig",
        "mlp_neuron_level": False,
        "mlp_hook": "mlp_out",
        "n_heads_loaded": len(q_head_scores),
        "n_mlp_loaded": len(mlp_scores),
    }

    # Component selection
    t0 = progress("Selecting components (circuit / random / baseline) ...")
    shared_sel = dict(
        head_scores=q_head_scores, mlp_scores=mlp_scores, metadata=score_meta,
        n_layers=arch["n_layers"], n_q_heads=arch["n_q_heads"],
        n_kv_heads=arch["n_kv_heads"], head_dim=arch["head_dim"],
    )
    circuit_sel = selector.select_components(**shared_sel, top_frac=TOP_FRAC, scope="both")
    random_sel = selector.random_selection(**shared_sel, circuit_result=circuit_sel, seed=SEED)
    baseline_sel = selector.build_baseline_selection(
        head_scores=q_head_scores, mlp_scores=mlp_scores, metadata=score_meta,
        scope="both", n_layers=arch["n_layers"],
    )
    selector.print_selection_summary(circuit_sel, random_sel, baseline_sel, head_dim=arch["head_dim"])
    done("Component selection done", t0)

    # Load model + tokenizer
    t0 = progress(f"Loading HF model ({MODEL}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    base_model.to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    done("Model loaded", t0)

    # Build dataloaders
    t0 = progress(f"Building dataloaders ({TASK_FT}, n_train={N_FT}, n_eval={N_EVAL}) ...")
    task_spec = load_task(TASK_FT)
    stripped = MODEL.split("/")[-1]

    finetune_dl = build_finetune_dataloader(
        task_spec=task_spec, tokenizer=tokenizer, model_name=stripped,
        discovery_cfg={}, device=device, n_examples=N_FT,
        max_length=MAX_LEN, batch_size=BS_FT, seed=SEED,
    )

    clean_texts, query_strings = task_spec.build_finetuning_dataset(
        tokenizer=tokenizer, model_name=stripped,
        n_examples=N_EVAL, discovery_cfg={}, seed=SEED + 100,
    )
    eval_dataset = LanguageModelingDataset(tokenizer, clean_texts, query_strings, MAX_LEN)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=BS_EVAL, shuffle=False)
    done("Dataloaders built", t0)

    # Eval function
    def eval_fn(model, dl):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in dl:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                ql = batch["query_length"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                bi = torch.arange(ids.size(0), device=device)
                preds = logits[bi, ql - 1, :].argmax(dim=-1)
                ans = ids[bi, ql]
                correct += (preds == ans).sum().item()
                total += preds.shape[0]
        return {"accuracy": correct / max(total, 1)}

    # Base accuracy
    base_model.eval()
    base_metrics = eval_fn(base_model, eval_dl)
    print(f"    Base {TASK_FT} accuracy: {base_metrics['accuracy']:.4f}")

    results: dict = {"base": base_metrics}
    epoch_logs: dict = {}

    # Train each condition
    conditions = [
        ("circuit", circuit_sel),
        ("random", random_sel),
        ("baseline", baseline_sel),
    ]

    for i, (cond_name, selection) in enumerate(conditions, 1):
        progress(f"Fine-tuning condition {i}/3: {cond_name.upper()} "
                 f"({N_EPOCHS} epochs, lr={LR}) ...")
        t0 = time.time()
        model_copy = None
        try:
            model_copy = copy.deepcopy(base_model)
            model_copy.to(device)

            trained, logs = run_finetuning(
                model=model_copy, selection=selection,
                finetune_dataloader=finetune_dl, device=device,
                n_epochs=N_EPOCHS, lr=LR, max_grad_norm=1.0,
                eval_dataloader=eval_dl, eval_fn=eval_fn, log_every=10,
            )
            trained.eval()
            metrics = eval_fn(trained, eval_dl)
            results[cond_name] = metrics
            epoch_logs[cond_name] = logs
            done(f"{cond_name}: accuracy={metrics['accuracy']:.4f}", t0)
        except Exception as exc:
            print(f"    ERROR ({cond_name}): {type(exc).__name__}: {exc}")
            results[cond_name] = {"error": str(exc)}
        finally:
            if model_copy is not None:
                del model_copy; free()

    del base_model; free()

    # Summary
    base_acc = results.get("base", {}).get("accuracy", float("nan"))
    rows = []
    for key in ("base", "circuit", "random", "baseline"):
        if key not in results or "error" in results[key]:
            continue
        acc = results[key].get("accuracy", float("nan"))
        delta = f"{acc - base_acc:+.4f}" if key != "base" else "—"
        rows.append({"Condition": key, "BoolQ Acc": f"{acc:.4f}", "Δ Base": delta})
    print_table("E7 Results — BoolQ Accuracy", rows, ["Condition", "BoolQ Acc", "Δ Base"])
    done("Phase 3 / E7 complete", total_t0)
    return {"results": results, "epoch_logs": epoch_logs}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wall_start = time.time()

    print("╔" + "═" * 62 + "╗")
    print("║  E5-E7 Downstream Applications Pipeline" + " " * 21 + "║")
    print(f"║  Model: {MODEL:<53}║")
    print("╚" + "═" * 62 + "╝")

    # Phase 0 — Discovery
    node_scores, q_head_scores, mlp_scores, eval_data, tl_model = phase_discovery()

    # Phase 1 / E5 — Pruning (reuse TL model if available)
    e5_results = phase_pruning(node_scores, tl_model=tl_model)

    # Phase 2 / E6 — Quantization (reuse discovery scores)
    e6_results = phase_quantization(q_head_scores, mlp_scores, eval_data)

    # Phase 3 / E7 — Selective fine-tuning (reuse discovery scores)
    e7_out = phase_finetuning(q_head_scores, mlp_scores)

    # Save combined results
    total_wall = time.time() - wall_start
    combined = {
        "experiment": "E5-E7",
        "model": MODEL,
        "seed": SEED,
        "total_wall_seconds": round(total_wall, 1),
        "discovery": {
            "task": TASK_DISC,
            "algorithm": "eap-ig",
            "level": "node",
            "alpha": ALPHA,
            "n_examples": N_DISC,
            "ig_steps": IG_STEPS,
            "n_nodes_scored": len(node_scores),
        },
        "e5_pruning": {
            "task": TASK_DISC, "alpha": ALPHA,
            "mmlu_fewshot": MMLU_FEWSHOT, "mmlu_limit": MMLU_LIMIT,
            "strategies": e5_results,
        },
        "e6_quantization": {
            "task": TASK_DISC, "high_fraction": HIGH_FRAC,
            "strategies": e6_results,
        },
        "e7_finetuning": {
            "task": TASK_FT, "top_frac": TOP_FRAC,
            "n_epochs": N_EPOCHS, "lr": LR,
            **e7_out,
        },
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    banner(f"ALL DONE — total wall time {total_wall/60:.1f} min")
    print(f"  Results saved to {out_path}")
    print(f"  Checkpoints in {OUTPUT_DIR / 'checkpoints'}/")


if __name__ == "__main__":
    main()
