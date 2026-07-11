"""Circuit-restricted soft healing using PEFT (HuggingFace) on Llama-3.2-3B-Instruct.

Pipeline per (method, seed):
  1. Discover circuit on MMLU (same setting as Q2 pruning).
  2. Measure base MMLU accuracy.
  3. Apply circuit-restricted LoRA healing via PEFT: LoRA adapters are
     restricted to the top-K circuit layers (attention + MLP), trained
     on a 1k-Alpaca slice for HEAL_EPOCHS epochs.
  4. Measure post-healing MMLU.
  5. Compare to unrestricted LoRA healing at the same rank and epoch budget.

Uses PEFT (HuggingFace) for LoRA -- standard library, no custom wrapper.
Circuit guidance is applied via ``layers_to_transform`` in LoraConfig, which
restricts adapter injection to the layers the circuit discovery identified
as most important.

Headline metric: circuit_advantage = post_circuit_heal_mmlu - post_unrestricted_heal_mmlu.
Positive values mean circuit guidance improves over blind LoRA.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status,
)

SCRIPT_NAME = "26_soft_healing_llama_circuit"
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TASK = "mmlu"
SEEDS = [42, 143, 256]
METHODS = ["eap", "eap-ig", "eap-clean-corrupted", "atp-gd",
           "eap-gp", "relp"]
HEAL_RANK = 16
HEAL_EPOCHS = 2
ALPACA_N = 1024
MMLU_N = 224
TOP_K_LAYERS = 8


# ---------------------------------------------------------------------------
# MMLU evaluation on HuggingFace model
# ---------------------------------------------------------------------------

def _eval_mmlu_hf(model, tokenizer, n_samples: int = MMLU_N,
                  device: str = "cuda") -> float:
    """5-shot MMLU accuracy on a HuggingFace CausalLM.

    Scores each answer letter (A/B/C/D) by the logit at the last prompt
    token position and picks the highest.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")

    rng = random.Random(42)
    indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))
    subjects = list({ds[i]["subject"] for i in indices})
    few_shot_map: Dict[str, str] = {}
    for subj in subjects:
        shots = [r for r in dev if r["subject"] == subj][:5]
        lines = []
        for s in shots:
            opts = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(s["choices"]))
            lines.append(f"Q: {s['question']}\n{opts}\nA: {chr(65+s['answer'])}")
        few_shot_map[subj] = "\n\n".join(lines)

    model.eval()
    correct = 0
    answer_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[0]
                     for c in "ABCD"]

    with torch.no_grad():
        for i in indices:
            row = ds[i]
            opts = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(row["choices"]))
            prefix = few_shot_map.get(row["subject"], "")
            prompt = (f"{prefix}\n\n" if prefix else "") + \
                     f"Q: {row['question']}\n{opts}\nA:"
            ids = tokenizer(prompt, return_tensors="pt",
                            truncation=True, max_length=512).input_ids.to(device)
            logits = model(ids).logits[0, -1]
            scores = [logits[t].item() for t in answer_tokens]
            pred = scores.index(max(scores))
            if pred == row["answer"]:
                correct += 1

    return correct / len(indices)


# ---------------------------------------------------------------------------
# Circuit layer extraction
# ---------------------------------------------------------------------------

def _circuit_layer_indices(node_scores: Dict[str, float],
                            top_k: int = TOP_K_LAYERS) -> List[int]:
    """Extract the top-K layer indices from circuit node scores.

    Combines MLP and attention scores per layer (max pooling), then
    returns the layer indices sorted by score descending.
    """
    layer_max: Dict[int, float] = {}
    for name, score in node_scores.items():
        parts = name.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            layer_idx = int(parts[-1])
        elif len(parts) >= 2 and "." in parts[-1]:
            layer_idx = int(parts[-1].split(".")[0])
        else:
            continue
        layer_max[layer_idx] = max(layer_max.get(layer_idx, 0.0), float(score))

    ranked = sorted(layer_max.items(), key=lambda kv: kv[1], reverse=True)
    return [idx for idx, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# PEFT LoRA setup
# ---------------------------------------------------------------------------

def _build_lora_model(base_model_name: str, device: str,
                      layers_to_transform: Optional[List[int]],
                      rank: int, seed: int):
    """Load a HF model and wrap it with a PEFT LoRA adapter.

    ``layers_to_transform=None`` applies LoRA to all transformer layers
    (unrestricted baseline). Passing a list restricts LoRA to those layer
    indices (circuit-restricted).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(base_model_name)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float32,
        device_map={"": device},
    )

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=layers_to_transform,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model, tok


# ---------------------------------------------------------------------------
# Alpaca training loop
# ---------------------------------------------------------------------------

def _train_alpaca(peft_model, tokenizer, n_samples: int = ALPACA_N,
                  epochs: int = HEAL_EPOCHS, seed: int = 42,
                  device: str = "cuda"):
    """Fine-tune the PEFT model on a random Alpaca slice."""
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    idxs = rng.sample(range(len(ds)), min(n_samples, len(ds)))
    texts = [ds[i]["text"] for i in idxs]

    tokenizer.padding_side = "right"
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=0.01,
    )
    peft_model.train()

    for epoch in range(epochs):
        rng.shuffle(texts)
        total_loss = 0.0
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=256, padding=False)
            ids = enc.input_ids.to(device)
            if ids.shape[1] < 4:
                continue
            optimizer.zero_grad()
            out = peft_model(ids, labels=ids)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft_model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            total_loss += out.loss.item()
        avg = total_loss / max(1, len(texts))
        print(f"    epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}")

    peft_model.eval()


# ---------------------------------------------------------------------------
# Per-cell experiment
# ---------------------------------------------------------------------------

def _cell(method: str, model_name: str, seed: int,
          out_dir: Path) -> Dict[str, object]:
    t0 = time.time()
    cell = {"method": method, "seed": seed}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        discovery = get_or_run_discovery(
            algorithm=method, model=model_name, task=TASK,
            num_examples=MMLU_N, batch_size=1,
        )
        node_scores = discovery["node_scores"]
        circuit_layers = _circuit_layer_indices(node_scores, top_k=TOP_K_LAYERS)
        cell["circuit_layers"] = circuit_layers

        # --- Base MMLU (no LoRA) ---
        base_model, base_tok = _build_lora_model(
            model_name, device, layers_to_transform=[], rank=HEAL_RANK, seed=seed
        )
        base_mmlu = _eval_mmlu_hf(base_model.base_model.model, base_tok,
                                   n_samples=MMLU_N, device=device)
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Circuit-restricted LoRA healing ---
        circ_model, circ_tok = _build_lora_model(
            model_name, device, layers_to_transform=circuit_layers,
            rank=HEAL_RANK, seed=seed,
        )
        _train_alpaca(circ_model, circ_tok, n_samples=ALPACA_N,
                      epochs=HEAL_EPOCHS, seed=seed, device=device)
        circuit_mmlu = _eval_mmlu_hf(circ_model, circ_tok,
                                     n_samples=MMLU_N, device=device)
        del circ_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Unrestricted LoRA healing (same rank, all layers) ---
        unres_model, unres_tok = _build_lora_model(
            model_name, device, layers_to_transform=None,
            rank=HEAL_RANK, seed=seed,
        )
        _train_alpaca(unres_model, unres_tok, n_samples=ALPACA_N,
                      epochs=HEAL_EPOCHS, seed=seed, device=device)
        unrestricted_mmlu = _eval_mmlu_hf(unres_model, unres_tok,
                                          n_samples=MMLU_N, device=device)
        del unres_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cell.update({
            "base_mmlu": round(base_mmlu, 4),
            "post_circuit_heal_mmlu": round(circuit_mmlu, 4),
            "post_unrestricted_heal_mmlu": round(unrestricted_mmlu, 4),
            "circuit_heal_gain": round(circuit_mmlu - base_mmlu, 4),
            "unrestricted_heal_gain": round(unrestricted_mmlu - base_mmlu, 4),
            "circuit_advantage": round(circuit_mmlu - unrestricted_mmlu, 4),
        })
    except Exception as exc:
        cell["error"] = f"{type(exc).__name__}: {exc}"

    cell["wall_s"] = round(time.time() - t0, 2)
    return cell


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    out_dir = make_results_dir(SCRIPT_NAME)
    rows: List[Dict[str, object]] = []
    for method in args.methods:
        for seed in args.seeds:
            cell = _cell(method, args.model, seed, out_dir)
            rows.append(cell)
            print(f"[{method} seed={seed}] circuit_adv={cell.get('circuit_advantage')} "
                  f"{cell.get('error') or 'ok'}")

    payload = {
        "script": SCRIPT_NAME, "model": args.model, "task": TASK,
        "heal_rank": HEAL_RANK, "heal_epochs": HEAL_EPOCHS,
        "alpaca_n": ALPACA_N, "top_k_layers": TOP_K_LAYERS,
        "method_seed_rows": rows,
    }
    write_status(out_dir, payload)
    print(f"Wrote {out_dir / 'status.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
