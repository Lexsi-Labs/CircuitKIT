"""
eval_utils.py — Task-level evaluation utilities for HuggingFace models.

Three evaluation surfaces:
  1. ``eval_hf_model_on_task`` — accuracy and cross-entropy loss measured on
     the same task data used for circuit discovery (IOI, MMLU, …).  Uses clean
     prompts; compares the model's last-position logit over {correct, wrong}
     token IDs.
  2. ``measure_latency``       — forward-pass wall-clock timing, reported as
     milliseconds per token and tokens per second.
  3. ``full_eval``             — combines (1) and (2) with a parameter count.

``print_results_table`` formats all five model variants into a comparison
table at the end of a pruning run.

Token-ID compatibility note
---------------------------
Both TransformerLens and HuggingFace models for the same model family use
the same underlying tokenizer vocabulary, so token IDs produced by circuitkit
(via TL) can be looked up directly in HF model logits.  The HF tokenizer is
used here to re-tokenize the clean prompt strings.  BOS handling follows the
``templated`` flag recorded by ``collect_eval_data``: raw prompts are
tokenized with ``add_special_tokens=True`` (BOS prepended, matching discovery's
``prepend_bos=True``), while chat-templated prompts already carry their own BOS
and are tokenized with ``add_special_tokens=False`` to avoid a double-BOS.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Core accuracy / loss evaluation
# ---------------------------------------------------------------------------


import logging

logger = logging.getLogger(__name__)

def eval_hf_model_on_task(
    model,
    tokenizer,
    eval_data: List[Dict],
    device: str = "cuda",
    batch_size: int = 8,
    max_examples: Optional[int] = None,
) -> Dict:
    """
    Evaluate a HuggingFace causal-LM on circuit-discovery task data.

    For each example the model receives the full clean prompt and we inspect
    the logits at the last non-padding position.  We check whether the correct
    completion token has the highest logit among all candidate token IDs
    (correct + incorrect).  Cross-entropy loss at that position against the
    correct token is also accumulated.

    Parameters
    ----------
    model        : HuggingFace CausalLM (any variant — LLaMA, Qwen3, …).
    tokenizer    : Matching HuggingFace tokenizer.
    eval_data    : List of dicts with keys ``clean``, ``correct_idx``,
                   ``incorrect_idx``.  Produced by
                   ``score_extractor.collect_eval_data``.
    device       : "cuda" or "cpu".
    batch_size   : Number of examples processed per forward pass.
    max_examples : Cap the number of examples evaluated.

    Returns
    -------
    Dict with keys: ``accuracy`` (float 0–1), ``loss`` (float, nats),
    ``n_examples`` (int).
    """
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = eval_data[:max_examples] if max_examples else eval_data
    total_correct = 0
    total_loss = 0.0
    n_evaluated = 0

    for batch_start in range(0, len(examples), batch_size):
        batch = examples[batch_start : batch_start + batch_size]
        texts = [ex["clean"] for ex in batch]
        # Chat-templated prompts already carry their own BOS; tokenizing them
        # with add_special_tokens=True would double-prepend it.  The flag is
        # recorded per-row by collect_eval_data.
        templated = bool(batch[0].get("templated", False))

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=not templated,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits  # (batch, seq, vocab)

        # Identify last non-padding position per example
        seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed last real token

        for i, ex in enumerate(batch):
            last_pos = seq_lengths[i].item()
            logit_vec = logits[i, last_pos, :]  # (vocab,)

            correct = ex["correct_idx"]
            wrong = ex["incorrect_idx"]
            candidate_ids = [correct] + wrong

            candidate_logits = logit_vec[candidate_ids]
            pred_idx = candidate_logits.argmax().item()
            total_correct += int(pred_idx == 0)  # index 0 is correct

            # Cross-entropy loss at this position
            loss = F.cross_entropy(
                logit_vec.unsqueeze(0),
                torch.tensor([correct], device=device),
            )
            total_loss += loss.item()
            n_evaluated += 1

    return {
        "accuracy": total_correct / n_evaluated if n_evaluated else 0.0,
        "loss": total_loss / n_evaluated if n_evaluated else float("nan"),
        "n_examples": n_evaluated,
    }


# ---------------------------------------------------------------------------
# Latency / throughput measurement
# ---------------------------------------------------------------------------


def measure_latency(
    model,
    tokenizer,
    eval_data: List[Dict],
    device: str = "cuda",
    n_warmup: int = 3,
    n_measure: int = 20,
    batch_size: int = 1,
) -> Dict:
    """
    Measure forward-pass latency on a sample of task prompts.

    Runs ``n_warmup`` warm-up passes (not timed), then ``n_measure`` timed
    passes.  Reports the mean latency per token and derived throughput.

    Parameters
    ----------
    n_warmup  : Number of un-timed warm-up iterations.
    n_measure : Number of timed iterations (mean is reported).
    batch_size: Batch size used for timing (default 1 for per-example latency).

    Returns
    -------
    Dict with keys:
        ``latency_ms_per_token`` : float — ms per input token.
        ``throughput_tokens_per_sec`` : float — tokens / second.
        ``mean_latency_ms`` : float — ms per forward pass (batch_size examples).
        ``seq_len`` : int — sequence length used for measurement.
    """
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use first batch_size examples, or cycle if fewer available
    sample = eval_data[:batch_size] if len(eval_data) >= batch_size else eval_data
    texts = [ex["clean"] for ex in sample] * (batch_size // max(len(sample), 1) + 1)
    texts = texts[:batch_size]
    # Chat-templated prompts already carry their own BOS; avoid a double-BOS.
    templated = bool(sample[0].get("templated", False)) if sample else False

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=not templated,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    n_tokens = batch_size * seq_len

    # Warm-up
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_measure):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_s = sum(times) / len(times)
    mean_ms = mean_s * 1000.0

    return {
        "latency_ms_per_token": mean_ms / seq_len,
        "throughput_tokens_per_sec": n_tokens / mean_s,
        "mean_latency_ms": mean_ms,
        "seq_len": seq_len,
    }


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def full_eval(
    model,
    tokenizer,
    eval_data: List[Dict],
    device: str = "cuda",
    batch_size: int = 8,
    latency_batch_size: int = 1,
    max_eval_examples: Optional[int] = None,
) -> Dict:
    """
    Run task accuracy + loss + latency in one call.

    Also counts total parameters and non-embedding parameters.

    Returns
    -------
    Dict with all keys from ``eval_hf_model_on_task`` and
    ``measure_latency``, plus:
        ``n_params``           : total parameter count.
        ``n_non_embed_params`` : non-embedding parameter count (rough proxy
                                  for computation cost of a forward pass).
    """
    task_metrics = eval_hf_model_on_task(
        model, tokenizer, eval_data, device, batch_size, max_eval_examples
    )
    latency_metrics = measure_latency(
        model,
        tokenizer,
        eval_data,
        device,
        batch_size=latency_batch_size,
    )

    # Parameter counts
    n_params = sum(p.numel() for p in model.parameters())
    n_non_embed = n_params
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        n_non_embed -= model.model.embed_tokens.weight.numel()
    if hasattr(model, "lm_head"):
        n_non_embed -= model.lm_head.weight.numel()

    return {
        **task_metrics,
        **latency_metrics,
        "n_params": n_params,
        "n_non_embed_params": n_non_embed,
    }


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------


def print_results_table(results: Dict[str, Dict]) -> None:
    """
    Print a formatted comparison table for multiple model variants.

    Parameters
    ----------
    results : Dict mapping variant name → metrics dict (output of full_eval).
              Expected keys: accuracy, loss, latency_ms_per_token,
              throughput_tokens_per_sec, n_params.

    Example
    -------
    results = {
        "Base":                   full_eval(base_model, ...),
        "Circuit (pre-FT)":       full_eval(circuit_pruned, ...),
        "Random  (pre-FT)":       full_eval(random_pruned, ...),
        "Circuit (post-FT)":      full_eval(circuit_finetuned, ...),
        "Random  (post-FT)":      full_eval(random_finetuned, ...),
    }
    print_results_table(results)
    """
    col_width = 22
    headers = ["Model", "Accuracy", "Loss", "Latency (ms/tok)", "Throughput (tok/s)", "Params (M)"]
    widths = [col_width, 10, 8, 18, 20, 12]

    def _fmt(val):
        if isinstance(val, float):
            return f"{val:.4f}"
        if isinstance(val, int):
            return f"{val:,}"
        return str(val)

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    logger.info("\n" + "=" * len(sep))
    logger.info("  PRUNING RESULTS COMPARISON")
    logger.info("=" * len(sep))
    logger.info(sep)
    logger.info(header_row)
    logger.info(sep)

    for name, m in results.items():
        acc = _fmt(m.get("accuracy", float("nan")))
        loss = _fmt(m.get("loss", float("nan")))
        lat = _fmt(m.get("latency_ms_per_token", float("nan")))
        thr = _fmt(m.get("throughput_tokens_per_sec", float("nan")))
        params = _fmt(m.get("n_params", 0) // 1_000_000)

        row_vals = [name, acc, loss, lat, thr, params]
        row = "| " + " | ".join(v.ljust(w) for v, w in zip(row_vals, widths)) + " |"
        logger.info(row)

    logger.info(sep)
    logger.info(
        "\nNote: Accuracy = fraction of task examples where the model's top logit\n"
        "      among candidate tokens matches the correct completion.\n"
        "      Loss     = mean cross-entropy at the prediction position (nats).\n"
        "      Latency  = mean wall-clock ms per input token (batch=1 forward pass).\n"
    )
