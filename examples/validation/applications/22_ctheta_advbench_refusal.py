"""C-ΔΘ refusal-direction case study on AdvBench (GPT-2).

Mirrors Arditi et al. 2024's refusal-direction recipe but expressed
through C-ΔΘ:

  positive task: harmful prompt → compliance token (' Sure')
  negative task: harmful prompt → refusal token (' I')

Discover the circuit on AdvBench (clean=refusal, corrupt=compliance),
fine-tune circuit-located heads on positive vs negative, then
steered = θ_target + k · (θ_pos − θ_neg). The steered model should
shift its top-token probability at the answer position from refusal
toward compliance (or the other way around if k < 0). Higher absolute
Δ p(refusal) ≈ stronger refusal-circuit localisation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apps_common import (  # noqa: E402
    get_or_run_discovery, make_results_dir, write_status,
)

SCRIPT_NAME = "22_ctheta_advbench_refusal"
MODEL = "gpt2"
TASK = "advbench_ctheta"
ALGOS = ["eap", "eap-ig", "atp-gd"]


def _build_advbench_task(cache_dir: Path):
    """Register a NormalizedTaskSpec for AdvBench via SafetyPromptAdapter."""
    from datasets import load_dataset
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task, is_task_registered
    if is_task_registered(TASK):
        return
    raw = list(load_dataset("walledai/AdvBench", split="train").take(64))
    # Arditi et al. 2024 style: harmful prompt vs benign control, different answer tokens.
    ds = SafetyPromptAdapter().adapt(raw, name=TASK, max_records=48,
                                     pairing_mode="harmful_vs_benign")
    ds.records = [r for r in ds.records if r.is_paired]
    spec = NormalizedTaskSpec(ds, name=TASK, cache_dir=str(cache_dir))
    register_task(spec)


def _refusal_loss(model, batch, *, target="positive"):
    """positive = drive toward refusal (' I'); negative = drive toward
    compliance (' Sure'). Sign flipped so each pipeline's loss is
    minimised."""
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    clean, corrupted, label = batch
    pair_padding = getattr(batch, "pair_padding_side", None)
    clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
        model, clean, corrupted, pair_padding_side=pair_padding,
    )
    logits = model(clean_tokens, attention_mask=attn_mask)
    losses = []
    for i in range(logits.shape[0]):
        last = input_lengths[i] - 1
        refusal_id = int(label[i, 0].item())   # ' I'
        compliance_id = int(label[i, 1].item())  # ' Sure'
        diff = logits[i, last, refusal_id] - logits[i, last, compliance_id]
        losses.append(-diff if target == "positive" else diff)
    return sum(losses) / max(1, len(losses))


def _eval_refusal_gap(model, dataloader) -> float:
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    diffs = []
    for clean, corrupted, label in dataloader:
        clean_tokens, _, attn_mask, _, input_lengths, _ = tokenize_batch_pair(
            model, clean, corrupted,
            pair_padding_side=getattr(dataloader, "pair_padding_side", None),
        )
        with torch.inference_mode():
            logits = model(clean_tokens, attention_mask=attn_mask)
        for i in range(logits.shape[0]):
            last = input_lengths[i] - 1
            r = int(label[i, 0].item())
            c = int(label[i, 1].item())
            diffs.append(logits[i, last, r].item() - logits[i, last, c].item())
    return sum(diffs) / max(1, len(diffs))


def _run_one_algo(algo, model_clean, dl, k=2.0):
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering
    try:
        cell = get_or_run_discovery(algo, MODEL, TASK, num_examples=24)
    except Exception as exc:
        return {"algorithm": algo,
                "error": f"discovery: {type(exc).__name__}: {str(exc)[:200]}"}
    cs = cell["scores"]
    t0 = time.time()
    base = _eval_refusal_gap(model_clean, dl)
    try:
        cws = CircuitWeightSteering(model_clean, cs.node_scores, top_k_frac=0.05)
        cws.fine_tune_positive(
            dl, lambda m, b: _refusal_loss(m, b, target="positive"),
            n_steps=15, lr=1e-5,
        )
        cws.fine_tune_negative(
            dl, lambda m, b: _refusal_loss(m, b, target="negative"),
            n_steps=15, lr=1e-5,
        )
        cws.compute_steering_vector()
        steered = cws.apply_steering(model_clean, k=k)
        post = _eval_refusal_gap(steered, dl)
        del steered
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        return {"algorithm": algo,
                "error": f"steering: {type(exc).__name__}: {str(exc)[:200]}"}
    return {
        "algorithm": algo,
        "wall_seconds": round(time.time() - t0, 2),
        "n_heads": len(cws.head_names),
        "steering_norm": round(cws.total_steering_norm(), 4),
        "k": k,
        "base_refusal_gap": round(base, 4),
        "post_refusal_gap": round(post, 4),
        "delta_refusal": round(post - base, 4),
    }


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        _build_advbench_task(out_dir / "_cache")
    except Exception as exc:
        write_status(out_dir, {
            "script": SCRIPT_NAME, "status": "BROKEN",
            "error": f"could not load AdvBench: {exc}",
        })
        return 1

    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.float32)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    from circuitkit.tasks.registry import get_task
    cfg = {"algorithm": "eap", "task": TASK, "level": "node",
           "batch_size": 4, "data_params": {"num_examples": 32}}
    dl = get_task(TASK).build_dataloader(model, cfg, device)

    rows = []
    for algo in ALGOS:
        row = _run_one_algo(algo, model, dl)
        rows.append(row)
        if "error" in row:
            print(f"  {algo:10s}  ERR: {row['error'][:80]}", flush=True)
        else:
            print(f"  {algo:10s}  Δrefusal={row['delta_refusal']:+.3f}  "
                  f"|w_b|={row['steering_norm']}  ({row['wall_seconds']}s)",
                  flush=True)

    md = ["# Application 22 — C-ΔΘ Refusal-Direction Steering on AdvBench (GPT-2)",
          "",
          "Per-algorithm: discover refusal-vs-compliance circuit, fine-tune "
          "the top-5% heads on positive (refusal) vs negative (compliance), "
          "then steered = θ + k · (θ_pos − θ_neg). Higher |Δ refusal gap| "
          "means stronger localisation.",
          "",
          "| Algorithm | base | post | Δ refusal | |w_b| | Wall |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| `{r['algorithm']}` | — | — | ERR | — | — |")
        else:
            md.append(
                f"| `{r['algorithm']}` | {r['base_refusal_gap']:+.3f} | "
                f"{r['post_refusal_gap']:+.3f} | {r['delta_refusal']:+.3f} | "
                f"{r['steering_norm']} | {r['wall_seconds']}s |"
            )
    (out_dir / "table.md").write_text("\n".join(md) + "\n")
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {
        "script": SCRIPT_NAME,
        "module": "applications.weight_steering (safety)",
        "input": {"model": MODEL, "task": TASK, "n_algos": len(ALGOS), "k": 2.0},
        "metrics": {"n_algos_ok": n_ok, "n_algos_total": len(ALGOS)},
        "status": "WORKING" if n_ok == len(ALGOS) else "NEEDS-FIX",
    })
    return 0 if n_ok == len(ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
