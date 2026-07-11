#!/usr/bin/env python3
"""Case study 24 — Safety refusal / behavior-contrast circuit via the custom-data path.

WHAT THIS VALIDATES (and what it does NOT)
------------------------------------------
This script validates CircuitKit's *custom-data path* end-to-end on a
safety-relevant, instruction-style task. It is the concrete answer to the
review question "can you run one safety dataset end-to-end through the
custom-data path?".

The honest capability being demonstrated:

  * Safety / instruction prompts are NOT syntactic templates (IOI-style named
    entities, subject-verb-object). CircuitKit's auto-corruption strategies
    (entity_swap, token_swap, paraphrase, distractor, role_swap) were designed
    for those syntactic templates and do NOT produce meaningful contrastive
    pairs for instruction-tuned / safety prompts. So for a task like this you
    MUST supply EXPLICIT contrastive pairs — a `corrupted_prompt` /
    `corrupted_answer` column that YOU author — rather than relying on a
    corruption strategy. That explicit-pair path is what this case study
    exercises.

  * The shipped sample dataset (`24_safety_sample.csv`) is deliberately BENIGN
    and illustrative. It is framed as a "refusal / behavior-contrast" circuit:
      - clean   = a benign instruction the model would comply with
                  (answer: a compliance token, e.g. " help")
      - corrupt = a minimally-different instruction (single-word swap) whose
                  intended completion is a refusal (answer: e.g. " refuse")
    The pairs are token-length aligned so `logit_diff` is a valid metric.
    NO real jailbreak strings are embedded. This is interpretability research
    on toy contrastive pairs, not a safety benchmark.

  * The run uses GPT-2 for REPRODUCIBILITY (small, CPU-friendly, cached). GPT-2
    is a base model with no safety training, so the discovered "circuit" here
    reflects only the toy lexical contrast in the sample data. REAL safety
    conclusions require (a) a safety-/instruction-tuned model and (b) the real
    AdvBench dataset (walledai/AdvBench). A commented
    "--- To run on real AdvBench ---" block at the bottom shows exactly how to
    swap those in.

WHAT IT DOES
------------
1. Loads `24_safety_sample.csv` as a custom task via the dict-config custom-data
   path (`config["data"]` with type "template", explicit clean/corrupt columns).
2. Runs circuit discovery with eap-ig on gpt2 (num_examples 16, batch_size 2,
   ig_steps 2 — kept tiny so this runs in a couple of minutes on CPU).
3. Evaluates faithfulness with the meaningful pillars for a signed-metric task:
   ["patching", "ablation", "baselines", "stability"] (n_stability_runs 2).
   The "robustness" pillar is intentionally EXCLUDED: its ratio is
   uninterpretable for signed metrics like logit_diff.
4. Repeats across 3 seeds and prints mean ± std of the Pillar-1 (patching)
   faithfulness score.

Run:
    python examples/case-studies/24-safety-refusal-custom-data.py
"""

import os
import statistics

from circuitkit import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "24_safety_sample.csv")
OUTPUT_DIR = os.path.join(HERE, "results", "24_safety_refusal")

# Explicit contrastive pairs. These template strings reference the CSV columns
# by name via Python .format() syntax. The corrupted side is the key part: we
# supply it explicitly (corrupt_prompt / corrupt_answer) instead of relying on
# a corruption strategy, because auto-corruption does not apply to
# instruction-style prompts.
TEMPLATE = {
    "clean_prompt": "{prompt}",
    "clean_answer": "{answer}",
    "corrupt_prompt": "{corrupted_prompt}",
    "corrupt_answer": "{corrupted_answer}",
}

SEEDS = [0, 1, 2]

# Pillars meaningful for a signed (logit_diff) metric. NOT "robustness": its
# ratio is uninterpretable when the underlying metric can be negative.
PILLARS = ["patching", "ablation", "baselines", "stability"]


def run_one_seed(seed: int) -> float:
    """Discover + evaluate a refusal/behavior-contrast circuit for one seed.

    Returns the Pillar-1 (causal patching) faithfulness score.
    """
    print(f"\n{'=' * 64}\n[seed {seed}] custom-data safety-contrast circuit\n{'=' * 64}")

    # --- 1. Load the CSV as a custom task via the dict-config path ---------
    # Pipeline.from_custom_data registers the CSV as a "template"-type task
    # (config["data"] with explicit clean/corrupt columns) and returns a
    # Pipeline whose .task points at the registered custom task. This is the
    # cleanest expression of the dict-config custom-data path.
    pipe = Pipeline.from_custom_data(
        "gpt2",
        CSV_PATH,
        clean_prompt=TEMPLATE["clean_prompt"],
        clean_answer=TEMPLATE["clean_answer"],
        # Explicit contrastive pairs — the loader keys are `corrupt_prompt` /
        # `corrupt_answer` at the *template* level; these map to the CSV's
        # `corrupted_prompt` / `corrupted_answer` columns.
        corrupt_prompt=TEMPLATE["corrupt_prompt"],
        corrupt_answer=TEMPLATE["corrupt_answer"],
        task_name="safety_refusal_contrast",
        precision="float32",  # float32 keeps the tiny CPU run numerically clean
        output_dir=OUTPUT_DIR,
    )

    # --- 2. Circuit discovery with eap-ig (small, reproducible) ------------
    pipe.discover(
        algorithm="eap-ig",
        level="node",
        sparsity=0.3,
        n_examples=16,
        batch_size=2,
        ig_steps=2,
        seed=seed,
    )
    print(f"[seed {seed}] discovered {len(pipe.circuit)} circuit nodes")

    # --- 3. Faithfulness evaluation (meaningful pillars only) --------------
    pipe.evaluate(
        pillars=PILLARS,
        n_examples=16,
        n_stability_runs=2,
    )

    report = pipe.report
    patching = report.patching_score
    print(
        f"[seed {seed}] Pillar-1 patching={patching:.4f}  "
        f"Pillar-2 ablation={report.ablation_score:.4f}"
    )
    return float(patching)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Safety refusal / behavior-contrast circuit — custom-data path validation")
    print("Model: gpt2 (base, for reproducibility) | metric: logit_diff (signed)")
    print(f"Explicit contrastive pairs from: {CSV_PATH}")

    scores = [run_one_seed(seed) for seed in SEEDS]

    mean = statistics.mean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0

    print(f"\n{'=' * 64}")
    print("PILLAR-1 (CAUSAL PATCHING) FAITHFULNESS ACROSS SEEDS")
    print(f"{'=' * 64}")
    for seed, s in zip(SEEDS, scores):
        print(f"  seed {seed}: patching = {s:.4f}")
    print(f"\n  mean +/- std = {mean:.4f} +/- {std:.4f}  (n={len(scores)} seeds)")
    print(
        "\nNote: GPT-2 is a base model; this validates the custom-data *path*, "
        "not a real safety claim. Use a safety-tuned model + real AdvBench for that."
    )


if __name__ == "__main__":
    main()


# =====================================================================
# --- To run on real AdvBench ---
# =====================================================================
#
# The block below is NOT executed. It shows how to point the SAME custom-data
# path at the real AdvBench harmful-behaviors dataset (walledai/AdvBench on the
# HuggingFace Hub) instead of the benign sample CSV above.
#
# AdvBench ships harmful prompts (`prompt`) each paired with the target
# affirmative-completion string a jailbreak tries to elicit (`target`, e.g.
# "Sure, here is ..."). It does NOT ship a benign contrastive partner, and
# auto-corruption will NOT synthesize a meaningful one for instruction prompts —
# so you must construct the contrastive pair yourself. The interpretable
# contrast for a safety-tuned model is:
#
#     clean_prompt  = the harmful AdvBench prompt
#     clean_answer  = a REFUSAL token the safe model emits, e.g. " I"  (as in
#                     "I can't help with that") or " Sorry"
#     corrupt_answer= the jailbreak TARGET's first token (compliance), derived
#                     from AdvBench's `target` column, e.g. " Sure"
#     corrupt_prompt= the same harmful prompt (the contrast is answer-only), OR
#                     a jailbreak-wrapped variant of it.
#
# Then the logit_diff (refusal_token - compliance_token) measures the refusal
# behavior, and the discovered circuit is the *refusal* circuit.
#
# IMPORTANT: run this on a SAFETY-TUNED model (e.g. an instruction-tuned chat
# model with refusal training). On GPT-2 there is no refusal behavior to find.
# Do NOT commit the materialized harmful prompts into the repo.
#
# ---------------------------------------------------------------------
# from datasets import load_dataset
# import pandas as pd
#
# # 1. Pull AdvBench harmful behaviors from the HF Hub.
# adv = load_dataset("walledai/AdvBench", split="train")   # columns: prompt, target
#
# # 2. Build explicit contrastive pairs. Refusal vs. the jailbreak target.
# def first_word(s: str) -> str:
#     return " " + s.strip().split()[0] if s.strip() else " Sure"
#
# rows = []
# for ex in adv.select(range(64)):        # subsample for a manageable run
#     harmful = ex["prompt"]
#     comply_tok = first_word(ex["target"])   # e.g. " Sure" — what a jailbreak elicits
#     rows.append({
#         "prompt":            harmful,
#         "answer":            " I",          # refusal-token the SAFE model emits
#         "corrupted_prompt":  harmful,       # answer-only contrast (same prompt)
#         "corrupted_answer":  comply_tok,    # compliance token from AdvBench target
#     })
# adv_csv = os.path.join(OUTPUT_DIR, "advbench_pairs.csv")   # kept OUT of git
# pd.DataFrame(rows).to_csv(adv_csv, index=False)
#
# # 3. Same custom-data path — just a safety-tuned model + the AdvBench CSV.
# #    (Token-length alignment is answer-side here since prompts are identical;
# #     for prompt-side jailbreak variants, use align_strategy="none" and the
# #     kl_divergence metric — see docs/user-guide/custom-data.md.)
# pipe = Pipeline.from_custom_data(
#     "meta-llama/Llama-3.2-1B-Instruct",   # a SAFETY-TUNED model — required
#     adv_csv,
#     clean_prompt="{prompt}",   clean_answer="{answer}",
#     corrupt_prompt="{corrupted_prompt}", corrupt_answer="{corrupted_answer}",
#     task_name="advbench_refusal",
#     output_dir=OUTPUT_DIR,
# )
# pipe.discover(algorithm="eap-ig", level="node", sparsity=0.3,
#               n_examples=64, batch_size=4, ig_steps=5, seed=0)
# pipe.evaluate(pillars=["patching", "ablation", "baselines", "stability"],
#               n_examples=64, n_stability_runs=3)
# print("AdvBench refusal-circuit patching:", pipe.report.patching_score)
