"""MMLU evaluation for HookedTransformer models.

Implements a lightweight 5-shot MMLU accuracy estimate that works with
any HookedTransformer checkpoint. Uses the lm-eval harness when it is
available; falls back to a direct HuggingFace datasets implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer


def evaluate_mmlu(
    model: "HookedTransformer",
    n_samples: int = 224,
    shots: int = 5,
    subjects: Optional[list] = None,
) -> float:
    """Estimate MMLU accuracy.

    Tries the lm-eval harness first (fast, standard). Falls back to a
    direct HuggingFace datasets implementation if lm-eval is unavailable
    or raises an error.

    Args:
        model: HookedTransformer model (in eval mode, on the correct device).
        n_samples: Maximum total examples to evaluate across all subjects.
        shots: Number of few-shot examples prepended per question.
        subjects: Optional list of MMLU subject names. If None, uses
                  all available subjects.

    Returns:
        Accuracy in [0, 1].
    """
    try:
        return _evaluate_via_lm_eval(model, n_samples=n_samples, shots=shots)
    except Exception:
        return _evaluate_direct(model, n_samples=n_samples, shots=shots, subjects=subjects)


def _evaluate_via_lm_eval(model, *, n_samples: int, shots: int) -> float:
    from circuitkit.evaluation.lm_eval_simple import evaluate_lm_eval

    results = evaluate_lm_eval(
        model,
        tasks=["mmlu"],
        fewshot=shots,
        limit=n_samples,
        verbosity="ERROR",
    )
    mmlu = results.get("results", {}).get("mmlu", {})
    # lm-eval 0.4+ uses the ",none" suffix; earlier versions use bare key.
    acc = mmlu.get("acc,none", mmlu.get("acc", None))
    if acc is None:
        raise ValueError(f"Unexpected lm-eval MMLU result keys: {list(mmlu.keys())}")
    return float(acc)


def _evaluate_direct(
    model,
    *,
    n_samples: int,
    shots: int,
    subjects: Optional[list],
) -> float:
    """Direct HuggingFace-datasets MMLU evaluation.

    Loads the hendrycks_test (MMLU) dataset, formats each question as a
    multiple-choice prompt, runs the model on all 4 answer choices, and
    picks the choice with the highest log-prob for the answer letter.
    """
    import random

    import torch
    from datasets import load_dataset

    _SUBJECTS = subjects or [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = model.cfg.device
    choices_tokens = [
        tok(" A", add_special_tokens=False)["input_ids"][0],
        tok(" B", add_special_tokens=False)["input_ids"][0],
        tok(" C", add_special_tokens=False)["input_ids"][0],
        tok(" D", add_special_tokens=False)["input_ids"][0],
    ]

    rng = random.Random(42)
    all_items: list = []
    per_subj = max(1, n_samples // len(_SUBJECTS))

    for subj in _SUBJECTS:
        try:
            ds_test = load_dataset("hendrycks_test", subj, split="test")
            ds_dev = load_dataset("hendrycks_test", subj, split="dev")
        except Exception:
            try:
                ds_test = load_dataset("cais/mmlu", subj, split="test")
                ds_dev = load_dataset("cais/mmlu", subj, split="dev")
            except Exception:
                continue
        dev_rows = list(ds_dev)[:shots]
        test_rows = list(ds_test)[:per_subj]
        all_items.append((subj, dev_rows, test_rows))

    if not all_items:
        return 0.0

    rng.shuffle(all_items)

    def _fmt_question(row, include_answer: bool = False) -> str:
        opts = "\n".join(f"{letter}. {ch}" for letter, ch in zip("ABCD", row["choices"]))
        q = f"Question: {row['question']}\n{opts}\nAnswer:"
        if include_answer:
            q += f" {row['answer']}"
        return q

    correct = total = 0
    model.eval()

    for subj, dev_rows, test_rows in all_items:
        few_shot = "\n\n".join(_fmt_question(r, include_answer=True) for r in dev_rows)
        for row in test_rows:
            prompt = few_shot + "\n\n" + _fmt_question(row) if few_shot else _fmt_question(row)
            try:
                in_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)[
                    "input_ids"
                ].to(device)
                with torch.no_grad():
                    logits = model(in_ids)
                last = logits[0, -1]
                scores = [last[t].item() for t in choices_tokens]
                pred = int(scores.index(max(scores)))
                label = (
                    int(row["answer"])
                    if isinstance(row["answer"], int)
                    else "ABCD".index(row["answer"])
                )
                correct += int(pred == label)
                total += 1
            except Exception:
                total += 1

    return correct / total if total > 0 else 0.0
