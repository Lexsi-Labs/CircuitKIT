"""Lazy registration of custom-HF-dataset TaskSpecs for the benchmark suite.

Each `register_*` function loads a real HuggingFace dataset, runs the
matching data-layer Adapter to get a NormalizedDataset, applies the
appropriate CorruptionStrategy to make it paired, wraps in
NormalizedTaskSpec, and registers it under a stable name so the
benchmark cells can reference the task by string.

Idempotent: if the task is already registered the function is a no-op.

Coverage in this file:
  arc_easy        — allenai/ai2_arc / ARC-Easy (MCQ)
  arc_challenge   — allenai/ai2_arc / ARC-Challenge (MCQ)
  hellaswag       — Rowan/hellaswag (MCQ)
  crows_pairs     — nyu-mll/crows-pairs (pairwise; from the project mirror)
  tofu            — locuslab/TOFU (forget/retain)

Datasets requiring custom paired construction (gsm8k, ifeval) are
deferred — they'd need free-form-generation corruption strategies that
aren't off-the-shelf in `circuitkit.data.corruption.*`.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[1] / "_cache" / "custom_tasks"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hf_take(dataset_path: str, *args, split="train", take=200, **kwargs):
    """Load `take` examples from an HF dataset in streaming mode."""
    from datasets import load_dataset
    candidates = [split, "validation", "test", "train"]
    last_err = None
    for s in candidates:
        try:
            return list(load_dataset(dataset_path, *args, split=s,
                                     streaming=True, **kwargs).take(take))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"Could not load {dataset_path}: {last_err}")


def _apply_strategy(ds, strategy):
    """Apply a CorruptionStrategy.apply() to every record and return a
    new NormalizedDataset of the records that successfully corrupted.

    The strategies are designed to operate per-record; this helper does
    the per-record map and reconstructs the dataset.
    """
    from circuitkit.data.normalized import NormalizedDataset
    new_records, failed = [], 0
    for r in ds.records:
        out = strategy.apply(r)
        if out.is_paired:
            new_records.append(out)
        else:
            failed += 1
    if failed:
        logger.info(f"_apply_strategy {strategy.name}: dropped {failed}/{len(ds)} records")
    return NormalizedDataset(
        name=ds.name, shape=ds.shape, records=new_records,
        source=ds.source, schema_version=ds.schema_version,
        meta={**ds.meta, "n_paired": len(new_records),
              "n_failed_corruption": failed},
    )


def _materialise_mcq(name: str, hf_path: str, hf_subset: Optional[str],
                     split: str, take: int):
    """Load HF MCQ data and return a paired NormalizedDataset.

    Choice-swap is the canonical corruption for circuit discovery on
    MCQ. Conmy et al. 2023 (ACDC) used an analogous token-swap
    intervention on Greater-Than; Lieberum et al. 2023 used MCQ format
    on Chinchilla. The clean prompt presents choices A/B/C/D; the
    corrupt prompt swaps the correct choice's CONTENT with a
    distractor's, so the correct LETTER label points to wrong content.
    Both prompts AND target letters differ — valid EAP setup.
    """
    from circuitkit.data.adapters.mcq import MCQAdapter
    from circuitkit.data.corruption.mcq_choice_swap import MCQChoiceSwap

    raw_args = (hf_path,)
    if hf_subset:
        raw_args = (hf_path, hf_subset)
    raw = _hf_take(*raw_args, split=split, take=take)
    ds = MCQAdapter().adapt(raw, name=name, max_records=take, source=hf_path)
    return _apply_strategy(ds, MCQChoiceSwap())


def _materialise_pairwise(name: str, csv_path: str):
    """CrowS-Pairs via PairwiseAdapter, filtered to records with a
    non-degenerate logit-difference signal.

    PairwiseAdapter strips the last word as the target. When both
    sentences end with the same word (e.g. "...short skirt"), the
    clean and corrupt targets are identical and the EAP metric is
    structurally zero. We drop those records. Vig et al. 2020 use
    pronoun-position swap for bias circuits, which is semantically
    cleaner but requires per-record parsing — last-word filtering is
    a coarse-but-tractable approximation.
    """
    from circuitkit.data.adapters.pairwise import PairwiseAdapter
    from circuitkit.data.normalized import NormalizedDataset

    ds = PairwiseAdapter().adapt(csv_path, name=name, source=csv_path,
                                 max_records=400)
    kept = [r for r in ds.records
            if r.is_paired
            and r.clean_answer.strip() != r.corrupt_answer.strip()
            and r.clean_prompt != r.corrupt_prompt]
    return NormalizedDataset(
        name=ds.name, shape=ds.shape, records=kept[:200],
        source=ds.source, schema_version=ds.schema_version,
        meta={**ds.meta, "n_after_filter": len(kept[:200]),
              "n_dropped_degenerate": len(ds.records) - len(kept)},
    )


def _materialise_math(name: str, hf_path: str, hf_subset: Optional[str],
                      split: str, take: int):
    """GSM8K with question-pair construction.

    Stolfo et al. 2023 (EMNLP, arxiv:2305.15054) define the canonical
    arithmetic-circuit corruption: pair two prompts with different
    operand values (e.g. "2 + 3 =" vs "4 + 5 ="). For GSM8K word
    problems we can't programmatically rewrite operands without a
    solver, so we approximate by pairing each question with another
    GSM8K question that has a DIFFERENT final answer. The clean
    prompt and target produce one numeric answer; the corrupt prompt
    and target produce a different numeric answer. Both prompts and
    targets differ — valid EAP setup.

    FinalAnswerSwap was the original choice but it does not change the
    prompt (only perturbs the target token), giving zero
    activation-difference. Documented as inappropriate for EAP
    discovery.
    """
    import random as _r
    from circuitkit.data.adapters.math import MathAdapter
    from circuitkit.data.normalized import NormalizedDataset, ContrastiveRecord, ContrastSource

    raw_args = (hf_path,)
    if hf_subset:
        raw_args = (hf_path, hf_subset)
    raw = _hf_take(*raw_args, split=split, take=take)

    ds = MathAdapter().adapt(raw, name=name, max_records=take, source=hf_path)
    if len(ds.records) < 2:
        raise RuntimeError(f"need >=2 records for pairing; got {len(ds)}")
    rng = _r.Random(42)
    by_answer: Dict[str, list] = {}
    for r in ds.records:
        by_answer.setdefault(r.clean_answer.strip(), []).append(r)
    answers = list(by_answer.keys())
    if len(answers) < 2:
        raise RuntimeError(f"need >=2 distinct answers; got {len(answers)}")

    pairs = []
    for r in ds.records:
        other_ans = rng.choice([a for a in answers if a != r.clean_answer.strip()])
        other = rng.choice(by_answer[other_ans])
        pairs.append(ContrastiveRecord(
            record_id=r.record_id,
            clean_prompt=r.clean_prompt, clean_answer=r.clean_answer,
            corrupt_prompt=other.clean_prompt, corrupt_answer=other.clean_answer,
            target_field="first_answer_token",
            contrast_source=ContrastSource.GENERATED,
            meta={**r.meta, "_pairing": "different_question_different_answer"},
        ))
    return NormalizedDataset(
        name=ds.name, shape=ds.shape, records=pairs,
        source=ds.source, schema_version=ds.schema_version,
        meta={**ds.meta, "n_paired": len(pairs)},
    )


def _materialise_forget_retain(name: str):
    """TOFU: pair forget questions vs retain questions.

    Following the CUD paper (arxiv:2601.09624) which discovers circuits
    on forget vs retain prompts. We pair each forget question's prompt
    (clean) with a randomly-selected retain question's prompt
    (corrupt). The discovered circuit reflects what differentiates
    forget-knowledge recall from retain-knowledge recall.
    """
    import random as _r
    from circuitkit.data.adapters.forget_retain import ForgetRetainAdapter
    from circuitkit.data.normalized import NormalizedDataset, ContrastiveRecord, ContrastSource

    forget = _hf_take("locuslab/TOFU", "forget01", split="train", take=120)
    retain = _hf_take("locuslab/TOFU", "retain99", split="train", take=120)
    raw = {"forget": forget, "retain": retain}
    ds = ForgetRetainAdapter().adapt(raw, name=name, max_records=240,
                                     source="locuslab/TOFU")
    forget_recs = [r for r in ds.records if r.meta.get("split") == "forget"]
    retain_recs = [r for r in ds.records if r.meta.get("split") == "retain"]
    if not forget_recs or not retain_recs:
        raise RuntimeError(
            f"TOFU pairing failed: forget={len(forget_recs)} "
            f"retain={len(retain_recs)}"
        )
    rng = _r.Random(42)
    pairs = []
    for i, f in enumerate(forget_recs):
        # Find a retain record with a DIFFERENT first-answer token so the
        # logit-diff metric is non-degenerate.
        candidates = [r for r in retain_recs
                      if r.clean_answer.strip() != f.clean_answer.strip()]
        if not candidates:
            continue
        r2 = rng.choice(candidates)
        pairs.append(ContrastiveRecord(
            record_id=f"forget-{i:05d}",
            clean_prompt=f.clean_prompt, clean_answer=f.clean_answer,
            corrupt_prompt=r2.clean_prompt, corrupt_answer=r2.clean_answer,
            target_field="first_answer_token",
            contrast_source=ContrastSource.GENERATED,
            meta={"forget_id": f.record_id, "retain_id": r2.record_id,
                  "_pairing": "forget_vs_retain"},
        ))
    return NormalizedDataset(
        name=ds.name, shape=ds.shape, records=pairs[:120],
        source=ds.source, schema_version=ds.schema_version,
        meta={**ds.meta, "n_paired": len(pairs[:120])},
    )


def _register(ds, task_name: str):
    """Register a paired NormalizedDataset under task_name."""
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task, is_task_registered
    if is_task_registered(task_name):
        return
    spec = NormalizedTaskSpec(ds, name=task_name,
                              cache_dir=str(_CACHE_DIR / task_name))
    register_task(spec)
    logger.info(f"Registered custom task: {task_name} ({len(ds)} records)")


# ---- per-dataset entrypoints --------------------------------------------

def register_arc_easy() -> None:
    from circuitkit.tasks.registry import is_task_registered
    if is_task_registered("arc_easy"):
        return
    ds = _materialise_mcq("arc_easy", "allenai/ai2_arc", "ARC-Easy",
                          split="test", take=200)
    _register(ds, "arc_easy")


def register_arc_challenge() -> None:
    from circuitkit.tasks.registry import is_task_registered
    if is_task_registered("arc_challenge"):
        return
    ds = _materialise_mcq("arc_challenge", "allenai/ai2_arc", "ARC-Challenge",
                          split="test", take=200)
    _register(ds, "arc_challenge")


def register_hellaswag() -> None:
    from circuitkit.tasks.registry import is_task_registered
    if is_task_registered("hellaswag"):
        return
    ds = _materialise_mcq("hellaswag", "Rowan/hellaswag", None,
                          split="validation", take=200)
    _register(ds, "hellaswag")


def register_crows_pairs() -> None:
    from circuitkit.tasks.registry import is_task_registered
    if is_task_registered("crows_pairs"):
        return
    csv_path = _CACHE_DIR / "crows_pairs.csv"
    if not csv_path.exists():
        url = ("https://raw.githubusercontent.com/nyu-mll/crows-pairs/"
               "master/data/crows_pairs_anonymized.csv")
        urllib.request.urlretrieve(url, csv_path)
    ds = _materialise_pairwise("crows_pairs", str(csv_path))
    _register(ds, "crows_pairs")


def register_tofu() -> None:
    from circuitkit.tasks.registry import is_task_registered
    if is_task_registered("tofu"):
        return
    ds = _materialise_forget_retain("tofu")
    _register(ds, "tofu")


def register_gsm8k() -> None:
    """GSM8K math word problems via MathAdapter + FinalAnswerSwap.

    The adapter extracts the final numeric answer from each example
    (text before/after '####'), and FinalAnswerSwap perturbs that number
    by +/-1 or +/-2 to produce a paired clean/corrupted target. Useful
    for testing whether the discovered circuit truly produces the right
    digit vs an adjacent one.
    """
    from circuitkit.tasks.registry import is_task_registered
    if is_task_registered("gsm8k"):
        return
    ds = _materialise_math("gsm8k", "openai/gsm8k", "main",
                           split="test", take=200)
    _register(ds, "gsm8k")


# IFEval is documented but NOT registered: it is a generation-evaluation
# task (each example asks the model to follow free-form instructions
# like "write a paragraph with no commas"). There is no single-token
# answer to pair against, so the EAP / AtP / RelP family — which rely
# on a 1-token logit-difference metric — does not apply directly.
# Discovering a circuit for "instruction following" requires either:
#   (a) a per-instruction probe head trained to predict compliance, or
#   (b) a fluency-vs-violation paired dataset constructed by an LLM
#       (see corruption/llm_counterfactual.py — needs an external LLM).
# Both are research-grade extensions and out of scope for this sweep.


# ---- bulk registration ---------------------------------------------------

CUSTOM_TASKS = [
    ("arc_easy",       register_arc_easy),
    ("arc_challenge",  register_arc_challenge),
    ("hellaswag",      register_hellaswag),
    ("crows_pairs",    register_crows_pairs),
    ("tofu",           register_tofu),
    ("gsm8k",          register_gsm8k),
]


def register_all() -> dict:
    """Register every custom task. Returns {name: ok_bool}."""
    results = {}
    for name, fn in CUSTOM_TASKS:
        try:
            fn()
            results[name] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to register {name}: {exc}")
            results[name] = False
    return results
