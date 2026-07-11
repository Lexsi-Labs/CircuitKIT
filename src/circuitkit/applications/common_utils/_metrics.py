# FILE: circuitkit/applications/common/_metrics.py
"""
Paper-canonical metrics for knowledge editing (Meng et al. 2022 / 2023).

This module implements the five metrics used to evaluate ROME and MEMIT in
the original papers, adapted to TransformerLens models. All probability-based
metrics route through `_tokenization.score_target` so multi-token targets,
BPE/SentencePiece quirks, and BOS handling stay consistent with the rest
of the editing pipeline.

Public surface
--------------
efficacy_metrics(model, prompt, target_new, target_true)
    -> EfficacyMetrics(success, magnitude, p_new, p_true)

paraphrase_metrics(model, paraphrases, target_new, target_true)
    -> AggregateMetrics(success_rate, mean_magnitude, n_used)

neighborhood_metrics(model, neighborhood_prompts, target_true, target_new)
    -> AggregateMetrics(success_rate, mean_magnitude, n_used)
    Note: the "target" used at each neighborhood prompt is `target_true`
    of the EDITED record (per CounterFact convention — neighborhood
    subjects share the relation, so their correct answer is the same
    real-world object as the edit's o^c). target_new is required to
    test P[o*] < P[o^c] at each prompt.

generation_entropy(model, prompt, max_new_tokens=100, n_gram=(2, 3))
    -> float (GE, Eqn 27 of MEMIT paper)

Aggregate helper
----------------
editing_score(es, ps, ns) -> float
    Harmonic mean of three success rates in [0, 1]. Returns 0.0 if any
    is zero (matches paper convention; harmonic mean is undefined / 0
    when any input is 0).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

from ._tokenization import ScoringError, _prepend_bos, _to_ids_1d, score_target

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class EfficacyMetrics:
    """Per-record efficacy metrics on a single prompt.

    Fields:
        success:    P[target_new] > P[target_true]  (ES, Eqn 24)
        magnitude:  P[target_new] - P[target_true]  (EM)
        p_new:      P[target_new | prompt]
        p_true:     P[target_true | prompt]
    """

    success: bool
    magnitude: float
    p_new: float
    p_true: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics over a list of prompts (paraphrases or neighborhood).

    Fields:
        success_rate:    mean of the per-prompt success indicator (PS / NS)
        mean_magnitude:  mean of the per-prompt magnitude (PM / NM)
        n_used:          number of prompts that scored successfully
                         (some may fail tokenisation; reported for diagnostics)
        n_total:         number of prompts attempted
    """

    success_rate: float
    mean_magnitude: float
    n_used: int
    n_total: int


# ── Efficacy (ES, EM) ─────────────────────────────────────────────────────────


def efficacy_metrics(
    model: Any,
    prompt: str,
    target_new: str,
    target_true: str,
) -> EfficacyMetrics:
    """Compute per-record Efficacy Success and Magnitude on a single prompt.

    Implements MEMIT paper Eqn 24:
        ES = P[o* | p(s,r)] > P[o^c | p(s,r)]
        EM = P[o* | p(s,r)] - P[o^c | p(s,r)]

    P[·] is the first-token probability under teacher-forcing — same metric
    used by ROME/MEMIT internals for the v/z optimisation, so consistent
    with how the edit was applied. Multi-token targets are scored via
    sequence_prob which handles them correctly.

    Args:
        model:        HookedTransformer.
        prompt:       The rewrite prompt p(s, r).
        target_new:   The new (counterfactual) object o*.
        target_true:  The original true object o^c.

    Returns:
        EfficacyMetrics. If either target fails to score (tokeniser boundary
        issue), magnitude is set to 0.0 and success to False; failures are
        logged as warnings rather than raised, because the benchmark needs
        per-record robustness across thousands of CounterFact entries.
    """
    p_new = _safe_first_token_prob(model, prompt, target_new)
    p_true = _safe_first_token_prob(model, prompt, target_true)
    return EfficacyMetrics(
        success=bool(p_new > p_true),
        magnitude=float(p_new - p_true),
        p_new=float(p_new),
        p_true=float(p_true),
    )


# ── Paraphrase (PS, PM) ───────────────────────────────────────────────────────


def paraphrase_metrics(
    model: Any,
    paraphrases: Sequence[str],
    target_new: str,
    target_true: str,
) -> AggregateMetrics:
    """Compute Paraphrase Success and Magnitude over a list of rephrasings.

    Implements MEMIT paper Eqn 25:
        PS = E_{p ∈ paraphrases} [P[o* | p] > P[o^c | p]]
        PM = E_{p ∈ paraphrases} [P[o* | p] - P[o^c | p]]

    Args:
        model:        HookedTransformer.
        paraphrases:  Rephrasings of the original (s, r) statement.
        target_new:   The new (counterfactual) object o*.
        target_true:  The original true object o^c.

    Returns:
        AggregateMetrics. If `paraphrases` is empty or all entries fail to
        score, success_rate and mean_magnitude are NaN — caller decides
        how to handle (skip aggregation, report per-record, etc.).
    """
    return _aggregate_pairwise(
        model=model,
        prompts=paraphrases,
        target_for_success=target_new,
        target_for_complement=target_true,
        success_direction=">",
    )


# ── Neighborhood (NS, NM) ─────────────────────────────────────────────────────


def neighborhood_metrics(
    model: Any,
    neighborhood_prompts: Sequence[str],
    target_true: str,
    target_new: str,
) -> AggregateMetrics:
    """Compute Neighborhood Success and Magnitude over related-but-distinct
    subjects sharing the relation.

    Implements MEMIT paper Eqn 26:
        NS = E_{p ∈ neighborhood} [P[o* | p] < P[o^c | p]]
        NM = E_{p ∈ neighborhood} [P[o^c | p] - P[o* | p]]

    Per CounterFact convention (Meng et al. 2022 §3.3, MEMIT §5.2.2):
    neighborhood prompts share the relation `r` of the edited record, so
    their correct answer is the same real-world object o^c as the edit's
    `target_true`. We test that the model still prefers o^c at these
    prompts — i.e. that the edit didn't bleed over to nearby subjects.

    Args:
        model:                  HookedTransformer.
        neighborhood_prompts:   Prompts about distinct subjects sharing the
                                relation (correct answer is target_true).
        target_true:            The original true object o^c (same for all
                                neighborhood prompts).
        target_new:             The counterfactual o* — required to test
                                P[o*] < P[o^c].

    Returns:
        AggregateMetrics.
    """
    return _aggregate_pairwise(
        model=model,
        prompts=neighborhood_prompts,
        target_for_success=target_true,
        target_for_complement=target_new,
        success_direction=">",
    )


# ── Generation Entropy (GE) ───────────────────────────────────────────────────


@torch.no_grad()
def generation_entropy(
    model: Any,
    prompt: str,
    max_new_tokens: int = 100,
    n_gram: Tuple[int, int] = (2, 3),
    weights: Tuple[float, float] = (2.0 / 3.0, 4.0 / 3.0),
) -> float:
    """Compute Generation Entropy (GE) of a free-form continuation.

    Implements MEMIT paper Eqn 27:
        GE = -[ (2/3) Σ f₂(k) log₂ f₂(k) + (4/3) Σ f₃(k) log₂ f₃(k) ]
    where fₙ(·) is the n-gram frequency distribution of the generated
    continuation. The leading negative sign and the inner Σ f log f
    combine to a positive number — degenerate "X X X X" generations
    drop GE toward zero (low entropy = repetitive); diverse text gives
    GE in the ~3–6 range for English on standard tokenisers.

    Generation strategy: greedy argmax (deterministic, reproducible).
    The original repos used greedy with no length penalty; we match.
    Stops early on EOS if encountered.

    Args:
        model:           HookedTransformer.
        prompt:          Generation prompt (e.g. CounterFact's
                         generation_prompts entries).
        max_new_tokens:  How many tokens to generate. Paper uses ~100.
        n_gram:          Which n-gram orders to entropy over. Default (2, 3)
                         matches Eqn 27. Customisable for ablations.
        weights:         Coefficients for each n-gram order. Default
                         (2/3, 4/3) sum to 2 and match Eqn 27 exactly.

    Returns:
        GE as a float. Returns 0.0 if generation produces fewer tokens
        than max(n_gram) (cannot form any n-gram of the largest order).
    """
    if len(n_gram) != len(weights):
        raise ValueError(
            f"n_gram and weights must have same length: " f"{len(n_gram)} vs {len(weights)}"
        )
    if max_new_tokens <= 0:
        return 0.0

    # ── Generate continuation greedily ──────────────────────────────────
    prepend_bos = _prepend_bos(model)
    try:
        ids = _to_ids_1d(model, prompt, prepend_bos=prepend_bos)
    except ScoringError as exc:
        logger.warning(f"generation_entropy: cannot tokenise prompt {prompt!r}: {exc}")
        return 0.0

    eos_id = _get_eos_id(model)
    generated_ids: List[int] = []
    cur = ids.unsqueeze(0)  # [1, L]

    for _ in range(max_new_tokens):
        logits = model(cur)
        next_id = int(logits[0, -1, :].argmax().item())
        if eos_id is not None and next_id == eos_id:
            break
        generated_ids.append(next_id)
        cur = torch.cat(
            [cur, torch.tensor([[next_id]], device=cur.device, dtype=cur.dtype)],
            dim=1,
        )

    # ── n-gram entropy ──────────────────────────────────────────────────
    # Guard: need at least max(n_gram) tokens to form even one of the
    # largest n-grams. With fewer than that, GE is undefined; return 0
    # (matches "fully collapsed / repetitive" downstream).
    if len(generated_ids) < max(n_gram):
        logger.warning(
            f"generation_entropy: generated {len(generated_ids)} tokens, "
            f"need at least {max(n_gram)} for the largest n-gram. Returning 0.0."
        )
        return 0.0

    total = 0.0
    for n, w in zip(n_gram, weights):
        if len(generated_ids) < n:
            continue  # skip this n-gram order entirely if not enough tokens
        ngrams = [tuple(generated_ids[i : i + n]) for i in range(len(generated_ids) - n + 1)]
        counts = Counter(ngrams)
        n_total_ngrams = sum(counts.values())  # renamed to avoid shadowing
        # Σ f(k) log₂ f(k) where f is the *frequency* (probability-like).
        # Eqn 27 writes f as the frequency distribution, so we normalise.
        h = 0.0
        for c in counts.values():
            f = c / n_total_ngrams
            h += f * math.log2(f)
        # Eqn 27 has a leading minus outside the brackets; the inner sum
        # (Σ f log f) is negative, so total = -Σ wᵢ · (Σ fᵢ log fᵢ)
        # = -wᵢ · h_inner. h above is the inner sum (negative).
        total += -w * h

    return float(total)


# ── Aggregator (Score) ────────────────────────────────────────────────────────


def editing_score(
    efficacy_success: float,
    paraphrase_success: float,
    neighborhood_success: float,
) -> float:
    """Harmonic mean of ES, PS, NS — the headline 'Score (S)' in both papers.

    Returns 0.0 if any input is 0.0 or NaN (harmonic mean is undefined
    in those cases and the paper convention is to report 0).
    """
    vals = [efficacy_success, paraphrase_success, neighborhood_success]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) or v <= 0.0 for v in vals):
        return 0.0
    return 3.0 / sum(1.0 / v for v in vals)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _safe_first_token_prob(model: Any, prompt: str, target: str) -> float:
    """Return P(target_first_token | prompt) or 0.0 on tokenisation failure.

    We use first_token_prob (not sequence_prob) because the papers'
    inequalities are all "P[o*] vs P[o^c]" at the prompt's last position
    — the model's prediction for the very next token. This matches the
    score that ROME/MEMIT optimise.
    """
    try:
        return float(score_target(model, prompt, target).first_token_prob)
    except ScoringError as exc:
        logger.warning(f"score_target failed for prompt={prompt!r} target={target!r}: {exc}")
        return 0.0


def _aggregate_pairwise(
    model: Any,
    prompts: Sequence[str],
    target_for_success: str,
    target_for_complement: str,
    success_direction: str,
) -> AggregateMetrics:
    """Common aggregator for paraphrase and neighborhood metrics.

    For each prompt p:
        p_a = P[target_for_success | p]
        p_b = P[target_for_complement | p]
        success = (p_a > p_b)        if success_direction == ">"
                  (p_a < p_b)        otherwise
        magnitude = p_a - p_b

    Then averages over all prompts that scored successfully (at least one
    of the two targets must have scored without ScoringError; if both
    return 0.0 from failure, the prompt still counts in n_total and
    contributes 0.0 magnitude / False success — same convention as the
    reference implementation, which treats a tokeniser failure as a
    zero-confidence outcome).
    """
    if not prompts:
        return AggregateMetrics(
            success_rate=float("nan"),
            mean_magnitude=float("nan"),
            n_used=0,
            n_total=0,
        )

    successes: List[bool] = []
    magnitudes: List[float] = []
    n_used = 0
    for p in prompts:
        try:
            p_a = float(score_target(model, p, target_for_success).first_token_prob)
            p_b = float(score_target(model, p, target_for_complement).first_token_prob)
            n_used += 1
        except ScoringError as exc:
            logger.warning(f"_aggregate_pairwise: skipping prompt {p!r} due to: {exc}")
            continue
        if success_direction == ">":
            successes.append(p_a > p_b)
        else:
            successes.append(p_a < p_b)
        magnitudes.append(p_a - p_b)

    if not successes:
        return AggregateMetrics(
            success_rate=float("nan"),
            mean_magnitude=float("nan"),
            n_used=0,
            n_total=len(prompts),
        )

    return AggregateMetrics(
        success_rate=sum(successes) / len(successes),
        mean_magnitude=sum(magnitudes) / len(magnitudes),
        n_used=n_used,
        n_total=len(prompts),
    )


def _get_eos_id(model: Any) -> Optional[int]:
    """
    Best-effort EOS id from the model's tokenizer. Returns None if
    the tokenizer doesn't expose one (some HF tokenisers don't), in
    which case generation runs to max_new_tokens.

    Note for GPT-2: tokenizer.eos_token_id == bos_token_id == 50256
    (`<|endoftext|>`). Generation may stop unusually early if the model
    happens to predict this token. This matches the reference repo's
    behaviour — the paper's GE numbers were computed under the same
    convention.
    """
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return None
    eos = getattr(tok, "eos_token_id", None)
    return int(eos) if eos is not None else None
