"""
circuitkit/applications/common/_tokenization.py

Single source of truth for every tokeniser-aware operation in
circuitkit.apply. All other modules import from here instead of
re-tokenising prompts/targets or re-locating subject tokens by hand.

Why one module?
---------------
Knowledge editing has four operations that touch the tokeniser:

    1. Score P(target | prompt)                          (verifiers, leakage)
    2. Locate the subject's last token in a prompt       (ROME u/v, MEMIT z/k)
    3. Build a teacher-forced (prompt, target) sequence  (ROME _optimize_v,
                                                          MEMIT _compute_z)
    4. Argmax-token-at-end of a prompt                   (efficacy baselines)

Historically these lived in ROME, MEMIT, the verifier, and the runner —
each with its own subtle drift around leading-space rules and BOS
handling. Centralising them here means:

    * the leading-space rule is applied once, not four times;
    * BOS handling honours `model.cfg.default_prepend_bos` everywhere
      (matters for Qwen and other `default_prepend_bos=False` models);
    * subject-finding has one implementation, not duplicates in
      RomeHandler and MemitHandler that were already byte-identical
      and would inevitably drift.

Public API
----------
Exceptions:
    ScoringError                Tokenisation/scoring failed.
    SubjectLocationError        Subject not found in prompt.

Result types:
    TargetScore                 Output of score_target.
    TeacherForcedSequence       Output of build_teacher_forced.

Functions:
    format_target(target, prompt=None) -> str
    tokenize_prompt(model, prompt) -> torch.Tensor      # [1, L]
    build_teacher_forced(model, prompt, target) -> TeacherForcedSequence
    score_target(model, prompt, target) -> TargetScore  # @torch.no_grad
    locate_subject_last_token(model, prompt, subject) -> int
    argmax_token_at_end(model, prompt) -> (int, str)    # @torch.no_grad

Design notes
------------
* The leading-space rule is correct for every tokeniser scheme
  TransformerLens supports (BPE-Ġ, SentencePiece-▁, WordPiece, byte-level).
  No per-family branching is needed.

* `build_teacher_forced` does NOT use `@torch.no_grad`: trainers need
  to run the resulting tensor through a differentiable forward pass.
  It only constructs token tensors; it never calls the model.

* `score_target` keeps its `@torch.no_grad` decorator. Callers that
  need gradients should call `build_teacher_forced` directly and run
  their own forward pass + cross-entropy.

* `locate_subject_last_token` preserves the two-tier strategy the
  existing trainer-private implementations used: offset-mapping first
  (handles BPE space artifacts cleanly on fast tokenisers), then
  token-id sequence matching as a fallback for slow tokenisers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Exceptions ────────────────────────────────────────────────────────────────


class ScoringError(ValueError):
    """Raised when a target cannot be scored (empty tokenisation,
    non-compositional tokeniser merging the prompt/target boundary, etc.)."""


class SubjectLocationError(ValueError):
    """Raised when the subject string cannot be located in the prompt's
    tokenisation. Distinct from ScoringError so callers can decide
    whether to fail the whole edit or fall back to a different prompt."""


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class TargetScore:
    """Result of scoring P(target | prompt) under a model.

    Field semantics:
        first_token_prob   P(target_token_0 | prompt). What ROME/MEMIT
                           optimise; the right value for EditResult.
                           confidence_before/after for backward compat.
        sequence_logprob   Σ log P(target_token_t | prompt, target_token_<t).
                           Correct for both single- and multi-token targets.
        sequence_prob      exp(sequence_logprob), clamped to [0, 1]. May
                           underflow to 0.0 for long rare targets; use
                           sequence_logprob when that happens.
        per_token_logprobs One entry per target token, in order.
        target_token_ids   Vocab ids that were scored.
        target_text_used   The string actually tokenised (may differ from
                           `target` if a leading space was prepended).
    """

    first_token_prob: float
    sequence_logprob: float
    sequence_prob: float
    per_token_logprobs: List[float] = field(default_factory=list)
    target_token_ids: List[int] = field(default_factory=list)
    target_text_used: str = ""


@dataclass
class TeacherForcedSequence:
    """Tokenised (prompt + target) ready for a teacher-forced training pass.

    Trainers consume:
        full_ids[:, :prompt_len]                         -> prompt-only forward
        full_ids                                          -> full forward pass
        target_ids                                        -> CE targets
        full_ids[0, prompt_len-1 : prompt_len-1 + T]     -> logit positions
                                                            that predict target

    Field semantics:
        full_ids            [1, L+T] LongTensor on the model's device.
        prompt_len          L: number of prompt tokens (incl. BOS if prepended).
        target_len          T: number of target tokens (>= 1).
        target_ids          [T] LongTensor; equals full_ids[0, L:L+T].
        target_text_used    The formatted target string (may have a leading
                            space prepended by `format_target`).
        prepend_bos_used    Whether BOS was prepended; honours
                            `model.cfg.default_prepend_bos`.
    """

    full_ids: torch.Tensor
    prompt_len: int
    target_len: int
    target_ids: torch.Tensor
    target_text_used: str
    prepend_bos_used: bool


# ── Target formatting ─────────────────────────────────────────────────────────


def format_target(target: str, prompt: str | None = None) -> str:
    """Return the target string in the form a tokeniser expects to see
    after `prompt`.

    Rule: prepend a single ASCII space unless one is already present at
    the boundary (either at the start of `target` or at the end of
    `prompt`). This is correct for every tokeniser scheme supported by
    TransformerLens:

        BPE-GPT2:        " Lyon"  ->  ĠLyon (one token)
        SentencePiece:   " Lyon"  ->  ▁Lyon (one token)
        WordPiece:       " Lyon"  ->  Lyon  (leading space stripped, fresh word)
        byte-level/other: " Lyon" ->  literal bytes; tokeniser handles it

    The boundary check uses ASCII space rather than `str.isspace()` because
    other whitespace (\\n, \\t) tokenises differently and shouldn't suppress
    the space.
    """
    if not target:
        raise ScoringError("Empty target string.")
    if target[0] == " ":
        return target
    if prompt and prompt[-1] == " ":
        return target
    return " " + target


# ── Internal helpers ──────────────────────────────────────────────────────────


def _prepend_bos(model: Any) -> bool:
    """Honour the model's own default rather than hardcoding True. Some
    models (notably Qwen via TransformerLens) set this to False; forcing
    True there shifts every position by one and silently corrupts every
    score and every subject-token lookup."""
    cfg = getattr(model, "cfg", None)
    return bool(getattr(cfg, "default_prepend_bos", True))


def _to_ids_1d(model: Any, text: str, prepend_bos: bool) -> torch.Tensor:
    """Tokenise to a 1-D LongTensor on the model's device. Raises
    ScoringError on empty output."""
    ids = model.to_tokens(text, prepend_bos=prepend_bos)
    if ids.numel() == 0 or ids.shape[-1] == 0:
        raise ScoringError(f"Tokenisation produced an empty sequence for: {text!r}")
    return ids[0]


# ── Public: prompt tokenisation ───────────────────────────────────────────────


def tokenize_prompt(model: Any, prompt: str) -> torch.Tensor:
    """Tokenise `prompt` into a [1, L] LongTensor on the model's device,
    honouring `model.cfg.default_prepend_bos`. Raises ScoringError on
    empty prompt or empty tokenisation.

    This exists so callers (e.g. `_compute_u`) stop hardcoding
    `prepend_bos=True` and stay aligned with the rest of this module.
    """
    if not prompt:
        raise ScoringError("Empty prompt.")
    prepend_bos = _prepend_bos(model)
    return _to_ids_1d(model, prompt, prepend_bos=prepend_bos).unsqueeze(0)


# ── Public: teacher-forced sequence construction ──────────────────────────────


def build_teacher_forced(model: Any, prompt: str, target: str) -> TeacherForcedSequence:
    """Tokenise (prompt + target) and return everything trainers need
    to run a teacher-forced cross-entropy pass.

    Algorithm:
      1. Format target (prepend space if needed).
      2. Tokenise prompt alone and (prompt + target) together, with BOS
         handling driven by `model.cfg.default_prepend_bos`.
      3. Verify the full tokenisation starts with the prompt's tokens.
         If it doesn't, the tokeniser merged the boundary
         non-compositionally and we cannot locate the target by length.
      4. Slice out target ids and return.

    No forward pass; safe to call inside or outside `torch.no_grad`.
    """
    if not prompt:
        raise ScoringError("Empty prompt.")

    formatted = format_target(target, prompt=prompt)
    prepend_bos = _prepend_bos(model)

    prompt_ids = _to_ids_1d(model, prompt, prepend_bos=prepend_bos)
    full_ids_1d = _to_ids_1d(model, prompt + formatted, prepend_bos=prepend_bos)

    L = int(prompt_ids.shape[0])
    T = int(full_ids_1d.shape[0]) - L

    if T <= 0:
        raise ScoringError(
            f"Target {target!r} added zero tokens after the prompt. "
            "The tokeniser may have absorbed it into the prompt's last token."
        )

    if not torch.equal(full_ids_1d[:L], prompt_ids):
        raise ScoringError(
            "Tokeniser merged the prompt/target boundary non-compositionally. "
            "Ensure the prompt ends at a natural token boundary (e.g. after a "
            "complete word or punctuation)."
        )

    target_ids = full_ids_1d[L : L + T]
    full_ids = full_ids_1d.unsqueeze(0)  # [1, L+T]

    return TeacherForcedSequence(
        full_ids=full_ids,
        prompt_len=L,
        target_len=T,
        target_ids=target_ids,
        target_text_used=formatted,
        prepend_bos_used=prepend_bos,
    )


# ── Public: scoring ───────────────────────────────────────────────────────────


@torch.no_grad()
def score_target(model: Any, prompt: str, target: str) -> TargetScore:
    """Score P(target | prompt) with one forward pass.

    Algorithm:
      1. Build a teacher-forced sequence (handles formatting + boundary
         integrity).
      2. One forward pass over the concatenation. Read log-probs for
         each target token from the position that predicts it.

    Note on the @torch.no_grad() decorator: this function intentionally
    does NOT support gradient flow. Callers that need gradients (e.g.
    UnlearningVerifier._check_gradient_unlearning, ROME _optimize_v,
    MEMIT _compute_z_vector) should call `build_teacher_forced`
    directly and run their own forward + cross-entropy.
    """
    seq = build_teacher_forced(model, prompt, target)

    logits = model(seq.full_ids)  # [1, L+T, V]

    # Logits at position p predict token at position p+1. To score target
    # token i (sitting at absolute position L+i), read logits[0, L+i-1, :].
    L, T = seq.prompt_len, seq.target_len
    if logits.dim() == 2:
        if logits.shape[0] >= L - 1 + T:
            pred_logits = logits[L - 1 : L - 1 + T, :]
        else:
            pred_logits = logits[-1:].expand(T, -1)
    else:
        pred_logits = logits[0, L - 1 : L - 1 + T, :]  # [T, V]
    log_probs = F.log_softmax(pred_logits, dim=-1)

    chosen = log_probs.gather(1, seq.target_ids.unsqueeze(1)).squeeze(1)  # [T]
    per_token_lp = chosen.tolist()
    seq_lp = float(chosen.sum().item())

    # exp() with clamps. math.exp on a Python float is faster and avoids
    # the tensor round-trip; underflow returns 0.0 cleanly.
    first_p = min(1.0, max(0.0, math.exp(per_token_lp[0])))
    seq_p = min(1.0, max(0.0, math.exp(seq_lp))) if seq_lp > -700 else 0.0

    return TargetScore(
        first_token_prob=first_p,
        sequence_logprob=seq_lp,
        sequence_prob=seq_p,
        per_token_logprobs=per_token_lp,
        target_token_ids=seq.target_ids.tolist(),
        target_text_used=seq.target_text_used,
    )


# ── Public: subject location ──────────────────────────────────────────────────


def locate_subject_last_token(model: Any, prompt: str, subject: str) -> int:
    """Return the position of the last token of `subject` in `prompt`.

    BOS handling honours `model.cfg.default_prepend_bos` so the returned
    index is consistent with tensors produced by `tokenize_prompt` and
    `build_teacher_forced`. Trainers should NOT pass an explicit
    `prepend_bos` flag — the model config is the single source of truth.

    Strategy:
      Method 1 (preferred, fast tokenisers): use the tokeniser's
        offset_mapping to map character span -> token index. Robust to
        BPE space artefacts and SentencePiece prefix markers because we
        index by character offsets, not token strings.

      Method 2 (fallback, slow tokenisers): tokenise the subject in
        isolation and search for its id sequence in the prompt's ids,
        scanning right-to-left (last occurrence wins).

    Raises:
        SubjectLocationError if neither method finds the subject.
    """
    if not prompt:
        raise SubjectLocationError("Empty prompt.")
    if not subject:
        raise SubjectLocationError("Empty subject.")

    prepend_bos = _prepend_bos(model)
    bos_offset = 1 if prepend_bos else 0

    # Method 1: Offset Mapping (Preferred, handles BPE spacing artifacts)
    try:
        subject_start = prompt.rfind(subject)
        if subject_start != -1:
            subject_end = subject_start + len(subject) - 1
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is not None:
                encoding = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
                offsets = encoding.get("offset_mapping")
                if offsets:
                    for i, (start, end) in enumerate(offsets):
                        if start == end:
                            continue
                        if start <= subject_end < end:
                            return i + bos_offset
    except Exception:
        pass  # Proceed to fallback

    # Method 2: Sequence Matching (Fallback for Slow tokenizers)
    try:
        prompt_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)[0].tolist()
        subject_ids = model.to_tokens(subject, prepend_bos=False)[0].tolist()

        if subject_ids and len(subject_ids) <= len(prompt_ids):
            for i in range(len(prompt_ids) - len(subject_ids), -1, -1):
                if prompt_ids[i : i + len(subject_ids)] == subject_ids:
                    return i + len(subject_ids) - 1
    except Exception:
        pass

    raise SubjectLocationError(f"Could not locate subject {subject!r} within prompt {prompt!r}.")


# ── Public: argmax at end ─────────────────────────────────────────────────────


@torch.no_grad()
def argmax_token_at_end(model: Any, prompt: str) -> Tuple[int, str]:
    """Return (token_id, decoded_string) of the model's top prediction
    at the end of `prompt`. Used to compute the "old target" baseline
    for an Efficacy Score (did the edit move the argmax old → new?).

    Decoding goes through the tokeniser directly when available, because
    HookedTransformer.to_string returns a list for batched input and a
    bare string for unbatched, which is too easy to get wrong at the
    call site.
    """
    if not prompt:
        raise ScoringError("Empty prompt.")

    prepend_bos = _prepend_bos(model)
    prompt_ids = _to_ids_1d(model, prompt, prepend_bos=prepend_bos)
    logits = model(prompt_ids.unsqueeze(0))  # [1, L, V]
    top_id = int(logits[0, -1, :].argmax().item())

    tok = getattr(model, "tokenizer", None)
    if tok is not None and hasattr(tok, "decode"):
        decoded = tok.decode([top_id])
    else:
        # Fall back to to_string and unwrap if it returns a list.
        s = model.to_string(torch.tensor([top_id], device=prompt_ids.device))
        decoded = s[0] if isinstance(s, list) else s

    return top_id, decoded


# ── Public: random-prefix sampling for compute_v / compute_z ──────────────────


def sample_random_prefixes(
    model: Any,
    n_short: int = 5,
    n_med: int = 5,
    len_short: int = 5,
    len_med: int = 10,
    seed: int = 0,
) -> List[str]:
    """Sample short random text prefixes for ROME/MEMIT optimization averaging.

    Both papers (Meng et al. 2022 / 2023) average the optimisation loss for v
    (ROME compute_v.py) and z (MEMIT compute_z.py) across short prefixes
    prepended to the prompt. The canonical recipe in the original repos is
    20 prefixes total — 10 of length 5 and 10 of length 10 — sampled once per
    fact at the start of optimization. ROME Appendix E.5 reports a +3 efficacy
    point ablation for this averaging.

    We source prefix material from the in-tree fallback corpus (re-exported
    from `_covariance._FALLBACK_TEXTS`) to avoid a second corpus surface. Each
    sampled prefix is a token-window decoded back to text, then trimmed of
    leading whitespace. The caller appends `" " + prompt` to form the full
    variant input.

    Args:
        model:     HookedTransformer (used only for its tokenizer).
        n_short:   Number of short-length prefixes to sample.
        n_med:     Number of medium-length prefixes to sample.
        len_short: Token length of the short prefixes (paper: 5).
        len_med:   Token length of the medium prefixes (paper: 10).
        seed:      Local RNG seed; does NOT touch the global random state.

    Returns:
        List of up to `n_short + n_med` decoded prefix strings. May be shorter
        if some windows decode to empty/whitespace-only strings, or empty if
        the corpus is too short to sample from at all (defensive).
    """
    from ._covariance import _FALLBACK_TEXTS

    rng = random.Random(seed)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return []

    # Tokenise the corpus once (no BOS — these are mid-stream snippets).
    pool: List[List[int]] = []
    for text in _FALLBACK_TEXTS:
        try:
            ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
            if len(ids) >= max(len_short, len_med) + 1:
                pool.append(ids)
        except Exception:
            continue

    if not pool:
        return []

    def _sample_one(n_tok: int) -> Optional[str]:
        # Pick a random source long enough, then a random window inside it.
        candidates = [ids for ids in pool if len(ids) >= n_tok]
        if not candidates:
            return None
        ids = rng.choice(candidates)
        start = rng.randint(0, len(ids) - n_tok)
        window = ids[start : start + n_tok]
        try:
            text = tokenizer.decode(window, skip_special_tokens=True)
        except Exception:
            return None
        text = text.strip()
        return text or None

    out: List[str] = []
    for _ in range(n_short):
        s = _sample_one(len_short)
        if s:
            out.append(s)
    for _ in range(n_med):
        s = _sample_one(len_med)
        if s:
            out.append(s)
    return out
