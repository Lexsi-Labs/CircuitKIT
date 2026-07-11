"""Template utility functions for placeholder-based prompt construction.

Pure utilities — no imports from strategy/adapter classes.
"""

import re
from dataclasses import dataclass
from typing import Set, Dict, List, Optional, Tuple


_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([^{}]+)\}(?!\})")


def parse_placeholders(template_str: str) -> List[str]:
    """Extract unique placeholder names from a template string.

    Ignores escaped braces ({{ / }}). Returns sorted unique list.
    """
    return sorted(set(_PLACEHOLDER_RE.findall(template_str)))


def validate_placeholders_against_columns(
    template_spec: Dict[str, str],
    columns: List[str],
) -> List[str]:
    """Return placeholders referenced in template_spec that are missing from columns.

    An empty return list means all placeholders are satisfied.
    """
    all_placeholders: Set[str] = set()
    for tmpl in template_spec.values():
        all_placeholders.update(parse_placeholders(tmpl))
    col_set = set(columns)
    return sorted(all_placeholders - col_set)


def detect_peer_columns(placeholders: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Separate placeholders into peer (other_*) and direct groups.

    Args:
        placeholders: flat list of all placeholder names.

    Returns:
        (peer_map, direct_list) where peer_map maps
        ``"other_X"`` → ``"X"`` for each peer placeholder.

    Raises:
        ValueError: if an ``other_X`` placeholder has no corresponding ``X``
                    in the direct list.
    """
    direct = [p for p in placeholders if not p.startswith("other_")]
    peers = [p for p in placeholders if p.startswith("other_")]
    direct_set = set(direct)

    peer_map: Dict[str, str] = {}
    for peer in peers:
        base = peer[len("other_"):]
        if base not in direct_set:
            raise ValueError(
                f"Placeholder '{peer}' has no corresponding '{base}' column. "
                f"Direct columns found: {sorted(direct_set)}"
            )
        peer_map[peer] = base

    return peer_map, direct


def resolve_template(template_str: str, values: Dict[str, str]) -> str:
    """Substitute {placeholder} tokens using values dict.

    Raises:
        KeyError: with a descriptive message if a placeholder has no value.
    """
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(
                f"Template placeholder '{{{key}}}' has no value. "
                f"Available keys: {sorted(values.keys())}"
            )
        return values[key]

    return _PLACEHOLDER_RE.sub(_replace, template_str)


def build_pair_from_templates(
    template_spec: Dict[str, str],
    clean_values: Dict[str, str],
    corrupt_values: Dict[str, str],
) -> Tuple[str, str, str, str]:
    """Render all four template strings into concrete text.

    Args:
        template_spec:  dict with keys clean_prompt, corrupt_prompt,
                        clean_answer, corrupt_answer.
        clean_values:   column values used for clean_prompt / clean_answer.
        corrupt_values: column values used for corrupt_prompt / corrupt_answer.

    Returns:
        (clean_prompt, corrupt_prompt, clean_answer, corrupt_answer)
    """
    clean_prompt = resolve_template(template_spec["clean_prompt"], clean_values)
    corrupt_prompt = resolve_template(template_spec["corrupt_prompt"], corrupt_values)
    clean_answer = resolve_template(template_spec["clean_answer"], clean_values)
    corrupt_answer = resolve_template(template_spec["corrupt_answer"], corrupt_values)
    return clean_prompt, corrupt_prompt, clean_answer, corrupt_answer

# ---------------------------------------------------------------------------
# Step 1: Alignment utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignmentResult:
    """Result of a token-length alignment check between a clean/corrupt pair."""
    aligned: bool
    clean_len: int
    corrupt_len: int
    diff: int  # corrupt_len - clean_len


def check_token_alignment(
    clean_str: str,
    corrupt_str: str,
    tokenizer,
) -> AlignmentResult:
    """Check whether clean and corrupt prompts tokenize to the same length.

    Tokenizes both strings without special tokens so the result reflects only
    the text content, independent of BOS/EOS injection.

    Args:
        clean_str:   Clean prompt string.
        corrupt_str: Corrupt prompt string.
        tokenizer:   Any HuggingFace-compatible tokenizer.

    Returns:
        AlignmentResult with aligned=True iff both strings produce the same
        token count.
    """
    clean_ids = tokenizer.encode(clean_str, add_special_tokens=False)
    corrupt_ids = tokenizer.encode(corrupt_str, add_special_tokens=False)
    clean_len = len(clean_ids)
    corrupt_len = len(corrupt_ids)
    return AlignmentResult(
        aligned=clean_len == corrupt_len,
        clean_len=clean_len,
        corrupt_len=corrupt_len,
        diff=corrupt_len - clean_len,
    )


@dataclass(frozen=True)
class AnswerCheckResult:
    """Result of an answer-discrimination check for a contrastive pair."""
    discriminative: bool
    clean_label_id: Optional[int]
    corrupt_label_id: Optional[int]
    shared_prefix_len: int           # number of shared leading answer tokens absorbed
    adjusted_clean_prompt: Optional[str]   # prompt with shared prefix absorbed; None if not discriminative
    adjusted_corrupt_prompt: Optional[str] # prompt with shared prefix absorbed; None if not discriminative


def check_answer_discriminative(
    clean_prompt: str,
    clean_answer: str,
    corrupt_prompt: str,
    corrupt_answer: str,
    tokenizer,
) -> AnswerCheckResult:
    """Determine whether a contrastive pair has a genuinely discriminative answer token.

    Tokenizes each side as ``prompt + answer`` jointly (not the answer
    standalone), then slices off the prompt tokens and walks the answer
    continuations to find the first index where they diverge. Any shared
    leading answer tokens are absorbed back into the prompt strings so the
    discriminative token lands at ``input_length - 1``.

    This mirrors the approach in ``gsm8k._align_answer_pair`` and is safe for
    tokenizers (e.g. Llama-3) that split a leading space into its own token —
    standalone tokenization of ``" 18"`` and ``" 27"`` would produce the same
    first token (the space), making the labels identical and useless.

    Args:
        clean_prompt:   Clean prompt string (no special tokens expected).
        clean_answer:   Clean answer string (may start with a space).
        corrupt_prompt: Corrupt prompt string.
        corrupt_answer: Corrupt answer string.
        tokenizer:      Any HuggingFace-compatible tokenizer.

    Returns:
        AnswerCheckResult. If ``discriminative`` is False, label IDs and
        adjusted prompts are None. If True, ``adjusted_*_prompt`` carries
        the prompt with shared prefix tokens decoded and appended, ready for
        direct use in the EAP CSV.
    """
    def _fail() -> AnswerCheckResult:
        return AnswerCheckResult(
            discriminative=False,
            clean_label_id=None,
            corrupt_label_id=None,
            shared_prefix_len=0,
            adjusted_clean_prompt=None,
            adjusted_corrupt_prompt=None,
        )

    clean_p = tokenizer.encode(clean_prompt, add_special_tokens=False)
    corr_p = tokenizer.encode(corrupt_prompt, add_special_tokens=False)
    clean_full = tokenizer.encode(clean_prompt + clean_answer, add_special_tokens=False)
    corr_full = tokenizer.encode(corrupt_prompt + corrupt_answer, add_special_tokens=False)

    # Require a clean prompt/answer boundary — if the tokenizer merged a token
    # across the boundary, joint tokenization diverges from standalone prompt
    # tokenization and we can't reliably slice off the prompt.
    if clean_full[: len(clean_p)] != clean_p or corr_full[: len(corr_p)] != corr_p:
        return _fail()

    clean_cont = clean_full[len(clean_p):]
    corr_cont = corr_full[len(corr_p):]

    if not clean_cont or not corr_cont:
        return _fail()

    # Walk to first diverging token.
    n = min(len(clean_cont), len(corr_cont))
    d = None
    for i in range(n):
        if clean_cont[i] != corr_cont[i]:
            d = i
            break

    if d is None:
        return _fail()

    # Absorb shared prefix tokens back into the prompts so the discriminative
    # token is at input_length - 1 for the EAP backend.
    adjusted_clean = tokenizer.decode(clean_p + clean_cont[:d])
    adjusted_corrupt = tokenizer.decode(corr_p + corr_cont[:d])

    return AnswerCheckResult(
        discriminative=True,
        clean_label_id=int(clean_cont[d]),
        corrupt_label_id=int(corr_cont[d]),
        shared_prefix_len=d,
        adjusted_clean_prompt=adjusted_clean,
        adjusted_corrupt_prompt=adjusted_corrupt,
    )


def pad_question_region(
    prompt: str,
    target_len: int,
    tokenizer,
    pad_boundary: str,
    neutral: str = " the",
    max_iterations: int = 50,
) -> Tuple[str, bool]:
    """Pad the question region of a prompt to reach a target token length.

    Inserts ``neutral`` tokens immediately before ``pad_boundary`` in an
    iterative loop (MMLU-style), stopping when the tokenized length matches
    ``target_len`` or ``max_iterations`` is reached.

    This is only meaningful for MCQ-like templates where the question/context
    sits before a fixed boundary string (e.g. ``"Answer:"``), and padding
    that region does not corrupt the answer-adjacent structure.

    Args:
        prompt:         The prompt string to pad (typically the corrupt prompt).
        target_len:     Desired token count (typically the clean prompt length).
        tokenizer:      Any HuggingFace-compatible tokenizer.
        pad_boundary:   Substring that marks the end of the paddable region.
                        Must appear exactly once in the prompt.
        neutral:        Token string to insert repeatedly. Defaults to ``" the"``.
        max_iterations: Maximum padding rounds before giving up.

    Returns:
        ``(padded_prompt, exact_match)`` where ``exact_match`` is True iff the
        padded prompt tokenizes to exactly ``target_len``.

    Raises:
        ValueError: if ``pad_boundary`` does not appear in ``prompt``.
    """
    if pad_boundary not in prompt:
        raise ValueError(
            f"pad_boundary {pad_boundary!r} not found in prompt. "
            f"Cannot pad question region."
        )

    current_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if current_len == target_len:
        return prompt, True

    # Only pad when the corrupt prompt is shorter than the clean prompt.
    # If it's already longer, no padding can help — return immediately.
    if current_len > target_len:
        return prompt, False

    boundary_idx = prompt.index(pad_boundary)
    pre = prompt[:boundary_idx]
    post = prompt[boundary_idx:]
    padded = prompt

    last_valid = prompt
    for _ in range(max_iterations):
        pre = pre + neutral
        padded = pre + post
        current_len = len(tokenizer.encode(padded, add_special_tokens=False))
        if current_len == target_len:
            return padded, True
        if current_len > target_len:
            return last_valid, False
        last_valid = padded

    return last_valid, False

__all__ = [
    "parse_placeholders",
    "validate_placeholders_against_columns",
    "detect_peer_columns",
    "resolve_template",
    "build_pair_from_templates",
    # Step 1 — alignment utilities
    "AlignmentResult",
    "check_token_alignment",
    "AnswerCheckResult",
    "check_answer_discriminative",
    "pad_question_region",
]
