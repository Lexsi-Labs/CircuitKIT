"""DataWorthinessReport — "is your data worthy of circuit discovery?"

Every NormalizedDataset goes through 8 core worthiness checks before discovery,
plus up to 2 alignment-quality checks for TEMPLATE datasets processed with
template_normalize() (alignment_metric_recommendation, discriminative_drop_rate).
The output is structured (green / yellow / red) with explicit fixes
suggested per-check. This is the single most defensible novel feature
versus competing libraries (TransformerLens, pyvene, repeng,
Interpreto): nobody else grades user data before attribution.

The 8 checks (per PLANS/PRDs/PRD_B_DATA.md):

    1. token_alignment           clean+corrupt token-counts match (>=99% pass)
    2. target_token_determinism  the answer is a single token (or fixed span)
    3. baseline_signal           model gets the answer right >=50% of the time
    4. logit_difference_signal   logit_correct - logit_distractor > 0 on >=50%
    5. class_balance             no single class > 90%
    6. pair_uniqueness           <10% duplicate pairs
    7. semantic_difference_contract  corruption strategy honours its length contract
    8. shape_specific            extra checks based on dataset shape
                                 (e.g. forget_retain: both splits non-empty)
                                 
    For TEMPLATE datasets with _alignment meta (from template_normalize):
    9. alignment_metric_recommendation  warn when align_strategy='none'
    10. discriminative_drop_rate         warn when >20% of pairs dropped as non-discriminative

Output: ``DataWorthinessReport`` with per-check ``CheckResult``,
overall verdict (GREEN / YELLOW / RED), and a list of suggested fixes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .normalized import DatasetShape, NormalizedDataset


class Verdict(str, Enum):
    GREEN = "GREEN"  # all checks pass; safe to run discovery
    YELLOW = "YELLOW"  # warnings only; discovery may be noisy
    RED = "RED"  # at least one hard failure; do not run discovery without --force


@dataclass
class CheckResult:
    """Outcome of one worthiness check."""

    name: str
    passed: bool
    severity: str  # "hard" or "soft"
    score: float  # check-specific numeric (e.g. fraction passing)
    message: str  # human-readable single-line summary
    fix: str = ""  # suggested fix if not passed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataWorthinessReport:
    """Structured worthiness report for a NormalizedDataset."""

    dataset_name: str
    dataset_shape: str
    n_records: int
    verdict: Verdict
    checks: List[CheckResult] = field(default_factory=list)
    artifact_safe_for: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.GREEN

    @property
    def hard_fails(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "hard"]

    @property
    def soft_fails(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "soft"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_shape": self.dataset_shape,
            "n_records": self.n_records,
            "verdict": self.verdict.value,
            "artifact_safe_for": self.artifact_safe_for,
            "suggested_fixes": self.suggested_fixes,
            "checks": [c.to_dict() for c in self.checks],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def render_terminal(self) -> str:
        """Pretty-print for the CLI / interactive use."""
        symbol = {Verdict.GREEN: "✓", Verdict.YELLOW: "⚠", Verdict.RED: "✗"}
        lines = [
            "DataWorthinessReport",
            f"  dataset: {self.dataset_name} ({self.dataset_shape})",
            f"  verdict: {self.verdict.value}    " f"records: {self.n_records}",
            "",
        ]
        for c in self.checks:
            mark = (
                symbol[Verdict.GREEN]
                if c.passed
                else (symbol[Verdict.RED] if c.severity == "hard" else symbol[Verdict.YELLOW])
            )
            lines.append(f"  {mark} {c.name}: {c.message}")
            if not c.passed and c.fix:
                lines.append(f"      → {c.fix}")
        if self.artifact_safe_for:
            lines += ["", f"  artifact_safe_for: {self.artifact_safe_for}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def _check_token_alignment(
    ds: NormalizedDataset,
    tokenizer: Optional[Any],
    threshold: float = 0.99,
    align_strategy: Optional[str] = None,
) -> CheckResult:
    """Fraction of paired records where clean and corrupt tokenise to
    the same length (after the prompt prefix / answer-position alignment).
    """
    # If alignment was already enforced upstream (filter or pad_question),
    # the token-length invariant is guaranteed — skip the redundant check.
    if ds.shape == DatasetShape.TEMPLATE and align_strategy in ("filter", "pad_question"):
        return CheckResult(
            name="token_alignment",
            passed=True,
            severity="hard",
            score=1.0,
            message=(
                f"skipped — alignment already enforced by align_strategy='{align_strategy}' "
                f"during template_normalize()"
            ),
        )
    paired = [r for r in ds.records if r.is_paired]
    if not paired:
        return CheckResult(
            name="token_alignment",
            passed=True,
            severity="hard",
            score=1.0,
            message="no paired records to check (will be paired by a strategy later)",
        )
    if tokenizer is None:
        # Without a tokenizer we fall back to a character-length heuristic.
        same = sum(1 for r in paired if abs(len(r.clean_prompt) - len(r.corrupt_prompt)) <= 4)
        frac = same / len(paired)
        msg = (
            f"{int(frac*len(paired))}/{len(paired)} pairs char-aligned "
            f"(no tokenizer; loose check)"
        )
        return CheckResult(
            name="token_alignment",
            passed=frac >= threshold,
            severity="hard",
            score=frac,
            message=msg,
            fix=(
                "Re-run with a tokenizer for an accurate token-length check, "
                "or pick a length-preserving corruption strategy "
                "(entity_swap, token_swap, role_swap)."
            ),
        )
    same = 0
    for r in paired:
        clean_ids = tokenizer.encode(r.clean_prompt)
        corrupt_ids = tokenizer.encode(r.corrupt_prompt)
        if len(clean_ids) == len(corrupt_ids):
            same += 1
    frac = same / len(paired)
    return CheckResult(
        name="token_alignment",
        passed=frac >= threshold,
        severity="hard",
        score=frac,
        message=f"{same}/{len(paired)} pairs aligned ({frac:.1%}, threshold {threshold:.0%})",
        fix=(
            "Use a length-preserving corruption strategy (entity_swap, "
            "token_swap, role_swap) or restructure prompts so the "
            "corrupt-able token is at a fixed position."
        ),
    )


def _check_target_token_determinism(
    ds: NormalizedDataset,
    tokenizer: Optional[Any],
) -> CheckResult:
    """Verify the answer is a single token (or fixed-length span)."""
    if ds.shape == DatasetShape.CLEAN_ONLY:
        has_answers = sum(1 for r in ds.records if (r.clean_answer or "").strip())
        return CheckResult(
            name="target_token_determinism",
            passed=True,
            severity="hard",
            score=1.0,
            message=(
                f"clean_only: {has_answers}/{len(ds.records)} records have answers "
                f"(required for IBCircuit, optional for CD-T)"
            ),
        )
    if tokenizer is None:
        # Fallback: assume single-word answers are likely single-token.
        pass_count = sum(
            1 for r in ds.records if r.clean_answer.strip() and " " not in r.clean_answer.strip()
        )
        frac = pass_count / max(1, len(ds.records))
        return CheckResult(
            name="target_token_determinism",
            passed=frac >= 0.95,
            severity="hard",
            score=frac,
            message=f"{pass_count}/{len(ds.records)} answers are whitespace-clean "
            f"single-token candidates (no tokenizer)",
            fix=(
                "Run with a tokenizer to verify single-token-ness, or "
                "restrict targets to a single token by selecting only the "
                "first sub-token of multi-token answers."
            ),
        )
    multi = []
    for r in ds.records:
        ids = tokenizer.encode(r.clean_answer)
        # Some HF tokenisers add bos/eos; we look for >1 *content* token.
        content = [
            i
            for i in ids
            if not getattr(tokenizer, "all_special_ids", None) or i not in tokenizer.all_special_ids
        ]
        if len(content) > 1:
            multi.append(r.record_id)
    pass_count = len(ds.records) - len(multi)
    frac = pass_count / max(1, len(ds.records))
    return CheckResult(
        name="target_token_determinism",
        passed=frac >= 0.95,
        severity="hard",
        score=frac,
        message=f"{pass_count}/{len(ds.records)} answers are single-token " f"under this tokenizer",
        fix=(
            "Either restructure prompts so answers are single tokens, or "
            "switch metric to KL-divergence which doesn't require single-token answers."
        ),
    )


def _check_class_balance(ds: NormalizedDataset) -> CheckResult:
    """No single answer class > 90% of dataset (else circuit may track class prior)."""
    if not ds.records:
        return CheckResult(
            name="class_balance",
            passed=False,
            severity="hard",
            score=0.0,
            message="empty dataset",
            fix="Provide at least 1 record.",
        )
    if ds.shape == DatasetShape.CLEAN_ONLY:
        return CheckResult(
            name="class_balance",
            passed=True,
            severity="soft",
            score=1.0,
            message="clean_only: class balance not applicable (no contrastive pairs)",
        )
    counts = Counter(r.clean_answer for r in ds.records)
    max_frac = max(counts.values()) / len(ds.records)
    return CheckResult(
        name="class_balance",
        passed=max_frac <= 0.9,
        severity="soft",
        score=1.0 - max_frac,
        message=f"largest class is {max_frac:.0%} of records ({len(counts)} unique answers)",
        fix=(
            "Stratify your data so no single class dominates >90%. Highly "
            "imbalanced data risks the circuit tracking the class prior, "
            "not the task."
        ),
    )


def _check_pair_uniqueness(ds: NormalizedDataset) -> CheckResult:
    """Verify <10% duplicate (clean_prompt, corrupt_prompt) pairs."""
    if not ds.records:
        return CheckResult(
            name="pair_uniqueness",
            passed=False,
            severity="hard",
            score=0.0,
            message="empty dataset",
            fix="Provide records.",
        )
    keys = [(r.clean_prompt, r.corrupt_prompt) for r in ds.records]
    n_unique = len(set(keys))
    dup_frac = 1 - n_unique / len(keys)
    return CheckResult(
        name="pair_uniqueness",
        passed=dup_frac <= 0.10,
        severity="soft",
        score=1.0 - dup_frac,
        message=f"{n_unique}/{len(keys)} unique pairs ({dup_frac:.1%} duplicates)",
        fix=(
            "Deduplicate the dataset; high duplicate rate inflates Pillar-3 " "stability metrics."
        ),
    )


def _check_semantic_difference_contract(
    ds: NormalizedDataset,
    expected_contract: Optional[str] = None,
    tokenizer: Optional[Any] = None,
) -> CheckResult:
    """Verify the corruption strategy honoured its length contract."""
    paired = [r for r in ds.records if r.is_paired]
    if not paired:
        return CheckResult(
            name="semantic_difference_contract",
            passed=True,
            severity="hard",
            score=1.0,
            message="no paired records to check",
        )
    if expected_contract is None:
        # Best-effort inference: if all clean+corrupt have same length, assume PRESERVE
        same_len = all(len(r.clean_prompt) == len(r.corrupt_prompt) for r in paired[:50])
        return CheckResult(
            name="semantic_difference_contract",
            passed=True,
            severity="soft",
            score=1.0,
            message=(
                "contract not declared; "
                + (
                    "clean+corrupt char-lengths match (looks PRESERVE)"
                    if same_len
                    else "clean+corrupt char-lengths differ (looks UNKNOWN)"
                )
            ),
            fix=(
                "Declare a length contract for your corruption strategy "
                "(preserve / extend / shrink) so this check can be tightened."
            ),
        )
    contract = expected_contract.lower()
    violations = 0
    for r in paired:
        if tokenizer is not None:
            cl = len(tokenizer.encode(r.clean_prompt))
            co = len(tokenizer.encode(r.corrupt_prompt))
        else:
            cl = len(r.clean_prompt.split())
            co = len(r.corrupt_prompt.split())
        if contract == "preserve" and cl != co:
            violations += 1
        elif contract == "extend" and co < cl:
            violations += 1
        elif contract == "shrink" and co > cl:
            violations += 1
    pass_count = len(paired) - violations
    frac = pass_count / max(1, len(paired))
    return CheckResult(
        name="semantic_difference_contract",
        passed=frac >= 0.95,
        severity="hard",
        score=frac,
        message=f"{pass_count}/{len(paired)} pairs honour contract '{contract}' " f"({frac:.1%})",
        fix=(
            f"Strategy declared '{contract}' but produced violations. "
            f"Either pick a different strategy or relax the contract to UNKNOWN."
        ),
    )


def _check_baseline_signal(
    ds: NormalizedDataset,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    *,
    rank_threshold: int = 5,
    sample_n: int = 16,
    fraction_passing_threshold: float = 0.5,
) -> CheckResult:
    """Run the model on a sample of clean prompts; require correct answer in top-k."""
    if model is None or tokenizer is None:
        return CheckResult(
            name="baseline_signal",
            passed=True,
            severity="soft",
            score=0.0,
            message="skipped (no model/tokenizer provided)",
            fix=(
                "Pass model+tokenizer to validate that the chosen model "
                "can perform the task at all. Weak baseline signal causes "
                "IBCircuit's IB-gradient to dominate the task-loss gradient."
            ),
        )
        
    if ds.shape == DatasetShape.CLEAN_ONLY and not any(
        (r.clean_answer or "").strip() for r in ds.records[:1]
    ):
        return CheckResult(
            name="baseline_signal",
            passed=True,
            severity="soft",
            score=1.0,
            message="clean_only: baseline signal check skipped (no answers; CD-T mode)",
        )
    
    import torch

    sample = ds.records[:sample_n]
    correct = 0
    for r in sample:
        ids = tokenizer.encode(r.clean_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        last = logits[0, -1]
        # Get the first token of the answer
        ans_ids = tokenizer.encode(r.clean_answer)
        if not ans_ids:
            continue
        ans_id = ans_ids[0]
        rank = (last > last[ans_id]).sum().item() + 1
        if rank <= rank_threshold:
            correct += 1
    frac = correct / max(1, len(sample))
    return CheckResult(
        name="baseline_signal",
        passed=frac >= fraction_passing_threshold,
        severity="soft",
        score=frac,
        message=f"correct answer in top-{rank_threshold} for "
        f"{correct}/{len(sample)} samples ({frac:.0%})",
        fix=(
            "Use a stronger model. Weak baseline signal means the "
            "task-preservation gradient is too small for IBCircuit; "
            "EAP / EAP-IG / CD-T are more robust."
        ),
    )


def _check_logit_difference_signal(
    ds: NormalizedDataset,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    *,
    sample_n: int = 16,
    fraction_passing_threshold: float = 0.5,
) -> CheckResult:
    """Verify logit(correct) > logit(distractor) on >=50% of paired records."""
    if ds.shape == DatasetShape.CLEAN_ONLY:
        return CheckResult(
            name="logit_difference_signal",
            passed=True,
            severity="soft",
            score=1.0,
            message="clean_only: logit difference check not applicable (no contrastive pairs)",
        )
    paired = [r for r in ds.records if r.is_paired][:sample_n]
    if not paired:
        return CheckResult(
            name="logit_difference_signal",
            passed=True,
            severity="soft",
            score=0.0,
            message="no paired records (will be checked after corruption strategy applied)",
        )
    if model is None or tokenizer is None:
        return CheckResult(
            name="logit_difference_signal",
            passed=True,
            severity="soft",
            score=0.0,
            message="skipped (no model/tokenizer provided)",
            fix=(
                "Pass model+tokenizer to verify that the contrastive pair "
                "actually elicits the expected logit gap."
            ),
        )
    import torch

    pos = 0
    for r in paired:
        ids = tokenizer.encode(r.clean_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        last = logits[0, -1]
        a_ids = tokenizer.encode(r.clean_answer)
        b_ids = tokenizer.encode(r.corrupt_answer)
        if not a_ids or not b_ids:
            continue
        if last[a_ids[0]].item() > last[b_ids[0]].item():
            pos += 1
    frac = pos / max(1, len(paired))
    return CheckResult(
        name="logit_difference_signal",
        passed=frac >= fraction_passing_threshold,
        severity="soft",
        score=frac,
        message=f"logit(correct) > logit(distractor) on {pos}/{len(paired)} "
        f"({frac:.0%}); paper-recommended >=50%",
        fix=(
            "Use a stronger model OR pick a more disambiguating distractor. "
            "Negative logit gap means the metric measures the wrong thing."
        ),
    )


def _check_shape_specific(ds: NormalizedDataset) -> CheckResult:
    """Shape-specific sanity: e.g. forget/retain need both splits non-empty."""
    if ds.shape == DatasetShape.FORGET_RETAIN:
        forget = sum(1 for r in ds.records if r.meta.get("split") == "forget")
        retain = sum(1 for r in ds.records if r.meta.get("split") == "retain")
        passed = forget > 0 and retain > 0
        return CheckResult(
            name="shape_specific",
            passed=passed,
            severity="hard",
            score=1.0 if passed else 0.0,
            message=f"forget_retain: forget={forget}, retain={retain}",
            fix="Provide records tagged with both split=forget and split=retain.",
        )
    if ds.shape == DatasetShape.MULTI_HOP:
        with_hops = sum(1 for r in ds.records if r.meta.get("hops"))
        passed = with_hops == len(ds.records) and with_hops > 0
        return CheckResult(
            name="shape_specific",
            passed=passed,
            severity="hard",
            score=with_hops / max(1, len(ds.records)),
            message=f"multi_hop: {with_hops}/{len(ds.records)} records have hop chain",
            fix="Each multi-hop record must carry meta['hops'] = [...] chain.",
        )
    if ds.shape == DatasetShape.TRIPLE:
        with_third = sum(1 for r in ds.records if r.meta.get("unrelated"))
        passed = with_third > 0
        return CheckResult(
            name="shape_specific",
            passed=passed,
            severity="hard",
            score=with_third / max(1, len(ds.records)),
            message=f"triple: {with_third}/{len(ds.records)} records have unrelated fill",
            fix="StereoSet intrasentence requires meta['unrelated'] field.",
        )
        
    if ds.shape == DatasetShape.TEMPLATE:
        import re as _re
        unresolved = 0
        identical_pairs = 0
        empty_answers = 0
        for r in ds.records:
            if _re.search(r"\{[a-zA-Z_]\w*\}", r.clean_prompt or ""):
                unresolved += 1
            if _re.search(r"\{[a-zA-Z_]\w*\}", r.corrupt_prompt or ""):
                unresolved += 1
            if r.clean_prompt and r.corrupt_prompt and r.clean_prompt == r.corrupt_prompt:
                identical_pairs += 1
            if not (r.clean_answer or "").strip() or not (r.corrupt_answer or "").strip():
                empty_answers += 1
        issues = []
        if unresolved:
            issues.append(f"{unresolved} fields have unresolved {{placeholders}}")
        if identical_pairs:
            issues.append(f"{identical_pairs} pairs have identical clean/corrupt prompts")
        if empty_answers:
            issues.append(f"{empty_answers} records have empty answers")
        passed = not issues
        return CheckResult(
            name="shape_specific",
            passed=passed,
            severity="hard",
            score=1.0 if passed else 0.0,
            message=(
                f"template: {'; '.join(issues)}" if issues else "template: all checks passed"
            ),
            fix=(
                "Check that template placeholders match CSV column names, and that "
                "clean_prompt and corrupt_prompt templates differ."
                if not passed else ""
            ),
        )
        
    if ds.shape == DatasetShape.CLEAN_ONLY:
        empty_prompts = sum(1 for r in ds.records if not (r.clean_prompt or "").strip())
        empty_answers = sum(1 for r in ds.records if not (r.clean_answer or "").strip())
        issues = []
        if empty_prompts:
            issues.append(f"{empty_prompts} records have empty clean_prompt")
        # empty_answers is a warning only — CD-T doesn't need answers
        passed = not issues
        msg_parts = ["clean_only: all checks passed"] if passed else [f"clean_only: {'; '.join(issues)}"]
        if empty_answers:
            msg_parts.append(f"({empty_answers} records have no answer — OK for CD-T, needed for IBCircuit)")
        return CheckResult(
            name="shape_specific",
            passed=passed,
            severity="hard",
            score=1.0 if passed else 0.0,
            message="; ".join(msg_parts),
            fix=(
                "Ensure clean_prompt column is populated for all records."
                if not passed else ""
            ),
        )
    
    return CheckResult(
        name="shape_specific",
        passed=True,
        severity="soft",
        score=1.0,
        message=f"no extra checks for shape {ds.shape.value}",
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def evaluate_worthiness(
    ds: NormalizedDataset,
    *,
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
    expected_length_contract: Optional[str] = None,
    token_alignment_threshold: float = 0.99,
) -> DataWorthinessReport:
    """Run all 8 worthiness checks on a NormalizedDataset.

    The expensive checks (baseline_signal, logit_difference_signal) only
    run when ``model`` and ``tokenizer`` are provided; they are skipped
    with a "soft pass" otherwise so the worthiness report stays useful
    in low-resource environments.
    """
    # Extract alignment metadata written by template_normalize() (Step 2).
    # Present for TEMPLATE datasets that went through the alignment pass;
    # absent for all other shapes and for datasets loaded without a tokenizer.
    _alignment = ds.meta.get("_alignment", {})
    align_strategy = _alignment.get("align_strategy")

    checks = [
        _check_token_alignment(ds, tokenizer, token_alignment_threshold, align_strategy),
        _check_target_token_determinism(ds, tokenizer),
        _check_baseline_signal(ds, model, tokenizer),
        _check_logit_difference_signal(ds, model, tokenizer),
        _check_class_balance(ds),
        _check_pair_uniqueness(ds),
        _check_semantic_difference_contract(ds, expected_length_contract, tokenizer),
        _check_shape_specific(ds),
    ]

    # --- alignment-quality synthetic checks (only when _alignment meta present) ---
    if _alignment:
        # 9. KL recommendation for align_strategy="none"
        if align_strategy == "none":
            checks.append(CheckResult(
                name="alignment_metric_recommendation",
                passed=False,
                severity="soft",
                score=0.0,
                message=(
                    "align_strategy='none': token lengths not enforced — "
                    "use metric='kl_divergence' instead of 'logit_diff'"
                ),
                fix=(
                    "Set metric='kl_divergence' in your discovery config, or re-run "
                    "template_normalize() with align_strategy='filter' or 'pad_question' "
                    "to enforce token alignment and unlock logit_diff."
                ),
            ))

        # 10. Non-discriminative drop-rate warning (>20% of total_input)
        total_input = _alignment.get("total_input", 0)
        dropped_nd = _alignment.get("dropped_nondiscriminative", 0)
        if total_input > 0 and dropped_nd / total_input > 0.20:
            pct = dropped_nd / total_input
            checks.append(CheckResult(
                name="discriminative_drop_rate",
                passed=False,
                severity="soft",
                score=max(0.0, 1.0 - pct),
                message=(
                    f"{dropped_nd}/{total_input} records ({pct:.0%}) dropped as "
                    f"non-discriminative — answers tokenise identically for clean/corrupt"
                ),
                fix=(
                    "Review your answer templates: if clean_answer and corrupt_answer "
                    "share a BPE prefix, the first divergent token is used. High drop "
                    "rates usually mean answers are too similar (e.g. 'Yes'/'Yes, ...') "
                    "or the tokenizer merges them. Try more lexically distinct answers."
                ),
            ))

    hard_fail = any(not c.passed and c.severity == "hard" for c in checks)
    soft_fail = any(not c.passed and c.severity == "soft" for c in checks)
    verdict = Verdict.RED if hard_fail else Verdict.YELLOW if soft_fail else Verdict.GREEN

    # Algorithms safe to apply, given the failure profile.
    safe = []
    token_aligned = next((c for c in checks if c.name == "token_alignment"), None)
    baseline_ok = next((c for c in checks if c.name == "baseline_signal"), None)
    if token_aligned and token_aligned.passed:
        safe.extend(["eap", "eap-ig", "acdc"])
    safe.append("cd-t")  # CD-T doesn't need contrastive pairs
    safe.append("eap-ifr")  # IFR doesn't need contrastive pairs either
    if baseline_ok and baseline_ok.passed:
        safe.append("ibcircuit")  # IB needs strong baseline signal

    fixes = [c.fix for c in checks if not c.passed and c.fix]

    return DataWorthinessReport(
        dataset_name=ds.name,
        dataset_shape=ds.shape.value,
        n_records=len(ds.records),
        verdict=verdict,
        checks=checks,
        artifact_safe_for=sorted(set(safe)),
        suggested_fixes=fixes,
    )


__all__ = [
    "Verdict",
    "CheckResult",
    "DataWorthinessReport",
    "evaluate_worthiness",
]
