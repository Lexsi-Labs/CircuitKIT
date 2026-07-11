"""SafetyPromptAdapter — universal harm-prompt adapter.

Recognises the common shape of safety / red-team datasets:
  AdvBench (walledai/AdvBench)        : {prompt, target}
  HarmBench (walledai/HarmBench)      : {prompt, behavior, category}
  Sorry-Bench (sorry-bench/sorry-bench-202410): {prompt, category}
  JailbreakBench (JailbreakBench/JBB-Behaviors): {Goal, Target, Category}
  TruthfulQA (truthful_qa)            : {question, best_answer, ...}
  StereoSet (Anthropic style)         : {prompt, stereotype}

The adapter pairs each harmful prompt with two answer tokens:
  clean_answer  : a refusal token (default " I", optionally " Sorry"/" As")
  corrupt_answer: a compliance token (default " Sure", optionally " Yes"/" Of")

EAP attribution then localises the circuit that *flips* the model from
refusal to compliance (or vice-versa). This is the canonical pairing
for the Refusal Direction Ablation safety case study (Arditi et al.
2024) — different prompts (one harmful, one paraphrased) AND different
target tokens, no need for an LLM-driven counterfactual.

Library users can plug in any HF dataset in two lines:

    from datasets import load_dataset
    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    raw = list(load_dataset("walledai/AdvBench", split="train").take(200))
    ds = SafetyPromptAdapter().adapt(raw, name="my_safety_set")

or via auto_detect with no hints — the adapter's ``fits()`` recognises
the column patterns of the major benchmarks listed above plus any
single-column harmful-prompt CSV.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
from .base import DataAdapter, register_adapter

# Column-name candidates that signal a harmful-prompt dataset.
_PROMPT_COLS = (
    "prompt",
    "Prompt",
    "goal",
    "Goal",
    "instruction",
    "harmful_prompt",
    "behavior",
    "Behavior",
    "question",
)
_TARGET_COLS = ("target", "Target", "harmful_response", "compliance")
_CATEGORY_COLS = ("category", "Category", "behavior_category", "harm_type")

# Default refusal / compliance tokens (single-token under most BPEs).
_DEFAULT_REFUSAL = " I"  # GPT-2 / Llama: "I'm sorry...", "I cannot..."
_DEFAULT_COMPLIANCE = " Sure"  # GPT-2 / Llama: "Sure, here's..."

# Benign control prompts paired with each harmful prompt. Different
# prompts (so EAP activation_difference is non-zero) and different
# expected target tokens. Following Arditi et al. 2024 — control
# prompts are everyday helpful queries the model should comply with.
_BENIGN_CONTROLS = [
    "How do I bake a chocolate cake?",
    "What's the capital of France?",
    "Recommend a good book about gardening.",
    "Translate 'good morning' to Spanish.",
    "What's the boiling point of water?",
    "Suggest three names for a pet rabbit.",
    "Summarise the plot of Romeo and Juliet.",
    "How does photosynthesis work?",
    "Give me a recipe for vegetable soup.",
    "What's the time difference between London and Tokyo?",
]


def _peek_columns(raw: Any) -> List[str]:
    if hasattr(raw, "column_names"):
        return list(raw.column_names)
    if hasattr(raw, "columns"):
        return list(raw.columns)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return list(raw[0].keys())
    return []


def _iter_rows(raw: Any):
    if isinstance(raw, list):
        yield from raw
        return
    if hasattr(raw, "iterrows"):
        for _, row in raw.iterrows():
            yield row.to_dict()
        return
    yield from raw


def _first_present(row: Dict, keys) -> Optional[str]:
    for k in keys:
        if k in row and row[k]:
            return str(row[k]).strip()
    return None


@register_adapter(DatasetShape.REFUSAL)
class SafetyPromptAdapter(DataAdapter):
    """Generic adapter for harm-prompt datasets (AdvBench / HarmBench /
    Sorry-Bench / JailbreakBench / TruthfulQA / custom CSV)."""

    description = (
        "Recognises the common shape of safety / red-team datasets: a "
        "single column of harmful prompts (or paired prompt + harmful "
        "response). Pairs each prompt with a refusal token (default ' I') "
        "as the clean answer and a compliance token (default ' Sure') as "
        "the corrupt answer. Used for refusal-direction circuit discovery."
    )

    @classmethod
    def fits(cls, raw: Any) -> bool:
        cols = _peek_columns(raw)
        if not cols:
            return False
        # Must have at least one prompt-shaped column.
        has_prompt = any(c in cols for c in _PROMPT_COLS)
        if not has_prompt:
            return False
        # Disambiguate from MCQ / pairwise / forget-retain: those have
        # additional structure (choices, sent_more, split, answer-with-work).
        if "choices" in cols or "endings" in cols:
            return False
        if "sent_more" in cols or "sent_less" in cols:
            return False
        if "split" in cols and "answer" in cols:
            return False  # forget/retain
        if "answer" in cols and "context" in cols:
            return False  # generic QA
        return True

    # ------------------------------------------------------------------ #
    # Pairing modes                                                        #
    # ------------------------------------------------------------------ #
    # "answer_contrastive" (default, recommended):                         #
    #   clean   = (harmful_prompt, refusal_token)                          #
    #   corrupt = (harmful_prompt, compliance_token)                       #
    #   Same prompt, different answer token. Isolates the refusal signal   #
    #   without confounding prompt-level differences. Correct for circuit  #
    #   localisation (Conmy et al. 2023, circuit faithfulness style).      #
    #                                                                       #
    # "harmful_vs_benign" (Arditi et al. 2024 style):                      #
    #   clean   = (harmful_prompt, refusal_token)                          #
    #   corrupt = (benign_control,  compliance_token)                      #
    #   Different prompts AND different answer tokens. Captures broader     #
    #   refusal-vs-compliance activation difference.                       #
    #                                                                       #
    # "benign_vs_harmful" (inverted):                                      #
    #   clean   = (benign_control,  compliance_token)                      #
    #   corrupt = (harmful_prompt, refusal_token)                          #
    #   Useful when the attribution direction should be flipped.           #
    # ------------------------------------------------------------------ #
    PAIRING_MODES = frozenset({"answer_contrastive", "harmful_vs_benign", "benign_vs_harmful"})

    def adapt(
        self,
        raw: Any,
        *,
        max_records: Optional[int] = None,
        name: Optional[str] = None,
        source: Optional[str] = None,
        refusal_token: str = _DEFAULT_REFUSAL,
        compliance_token: str = _DEFAULT_COMPLIANCE,
        benign_controls: Optional[List[str]] = None,
        pairing_mode: str = "answer_contrastive",
        **_unused: Any,
    ) -> NormalizedDataset:
        """Adapt a harm-prompt dataset into a NormalizedDataset.

        Args:
            raw: HF dataset, list of dicts, or pandas DataFrame.
            max_records: Cap on output records.
            name: Task name for registration.
            source: Human-readable source label.
            refusal_token: Answer token the model gives when refusing.
            compliance_token: Answer token the model gives when complying.
            benign_controls: List of benign prompts for harmful_vs_benign /
                benign_vs_harmful modes.  Ignored for answer_contrastive.
            pairing_mode: One of "answer_contrastive", "harmful_vs_benign",
                "benign_vs_harmful".  Default: "answer_contrastive".
        """
        if pairing_mode not in self.PAIRING_MODES:
            raise ValueError(
                f"pairing_mode must be one of {self.PAIRING_MODES}, " f"got {pairing_mode!r}"
            )

        controls = benign_controls if benign_controls is not None else _BENIGN_CONTROLS
        if pairing_mode != "answer_contrastive" and not controls:
            raise ValueError(
                f"pairing_mode={pairing_mode!r} requires at least one benign "
                "control prompt (benign_controls or the default list)."
            )

        records: List[ContrastiveRecord] = []
        import random as _r

        rng = _r.Random(42)

        for i, row in enumerate(_iter_rows(raw)):
            if not isinstance(row, dict):
                row = dict(row) if hasattr(row, "items") else {"prompt": str(row)}
            prompt = _first_present(row, _PROMPT_COLS)
            if not prompt:
                continue
            category = _first_present(row, _CATEGORY_COLS) or ""
            target = _first_present(row, _TARGET_COLS) or ""

            if pairing_mode == "answer_contrastive":
                # Same prompt, different answer tokens -- cleanest signal.
                clean_p, clean_a = prompt, refusal_token
                corrupt_p, corrupt_a = prompt, compliance_token
            elif pairing_mode == "harmful_vs_benign":
                control = rng.choice(controls)
                clean_p, clean_a = prompt, refusal_token
                corrupt_p, corrupt_a = control, compliance_token
            else:  # benign_vs_harmful
                control = rng.choice(controls)
                clean_p, clean_a = control, compliance_token
                corrupt_p, corrupt_a = prompt, refusal_token

            records.append(
                ContrastiveRecord(
                    record_id=str(row.get("id", row.get("Behavior", f"{i:06d}"))),
                    clean_prompt=clean_p,
                    clean_answer=clean_a,
                    corrupt_prompt=corrupt_p,
                    corrupt_answer=corrupt_a,
                    target_field="refusal_vs_compliance",
                    contrast_source=ContrastSource.GENERATED,
                    meta={
                        "category": category,
                        "harmful_target": target,
                        "pairing_mode": pairing_mode,
                    },
                )
            )
            if max_records and len(records) >= max_records:
                break

        return NormalizedDataset(
            name=name or "safety_prompt",
            shape=DatasetShape.REFUSAL,
            records=records,
            source=source or "raw",
            meta={
                "n_loaded": len(records),
                "refusal_token": refusal_token,
                "compliance_token": compliance_token,
                "pairing_mode": pairing_mode,
            },
        )
