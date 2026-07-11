"""Auto-detect a dataset's shape and pick the right Adapter + default
CorruptionStrategy.

Usage:

    from circuitkit.data.auto_detect import auto_normalize
    ds = auto_normalize(raw_hf_or_csv_or_records,
                        max_records=128,
                        name="my-task")
    # ds is a NormalizedDataset; the right Adapter ran automatically.

The detector walks through registered Adapters in priority order (most
specific first) and uses the first one whose ``fits()`` returns True.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

from .adapters import ADAPTER_REGISTRY, get_adapter
from .normalized import DatasetShape, NormalizedDataset

# Force-load all built-in adapters and strategies so they self-register.
_ADAPTER_MODULES = (
    "circuitkit.data.adapters.pairwise",
    "circuitkit.data.adapters.instruction",
    "circuitkit.data.adapters.conversational",
    "circuitkit.data.adapters.mcq",
    "circuitkit.data.adapters.forget_retain",
    "circuitkit.data.adapters.math",
    "circuitkit.data.adapters.code",
    "circuitkit.data.adapters.safety_prompt",
)
_STRATEGY_MODULES = (
    "circuitkit.data.corruption.entity_swap",
    "circuitkit.data.corruption.token_swap",
    "circuitkit.data.corruption.mcq_choice_swap",
    "circuitkit.data.corruption.final_answer_swap",
    "circuitkit.data.corruption.logical_negation",
    "circuitkit.data.corruption.resample",
    "circuitkit.data.corruption.llm_counterfactual",
    "circuitkit.data.corruption.benign_rewrite",
    "circuitkit.data.corruption.code_syntax_corrupt",
    "circuitkit.data.corruption.math_step_corrupt",
    "circuitkit.data.corruption.operand_swap",
    "circuitkit.data.corruption.instruction_swap",
    "circuitkit.data.corruption.profession_swap",
    "circuitkit.data.corruption.template",
)
for _m in _ADAPTER_MODULES + _STRATEGY_MODULES:
    try:
        importlib.import_module(_m)
    except Exception:
        # Tolerate missing optional deps; CLI will report clearly later.
        pass


# Detection priority: more-specific shapes win. CrowS-Pairs (pairwise)
# is detected before generic QA. ForgetRetain is detected before MCQ
# (since some unlearning sets also have ans columns).
_DETECTION_ORDER: Tuple[DatasetShape, ...] = (
    DatasetShape.PAIRWISE,
    DatasetShape.FORGET_RETAIN,
    DatasetShape.CONVERSATIONAL,
    DatasetShape.INSTRUCTION,
    DatasetShape.MCQ,
    DatasetShape.MATH,
    DatasetShape.CODE,
    # REFUSAL shape: AdvBench / HarmBench / Sorry-Bench / JailbreakBench
    # detected by SafetyPromptAdapter.fits(). Tried last because it
    # matches a single-column-prompt fallback that other adapters might
    # also accept; this priority gives more specific shapes first crack.
    DatasetShape.REFUSAL,
)


# Default corruption strategy per shape, chosen from the literature scan:
# - PAIRWISE: NATIVE_PAIR; no strategy needed.
# - INSTRUCTION/CONVERSATIONAL: resample (paired with peer record).
# - MCQ: mcq_choice_swap (length-preserving STR per Zhang & Nanda 2023).
# - FORGET_RETAIN: resample (peer from the *opposite* split is paired by
#   a future ``forget_swap`` strategy; for now use generic resample).
DEFAULT_STRATEGY_BY_SHAPE: Dict[DatasetShape, Optional[str]] = {
    DatasetShape.PAIRWISE: None,  # already paired natively
    DatasetShape.FORGET_RETAIN: "resample",
    DatasetShape.CONVERSATIONAL: "resample",
    DatasetShape.INSTRUCTION: "resample",
    DatasetShape.MCQ: "mcq_choice_swap",
    DatasetShape.QA: "entity_swap",
    DatasetShape.MATH: "math_step_corrupt",
    DatasetShape.CODE: "code_syntax_corrupt",
    DatasetShape.CLASSIFICATION: "token_swap",
    DatasetShape.REFUSAL: "benign_rewrite",
}


def detect_shape(raw: Any) -> DatasetShape:
    """Walk registered adapters in priority order; return the first match."""
    for shape in _DETECTION_ORDER:
        cls = ADAPTER_REGISTRY.get(shape)
        if cls is None:
            continue
        try:
            if cls.fits(raw):
                return shape
        except Exception:
            continue
    return DatasetShape.UNKNOWN


def auto_normalize(
    raw: Any,
    *,
    max_records: Optional[int] = None,
    name: Optional[str] = None,
    source: Optional[str] = None,
    force_shape: Optional[DatasetShape] = None,
    apply_default_strategy: bool = False,
    rng=None,
    **adapter_kwargs: Any,
) -> NormalizedDataset:
    """Pick the right adapter automatically and produce a NormalizedDataset.

    Args:
        raw:           a HF Dataset, list of dicts, pandas DataFrame, CSV
                       path, or a {split: raw} dict (forget/retain).
        max_records:   cap on number of records returned.
        name / source: passed through to the adapter.
        force_shape:   bypass auto-detection and use this shape's adapter.
        **adapter_kwargs: forwarded to the adapter's ``adapt()`` method.

    Raises:
        ValueError: if no adapter fits the input.
    """
    shape = force_shape or detect_shape(raw)
    if shape == DatasetShape.UNKNOWN:
        raise ValueError(
            "Could not auto-detect dataset shape. Inspect column names; "
            "you may need to pass ``force_shape=`` explicitly. "
            f"Registered shapes: {sorted(s.value for s in ADAPTER_REGISTRY)}"
        )
    adapter = get_adapter(shape)()
    ds = adapter.adapt(
        raw,
        max_records=max_records,
        name=name,
        source=source,
        **adapter_kwargs,
    )
    if apply_default_strategy and not ds.fully_paired:
        strategy_name = default_strategy_for(ds.shape)
        if strategy_name:
            from .corruption.base import get_strategy
            strat = get_strategy(strategy_name)()
            ds = strat.apply_to_dataset(ds, rng=rng)
    return ds


def default_strategy_for(shape: DatasetShape) -> Optional[str]:
    """Return the default corruption strategy name for a given shape, if any."""
    return DEFAULT_STRATEGY_BY_SHAPE.get(shape)


def list_supported_shapes() -> List[Dict[str, Any]]:
    """For CLI: registered shapes + default strategies + adapter descriptions."""
    out = []
    for shape in DatasetShape:
        if shape in ADAPTER_REGISTRY:
            cls = ADAPTER_REGISTRY[shape]
            out.append(
                {
                    "shape": shape.value,
                    "adapter": cls.__name__,
                    "description": cls.description,
                    "default_strategy": default_strategy_for(shape),
                }
            )
    return out


__all__ = [
    "detect_shape",
    "auto_normalize",
    "default_strategy_for",
    "list_supported_shapes",
]
