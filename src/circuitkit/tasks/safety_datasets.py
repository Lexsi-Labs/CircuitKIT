"""Lazy convenience registrations for popular safety / red-team
datasets. None of these download anything at import time — each
dataset is fetched from HuggingFace on first use of the
corresponding registered task name.

Public API:

    from circuitkit.tasks.safety_datasets import register_all_safety
    register_all_safety()                # registers all six
    # or selectively:
    from circuitkit.tasks.safety_datasets import register_advbench
    register_advbench()

After registration, any of the 11 algorithms can run on the dataset:

    from circuitkit.api import discover_circuit
    discover_circuit({
        'model': {'name': 'gpt2', 'precision': 'float32'},
        'discovery': {'algorithm': 'eap-ig', 'task': 'advbench',
                      'level': 'node', 'data_params': {'num_examples': 64}},
        'pruning': {'target_sparsity': 0.1, 'scope': 'heads'},
        'output_path': './advbench_circuit.pt',
    })

Each registration uses ``SafetyPromptAdapter`` which pairs each
harmful prompt with a benign control. Users can plug in their OWN
safety dataset (CSV / parquet / HF) without modifying the library —
this module just shows the recommended HF identifiers and provides
ready-made entry points for the common ones.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _register(
    hf_path: str,
    hf_subset: Optional[str],
    hf_split: str,
    task_name: str,
    max_records: int = 200,
    pairing_mode: str = "answer_contrastive",
) -> None:
    """Lazy: load HF data, run SafetyPromptAdapter, register as task_name.

    Args:
        pairing_mode: Passed through to SafetyPromptAdapter.adapt().
            "answer_contrastive" (default) -- same prompt, refusal vs
            compliance token.  "harmful_vs_benign" -- different prompts
            (Arditi et al. 2024 style).
    """
    from circuitkit.tasks.registry import is_task_registered, register_task

    if is_task_registered(task_name):
        return
    from datasets import load_dataset

    from circuitkit.data.adapters.safety_prompt import SafetyPromptAdapter
    from circuitkit.data.normalized_task import NormalizedTaskSpec

    args = (hf_path,) if hf_subset is None else (hf_path, hf_subset)
    raw = list(load_dataset(*args, split=hf_split, streaming=True).take(max_records))
    ds = SafetyPromptAdapter().adapt(
        raw,
        name=task_name,
        source=hf_path,
        max_records=max_records,
        pairing_mode=pairing_mode,
    )
    if not ds.fully_paired:
        raise RuntimeError(
            f"SafetyPromptAdapter on {hf_path} produced "
            f"{ds.n_paired}/{len(ds)} paired records; cannot register."
        )
    register_task(NormalizedTaskSpec(ds, name=task_name))
    logger.info(
        f"Registered safety task: {task_name} ({len(ds)} records, "
        f"pairing_mode={pairing_mode!r})"
    )


def register_advbench(max_records: int = 200) -> None:
    """AdvBench (Zou et al. 2023, walledai/AdvBench)."""
    _register("walledai/AdvBench", None, "train", "advbench", max_records)


def register_harmbench(max_records: int = 200) -> None:
    """HarmBench (Mazeika et al. 2024, walledai/HarmBench)."""
    _register("walledai/HarmBench", "standard", "train", "harmbench", max_records)


def register_sorrybench(max_records: int = 200) -> None:
    """Sorry-Bench (sorry-bench/sorry-bench-202410)."""
    _register("sorry-bench/sorry-bench-202410", None, "train", "sorrybench", max_records)


def register_jailbreakbench(max_records: int = 200) -> None:
    """JailbreakBench-Behaviors (JailbreakBench/JBB-Behaviors)."""
    _register("JailbreakBench/JBB-Behaviors", "behaviors", "harmful", "jailbreakbench", max_records)


def register_truthfulqa(max_records: int = 200) -> None:
    """TruthfulQA (truthful_qa/generation)."""
    _register("truthful_qa", "generation", "validation", "truthfulqa", max_records)


def register_stereoset(max_records: int = 200) -> None:
    """StereoSet (Anthropic style stereotype benchmark)."""
    _register("McGill-NLP/stereoset", "intersentence", "validation", "stereoset", max_records)


_ALL = [
    register_advbench,
    register_harmbench,
    register_sorrybench,
    register_jailbreakbench,
    register_truthfulqa,
    register_stereoset,
]


def register_all_safety(max_records: int = 200) -> dict:
    """Register every safety dataset; returns {task_name: ok_bool}.
    Errors are logged but don't propagate so partial failures (e.g.
    HF gating) don't take down the others."""
    results = {}
    for fn in _ALL:
        name = fn.__name__.removeprefix("register_")
        try:
            fn(max_records=max_records)
            results[name] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to register {name}: {exc}")
            results[name] = False
    return results
