"""Shared helpers for the data-layer validation suite.

Every script under validation/data/ exercises one Adapter or Strategy
on a real HuggingFace (or filesystem) dataset, runs the full worthiness
report, and saves the report + a 4-record preview JSON.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Optional

# Make the library importable when running scripts standalone
_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Re-use the visualizations runner's results / make_results_dir helpers.
sys.path.insert(0, str(_REPO / "validation"))
from _common import make_results_dir, write_status  # noqa: E402


def fetch_hf(name: str, *args: Any, split: str = "test",
             take: int = 50, **kwargs: Any) -> list:
    """Load `take` records from a HF dataset in streaming mode.

    Tries the requested split first, falls back to common alternates.
    Returns a list of dicts.
    """
    from datasets import load_dataset
    cands = [split, "train", "validation", "test"]
    for s in cands:
        try:
            return list(load_dataset(name, *args, split=s, streaming=True,
                                     **kwargs).take(take))
        except Exception:
            continue
    raise RuntimeError(f"Could not load {name} on any common split.")
