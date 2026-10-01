"""``lexsi_provenance.json``: the provenance record shared by the Lexsi libraries.

Schema ``lexsi.provenance/1``. CircuitKIT writes it into every checkpoint directory it
exports and embeds it under ``"provenance"`` in its ``*_scores.json`` files. When the
model came from a local folder that carries its own ``lexsi_provenance.json`` (a
SafeTune or AlignTune output), that record is nested under ``inputs[].provenance``
so lineage chains across libraries. Readers tolerate a missing or unreadable file.
"""

from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

FILENAME = "lexsi_provenance.json"
SCHEMA = "lexsi.provenance/1"


def read_provenance(path: Any) -> Optional[Dict[str, Any]]:
    """The record in ``<path>/lexsi_provenance.json``, or ``None`` (Hub ids, no file, bad JSON)."""
    try:
        return json.loads((Path(path) / FILENAME).read_text())
    except (OSError, TypeError, ValueError):
        return None


def model_input(model: Any) -> Dict[str, Any]:
    """The ``inputs[]`` entry for a loaded model (set by ``load_model``, else from its config)."""
    if getattr(model, "lexsi_input", None):
        return model.lexsi_input
    ref = getattr(getattr(model, "config", None), "_name_or_path", None) or getattr(
        getattr(model, "cfg", None), "tokenizer_name", None
    )
    return {"kind": "model", "ref": ref, "provenance": read_provenance(ref) if ref else None}


def make_provenance(
    method: str,
    *,
    inputs: Optional[list] = None,
    base_model: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A ``lexsi.provenance/1`` record. ``base_model`` defaults to the first input's lineage."""
    inputs = list(inputs or [])
    if base_model is None and inputs:
        first = inputs[0]
        base_model = (first.get("provenance") or {}).get("base_model") or first.get("ref")
    try:
        version = importlib.metadata.version("circuitkit")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "schema": SCHEMA,
        "library": "circuitkit",
        "version": version,
        "git_sha": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base_model": base_model,
        "method": method,
        "inputs": inputs,
        "params": dict(params or {}),
    }


def write_provenance(output_dir: Any, method: str, **kw: Any) -> Dict[str, Any]:
    """Write ``make_provenance(method, **kw)`` to ``<output_dir>/lexsi_provenance.json``."""
    record = make_provenance(method, **kw)
    (Path(output_dir) / FILENAME).write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record
