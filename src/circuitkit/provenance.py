"""``lexsi_provenance.json``: the provenance record shared by the Lexsi libraries.

Schema ``lexsi.provenance/1``. CircuitKIT writes it into every checkpoint directory it
exports and embeds it under ``"provenance"`` in its ``*_scores.json`` files. When the
model came from a local folder that carries its own ``lexsi_provenance.json`` (a
SafeTune or AlignTune output), that record is nested under ``inputs[].provenance``
so lineage chains across libraries. ``base_model`` names the immediate parent (the
model that was loaded), not the original model; see :func:`make_provenance`. Readers
tolerate a missing or unreadable file.
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
        return json.loads((Path(path) / FILENAME).read_text(encoding="utf-8"))
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


RELATIONS = ("finetune", "adapter", "quantized", "merge")


def make_provenance(
    method: str,
    *,
    inputs: Optional[list] = None,
    base_model: Optional[str] = None,
    base_model_relation: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A ``lexsi.provenance/1`` record.

    ``base_model`` is the immediate parent: the model this step loaded. It is the
    ``ref`` of a ``kind: "model"`` entry in ``inputs``, so the parent's own record
    (``inputs[].provenance``) is embedded and older ancestors are found by following
    it. It is ``None`` when ``inputs`` has no model entry with a ``ref``.

    Args:
        method: What produced the output, for example ``"export_checkpoint.pruning"``.
        inputs: ``inputs[]`` entries (``kind``, ``ref``, ``provenance``).
        base_model: Which model input is the parent. Defaults to the first model
            entry in ``inputs``; only needed when there are several (a merge).
        base_model_relation: How the output relates to ``base_model``: one of
            ``RELATIONS``. Leave it ``None`` when the step did not produce a model
            or the relation is not known; the key is then not written.
        params: Free-form settings of the step.

    Raises:
        ValueError: ``base_model`` is not the ``ref`` of a model entry in ``inputs``,
            or ``base_model_relation`` is not one of ``RELATIONS``.
    """
    inputs = list(inputs or [])
    refs = [e.get("ref") for e in inputs if e.get("kind") == "model" and e.get("ref")]
    if base_model is None:
        base_model = refs[0] if refs else None
    elif base_model not in refs:
        raise ValueError(
            f"base_model {base_model!r} must be the ref of a kind='model' entry in inputs "
            f"(found {refs}): it names the model this step loaded."
        )
    if base_model_relation is not None and base_model_relation not in RELATIONS:
        raise ValueError(
            f"base_model_relation must be one of {RELATIONS} or None, got {base_model_relation!r}"
        )
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
        **({"base_model_relation": base_model_relation} if base_model_relation else {}),
        "method": method,
        "inputs": inputs,
        "params": dict(params or {}),
    }


def write_provenance(output_dir: Any, method: str, **kw: Any) -> Dict[str, Any]:
    """Write ``make_provenance(method, **kw)`` to ``<output_dir>/lexsi_provenance.json``."""
    record = make_provenance(method, **kw)
    (Path(output_dir) / FILENAME).write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record
