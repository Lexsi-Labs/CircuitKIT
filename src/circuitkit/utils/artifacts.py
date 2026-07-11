"""
Artifact utilities for CircuitKit.

Provides helpers to write and read sidecar metadata JSON next to .pt artifacts.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict


def write_sidecar_metadata(artifact_path: str, metadata: Dict[str, Any]) -> str:
    """
    Write a sidecar JSON file next to the artifact path.

    The JSON file will have the same base name as the artifact with a .json extension.
    Returns the path to the JSON file written.
    """
    base, _ = os.path.splitext(artifact_path)
    json_path = base + ".json"
    enriched = dict(metadata)
    if "created_at" not in enriched:
        enriched["created_at"] = datetime.utcnow().isoformat() + "Z"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)
    return json_path


def read_sidecar_metadata(artifact_path: str) -> Dict[str, Any]:
    """
    Read sidecar JSON metadata if present; return empty dict if missing.
    """
    base, _ = os.path.splitext(artifact_path)
    json_path = base + ".json"
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
