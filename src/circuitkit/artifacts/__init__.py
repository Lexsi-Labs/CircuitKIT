"""CircuitKit artifacts module for unified score formats and circuit representation."""

from .circuit_artifact import CircuitArtifact, Edge, Node, NodeType
from .converters import (
    acdc_to_artifact,
    eap_to_artifact,
    ibcircuit_to_artifact,
    normalize_importance_scores,
)
from .scores import CircuitScores

__all__ = [
    # Scores
    "CircuitScores",
    # Artifact schema
    "CircuitArtifact",
    "Node",
    "Edge",
    "NodeType",
    # Converters
    "acdc_to_artifact",
    "eap_to_artifact",
    "ibcircuit_to_artifact",
    "normalize_importance_scores",
]
