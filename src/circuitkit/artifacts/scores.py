"""
Unified CircuitScores artifact schema across all CircuitKit backends.

All algorithms (EAP, EAP-IG, ACDC, IBCircuit) emit this format for
node-level circuit discovery. This provides a single contract for:
- JSON serialization/deserialization
- Score access and normalization
- Metadata tracking (algorithm, task, model, timestamp)
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CircuitScores:
    """
    Unified scores artifact across all CircuitKit algorithms.

    Represents node-level importance scores from circuit discovery.
    All backends (EAP, ACDC, IBCircuit) convert their outputs to this
    format for consistent downstream processing.

    Attributes:
        task (str): Task name (e.g., 'ioi', 'mmlu', 'capital_country').
        model (str): Model identifier (e.g., 'gpt2', 'pythia-70m').
        algorithm (str): Discovery algorithm used ('eap', 'eap-ig', 'acdc', 'ibcircuit').
        level (str): Granularity of scores ('node' or 'neuron').

        node_scores (Dict[str, float]): Importance scores keyed by node name.
            - Attention heads: 'A{layer}.{head}' (e.g., 'A0.1')
            - MLPs: 'MLP {layer}' (e.g., 'MLP 3')
            - Values are absolute importance scores (non-negative).

        timestamp (str): ISO 8601 datetime when scores were generated.
        version (str): Schema version for backward compatibility (default '1.0').
        discovery_cfg (Optional[Dict]): Full discovery configuration for reproducibility.
    """

    task: str
    model: str
    algorithm: str
    level: str
    node_scores: Dict[str, float]
    timestamp: str
    version: str = "1.0"
    discovery_cfg: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate schema constraints."""
        from ..utils.exceptions import SUPPORTED_ALGORITHMS

        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {sorted(SUPPORTED_ALGORITHMS)}, "
                f"got {self.algorithm!r}"
            )
        if self.level not in {"node", "neuron"}:
            raise ValueError(f"level must be 'node' or 'neuron', got {self.level!r}")
        # Neuron-level scores use the same node_scores dict with keys of
        # the form "L{layer}.{neuron_idx}" (e.g. "L3.42"). No additional
        # schema validation is required beyond the numeric check below.

        # Validate node_scores format
        for name, score in self.node_scores.items():
            if not isinstance(score, (int, float)):
                raise TypeError(
                    f"All scores must be numeric; got {type(score).__name__} " f"for node {name!r}"
                )
            if score < 0:
                raise ValueError(
                    f"All scores must be non-negative; got {score} " f"for node {name!r}"
                )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitScores":
        """
        Construct CircuitScores from a dictionary.

        Handles schema evolution: if version is absent, assumes '1.0'.
        """
        data_copy = dict(data)
        if "version" not in data_copy:
            data_copy["version"] = "1.0"
        if "discovery_cfg" not in data_copy:
            data_copy["discovery_cfg"] = {}
        return cls(**data_copy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """
        Save CircuitScores to JSON file.

        Args:
            path (Path): Output path (typically .json).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: Path) -> "CircuitScores":
        """
        Load CircuitScores from JSON file.

        Args:
            path (Path): Input path (typically .json).

        Returns:
            CircuitScores instance.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def normalize_scores(self, method: str = "minmax") -> Dict[str, float]:
        """
        Return normalized scores (original scores unchanged).

        Args:
            method (str): Normalization method.
                - 'minmax': Rescale to [0, 1] using min-max normalization.
                - 'zscore': Standardize to mean=0, std=1.

        Returns:
            Dict[str, float]: Normalized scores by node name.
        """
        if not self.node_scores:
            return {}

        scores = list(self.node_scores.values())

        if method == "minmax":
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                # All scores equal; return 1.0
                return {name: 1.0 for name in self.node_scores.keys()}
            return {
                name: (score - min_score) / (max_score - min_score)
                for name, score in self.node_scores.items()
            }

        elif method == "zscore":
            import numpy as np

            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                # All scores equal; return 0.0
                return {name: 0.0 for name in self.node_scores.keys()}
            return {name: (score - mean) / std for name, score in self.node_scores.items()}

        else:
            raise ValueError(f"method must be 'minmax' or 'zscore', got {method!r}")

    def top_k_nodes(self, k: int) -> Dict[str, float]:
        """
        Return the top-k highest-scoring nodes.

        Args:
            k (int): Number of top nodes to return.

        Returns:
            Dict[str, float]: Top-k nodes by importance (descending).
        """
        sorted_nodes = sorted(self.node_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_nodes[:k])

    def bottom_k_nodes(self, k: int) -> Dict[str, float]:
        """
        Return the bottom-k lowest-scoring nodes (candidates for pruning).

        Args:
            k (int): Number of bottom nodes to return.

        Returns:
            Dict[str, float]: Bottom-k nodes by importance (ascending).
        """
        sorted_nodes = sorted(self.node_scores.items(), key=lambda x: x[1], reverse=False)
        return dict(sorted_nodes[:k])

    @staticmethod
    def create_timestamp() -> str:
        """Generate ISO 8601 timestamp."""
        return datetime.utcnow().isoformat() + "Z"
