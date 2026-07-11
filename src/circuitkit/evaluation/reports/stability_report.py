"""
E2: Stability Report

Structured output for circuit stability evaluation (Pillar 3).

Captures metrics from stability evaluation including:
- Jaccard and Dice overlap coefficients
- Layer-wise overlap breakdown
- Bootstrap stability metrics
- All values are JSON-serializable for reporting and storage.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    """
    Structured report for circuit stability evaluation.

    Attributes:
        mean_jaccard: Mean Jaccard similarity across all circuit pairs.
            Range [0, 1]. Higher = more stable.
        std_jaccard: Standard deviation of Jaccard similarities.
        mean_dice: Mean Dice coefficient across circuit pairs.
        std_dice: Standard deviation of Dice coefficients.
        n_runs: Number of discovery runs used for stability assessment.
        n_stable_nodes: Number of nodes present in all runs (fully stable).
        overlap_per_layer: Dict mapping layer index to mean layer-wise Jaccard.
        jaccard_matrix: Optional full pairwise Jaccard similarity matrix.
        dice_matrix: Optional full pairwise Dice coefficient matrix.
        bootstrap_results: Optional results from data resampling stability.
            Keys: 'mean_score', 'std_score', 'performance_stability', etc.
        metadata: Arbitrary metadata about the stability evaluation.
            Keys typically include:
            - algorithm: Discovery algorithm used
            - model: Model name
            - task: Task name
            - timestamp: Evaluation timestamp
            - config_path: Path to config file
    """

    # Overlap metrics (required)
    mean_jaccard: float
    std_jaccard: float
    mean_dice: float
    std_dice: float
    n_runs: int

    # Node stability
    n_stable_nodes: Optional[int] = None

    # Layer-wise breakdown
    overlap_per_layer: Dict[int, float] = field(default_factory=dict)

    # Optional full matrices (can be large)
    jaccard_matrix: Optional[List[List[float]]] = None
    dice_matrix: Optional[List[List[float]]] = None

    # Bootstrap stability on data resampling
    bootstrap_results: Optional[Dict[str, float]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """
        Save report as JSON file.

        Args:
            path: Path to save JSON file to.

        Raises:
            IOError: If file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling numpy arrays
        report_dict = asdict(self)

        # Convert numpy arrays to lists for JSON serialization
        if self.jaccard_matrix is not None:
            if isinstance(self.jaccard_matrix, np.ndarray):
                report_dict["jaccard_matrix"] = self.jaccard_matrix.tolist()

        if self.dice_matrix is not None:
            if isinstance(self.dice_matrix, np.ndarray):
                report_dict["dice_matrix"] = self.dice_matrix.tolist()

        try:
            with open(path, "w") as f:
                json.dump(report_dict, f, indent=2)
            logger.info(f"Saved stability report to {path}")
        except IOError as e:
            logger.error(f"Failed to save report to {path}: {e}")
            raise

    @classmethod
    def from_json(cls, path: Path) -> "StabilityReport":
        """
        Load report from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            StabilityReport instance.

        Raises:
            FileNotFoundError: If file does not exist.
            json.JSONDecodeError: If JSON is invalid.
            ValueError: If JSON structure is invalid.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Report file not found: {path}")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Failed to parse JSON from {path}: {e}", e.doc, e.pos)

        # Validate required fields
        required_fields = ["mean_jaccard", "std_jaccard", "mean_dice", "std_dice", "n_runs"]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Report JSON missing required field '{field_name}'")

        try:
            report = cls(**data)
            logger.info(f"Loaded stability report from {path}")
            return report
        except TypeError as e:
            raise ValueError(f"Invalid report JSON structure: {e}")

    def __repr__(self) -> str:
        """
        Pretty-print the report.

        Returns:
            Human-readable report string.
        """
        lines = [
            "=" * 60,
            "STABILITY EVALUATION REPORT (Pillar 3)",
            "=" * 60,
        ]

        # Overlap metrics
        lines.extend(
            [
                "",
                "OVERLAP METRICS (across all discovery runs)",
                f"  Mean Jaccard:     {self.mean_jaccard:.4f} ± {self.std_jaccard:.4f}",
                f"  Mean Dice:        {self.mean_dice:.4f} ± {self.std_dice:.4f}",
                f"  Number of Runs:   {self.n_runs}",
            ]
        )

        # Node stability
        if self.n_stable_nodes is not None:
            lines.extend(
                [
                    "",
                    "NODE STABILITY",
                    f"  Nodes in All Runs: {self.n_stable_nodes}",
                ]
            )

        # Layer-wise breakdown
        if self.overlap_per_layer:
            lines.extend(
                [
                    "",
                    "LAYER-WISE OVERLAP",
                ]
            )
            for layer, overlap in sorted(self.overlap_per_layer.items()):
                lines.append(f"  Layer {layer}: {overlap:.4f}")

        # Bootstrap stability
        if self.bootstrap_results:
            lines.extend(
                [
                    "",
                    "BOOTSTRAP STABILITY (data resampling)",
                    *self._format_dict(self.bootstrap_results),
                ]
            )

        # Metadata
        if self.metadata:
            lines.extend(
                [
                    "",
                    "METADATA",
                    *self._format_dict(self.metadata),
                ]
            )

        # Summary interpretation
        lines.extend(
            [
                "",
                "INTERPRETATION",
                f"  {'High' if self.mean_jaccard > 0.7 else 'Low' if self.mean_jaccard < 0.3 else 'Moderate'} stability: "
                f"Circuit {'changes significantly' if self.mean_jaccard < 0.3 else 'is quite consistent' if self.mean_jaccard > 0.7 else 'shows moderate consistency'} across runs.",
            ]
        )

        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def _format_dict(d: Dict[str, Any], indent: int = 2) -> list:
        """Format a dictionary for pretty-printing."""
        lines = []
        indent_str = " " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.extend(StabilityReport._format_dict(value, indent + 2))
            elif isinstance(value, (int, float)):
                lines.append(f"{indent_str}{key}: {value:.4f}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return lines

    def summary(self) -> Dict[str, float]:
        """
        Get a summary of key stability metrics.

        Returns:
            Dict with stability-specific summaries:
            - 'mean_jaccard': Mean Jaccard overlap
            - 'mean_dice': Mean Dice overlap
            - 'stability_score': Overall stability (mean of normalized metrics)
        """
        # Normalize metrics to [0, 1] and average
        stability_score = (self.mean_jaccard + self.mean_dice) / 2

        return {
            "mean_jaccard": self.mean_jaccard,
            "mean_dice": self.mean_dice,
            "stability_score": stability_score,
            "n_runs": self.n_runs,
        }

    @classmethod
    def from_pillar3_output(
        cls, pillar_result: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> "StabilityReport":
        """
        Create a StabilityReport from Pillar3_Stability.run() output.

        Args:
            pillar_result: Dict returned from Pillar3_Stability.run().
            metadata: Optional metadata to include in report.

        Returns:
            StabilityReport instance.
        """
        if metadata is None:
            metadata = {}

        # Convert numpy arrays to lists for JSON serialization
        jaccard_matrix = pillar_result.get("jaccard_matrix")
        dice_matrix = pillar_result.get("dice_matrix")

        if isinstance(jaccard_matrix, np.ndarray):
            jaccard_matrix = jaccard_matrix.tolist()
        if isinstance(dice_matrix, np.ndarray):
            dice_matrix = dice_matrix.tolist()

        return cls(
            mean_jaccard=pillar_result["mean_jaccard"],
            std_jaccard=pillar_result["std_jaccard"],
            mean_dice=pillar_result["mean_dice"],
            std_dice=pillar_result["std_dice"],
            n_runs=pillar_result["n_runs"],
            n_stable_nodes=pillar_result.get("n_stable_nodes"),
            overlap_per_layer=pillar_result.get("overlap_per_layer", {}),
            jaccard_matrix=jaccard_matrix,
            dice_matrix=dice_matrix,
            metadata=metadata,
        )
