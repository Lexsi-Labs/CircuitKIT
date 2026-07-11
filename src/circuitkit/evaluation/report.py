"""
FaithfulnessReport: Structured output from faithfulness evaluation.

Provides a comprehensive dataclass for capturing evaluation results across
all six pillars of the faithfulness framework:

1. Causal Patching: Does the circuit explain model behavior via intervention?
2. Ablation: Does the circuit support learned behavior when ablated?
3. Stability: Is the circuit's explanation robust to dataset variation?
4. Robustness: Does the explanation hold under input corruptions?
5. Baseline Comparison: How does circuit performance compare to baseline?
6. Generalization: Does the circuit transfer to related tasks/datasets?
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class _ReportEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy arrays and scalars to Python natives."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


@dataclass
class FaithfulnessReport:
    """
    Comprehensive faithfulness evaluation report for a circuit.

    Attributes:
        patching_score: Pillar 1 - Causal Patching validity.
            Normalized faithfulness ratio F = (y_circuit - y_corrupt) /
            (y_clean - y_corrupt), clamped at 1.0. 1.0 = circuit fully recovers
            the clean (full-model) behavior; 0.0 = circuit no better than the
            corrupt baseline (it may be negative if the patched circuit
            underperforms the corrupt baseline). NOT the raw logit-diff metric.
            Measures how well the circuit explains model behavior via intervention.

        ablation_score: Pillar 2 - Faithfulness under ablation.
            Normalized faithfulness ratio F = (y_circuit - y_corrupt) /
            (y_clean - y_corrupt), clamped at 1.0, computed on the ablated
            circuit. 1.0 = ablated circuit fully recovers clean behavior;
            0.0 = no better than the corrupt baseline. NOT the raw metric.

        stability: Pillar 3 - Stability and robustness to distribution shift.
            Optional dict with keys like:
            - jaccard_overlap: Jaccard similarity across dataset samples
            - consistency_score: How consistent is the circuit across runs
            - mean_delta_jitter: Variation in effect sizes

        robustness: Pillar 4 - Robustness to input corruptions.
            Optional dict with keys like:
            - delta_vs_paraphrase: Effect size under paraphrasing
            - delta_vs_entity_swap: Effect size under entity swapping
            - consistency_under_corruption: Avg consistency score

        baseline_comparison: Pillar 5 - Comparison to baselines.
            Optional dict with keys like:
            - circuit_vs_full_model: Performance gap
            - circuit_vs_random: Circuit vs random circuit
            - improvement_over_baseline: Improvement metric

        generalization: Pillar 6 - Transfer and generalization.
            Optional dict with keys like:
            - transfer_to_related_task: How well does circuit transfer
            - in_distribution_accuracy: On original distribution
            - out_of_distribution_accuracy: On shifted distribution

        intervention_reliability: Pillar 7 - Intervention reliability.
            Optional dict with keys:
            - r1_seed_consistency: Mean Spearman rho across seed pairs [0,1]
            - r2_effect_magnitude: Mean (circuit-baseline)/baseline
            - r3_effect_variance: 1 - CV of deltas across seeds [0,1]
            - reliability_index: Harmonic mean of the three sub-scores [0,1]
            - n_seeds: Number of seeds successfully evaluated
            - per_seed: List of per-seed dicts

        metadata: Arbitrary metadata about the evaluation.
            Keys typically include:
            - algorithm: Discovery algorithm used (eap, acdc, etc.)
            - model: Model name
            - task: Task name
            - level: Discovery level (node/neuron)
            - sparsity: Target sparsity
            - timestamp: Evaluation timestamp
            - config_path: Path to config file
    """

    # Pillar scores
    patching_score: Optional[float] = None
    ablation_score: Optional[float] = None

    # Optional pillar scores
    stability: Optional[Dict[str, Any]] = None
    robustness: Optional[Dict[str, Any]] = None
    baseline_comparison: Optional[Dict[str, Any]] = None
    generalization: Optional[Dict[str, Any]] = None
    intervention_reliability: Optional[Dict[str, Any]] = None

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

        # Convert to dict (dataclass asdict handles nested structures)
        report_dict = asdict(self)

        try:
            with open(path, "w") as f:
                json.dump(report_dict, f, indent=2, cls=_ReportEncoder)
            logger.info(f"Saved faithfulness report to {path}")
        except IOError as e:
            logger.error(f"Failed to save report to {path}: {e}")
            raise

    @classmethod
    def from_json(cls, path: Path) -> "FaithfulnessReport":
        """
        Load report from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            FaithfulnessReport instance.

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
        required_fields = ["patching_score", "ablation_score"]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Report JSON missing required field '{field_name}'")

        try:
            report = cls(**data)
            logger.info(f"Loaded faithfulness report from {path}")
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
            "FAITHFULNESS EVALUATION REPORT",
            "=" * 60,
        ]

        p_score = f"{self.patching_score:.4f}" if self.patching_score is not None else "N/A"
        a_score = f"{self.ablation_score:.4f}" if self.ablation_score is not None else "N/A"

        # Pillar 1 & 2
        lines.extend(
            [
                "",
                "PILLAR 1: Causal Patching Validity",
                f"  Score: {p_score}",
                "",
                "PILLAR 2: Ablation Faithfulness",
                f"  Score: {a_score}",
            ]
        )

        # Pillar 3: Stability
        if self.stability:
            lines.extend(
                [
                    "",
                    "PILLAR 3: Stability & Robustness to Distribution Shift",
                    *self._format_dict(self.stability),
                ]
            )

        # Pillar 4: Robustness
        if self.robustness:
            lines.extend(
                [
                    "",
                    "PILLAR 4: Robustness to Input Corruptions",
                    *self._format_dict(self.robustness),
                ]
            )

        # Pillar 5: Baseline Comparison
        if self.baseline_comparison:
            lines.extend(
                [
                    "",
                    "PILLAR 5: Baseline Comparison",
                    *self._format_dict(self.baseline_comparison),
                ]
            )

        # Pillar 6: Generalization
        if self.generalization:
            lines.extend(
                [
                    "",
                    "PILLAR 6: Generalization & Transfer",
                    *self._format_dict(self.generalization),
                ]
            )

        # Pillar 7: Intervention Reliability
        if self.intervention_reliability:
            lines.extend(
                [
                    "",
                    "PILLAR 7: Intervention Reliability",
                    *self._format_dict(
                        {k: v for k, v in self.intervention_reliability.items() if k != "per_seed"}
                    ),
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

        # Summary statistics
        lines.extend(
            [
                "",
                "SUMMARY",
            ]
        )
        if self.patching_score is not None and self.ablation_score is not None:
            lines.append(
                f"  Average Pillar 1-2 Score: {(self.patching_score + self.ablation_score) / 2:.4f}"
            )
        else:
            lines.append("  Average Pillar 1-2 Score: N/A")

        lines.append("=" * 60)
        return "\n".join(lines)

    def summary(self) -> Dict[str, Optional[float]]:
        avg = None
        if self.patching_score is not None and self.ablation_score is not None:
            avg = (self.patching_score + self.ablation_score) / 2

        return {
            "pillar_1_patching": self.patching_score,
            "pillar_2_ablation": self.ablation_score,
            "overall_average": avg,
        }

    @staticmethod
    def _format_dict(d: Dict[str, Any], indent: int = 2) -> list:
        """Format a dictionary for pretty-printing."""
        import numpy as np

        lines = []
        indent_str = " " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.extend(FaithfulnessReport._format_dict(value, indent + 2))
            elif isinstance(value, np.ndarray):
                lines.append(f"{indent_str}{key}: ndarray{list(value.shape)}")
            elif isinstance(value, list):
                lines.append(f"{indent_str}{key}: [{len(value)} items]")
            elif isinstance(value, (int, float, np.integer, np.floating)):
                lines.append(f"{indent_str}{key}: {float(value):.4f}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return lines
