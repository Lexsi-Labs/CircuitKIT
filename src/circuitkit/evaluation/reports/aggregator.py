"""
E4: Report Aggregator

Combines individual stability and robustness reports into a unified report
structure. Handles JSON serialization and provides summary statistics.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

logger = logging.getLogger(__name__)

# Type hints without circular imports
if TYPE_CHECKING:
    from ..report import FaithfulnessReport
    from .robustness_report import RobustnessReport
    from .stability_report import StabilityReport


@dataclass
class StabilityRobustnessReport:
    """
    Combined report aggregating stability and robustness evaluations.

    This report brings together Pillar 3 (Stability) and Pillar 4 (Robustness)
    evaluations with their supporting metadata and interpretation.

    Attributes:
        stability_report: StabilityReport from Pillar 3 evaluation.
        robustness_report: RobustnessReport from Pillar 4 evaluation.
        timestamp: ISO timestamp when report was generated.
        metadata: Shared metadata for both evaluations.
            Keys typically include:
            - model: Model name
            - task: Task name
            - algorithm: Discovery algorithm
            - discovered_nodes: Number of nodes in discovered circuit
            - circuit_sparsity: Sparsity of discovered circuit
    """

    stability_report: Optional["StabilityReport"] = None
    robustness_report: Optional["RobustnessReport"] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """
        Save aggregated report as JSON file.

        Args:
            path: Path to save JSON file to.

        Raises:
            IOError: If file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build report dict with nested structure
        report_dict = {
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "stability_report": asdict(self.stability_report) if self.stability_report else None,
            "robustness_report": asdict(self.robustness_report) if self.robustness_report else None,
        }

        try:
            with open(path, "w") as f:
                json.dump(report_dict, f, indent=2)
            logger.info(f"Saved combined stability-robustness report to {path}")
        except IOError as e:
            logger.error(f"Failed to save report to {path}: {e}")
            raise

    @classmethod
    def from_json(cls, path: Path) -> "StabilityRobustnessReport":
        """
        Load aggregated report from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            StabilityRobustnessReport instance.

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

        try:
            # Lazy import to avoid circular dependencies
            try:
                from .robustness_report import RobustnessReport
                from .stability_report import StabilityReport
            except ImportError:
                # Fallback for when module is not in a package context
                import sys

                mod_path = Path(__file__).parent
                sys.path.insert(0, str(mod_path))
                from robustness_report import RobustnessReport
                from stability_report import StabilityReport

                sys.path.pop(0)

            # Reconstruct nested report objects
            stability_data = data.get("stability_report")
            robustness_data = data.get("robustness_report")

            stability_report = None
            robustness_report = None

            if stability_data is not None:
                stability_report = StabilityReport(**stability_data)

            if robustness_data is not None:
                robustness_report = RobustnessReport(**robustness_data)

            report = cls(
                stability_report=stability_report,
                robustness_report=robustness_report,
                timestamp=data.get("timestamp"),
                metadata=data.get("metadata", {}),
            )
            logger.info(f"Loaded combined stability-robustness report from {path}")
            return report
        except TypeError as e:
            raise ValueError(f"Invalid report JSON structure: {e}")

    def __repr__(self) -> str:
        """
        Pretty-print the aggregated report.

        Returns:
            Human-readable report string.
        """
        lines = [
            "=" * 80,
            "COMBINED STABILITY & ROBUSTNESS REPORT",
            "=" * 80,
        ]

        if self.timestamp:
            lines.extend(
                [
                    "",
                    f"Timestamp: {self.timestamp}",
                ]
            )

        if self.metadata:
            lines.extend(
                [
                    "",
                    "EVALUATION METADATA",
                    *self._format_dict(self.metadata),
                ]
            )

        # Stability report section
        if self.stability_report:
            lines.extend(
                [
                    "",
                    "PILLAR 3: STABILITY EVALUATION",
                    "-" * 80,
                    str(self.stability_report),
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "PILLAR 3: STABILITY EVALUATION",
                    "  [Not evaluated]",
                ]
            )

        # Robustness report section
        if self.robustness_report:
            lines.extend(
                [
                    "",
                    "PILLAR 4: ROBUSTNESS EVALUATION",
                    "-" * 80,
                    str(self.robustness_report),
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "PILLAR 4: ROBUSTNESS EVALUATION",
                    "  [Not evaluated]",
                ]
            )

        # Overall summary
        lines.extend(
            [
                "",
                "OVERALL SUMMARY",
            ]
        )

        if self.stability_report and self.robustness_report:
            stability_summary = self.stability_report.summary()
            robustness_summary = self.robustness_report.summary()

            lines.extend(
                [
                    f"  Stability Score (avg overlap):  {stability_summary['stability_score']:.4f}",
                    f"  Robustness Ratio:               {robustness_summary['robustness_ratio']:.4f}",
                    "",
                    "  Combined Assessment:",
                    self._get_combined_assessment(stability_summary, robustness_summary),
                ]
            )
        elif self.stability_report:
            lines.append("  [Stability evaluated, robustness not evaluated]")
        elif self.robustness_report:
            lines.append("  [Robustness evaluated, stability not evaluated]")
        else:
            lines.append("  [No evaluations performed]")

        lines.append("=" * 80)

        return "\n".join(lines)

    @staticmethod
    def _format_dict(d: Dict[str, Any], indent: int = 2) -> list:
        """Format a dictionary for pretty-printing."""
        lines = []
        indent_str = " " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.extend(StabilityRobustnessReport._format_dict(value, indent + 2))
            elif isinstance(value, (int, float)):
                lines.append(f"{indent_str}{key}: {value:.4f}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return lines

    @staticmethod
    def _get_combined_assessment(stability_summary: Dict, robustness_summary: Dict) -> str:
        """Generate a qualitative assessment of combined metrics."""
        stability_score = stability_summary["stability_score"]
        robustness_ratio = robustness_summary["robustness_ratio"]

        stability_quality = (
            "High" if stability_score > 0.7 else "Low" if stability_score < 0.3 else "Moderate"
        )

        robustness_quality = (
            "High" if robustness_ratio > 0.7 else "Low" if robustness_ratio < 0.3 else "Moderate"
        )

        combined = (
            "Excellent (stable AND robust)"
            if stability_score > 0.7 and robustness_ratio > 0.7
            else (
                "Good (stable or robust)"
                if (stability_score > 0.6 or robustness_ratio > 0.6)
                else (
                    "Fair (moderate stability/robustness)"
                    if (stability_score > 0.4 and robustness_ratio > 0.4)
                    else "Poor (low stability and/or robustness)"
                )
            )
        )

        return f"    {stability_quality} stability + {robustness_quality} robustness = {combined}"

    def summary(self) -> Dict[str, Any]:
        """
        Get a combined summary of key metrics.

        Returns:
            Dict with summaries from both reports.
        """
        result = {
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "stability": None,
            "robustness": None,
        }

        if self.stability_report:
            result["stability"] = self.stability_report.summary()

        if self.robustness_report:
            result["robustness"] = self.robustness_report.summary()

        return result


@dataclass
class ComprehensiveEvaluationReport:
    """
    Top-level report combining all evaluation reports (Faithfulness + Stability + Robustness).

    Attributes:
        faithfulness_report: FaithfulnessReport (Pillars 1, 2, 5, 6).
        stability_robustness_report: StabilityRobustnessReport (Pillars 3, 4).
        timestamp: ISO timestamp when report was generated.
        metadata: Shared metadata for all evaluations.
    """

    faithfulness_report: Optional["FaithfulnessReport"] = None
    stability_robustness_report: Optional[StabilityRobustnessReport] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """
        Save comprehensive report as JSON file.

        Args:
            path: Path to save JSON file to.

        Raises:
            IOError: If file cannot be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build comprehensive report dict
        report_dict = {
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "faithfulness_report": (
                asdict(self.faithfulness_report) if self.faithfulness_report else None
            ),
            "stability_robustness_report": None,
        }

        if self.stability_robustness_report:
            sr = self.stability_robustness_report
            report_dict["stability_robustness_report"] = {
                "timestamp": sr.timestamp,
                "metadata": sr.metadata,
                "stability_report": asdict(sr.stability_report) if sr.stability_report else None,
                "robustness_report": asdict(sr.robustness_report) if sr.robustness_report else None,
            }

        try:
            with open(path, "w") as f:
                json.dump(report_dict, f, indent=2)
            logger.info(f"Saved comprehensive evaluation report to {path}")
        except IOError as e:
            logger.error(f"Failed to save report to {path}: {e}")
            raise

    @classmethod
    def from_json(cls, path: Path) -> "ComprehensiveEvaluationReport":
        """
        Load comprehensive report from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            ComprehensiveEvaluationReport instance.

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

        try:
            # Lazy imports to avoid circular dependencies
            try:
                from ..report import FaithfulnessReport
                from .robustness_report import RobustnessReport
                from .stability_report import StabilityReport
            except ImportError:
                # Fallback for when module is not in a package context
                import sys

                mod_path = Path(__file__).parent
                sys.path.insert(0, str(mod_path))
                from robustness_report import RobustnessReport
                from stability_report import StabilityReport

                sys.path.pop(0)

                # FaithfulnessReport is in parent directory
                parent_path = Path(__file__).parent.parent
                sys.path.insert(0, str(parent_path))
                from report import FaithfulnessReport

                sys.path.pop(0)

            # Reconstruct nested report objects
            faithfulness_data = data.get("faithfulness_report")
            sr_data = data.get("stability_robustness_report")

            faithfulness_report = None
            stability_robustness_report = None

            if faithfulness_data is not None:
                faithfulness_report = FaithfulnessReport(**faithfulness_data)

            if sr_data is not None:
                stability_data = sr_data.get("stability_report")
                robustness_data = sr_data.get("robustness_report")

                stability_report = None
                robustness_report = None

                if stability_data is not None:
                    stability_report = StabilityReport(**stability_data)

                if robustness_data is not None:
                    robustness_report = RobustnessReport(**robustness_data)

                stability_robustness_report = StabilityRobustnessReport(
                    stability_report=stability_report,
                    robustness_report=robustness_report,
                    timestamp=sr_data.get("timestamp"),
                    metadata=sr_data.get("metadata", {}),
                )

            report = cls(
                faithfulness_report=faithfulness_report,
                stability_robustness_report=stability_robustness_report,
                timestamp=data.get("timestamp"),
                metadata=data.get("metadata", {}),
            )
            logger.info(f"Loaded comprehensive evaluation report from {path}")
            return report
        except TypeError as e:
            raise ValueError(f"Invalid report JSON structure: {e}")

    def __repr__(self) -> str:
        """Pretty-print the comprehensive report."""
        lines = [
            "=" * 80,
            "COMPREHENSIVE CIRCUIT EVALUATION REPORT",
            "=" * 80,
        ]

        if self.timestamp:
            lines.append(f"Timestamp: {self.timestamp}")

        if self.metadata:
            lines.extend(
                [
                    "",
                    "METADATA",
                    *self._format_dict(self.metadata),
                ]
            )

        # Faithfulness report
        if self.faithfulness_report:
            lines.extend(
                [
                    "",
                    str(self.faithfulness_report),
                ]
            )

        # Stability & Robustness report
        if self.stability_robustness_report:
            lines.extend(
                [
                    "",
                    str(self.stability_robustness_report),
                ]
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    @staticmethod
    def _format_dict(d: Dict[str, Any], indent: int = 2) -> list:
        """Format a dictionary for pretty-printing."""
        lines = []
        indent_str = " " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                lines.extend(ComprehensiveEvaluationReport._format_dict(value, indent + 2))
            elif isinstance(value, (int, float)):
                lines.append(f"{indent_str}{key}: {value:.4f}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return lines

    def summary(self) -> Dict[str, Any]:
        """Get a combined summary of all evaluations."""
        result = {
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "faithfulness": None,
            "stability_robustness": None,
        }

        if self.faithfulness_report:
            result["faithfulness"] = self.faithfulness_report.summary()

        if self.stability_robustness_report:
            result["stability_robustness"] = self.stability_robustness_report.summary()

        return result
