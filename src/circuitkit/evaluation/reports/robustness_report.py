"""
E3: Robustness Report

Structured output for circuit robustness evaluation (Pillar 4).

Captures metrics from robustness evaluation including:
- Performance under various corruption types (paraphrase, entity_swap, etc.)
- Relative performance degradation
- Comparison with baseline performance
- All values are JSON-serializable for reporting and storage.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _fmt(value) -> str:
    """Format a metric value for display; None / non-numeric -> 'N/A'."""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "N/A"


@dataclass
class RobustnessReport:
    """
    Structured report for circuit robustness evaluation.

    Attributes:
        corruption_variant: Type of corruption evaluated (e.g., 'paraphrase', 'entity_swap').
        original_score: Circuit performance on original (clean) data.
        variant_score: Circuit performance on corrupted/variant data.
        delta: Absolute performance drop (original_score - variant_score).
        relative_drop: Relative performance drop ((original - variant) / original).
            Range [0, 1]. Lower is more robust.
        robustness_ratio: Ratio of variant to original score.
            Range [0, 1]. Higher is more robust (closer to 1).
        multi_variant_results: Optional dict with results for multiple corruption variants.
            Keys: corruption variant names
            Values: result dicts with 'original_score', 'variant_score', 'delta', etc.
        baseline_comparison: Optional comparison with full model robustness.
            Keys: 'baseline_original_score', 'baseline_variant_score',
                  'baseline_relative_drop', 'circuit_vs_baseline_relative_drop_difference'
        metadata: Arbitrary metadata about the robustness evaluation.
            Keys typically include:
            - algorithm: Discovery algorithm used
            - model: Model name
            - task: Task name
            - intervention: Type of intervention used
            - timestamp: Evaluation timestamp
            - config_path: Path to config file
    """

    # Single corruption variant result
    corruption_variant: str
    original_score: float
    variant_score: float
    delta: float
    relative_drop: Optional[float]
    robustness_ratio: Optional[float]
    # Set when the pillar reported the ratio as undefined (status='invalid':
    # signed metric with non-positive original / negative variant) or skipped.
    status: Optional[str] = None
    reason: Optional[str] = None

    # Multi-variant results
    multi_variant_results: Optional[Dict[str, Dict[str, float]]] = None

    # Baseline comparison results
    baseline_comparison: Optional[Dict[str, float]] = None

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

        # Convert to dict
        report_dict = asdict(self)

        try:
            with open(path, "w") as f:
                json.dump(report_dict, f, indent=2)
            logger.info(f"Saved robustness report to {path}")
        except IOError as e:
            logger.error(f"Failed to save report to {path}: {e}")
            raise

    @classmethod
    def from_json(cls, path: Path) -> "RobustnessReport":
        """
        Load report from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            RobustnessReport instance.

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
        required_fields = [
            "corruption_variant",
            "original_score",
            "variant_score",
            "delta",
            "relative_drop",
            "robustness_ratio",
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Report JSON missing required field '{field_name}'")

        try:
            report = cls(**data)
            logger.info(f"Loaded robustness report from {path}")
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
            "ROBUSTNESS EVALUATION REPORT (Pillar 4)",
            "=" * 60,
        ]

        # Single corruption variant. _fmt renders None / non-numeric values
        # as "N/A" — invalid/skipped pillar results carry no defined ratios.
        lines.extend(
            [
                "",
                f"CORRUPTION VARIANT: {self.corruption_variant.upper()}",
                f"  Original Score:    {_fmt(self.original_score)}",
                f"  Variant Score:     {_fmt(self.variant_score)}",
                f"  Absolute Delta:    {_fmt(self.delta)}",
                f"  Relative Drop:     {_fmt(self.relative_drop)}",
                f"  Robustness Ratio:  {_fmt(self.robustness_ratio)}",
            ]
        )
        if self.status:
            lines.append(f"  Status:            {self.status.upper()}")
            if self.reason:
                lines.append(f"  Reason:            {self.reason}")

        # Multi-variant results
        if self.multi_variant_results:
            lines.extend(
                [
                    "",
                    "MULTI-VARIANT ROBUSTNESS COMPARISON",
                ]
            )
            for variant, result in self.multi_variant_results.items():
                if not isinstance(result, dict) or "error" in result:
                    lines.append(f"  {variant}: [Error or skipped]")
                else:
                    lines.append(
                        f"  {variant}: "
                        f"delta={_fmt(result.get('delta'))}, "
                        f"rel_drop={_fmt(result.get('relative_drop'))}"
                    )

        # Baseline comparison
        if self.baseline_comparison:
            lines.extend(
                [
                    "",
                    "BASELINE COMPARISON",
                    f"  Baseline Original Score:    {_fmt(self.baseline_comparison.get('baseline_original_score'))}",
                    f"  Baseline Variant Score:     {_fmt(self.baseline_comparison.get('baseline_variant_score'))}",
                    f"  Baseline Relative Drop:     {_fmt(self.baseline_comparison.get('baseline_relative_drop'))}",
                ]
            )

            diff = self.baseline_comparison.get("circuit_vs_baseline_relative_drop_difference")
            is_more_robust = self.baseline_comparison.get("is_circuit_more_robust")
            if diff is not None and is_more_robust is not None:
                lines.append(
                    f"  Circuit vs Baseline:        "
                    f"{'[MORE ROBUST]' if is_more_robust else '[LESS ROBUST]'} "
                    f"(diff={diff:.4f})"
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

        # Summary interpretation. A None ratio means the pillar reported the
        # value as undefined — say so instead of crashing or mislabeling.
        if self.robustness_ratio is None:
            lines.extend(
                [
                    "",
                    "INTERPRETATION",
                    f"  Robustness under {self.corruption_variant} corruption is "
                    f"{self.status or 'undefined'}: "
                    f"{self.reason or 'the robustness ratio could not be computed.'}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "INTERPRETATION",
                    f"  {'High' if self.robustness_ratio > 0.7 else 'Low' if self.robustness_ratio < 0.3 else 'Moderate'} robustness: "
                    f"Circuit {'degrades significantly' if self.robustness_ratio < 0.3 else 'maintains performance well' if self.robustness_ratio > 0.7 else 'shows moderate degradation'} "
                    f"under {self.corruption_variant} corruption.",
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
                lines.extend(RobustnessReport._format_dict(value, indent + 2))
            elif isinstance(value, (int, float)):
                lines.append(f"{indent_str}{key}: {value:.4f}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
        return lines

    def summary(self) -> Dict[str, float]:
        """
        Get a summary of key robustness metrics.

        Returns:
            Dict with robustness-specific summaries:
            - 'original_score': Original performance
            - 'variant_score': Performance under corruption
            - 'relative_drop': Relative performance drop
            - 'robustness_ratio': Ratio metric (higher is better)
        """
        return {
            "original_score": self.original_score,
            "variant_score": self.variant_score,
            "delta": self.delta,
            "relative_drop": self.relative_drop,
            "robustness_ratio": self.robustness_ratio,
            "corruption_variant": self.corruption_variant,
        }

    @classmethod
    def from_pillar4_output(
        cls,
        pillar_result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        multi_variant_results: Optional[Dict[str, Dict[str, float]]] = None,
        baseline_comparison: Optional[Dict[str, float]] = None,
    ) -> "RobustnessReport":
        """
        Create a RobustnessReport from Pillar4_Robustness.run() output.

        Args:
            pillar_result: Dict returned from Pillar4_Robustness.run().
            metadata: Optional metadata to include in report.
            multi_variant_results: Optional dict with results for multiple corruption variants.
            baseline_comparison: Optional baseline comparison dict.

        Returns:
            RobustnessReport instance.
        """
        if metadata is None:
            metadata = {}

        # status='invalid' / 'skipped' markers omit relative_drop and
        # robustness_ratio entirely (there is no defined value to report), so
        # read every optional key with .get — indexing crashed with KeyError.
        return cls(
            corruption_variant=pillar_result["corruption_variant"],
            original_score=pillar_result.get("original_score", float("nan")),
            variant_score=pillar_result.get("variant_score", float("nan")),
            delta=pillar_result.get("delta", float("nan")),
            relative_drop=pillar_result.get("relative_drop"),
            robustness_ratio=pillar_result.get("robustness_ratio"),
            status=pillar_result.get("status"),
            reason=pillar_result.get("reason"),
            multi_variant_results=multi_variant_results,
            baseline_comparison=baseline_comparison,
            metadata=metadata,
        )
