"""
Tests for stability and robustness report classes (Workstream E.2-E4).

Tests StabilityReport, RobustnessReport, and aggregator classes.
These are standalone unit tests that test JSON serialization and basic functionality.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from circuitkit.evaluation.reports.aggregator import (  # noqa: E402 - import after intentional pre-import setup
    StabilityRobustnessReport,
)
from circuitkit.evaluation.reports.robustness_report import (  # noqa: E402 - import after intentional pre-import setup
    RobustnessReport,
)

# Direct imports to avoid circular dependencies
from circuitkit.evaluation.reports.stability_report import (  # noqa: E402 - import after intentional pre-import setup
    StabilityReport,
)


class TestStabilityReportBasics:
    """Basic tests for StabilityReport."""

    def test_stability_report_creation(self):
        """Test creating a StabilityReport."""
        report = StabilityReport(
            mean_jaccard=0.85,
            std_jaccard=0.05,
            mean_dice=0.87,
            std_dice=0.04,
            n_runs=5,
        )

        assert report.mean_jaccard == 0.85
        assert report.std_jaccard == 0.05
        assert report.mean_dice == 0.87
        assert report.std_dice == 0.04
        assert report.n_runs == 5

    def test_stability_report_with_optional_fields(self):
        """Test StabilityReport with optional fields."""
        report = StabilityReport(
            mean_jaccard=0.75,
            std_jaccard=0.08,
            mean_dice=0.78,
            std_dice=0.07,
            n_runs=5,
            n_stable_nodes=42,
            overlap_per_layer={0: 0.80, 1: 0.75},
            metadata={"model": "gpt2", "task": "ioi"},
        )

        assert report.n_stable_nodes == 42
        assert len(report.overlap_per_layer) == 2
        assert report.metadata["model"] == "gpt2"

    def test_stability_report_summary(self):
        """Test StabilityReport summary method."""
        report = StabilityReport(
            mean_jaccard=0.80,
            std_jaccard=0.05,
            mean_dice=0.82,
            std_dice=0.04,
            n_runs=5,
        )

        summary = report.summary()

        assert "mean_jaccard" in summary
        assert "mean_dice" in summary
        assert "stability_score" in summary
        assert "n_runs" in summary
        # Stability score should be average of jaccard and dice
        expected_score = (0.80 + 0.82) / 2
        assert abs(summary["stability_score"] - expected_score) < 0.001


class TestStabilityReportSerialization:
    """Test JSON serialization for StabilityReport."""

    def test_stability_report_to_json(self):
        """Test saving StabilityReport to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stability_report.json"

            report = StabilityReport(
                mean_jaccard=0.75,
                std_jaccard=0.08,
                mean_dice=0.78,
                std_dice=0.07,
                n_runs=5,
                overlap_per_layer={0: 0.80, 1: 0.75},
            )

            report.to_json(path)

            assert path.exists()
            assert path.stat().st_size > 0

    def test_stability_report_from_json(self):
        """Test loading StabilityReport from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stability_report.json"

            original = StabilityReport(
                mean_jaccard=0.75,
                std_jaccard=0.08,
                mean_dice=0.78,
                std_dice=0.07,
                n_runs=5,
            )

            original.to_json(path)
            loaded = StabilityReport.from_json(path)

            assert loaded.mean_jaccard == original.mean_jaccard
            assert loaded.std_jaccard == original.std_jaccard
            assert loaded.n_runs == original.n_runs

    def test_stability_report_json_roundtrip(self):
        """Test complete JSON roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stability_report.json"

            original = StabilityReport(
                mean_jaccard=0.85,
                std_jaccard=0.05,
                mean_dice=0.87,
                std_dice=0.04,
                n_runs=5,
                n_stable_nodes=42,
                overlap_per_layer={0: 0.9, 1: 0.85},
                metadata={"model": "gpt2", "task": "ioi"},
            )

            original.to_json(path)
            loaded = StabilityReport.from_json(path)

            assert loaded.mean_jaccard == original.mean_jaccard
            assert loaded.std_jaccard == original.std_jaccard
            assert loaded.mean_dice == original.mean_dice
            assert loaded.std_dice == original.std_dice
            assert loaded.n_runs == original.n_runs
            assert loaded.n_stable_nodes == original.n_stable_nodes
            assert loaded.overlap_per_layer == {
                str(k): v for k, v in original.overlap_per_layer.items()
            }
            assert loaded.metadata == original.metadata

    def test_stability_report_numpy_array_conversion(self):
        """Test that numpy arrays are properly converted for JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stability_report.json"

            report = StabilityReport(
                mean_jaccard=0.85,
                std_jaccard=0.05,
                mean_dice=0.87,
                std_dice=0.04,
                n_runs=2,
                jaccard_matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
                dice_matrix=np.array([[1.0, 0.85], [0.85, 1.0]]),
            )

            report.to_json(path)

            # Verify the JSON is valid and contains lists, not arrays
            with open(path) as f:
                data = json.load(f)

            assert isinstance(data["jaccard_matrix"], list)
            assert isinstance(data["dice_matrix"], list)

    def test_stability_report_from_pillar3_output(self):
        """Test creating StabilityReport from Pillar3 output."""
        pillar_output = {
            "mean_jaccard": 0.85,
            "std_jaccard": 0.05,
            "mean_dice": 0.87,
            "std_dice": 0.04,
            "n_runs": 5,
            "n_stable_nodes": 42,
            "overlap_per_layer": {0: 0.9, 1: 0.85},
            "jaccard_matrix": np.array([[1.0, 0.85], [0.85, 1.0]]),
        }

        metadata = {"model": "gpt2", "task": "ioi"}
        report = StabilityReport.from_pillar3_output(pillar_output, metadata)

        assert report.mean_jaccard == 0.85
        assert report.n_stable_nodes == 42
        assert report.metadata == metadata
        # Check numpy array was converted
        assert isinstance(report.jaccard_matrix, list)


class TestRobustnessReportBasics:
    """Basic tests for RobustnessReport."""

    def test_robustness_report_creation(self):
        """Test creating a RobustnessReport."""
        report = RobustnessReport(
            corruption_variant="paraphrase",
            original_score=0.95,
            variant_score=0.88,
            delta=0.07,
            relative_drop=0.0737,
            robustness_ratio=0.9263,
        )

        assert report.corruption_variant == "paraphrase"
        assert report.original_score == 0.95
        assert report.variant_score == 0.88

    def test_robustness_report_with_multi_variant(self):
        """Test RobustnessReport with multiple variants."""
        multi_variant = {
            "paraphrase": {
                "original_score": 0.95,
                "variant_score": 0.88,
                "delta": 0.07,
            },
            "entity_swap": {
                "original_score": 0.95,
                "variant_score": 0.91,
                "delta": 0.04,
            },
        }

        report = RobustnessReport(
            corruption_variant="paraphrase",
            original_score=0.95,
            variant_score=0.88,
            delta=0.07,
            relative_drop=0.0737,
            robustness_ratio=0.9263,
            multi_variant_results=multi_variant,
        )

        assert len(report.multi_variant_results) == 2

    def test_robustness_report_summary(self):
        """Test RobustnessReport summary method."""
        report = RobustnessReport(
            corruption_variant="paraphrase",
            original_score=0.95,
            variant_score=0.88,
            delta=0.07,
            relative_drop=0.0737,
            robustness_ratio=0.9263,
        )

        summary = report.summary()

        assert summary["original_score"] == 0.95
        assert summary["variant_score"] == 0.88
        assert summary["corruption_variant"] == "paraphrase"


class TestRobustnessReportSerialization:
    """Test JSON serialization for RobustnessReport."""

    def test_robustness_report_json_roundtrip(self):
        """Test complete JSON roundtrip for RobustnessReport."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "robustness_report.json"

            original = RobustnessReport(
                corruption_variant="entity_swap",
                original_score=0.92,
                variant_score=0.85,
                delta=0.07,
                relative_drop=0.076,
                robustness_ratio=0.924,
                metadata={"model": "gpt2", "task": "ioi"},
            )

            original.to_json(path)
            loaded = RobustnessReport.from_json(path)

            assert loaded.corruption_variant == original.corruption_variant
            assert loaded.original_score == original.original_score
            assert loaded.variant_score == original.variant_score
            assert loaded.metadata == original.metadata

    def test_robustness_report_from_pillar4_output(self):
        """Test creating RobustnessReport from Pillar4 output."""
        pillar_output = {
            "original_score": 0.95,
            "variant_score": 0.88,
            "delta": 0.07,
            "relative_drop": 0.0737,
            "robustness_ratio": 0.9263,
            "corruption_variant": "paraphrase",
        }

        metadata = {"model": "gpt2", "task": "ioi"}
        report = RobustnessReport.from_pillar4_output(pillar_output, metadata)

        assert report.original_score == 0.95
        assert report.corruption_variant == "paraphrase"
        assert report.metadata == metadata


class TestCombinedReportSerialization:
    """Test combined stability and robustness reports."""

    def test_combined_report_creation(self):
        """Test creating a combined report."""
        stability = StabilityReport(
            mean_jaccard=0.85,
            std_jaccard=0.05,
            mean_dice=0.87,
            std_dice=0.04,
            n_runs=5,
        )

        robustness = RobustnessReport(
            corruption_variant="paraphrase",
            original_score=0.95,
            variant_score=0.88,
            delta=0.07,
            relative_drop=0.0737,
            robustness_ratio=0.9263,
        )

        combined = StabilityRobustnessReport(
            stability_report=stability,
            robustness_report=robustness,
            metadata={"model": "gpt2"},
        )

        assert combined.stability_report is not None
        assert combined.robustness_report is not None

    def test_combined_report_json_roundtrip(self):
        """Test JSON roundtrip for combined report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "combined_report.json"

            stability = StabilityReport(
                mean_jaccard=0.85,
                std_jaccard=0.05,
                mean_dice=0.87,
                std_dice=0.04,
                n_runs=5,
            )

            robustness = RobustnessReport(
                corruption_variant="paraphrase",
                original_score=0.95,
                variant_score=0.88,
                delta=0.07,
                relative_drop=0.0737,
                robustness_ratio=0.9263,
            )

            combined = StabilityRobustnessReport(
                stability_report=stability,
                robustness_report=robustness,
            )

            combined.to_json(path)
            loaded = StabilityRobustnessReport.from_json(path)

            assert loaded.stability_report is not None
            assert loaded.robustness_report is not None
            assert loaded.stability_report.mean_jaccard == 0.85
            assert loaded.robustness_report.original_score == 0.95


class TestJSONCompliance:
    """Test JSON serialization compliance."""

    def test_all_reports_json_serializable(self):
        """Test that all reports produce valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Stability
            stability = StabilityReport(
                mean_jaccard=0.85,
                std_jaccard=0.05,
                mean_dice=0.87,
                std_dice=0.04,
                n_runs=5,
            )
            stability.to_json(Path(tmpdir) / "stability.json")

            # Robustness
            robustness = RobustnessReport(
                corruption_variant="paraphrase",
                original_score=0.95,
                variant_score=0.88,
                delta=0.07,
                relative_drop=0.0737,
                robustness_ratio=0.9263,
            )
            robustness.to_json(Path(tmpdir) / "robustness.json")

            # Verify all are valid JSON
            for filename in ["stability.json", "robustness.json"]:
                with open(Path(tmpdir) / filename) as f:
                    data = json.load(f)
                    assert isinstance(data, dict)


class TestRobustnessReportInvalidPillarResult:
    """from_pillar4_output must accept the status='invalid'/'skipped' markers
    that Pillar4_Robustness.run() emits for undefined ratios (signed metric
    with non-positive original / negative variant). It previously indexed
    pillar_result["relative_drop"] / ["robustness_ratio"] unconditionally and
    crashed with KeyError; the interpretation text also assumed a float ratio.
    """

    _INVALID_MARKER = {
        "original_score": -0.5,
        "variant_score": -1.5,
        "delta": 1.0,
        "status": "invalid",
        "reason": (
            "robustness_ratio undefined: signed/unbounded faithfulness metric "
            "with non-positive original or negative variant score."
        ),
        "corruption_variant": "paraphrase",
        "corruption_source": "generated",
    }

    def test_from_pillar4_output_accepts_invalid_marker(self):
        from circuitkit.evaluation.reports.robustness_report import RobustnessReport

        report = RobustnessReport.from_pillar4_output(dict(self._INVALID_MARKER))
        assert report.status == "invalid"
        assert report.relative_drop is None
        assert report.robustness_ratio is None
        assert abs(report.original_score - (-0.5)) < 1e-9

    def test_repr_renders_invalid_without_crashing(self):
        from circuitkit.evaluation.reports.robustness_report import RobustnessReport

        report = RobustnessReport.from_pillar4_output(dict(self._INVALID_MARKER))
        text = repr(report)
        assert "N/A" in text  # undefined ratios shown as N/A, not fabricated
        assert "INVALID" in text  # status surfaced
        assert "undefined" in text.lower()  # interpretation names the cause
        # The old interpretation branch would have raised TypeError on None.

    def test_repr_handles_none_baseline_relative_drop(self):
        """P4 compare_with_baseline now emits baseline_relative_drop=None for
        undefined comparisons; the old f-string formatted .get(..., 'N/A')
        with :.4f, which raises on both 'N/A' and None."""
        from circuitkit.evaluation.reports.robustness_report import RobustnessReport

        report = RobustnessReport(
            corruption_variant="paraphrase",
            original_score=0.9,
            variant_score=0.8,
            delta=0.1,
            relative_drop=0.11,
            robustness_ratio=0.89,
            baseline_comparison={
                "baseline_original_score": -0.5,
                "baseline_variant_score": -1.5,
                "baseline_relative_drop": None,
                "status": "invalid",
            },
        )
        text = repr(report)
        assert "Baseline Relative Drop:     N/A" in text
