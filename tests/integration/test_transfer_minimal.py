"""
Tests for transfer matrix functionality.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from circuitkit.evaluation.transfer import TransferMatrix


class TestTransferMatrixCore:
    """Core functionality tests for TransferMatrix."""

    def test_initialization(self):
        """Test basic initialization."""
        tasks = ["ioi", "sva", "greater_than"]
        tm = TransferMatrix(tasks)

        assert tm.task_names == tasks
        assert tm.n_tasks == 3
        assert tm.matrix is None

    def test_matrix_manual_build(self):
        """Test manual matrix construction."""
        tasks = ["ioi", "sva", "greater_than"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.85, 0.60, 0.45],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        assert tm.matrix is not None
        assert tm.matrix.shape == (3, 3)
        assert np.all(tm.matrix >= 0) and np.all(tm.matrix <= 1)

    def test_analyze_basic(self):
        """Test basic analysis."""
        tasks = ["ioi", "sva", "greater_than"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.85, 0.60, 0.45],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        analysis = tm.analyze()

        # Check all required keys
        assert "source_avg" in analysis
        assert "target_avg" in analysis
        assert "best_transfer" in analysis
        assert "worst_transfer" in analysis
        assert "overall_mean" in analysis
        assert "overall_std" in analysis
        assert "high_transfer_pairs" in analysis

        # Check values
        assert len(analysis["source_avg"]) == 3
        assert len(analysis["target_avg"]) == 3

    def test_source_averages(self):
        """Test per-source task averages."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.90, 0.80, 0.70],
                [0.50, 0.60, 0.70],
                [0.30, 0.40, 0.50],
            ]
        )

        analysis = tm.analyze()

        # Task 'a' should have highest average
        assert analysis["source_avg"]["a"] == pytest.approx((0.90 + 0.80 + 0.70) / 3)
        assert analysis["source_avg"]["a"] > analysis["source_avg"]["b"]
        assert analysis["source_avg"]["b"] > analysis["source_avg"]["c"]

    def test_target_averages(self):
        """Test per-target task averages."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.90, 0.80, 0.70],
                [0.50, 0.60, 0.70],
                [0.30, 0.40, 0.50],
            ]
        )

        analysis = tm.analyze()

        # Column 0 should have highest average
        col0_avg = (0.90 + 0.50 + 0.30) / 3
        (0.70 + 0.70 + 0.50) / 3
        assert analysis["target_avg"]["a"] == pytest.approx(col0_avg)
        assert analysis["target_avg"]["a"] < analysis["target_avg"]["c"]

    def test_best_worst_transfers(self):
        """Test identification of best and worst transfers."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.50, 0.80, 0.30],
                [0.40, 0.50, 0.20],
                [0.60, 0.70, 0.40],
            ]
        )

        analysis = tm.analyze()

        best = analysis["best_transfer"]
        worst = analysis["worst_transfer"]

        assert best[2] == 0.80  # Highest value at [0,1]
        assert worst[2] == 0.20  # Lowest value at [1,2]

    def test_high_transfer_pairs(self):
        """Test filtering of high-transfer pairs."""
        tasks = ["a", "b", "c", "d"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.95, 0.55, 0.35, 0.25],
                [0.45, 0.92, 0.65, 0.40],
                [0.35, 0.48, 0.88, 0.52],
                [0.25, 0.35, 0.55, 0.90],
            ]
        )

        analysis = tm.analyze()

        # All high-transfer pairs should be >= 0.5
        for src, tgt, score in analysis["high_transfer_pairs"]:
            assert score >= 0.5
            assert src in tasks
            assert tgt in tasks

    def test_nan_handling(self):
        """Test NaN handling in analysis."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.85, 0.60, np.nan],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        analysis = tm.analyze()

        # Should still compute stats
        assert analysis["num_successful"] == 8
        assert not np.isnan(analysis["overall_mean"])

        # Source averages should handle NaN
        assert not np.isnan(analysis["source_avg"]["a"])
        assert not np.isnan(analysis["source_avg"]["b"])
        assert not np.isnan(analysis["source_avg"]["c"])

    def test_summary_generation(self):
        """Test human-readable summary."""
        tasks = ["ioi", "sva"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.90, 0.45],
                [0.40, 0.88],
            ]
        )

        summary = tm.summary(threshold=0.5)

        assert "Transfer Matrix" in summary
        assert "Best Transfer" in summary
        assert "Worst Transfer" in summary
        assert "ioi" in summary.lower()
        assert "sva" in summary.lower()

    def test_to_dict(self):
        """Test serialization to dictionary."""
        tasks = ["a", "b"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.90, 0.45],
                [0.40, 0.88],
            ]
        )

        result = tm.to_dict()

        assert "task_names" in result
        assert "matrix" in result
        assert "analysis" in result
        assert result["task_names"] == tasks
        assert np.array_equal(result["matrix"], tm.matrix)

    def test_to_json_serialization(self):
        """Test JSON serialization."""
        tasks = ["ioi", "sva"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.85, np.nan],
                [0.50, 0.92],
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "matrix.json"
            tm.to_json(json_path)

            assert json_path.exists()

            # Load and verify
            with open(json_path) as f:
                data = json.load(f)

            assert "task_names" in data
            assert "matrix" in data
            assert data["task_names"] == tasks
            # NaN should be None in JSON
            assert data["matrix"][0][1] is None
            # Other values preserved
            assert data["matrix"][0][0] == 0.85

    def test_singleton_matrix(self):
        """Test with single task."""
        tasks = ["single"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array([[0.85]])

        analysis = tm.analyze()

        assert analysis["source_avg"]["single"] == 0.85
        assert analysis["target_avg"]["single"] == 0.85
        assert analysis["best_transfer"] == ("single", "single", 0.85)

    def test_all_nan_matrix(self):
        """Test all NaN matrix."""
        tasks = ["a", "b"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )

        analysis = tm.analyze()

        assert analysis["num_successful"] == 0
        assert np.isnan(analysis["overall_mean"])

    def test_perfect_transfer_matrix(self):
        """Test perfect transfer (all 1.0)."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.ones((3, 3))

        analysis = tm.analyze()

        assert analysis["overall_mean"] == 1.0
        assert analysis["overall_std"] == 0.0
        assert len(analysis["high_transfer_pairs"]) == 9  # All pairs >= 0.5

    def test_empty_matrix_error(self):
        """Test error when analyzing unbuilt matrix."""
        tm = TransferMatrix(["a", "b"])

        with pytest.raises(RuntimeError):
            tm.analyze()

    def test_summary_empty_matrix(self):
        """Test summary on empty matrix."""
        tm = TransferMatrix(["task1", "task2"])
        summary = tm.summary()

        assert "not built" in summary.lower()

    def test_asymmetric_matrix(self):
        """Test with asymmetric transfer (task-specific circuits)."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        # Task 'a' circuit transfers well to 'b' but not vice versa
        tm.matrix = np.array(
            [
                [0.85, 0.75, 0.40],  # 'a' circuit transfers well to 'b'
                [0.30, 0.90, 0.35],  # 'b' circuit doesn't transfer to 'a'
                [0.50, 0.45, 0.88],
            ]
        )

        analysis = tm.analyze()

        # Source averages should differ from target averages
        assert analysis["source_avg"]["a"] != analysis["target_avg"]["a"]
        assert analysis["source_avg"]["b"] < analysis["target_avg"]["b"]

    def test_matrix_dimensions(self):
        """Test with various matrix dimensions."""
        for n_tasks in [1, 2, 3, 5]:
            tasks = [f"task_{i}" for i in range(n_tasks)]
            tm = TransferMatrix(tasks)

            tm.matrix = np.random.rand(n_tasks, n_tasks)

            analysis = tm.analyze()

            assert len(analysis["source_avg"]) == n_tasks
            assert len(analysis["target_avg"]) == n_tasks

    def test_partial_success_matrix(self):
        """Test with some successful and some failed evaluations."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.85, np.nan, 0.45],
                [0.50, 0.92, np.nan],
                [np.nan, 0.35, 0.88],
            ]
        )

        analysis = tm.analyze()

        # Should compute stats only on successful ones
        assert analysis["num_successful"] == 6
        assert not np.isnan(analysis["overall_mean"])

        # High-transfer pairs should skip NaN
        for src, tgt, score in analysis["high_transfer_pairs"]:
            assert not np.isnan(score)


class TestTransferMatrixEdgeCases:
    """Edge case tests."""

    def test_very_small_values(self):
        """Test with very small transfer scores."""
        tasks = ["a", "b"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.001, 0.0001],
                [0.0002, 0.0003],
            ]
        )

        analysis = tm.analyze()

        assert analysis["num_successful"] == 4
        assert analysis["overall_mean"] < 0.001

    def test_identical_rows(self):
        """Test with constant per-source transfer profiles.

        Every row is internally constant, so each source task transfers
        equally to all targets. The overall mean of all nine values is
        (0.5 + 0.8 + 0.3) / 3 = 0.5333.
        """
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.50, 0.50, 0.50],
                [0.80, 0.80, 0.80],
                [0.30, 0.30, 0.30],
            ]
        )

        analysis = tm.analyze()

        assert analysis["overall_mean"] == pytest.approx(0.5333, rel=1e-2)
        # Each source row is constant, so every source average matches its row.
        assert analysis["source_avg"]["a"] == pytest.approx(0.50)
        assert analysis["source_avg"]["b"] == pytest.approx(0.80)
        assert analysis["source_avg"]["c"] == pytest.approx(0.30)

    def test_identical_columns(self):
        """Test with identical source transfer profiles.

        Every source row is the same, so all source averages are equal
        while each target average matches its (constant) column value.
        """
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.50, 0.80, 0.30],
                [0.50, 0.80, 0.30],
                [0.50, 0.80, 0.30],
            ]
        )

        analysis = tm.analyze()

        # All source rows are identical, so all source averages match.
        assert analysis["source_avg"]["a"] == analysis["source_avg"]["b"]
        assert analysis["source_avg"]["b"] == analysis["source_avg"]["c"]
        # Each target column is constant.
        assert analysis["target_avg"]["a"] == pytest.approx(0.50)
        assert analysis["target_avg"]["b"] == pytest.approx(0.80)
        assert analysis["target_avg"]["c"] == pytest.approx(0.30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
