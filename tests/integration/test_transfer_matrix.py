"""
Tests for cross-task transfer matrix (Workstream F.1).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from circuitkit.evaluation.transfer import (  # noqa: E402 - import after intentional pre-import setup
    TransferMatrix,
)


class TestTransferMatrix:
    """Test TransferMatrix initialization and utilities."""

    def test_transfer_matrix_init(self):
        """Test basic initialization."""
        tasks = ["ioi", "sva", "greater_than"]
        tm = TransferMatrix(tasks)

        assert tm.task_names == tasks
        assert tm.n_tasks == 3
        assert tm.matrix is None

    def test_transfer_matrix_analyze_empty(self):
        """Test that analyze raises error if matrix not built."""
        tm = TransferMatrix(["task1", "task2"])

        with pytest.raises(RuntimeError):
            tm.analyze()

    def test_transfer_matrix_summary_empty(self):
        """Test summary on unbuilt matrix."""
        tm = TransferMatrix(["task1", "task2"])
        summary = tm.summary()

        assert "not built" in summary.lower()

    def test_transfer_matrix_manual_build(self):
        """Test manual matrix construction and analysis."""
        tasks = ["ioi", "sva", "greater_than"]
        tm = TransferMatrix(tasks)

        # Manually create a 3x3 matrix with example scores
        tm.matrix = np.array(
            [
                [0.85, 0.60, 0.45],  # IOI circuit: strong on IOI, OK on SVA, weak on GT
                [0.50, 0.92, 0.40],  # SVA circuit: OK on IOI, strong on SVA, weak on GT
                [0.55, 0.35, 0.88],  # GT circuit: weak on IOI/SVA, strong on GT
            ]
        )

        # Test analyze
        analysis = tm.analyze()

        # Check source averages
        assert "source_avg" in analysis
        assert len(analysis["source_avg"]) == 3
        assert analysis["source_avg"]["ioi"] == pytest.approx(0.6333, rel=1e-3)
        assert analysis["source_avg"]["sva"] == pytest.approx(0.6067, rel=1e-3)
        assert analysis["source_avg"]["greater_than"] == pytest.approx(0.5933, rel=1e-3)

        # Check target averages
        assert "target_avg" in analysis
        assert analysis["target_avg"]["ioi"] == pytest.approx(0.6333, rel=1e-3)

        # Check best/worst
        assert analysis["best_transfer"] == ("sva", "sva", 0.92)
        assert analysis["worst_transfer"] == ("greater_than", "sva", 0.35)

        # Check overall statistics
        assert analysis["overall_mean"] == pytest.approx(0.6, rel=1e-1)
        assert analysis["num_successful"] == 9

    def test_transfer_matrix_high_transfer_pairs(self):
        """Test identification of high-transfer pairs."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.90, 0.45, 0.35],
                [0.40, 0.88, 0.52],
                [0.30, 0.50, 0.85],
            ]
        )

        analysis = tm.analyze()

        # Should find pairs >= 0.5
        high_pairs = analysis["high_transfer_pairs"]
        assert len(high_pairs) >= 3

        # Check that all returned pairs have score >= 0.5
        for src, tgt, score in high_pairs:
            assert score >= 0.5

    def test_transfer_matrix_with_nan(self):
        """Test handling of NaN values (failed evaluations)."""
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

        # Should handle NaN gracefully
        assert analysis["num_successful"] == 8
        assert not np.isnan(analysis["overall_mean"])

    def test_transfer_matrix_summary(self):
        """Test summary text generation."""
        tasks = ["ioi", "sva"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.90, 0.45],
                [0.40, 0.88],
            ]
        )

        summary = tm.summary(threshold=0.5)

        # Check that summary contains expected info
        assert "Transfer Matrix" in summary
        assert "Best Transfer" in summary
        assert "Worst Transfer" in summary
        assert "ioi" in summary.lower()
        assert "sva" in summary.lower()

    def test_transfer_matrix_to_dict(self):
        """Test serialization to dict."""
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

    def test_transfer_matrix_symmetry(self):
        """Test analysis of symmetric transfer matrix (all tasks similar)."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        # Symmetric matrix: all diagonal elements high, off-diagonal similar
        tm.matrix = np.array(
            [
                [0.95, 0.70, 0.65],
                [0.75, 0.92, 0.68],
                [0.72, 0.69, 0.90],
            ]
        )

        analysis = tm.analyze()

        # All source/target averages should be similar
        source_avgs = list(analysis["source_avg"].values())
        assert max(source_avgs) - min(source_avgs) < 0.1

    def test_transfer_matrix_singleton(self):
        """Test with single task (1x1 matrix)."""
        tasks = ["single"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array([[0.85]])

        analysis = tm.analyze()

        assert analysis["source_avg"]["single"] == 0.85
        assert analysis["target_avg"]["single"] == 0.85
        assert analysis["best_transfer"] == ("single", "single", 0.85)

    def test_transfer_matrix_all_nan(self):
        """Test behavior when all evaluations fail."""
        tasks = ["a", "b"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [np.nan, np.nan],
                [np.nan, np.nan],
            ]
        )

        analysis = tm.analyze()

        # Should handle gracefully
        assert analysis["num_successful"] == 0
        assert np.isnan(analysis["overall_mean"])

    def test_transfer_matrix_partial_success(self):
        """Test with some successful and some failed evaluations."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        # Mix of successful and failed
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

        # Check that high-transfer pairs skips NaNs
        for src, tgt, score in analysis["high_transfer_pairs"]:
            assert not np.isnan(score)


class TestTransferMatrixSerialization:
    """Test saving and loading of transfer matrices."""

    def test_save_and_load_matrix(self):
        """Test that matrix can be saved and loaded."""
        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.85, 0.60, 0.45],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            matrix_path = Path(tmpdir) / "transfer_matrix.npy"

            # Save manually (TransferMatrix.build() does this automatically)
            np.save(matrix_path, tm.matrix)

            # Load
            loaded = np.load(matrix_path)

            assert np.array_equal(loaded, tm.matrix)

    def test_save_analysis_as_dict(self):
        """Test that analysis dict can be serialized."""
        import json

        tasks = ["a", "b"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.array(
            [
                [0.90, 0.45],
                [0.40, 0.88],
            ]
        )

        analysis = tm.analyze()

        # Convert to JSON-serializable format
        json_data = json.dumps(analysis, default=str)

        # Should be serializable
        assert '"source_avg"' in json_data
        assert '"best_transfer"' in json_data


class TestTransferMatrixMockBuild:
    """
    Mock tests for TransferMatrix.build() without actual discovery/eval.
    Full integration tests deferred to separate test file (requires expensive compute).
    """

    def test_build_mock_interface(self):
        """
        Test that TransferMatrix.build() has correct interface.
        Does not actually run discovery/eval (mocked).
        """
        # This is a placeholder test verifying the method signature
        # Real integration tests should be in a separate file that can be
        # marked as slow/expensive and skipped in CI

        tasks = ["ioi", "sva", "greater_than"]
        tm = TransferMatrix(tasks)

        # Just verify the method exists and has correct signature
        assert hasattr(tm, "build")
        assert callable(tm.build)

        # Verify analyze is also present
        assert hasattr(tm, "analyze")
        assert callable(tm.analyze)


class TestTransferMatrixBuildContract:
    """Regression tests for build()'s real config path (the other build() tests
    mock at the signature level, so the config handed to discover_circuit was
    never asserted)."""

    def _run_build(self, monkeypatch, eval_return):
        from circuitkit import api

        captured = {"discover_cfgs": []}

        def fake_discover(cfg):
            captured["discover_cfgs"].append(cfg)
            return {"nodes": []}

        def fake_evaluate(config=None, pruned_artifact_path=None, **kw):
            return eval_return

        monkeypatch.setattr(api, "discover_circuit", fake_discover)
        monkeypatch.setattr(api, "evaluate_circuit", fake_evaluate)

        tm = TransferMatrix(["ioi", "sva"])
        template = {
            "model": {"name": "gpt2"},
            "discovery": {"algorithm": "eap-ig", "level": "node"},
            "pruning": {"target_sparsity": 0.3},
        }
        matrix = tm.build(model=None, discovery_cfg_template=template)
        return matrix, captured

    def test_task_set_under_discovery_not_toplevel(self, monkeypatch):
        class _Report:
            ablation_score = 0.5

        _, captured = self._run_build(monkeypatch, _Report())
        assert captured["discover_cfgs"], "discover_circuit was never called"
        for cfg in captured["discover_cfgs"]:
            assert "task" in cfg["discovery"], "task must live under discovery"
            assert "task" not in cfg, "task must NOT be a top-level key"
        assert {c["discovery"]["task"] for c in captured["discover_cfgs"]} == {"ioi", "sva"}

    def test_build_bootstraps_builtin_tasks(self, monkeypatch):
        class _Report:
            ablation_score = 0.5

        matrix, _ = self._run_build(monkeypatch, _Report())
        assert matrix.shape == (2, 2)
        assert not np.isnan(matrix).any()

    def test_score_extraction_is_ablation_score(self, monkeypatch):
        class _Report:
            ablation_score = 0.7

        m_report, _ = self._run_build(monkeypatch, _Report())
        assert np.allclose(m_report, 0.7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
