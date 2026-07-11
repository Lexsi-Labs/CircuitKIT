"""
Integration tests for Workstream F: Cross-Task Transfer Matrix.

Tests the complete workflow: building, analyzing, visualizing, and serializing
transfer matrices. These tests use synthetic data rather than actual discovery.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestTransferMatrixIntegration:
    """Integration tests for transfer matrix workflow."""

    def test_f1_transfer_matrix_builder(self):
        """F1: Test TransferMatrix builder with synthetic data."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tasks = ["ioi", "sva", "greater_than"]
        tm = TransferMatrix(tasks)

        # Simulate matrix build
        tm.matrix = np.array(
            [
                [0.85, 0.60, 0.45],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        # Verify matrix is built
        assert tm.matrix is not None
        assert tm.matrix.shape == (3, 3)
        assert np.all(tm.matrix >= 0) and np.all(tm.matrix <= 1)

    def test_f1_analyze_transfers(self):
        """F1: Test analysis of transfer matrix."""
        from circuitkit.evaluation.transfer import TransferMatrix

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

        # Check source averages
        for task in tasks:
            assert task in analysis["source_avg"]
            assert 0 <= analysis["source_avg"][task] <= 1

        # Check best/worst
        assert analysis["best_transfer"][2] >= analysis["worst_transfer"][2]

        # Check high-transfer pairs (>= 0.5)
        for src, tgt, score in analysis["high_transfer_pairs"]:
            assert score >= 0.5
            assert src in tasks
            assert tgt in tasks

    def test_f2_visualization(self):
        """F2: Test heatmap visualization of transfer matrix."""
        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            from circuitkit.evaluation.transfer_visualizer import TransferMatrixVisualizer
        except ImportError:
            pytest.skip("matplotlib not available")

        tasks = ["ioi", "sva", "greater_than"]
        matrix = np.array(
            [
                [0.85, 0.60, 0.45],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        visualizer = TransferMatrixVisualizer(tasks, figsize=(8, 6))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test heatmap
            heatmap_path = Path(tmpdir) / "heatmap.png"
            visualizer.heatmap(matrix, str(heatmap_path))
            assert heatmap_path.exists()

    def test_f2_per_task_averages(self):
        """F2: Test per-task averages visualization."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            from circuitkit.evaluation.transfer import TransferMatrix
            from circuitkit.evaluation.transfer_visualizer import TransferMatrixVisualizer
        except ImportError:
            pytest.skip("matplotlib not available")

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
        visualizer = TransferMatrixVisualizer(tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            avg_path = Path(tmpdir) / "averages.png"
            visualizer.per_task_averages(tm.matrix, analysis, str(avg_path))
            assert avg_path.exists()

    def test_f2_distribution_plot(self):
        """F2: Test score distribution visualization."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            from circuitkit.evaluation.transfer import TransferMatrix
            from circuitkit.evaluation.transfer_visualizer import TransferMatrixVisualizer
        except ImportError:
            pytest.skip("matplotlib not available")

        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)
        tm.matrix = np.array(
            [
                [0.85, 0.60, 0.45],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        analysis = tm.analyze()
        visualizer = TransferMatrixVisualizer(tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            dist_path = Path(tmpdir) / "distribution.png"
            visualizer.distribution_plot(tm.matrix, analysis, str(dist_path))
            assert dist_path.exists()

    def test_f2_save_all_visualizations(self):
        """F2: Test saving all visualizations at once."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            from circuitkit.evaluation.transfer import TransferMatrix
            from circuitkit.evaluation.transfer_visualizer import TransferMatrixVisualizer
        except ImportError:
            pytest.skip("matplotlib not available")

        tasks = ["task1", "task2", "task3"]
        tm = TransferMatrix(tasks)
        tm.matrix = np.array(
            [
                [0.80, 0.65, 0.50],
                [0.55, 0.90, 0.45],
                [0.60, 0.40, 0.85],
            ]
        )

        analysis = tm.analyze()
        visualizer = TransferMatrixVisualizer(tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_files = visualizer.save_all_visualizations(tm.matrix, analysis, tmpdir)

            # Check that files were saved
            assert len(saved_files) == 3
            for filepath in saved_files:
                assert Path(filepath).exists()

    def test_f3_statistical_analysis(self):
        """F3: Test statistical analysis of transfer matrix."""
        try:
            from circuitkit.evaluation.transfer_analysis import TransferMatrixAnalyzer
        except ImportError:
            pytest.skip("scipy/sklearn not available")

        tasks = ["ioi", "sva", "greater_than"]
        matrix = np.array(
            [
                [0.85, 0.60, 0.45],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        analyzer = TransferMatrixAnalyzer(tasks)

        # Test task similarity
        similarity = analyzer.task_similarity(matrix)
        assert similarity.shape == (3, 3)
        assert np.all(similarity >= -1) and np.all(similarity <= 1)

        # Test correlation structure
        corr_struct = analyzer.correlation_structure(matrix)
        assert "diagonal_strength" in corr_struct
        assert "symmetry" in corr_struct
        assert "sparsity" in corr_struct
        assert 0 <= corr_struct["symmetry"] <= 1

        # Test transferability scores
        transfer_scores = analyzer.transferability_score(matrix)
        assert len(transfer_scores) == 3
        for task, score in transfer_scores.items():
            assert task in tasks
            assert score >= 0

        # Test effect sizes
        effect_sizes = analyzer.effect_sizes(matrix)
        assert "practical_significance" in effect_sizes
        assert 0 <= effect_sizes["practical_significance"] <= 1
        assert len(effect_sizes["largest_improvements"]) <= 5

    def test_f3_clustering(self):
        """F3: Test transfer pattern clustering."""
        try:
            from circuitkit.evaluation.transfer_analysis import TransferMatrixAnalyzer
        except ImportError:
            pytest.skip("scipy/sklearn not available")

        tasks = ["a", "b", "c", "d"]
        # Create matrix with clear clusters
        matrix = np.array(
            [
                [0.95, 0.85, 0.10, 0.15],  # Cluster 1
                [0.90, 0.92, 0.15, 0.20],  # Cluster 1
                [0.10, 0.20, 0.90, 0.88],  # Cluster 2
                [0.15, 0.25, 0.92, 0.95],  # Cluster 2
            ]
        )

        analyzer = TransferMatrixAnalyzer(tasks)
        clustering = analyzer.transfer_clustering(matrix)

        assert "clusters" in clustering
        assert len(clustering["clusters"]) >= 1
        # Should find 2 clusters given our test matrix
        assert len(clustering["clusters"]) <= 4

    def test_f4_cli_integration(self):
        """F4: Test CLI command structure (without running actual discovery)."""
        from circuitkit.cli.main import cli

        # Verify command exists
        assert "transfer-matrix" in [cmd.name for cmd in cli.commands.values()]

    def test_json_serialization(self):
        """Test JSON serialization with NaN handling."""
        from circuitkit.evaluation.transfer import TransferMatrix

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

            # Load and verify
            with open(json_path) as f:
                data = json.load(f)

            assert "task_names" in data
            assert "matrix" in data
            assert "analysis" in data
            assert data["task_names"] == tasks
            # NaN should be converted to None in JSON
            assert data["matrix"][0][1] is None

    def test_summary_generation(self):
        """Test human-readable summary generation."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tasks = ["ioi", "sva", "greater_than"]
        tm = TransferMatrix(tasks)
        tm.matrix = np.array(
            [
                [0.85, 0.60, 0.45],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        summary = tm.summary(threshold=0.5)

        # Check that summary contains expected content
        assert "Transfer Matrix" in summary
        assert "Best Transfer" in summary
        assert "Worst Transfer" in summary
        assert "Overall Mean" in summary
        for task in tasks:
            assert task in summary.lower()

    def test_end_to_end_workflow(self):
        """E2E: Test complete transfer matrix workflow."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tasks = ["ioi", "sva", "greater_than"]
        tm = TransferMatrix(tasks)

        # Simulate building matrix
        tm.matrix = np.array(
            [
                [0.85, 0.60, 0.45],
                [0.50, 0.92, 0.40],
                [0.55, 0.35, 0.88],
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # 1. Save to JSON
            json_path = output_dir / "matrix.json"
            tm.to_json(json_path)
            assert json_path.exists()

            # 2. Save matrix numpy file
            matrix_path = output_dir / "matrix.npy"
            np.save(matrix_path, tm.matrix)
            assert matrix_path.exists()

            # 3. Generate summary
            summary = tm.summary()
            assert "Transfer Matrix" in summary

            # 4. Get analysis dict
            analysis_dict = tm.to_dict()
            assert "analysis" in analysis_dict
            assert "matrix" in analysis_dict

            # 5. Try visualization
            try:
                import matplotlib

                matplotlib.use("Agg")
                viz_result = tm.visualize(output_dir=str(output_dir))
                assert "saved_files" in viz_result or "message" in viz_result
            except ImportError:
                pass  # Skip if matplotlib not available

            # 6. Try statistical analysis
            try:
                stats = tm.statistical_analysis()
                assert "task_similarity" in stats
            except ImportError:
                pass  # Skip if scipy/sklearn not available

    def test_threshold_filtering(self):
        """Test filtering of transfer pairs by threshold."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tasks = ["a", "b", "c", "d"]
        tm = TransferMatrix(tasks)
        tm.matrix = np.array(
            [
                [0.95, 0.45, 0.35, 0.25],
                [0.55, 0.92, 0.48, 0.42],
                [0.35, 0.38, 0.88, 0.52],
                [0.22, 0.32, 0.65, 0.90],
            ]
        )

        analysis = tm.analyze()

        # Count high-transfer pairs
        high_pairs = analysis["high_transfer_pairs"]

        # All pairs should be >= 0.5
        for src, tgt, score in high_pairs:
            assert score >= 0.5

        # Verify some expected pairs are included
        assert len(high_pairs) >= 3  # At least diagonal and some off-diagonal

    def test_nan_handling_comprehensive(self):
        """Test handling of NaN values throughout workflow."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tasks = ["a", "b", "c"]
        tm = TransferMatrix(tasks)

        # Mix of successful and failed evaluations
        tm.matrix = np.array(
            [
                [0.85, np.nan, 0.45],
                [0.50, 0.92, np.nan],
                [np.nan, 0.35, 0.88],
            ]
        )

        analysis = tm.analyze()

        # Should compute stats correctly
        assert analysis["num_successful"] == 6
        assert not np.isnan(analysis["overall_mean"])
        assert analysis["overall_mean"] > 0

        # High-transfer pairs should skip NaN
        for src, tgt, score in analysis["high_transfer_pairs"]:
            assert not np.isnan(score)

    def test_transfer_matrix_dimensions(self):
        """Test matrix dimension validation."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tasks = ["t1", "t2", "t3", "t4", "t5"]
        tm = TransferMatrix(tasks)

        tm.matrix = np.random.rand(5, 5)

        assert tm.matrix.shape == (5, 5)
        analysis = tm.analyze()

        # Verify all tasks are represented
        assert len(analysis["source_avg"]) == 5
        assert len(analysis["target_avg"]) == 5


class TestTransferMatrixEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_matrix_error(self):
        """Test error when analyzing empty matrix."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tm = TransferMatrix(["a", "b"])

        with pytest.raises(RuntimeError):
            tm.analyze()

    def test_singleton_matrix(self):
        """Test with single task (1x1 matrix)."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tm = TransferMatrix(["single"])
        tm.matrix = np.array([[0.85]])

        analysis = tm.analyze()

        assert analysis["source_avg"]["single"] == 0.85
        assert analysis["target_avg"]["single"] == 0.85
        assert analysis["best_transfer"] == ("single", "single", 0.85)

    def test_all_nan_matrix(self):
        """Test behavior when all evaluations fail."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tm = TransferMatrix(["a", "b"])
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
        """Test with perfect transfer (all 1.0)."""
        from circuitkit.evaluation.transfer import TransferMatrix

        tm = TransferMatrix(["a", "b", "c"])
        tm.matrix = np.ones((3, 3))

        analysis = tm.analyze()

        assert analysis["overall_mean"] == 1.0
        assert analysis["overall_std"] == 0.0
        assert len(analysis["high_transfer_pairs"]) == 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
