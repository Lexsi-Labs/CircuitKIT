"""
Tests for bootstrap utility (Workstream E.1).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path for direct imports (avoid loading full circuitkit)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import only the bootstrap module, not the full circuitkit package
import importlib.util  # noqa: E402 - import after intentional pre-import setup

bootstrap_spec = importlib.util.spec_from_file_location(
    "bootstrap",
    Path(__file__).parent.parent.parent / "src" / "circuitkit" / "utils" / "bootstrap.py",
)
bootstrap_mod = importlib.util.module_from_spec(bootstrap_spec)
bootstrap_spec.loader.exec_module(bootstrap_mod)
bootstrap = bootstrap_mod.bootstrap
bootstrap_metric_parallel = bootstrap_mod.bootstrap_metric_parallel


class TestBootstrap:
    """Test basic bootstrap functionality."""

    def test_bootstrap_simple_metric(self):
        """Test bootstrap on a simple mean metric."""
        # Simple dataset: [1, 2, 3, 4, 5]
        data = [1, 2, 3, 4, 5]

        def mean_metric(subset):
            return float(np.mean(subset))

        result = bootstrap(mean_metric, data, n_samples=50, seed=42, quiet=True)

        # Should have expected keys
        assert "mean" in result
        assert "std" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "median" in result
        assert "min" in result
        assert "max" in result
        assert "all_values" not in result  # return_all=False by default

        # Mean should be close to 3.0
        assert 2.5 < result["mean"] < 3.5

        # CI should be a valid interval
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_bootstrap_with_return_all(self):
        """Test bootstrap with return_all=True."""
        data = list(range(10))

        def sum_metric(subset):
            return float(np.sum(subset))

        result = bootstrap(sum_metric, data, n_samples=50, seed=42, return_all=True, quiet=True)

        assert "all_values" in result
        assert len(result["all_values"]) == 50

    def test_bootstrap_tensor_output(self):
        """Test bootstrap when metric_fn returns a tensor."""
        data = list(range(5))

        def tensor_metric(subset):
            return torch.tensor(float(np.mean(subset)))

        result = bootstrap(tensor_metric, data, n_samples=30, seed=42, quiet=True)

        assert isinstance(result["mean"], float)
        assert isinstance(result["std"], float)

    def test_bootstrap_deterministic(self):
        """Test that bootstrap is deterministic with fixed seed."""
        data = list(range(20))

        def metric(subset):
            return float(np.var(subset))

        result1 = bootstrap(metric, data, n_samples=30, seed=99, quiet=True)
        result2 = bootstrap(metric, data, n_samples=30, seed=99, quiet=True)

        assert result1["mean"] == result2["mean"]
        assert result1["std"] == result2["std"]

    def test_bootstrap_empty_data(self):
        """Test that bootstrap raises error on empty data."""
        with pytest.raises(ValueError):
            bootstrap(lambda x: 0, [], n_samples=10)

    def test_bootstrap_invalid_metric_output(self):
        """Test that bootstrap raises error if metric doesn't return scalar."""
        data = [1, 2, 3]

        def bad_metric(subset):
            return [1, 2, 3]  # Returns list, not scalar

        with pytest.raises(TypeError):
            bootstrap(bad_metric, data, n_samples=10, quiet=True)

    def test_bootstrap_custom_sample_size(self):
        """Test bootstrap with custom sample size."""
        data = list(range(100))

        def metric(subset):
            return float(np.mean(subset))

        # Sample size smaller than full dataset
        result = bootstrap(metric, data, n_samples=50, sample_size=10, seed=42, quiet=True)

        assert isinstance(result["mean"], float)
        assert result["std"] > 0  # Should have variance

    def test_bootstrap_ci_bounds(self):
        """Test that CI bounds are correct for different CI levels."""
        data = list(range(50))

        def metric(subset):
            return float(np.mean(subset))

        result_95 = bootstrap(metric, data, n_samples=100, seed=42, ci=0.95, quiet=True)
        result_99 = bootstrap(metric, data, n_samples=100, seed=42, ci=0.99, quiet=True)

        # 99% CI should be wider than 95% CI
        ci_width_95 = result_95["ci_upper"] - result_95["ci_lower"]
        ci_width_99 = result_99["ci_upper"] - result_99["ci_lower"]
        assert ci_width_99 > ci_width_95


class TestBootstrapMetricParallel:
    """Test bootstrap_metric_parallel for multi-task metrics."""

    def test_parallel_bootstrap_basic(self):
        """Test bootstrap on multiple data splits."""
        split1 = [1, 2, 3, 4, 5]
        split2 = [10, 20, 30, 40, 50]
        data_splits = [split1, split2]

        def mean_metric(subset):
            return float(np.mean(subset))

        result = bootstrap_metric_parallel(
            mean_metric, data_splits, n_samples=50, seed=42, quiet=True
        )

        assert "mean" in result
        assert "std" in result
        assert "all_values" in result

        # Overall mean should be close to mean of both splits
        # split1 mean = 3, split2 mean = 30, aggregate mean = 16.5
        assert 10 < result["mean"] < 25

    def test_parallel_bootstrap_custom_aggregate(self):
        """Test bootstrap with custom aggregation function."""
        data_splits = [
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
        ]

        def metric(subset):
            return float(np.mean(subset))

        # Use max instead of mean for aggregation
        result = bootstrap_metric_parallel(
            metric,
            data_splits,
            aggregate_fn=np.max,
            n_samples=50,
            seed=42,
            quiet=True,
        )

        # Max of means should be close to 3.1
        assert result["mean"] > 2.5

    def test_parallel_bootstrap_empty(self):
        """Test that bootstrap_metric_parallel raises error on empty splits."""
        with pytest.raises(ValueError):
            bootstrap_metric_parallel(lambda x: 0, [], n_samples=10)


class TestBootstrapOnIOI:
    """Integration test: bootstrap patching score on IOI task."""

    @pytest.mark.skip(reason="Integration test - requires model loading")
    def test_bootstrap_ioi_score(self):
        """
        Test bootstrap on a real IOI patching score metric.

        This is an integration test that requires:
        - Model loading (gpt2)
        - IOI data generation
        - Graph construction
        - Metric computation

        Marks as skipped since it's expensive and requires external dependencies.
        """


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
