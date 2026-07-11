"""
Tests for PEFT benchmarking framework (Phase 3 Week 10).

Tests benchmark metrics collection, cross-architecture comparison,
and report generation.
"""

import pytest
import torch.nn as nn

from circuitkit.applications.finetuning.benchmark_peft import (
    BenchmarkMetrics,
    CrossArchitectureBenchmark,
    PEFTBenchmark,
)

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def simple_model():
    """Create a simple test model."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"model_type": "test_model"})()
            self.linear1 = nn.Linear(256, 256)
            self.linear2 = nn.Linear(256, 256)

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    return SimpleModel()


@pytest.fixture
def multi_model_dict():
    """Create multiple test models for cross-architecture benchmark."""
    models = {}

    class TestModel(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.config = type("Config", (), {"model_type": name})()
            self.linear1 = nn.Linear(256, 256)
            self.linear2 = nn.Linear(256, 256)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    for model_name in ["llama", "gemma", "qwen", "gpt2"]:
        models[model_name] = TestModel(model_name)

    return models


# ==============================================================================
# TESTS: BenchmarkMetrics
# ==============================================================================


class TestBenchmarkMetrics:
    """Test benchmark metrics dataclass."""

    def test_metrics_creation(self):
        """Test creating benchmark metrics."""
        metrics = BenchmarkMetrics(
            method_name="lora",
            model_arch="llama",
            total_params=1000,
            trainable_params=100,
            param_efficiency=0.1,
        )

        assert metrics.method_name == "lora"
        assert metrics.model_arch == "llama"
        assert metrics.total_params == 1000
        assert metrics.trainable_params == 100
        assert metrics.param_efficiency == 0.1

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = BenchmarkMetrics(
            method_name="adapter",
            model_arch="gemma",
            total_params=2000,
            trainable_params=150,
        )

        result = metrics.to_dict()
        assert isinstance(result, dict)
        assert result["method_name"] == "adapter"
        assert result["model_arch"] == "gemma"

    def test_metrics_summary(self):
        """Test summary string generation."""
        metrics = BenchmarkMetrics(
            method_name="prefix",
            model_arch="qwen",
            total_params=5000,
            trainable_params=200,
            param_efficiency=0.04,
            peak_memory_mb=512.5,
            training_time_sec=10.2,
            batches_per_second=9.8,
            inference_latency_ms=2.1,
        )

        summary = metrics.summary()
        assert "prefix" in summary
        assert "qwen" in summary
        assert "5,000" in summary
        assert "200" in summary


# ==============================================================================
# TESTS: PEFTBenchmark
# ==============================================================================


class TestPEFTBenchmark:
    """Test PEFT method benchmarking."""

    def test_benchmark_initialization(self, simple_model):
        """Test benchmark initialization."""
        benchmark = PEFTBenchmark(simple_model, method="lora", device="cpu")

        assert benchmark.method == "lora"
        assert benchmark.device == "cpu"
        assert benchmark.model_arch == "test_model"

    def test_architecture_detection(self, simple_model):
        """Test automatic architecture detection."""
        benchmark = PEFTBenchmark(simple_model, device="cpu")
        assert benchmark.model_arch == "test_model"

    def test_parameter_counting(self, simple_model):
        """Test parameter counting."""
        benchmark = PEFTBenchmark(simple_model, device="cpu")
        total, trainable = benchmark._count_parameters(simple_model)

        assert total > 0
        assert trainable > 0
        assert trainable == total  # All params trainable by default

    def test_benchmark_run_lora(self, simple_model):
        """Test running LoRA benchmark."""
        benchmark = PEFTBenchmark(simple_model, method="lora", device="cpu")
        metrics = benchmark.run(num_batches=5, batch_size=4)

        assert metrics.method_name == "lora"
        assert metrics.model_arch == "test_model"
        assert metrics.total_params > 0
        assert metrics.trainable_params > 0
        assert metrics.param_efficiency > 0
        assert metrics.num_batches == 5

    def test_benchmark_run_adapter(self, simple_model):
        """Test running Adapter benchmark."""
        benchmark = PEFTBenchmark(simple_model, method="adapter", device="cpu")
        metrics = benchmark.run(num_batches=3, batch_size=4)

        assert metrics.method_name == "adapter"
        assert metrics.num_batches == 3

    def test_benchmark_run_prefix(self, simple_model):
        """Test running Prefix tuning benchmark."""
        benchmark = PEFTBenchmark(simple_model, method="prefix", device="cpu")
        metrics = benchmark.run(num_batches=3, batch_size=4)

        assert metrics.method_name == "prefix"

    def test_benchmark_run_bitfit(self, simple_model):
        """Test running BitFit benchmark."""
        benchmark = PEFTBenchmark(simple_model, method="bitfit", device="cpu")
        metrics = benchmark.run(num_batches=3, batch_size=4)

        assert metrics.method_name == "bitfit"

    def test_inference_latency_measurement(self, simple_model):
        """Test inference latency measurement."""
        benchmark = PEFTBenchmark(simple_model, device="cpu")
        latency = benchmark._measure_inference_latency(num_runs=3)

        assert latency >= 0  # Should be non-negative
        assert isinstance(latency, float)

    def test_benchmark_with_different_batch_sizes(self, simple_model):
        """Test benchmarking with different batch sizes."""
        benchmark = PEFTBenchmark(simple_model, device="cpu")

        metrics_small = benchmark.run(num_batches=5, batch_size=1)
        metrics_large = benchmark.run(num_batches=5, batch_size=8)

        assert metrics_small.batch_size == 1
        assert metrics_large.batch_size == 8


# ==============================================================================
# TESTS: CrossArchitectureBenchmark
# ==============================================================================


class TestCrossArchitectureBenchmark:
    """Test cross-architecture benchmarking."""

    def test_initialization(self, multi_model_dict):
        """Test cross-architecture benchmark initialization."""
        benchmark = CrossArchitectureBenchmark(multi_model_dict, device="cpu")

        assert len(benchmark.models) == 4
        assert len(benchmark.peft_methods) == 4

    def test_run_all(self, multi_model_dict):
        """Test running all benchmarks."""
        benchmark = CrossArchitectureBenchmark(multi_model_dict, device="cpu")
        results = benchmark.run_all(num_batches=2, rank=4)

        assert "models" in results
        assert "methods" in results
        assert "results" in results
        assert len(results["models"]) == 4
        assert len(results["methods"]) == 4

    def test_results_structure(self, multi_model_dict):
        """Test results structure after running benchmark."""
        benchmark = CrossArchitectureBenchmark(multi_model_dict, device="cpu")
        benchmark.run_all(num_batches=2)

        # Check structure
        for model_name in multi_model_dict.keys():
            assert model_name in benchmark.results
            for method in benchmark.peft_methods:
                assert method in benchmark.results[model_name]
                metrics = benchmark.results[model_name][method]
                assert isinstance(metrics, BenchmarkMetrics)

    def test_report_generation(self, multi_model_dict):
        """Test report generation."""
        benchmark = CrossArchitectureBenchmark(multi_model_dict, device="cpu")
        benchmark.run_all(num_batches=2)
        report = benchmark.generate_report()

        assert isinstance(report, str)
        assert "CROSS-ARCHITECTURE" in report
        assert "PEFT BENCHMARK" in report
        # Check that all models are mentioned
        for model_name in multi_model_dict.keys():
            assert model_name.upper() in report or model_name in report

    def test_report_comparison_table(self, multi_model_dict):
        """Test that report includes comparison table."""
        benchmark = CrossArchitectureBenchmark(multi_model_dict, device="cpu")
        benchmark.run_all(num_batches=2)
        report = benchmark.generate_report()

        assert "PARAMETER EFFICIENCY" in report
        assert "COMPARISON" in report
        # Check for method names in report
        for method in benchmark.peft_methods:
            assert method in report

    def test_all_methods_benchmarked(self, multi_model_dict):
        """Test that all PEFT methods are benchmarked."""
        benchmark = CrossArchitectureBenchmark(multi_model_dict, device="cpu")
        benchmark.run_all(num_batches=1)

        # Check that all combinations exist
        assert len(benchmark.results) == 4  # 4 models
        for model_name in multi_model_dict.keys():
            assert len(benchmark.results[model_name]) == 4  # 4 methods per model


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestBenchmarkIntegration:
    """Integration tests for benchmarking framework."""

    def test_single_method_on_multiple_models(self, multi_model_dict):
        """Test benchmarking single method across all models."""
        benchmark = CrossArchitectureBenchmark(multi_model_dict, device="cpu")

        # Run full benchmark
        benchmark.run_all(num_batches=2)

        # Check LoRA results
        lora_results = {
            model_name: benchmark.results[model_name]["lora"]
            for model_name in multi_model_dict.keys()
        }

        assert len(lora_results) == 4
        for model_name, metrics in lora_results.items():
            assert metrics.method_name == "lora"
            assert metrics.model_arch == model_name

    def test_metrics_consistency(self, multi_model_dict):
        """Test that metrics are consistent across runs."""
        benchmark = CrossArchitectureBenchmark(multi_model_dict, device="cpu")
        benchmark.run_all(num_batches=2, rank=8)

        # All metrics should have same rank
        for model_results in benchmark.results.values():
            for metrics in model_results.values():
                assert metrics.num_batches == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
