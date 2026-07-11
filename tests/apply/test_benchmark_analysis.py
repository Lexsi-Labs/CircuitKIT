"""
Tests for PEFT benchmark analysis (Phase 3 Week 11).

Tests analysis, recommendations, and report generation.
"""

import pytest

from circuitkit.applications.common_utils.benchmark_analysis import (
    BenchmarkAnalysis,
    MethodRecommendation,
    run_week11_analysis,
)
from circuitkit.applications.finetuning.benchmark_peft import BenchmarkMetrics

# ==============================================================================
# TEST FIXTURES
# ==============================================================================


@pytest.fixture
def mock_benchmark_results():
    """Create mock benchmark results."""
    return {
        "llama": {
            "lora": BenchmarkMetrics(
                method_name="lora",
                model_arch="llama",
                total_params=7000000000,
                trainable_params=140000000,
                param_efficiency=0.02,
                peak_memory_mb=512.0,
                training_time_sec=10.5,
                batches_per_second=9.5,
                inference_latency_ms=2.1,
            ),
            "adapter": BenchmarkMetrics(
                method_name="adapter",
                model_arch="llama",
                total_params=7000000000,
                trainable_params=140000000,
                param_efficiency=0.02,
                peak_memory_mb=480.0,
                training_time_sec=11.2,
                batches_per_second=8.9,
                inference_latency_ms=2.8,
            ),
            "prefix": BenchmarkMetrics(
                method_name="prefix",
                model_arch="llama",
                total_params=7000000000,
                trainable_params=84000000,
                param_efficiency=0.012,
                peak_memory_mb=256.0,
                training_time_sec=15.8,
                batches_per_second=6.3,
                inference_latency_ms=4.2,
            ),
            "bitfit": BenchmarkMetrics(
                method_name="bitfit",
                model_arch="llama",
                total_params=7000000000,
                trainable_params=7000000,
                param_efficiency=0.0001,
                peak_memory_mb=128.0,
                training_time_sec=9.8,
                batches_per_second=10.2,
                inference_latency_ms=1.9,
            ),
        },
        "gemma": {
            "lora": BenchmarkMetrics(
                method_name="lora",
                model_arch="gemma",
                total_params=2000000000,
                trainable_params=40000000,
                param_efficiency=0.02,
                peak_memory_mb=256.0,
                training_time_sec=5.2,
                batches_per_second=9.6,
                inference_latency_ms=1.8,
            ),
            "adapter": BenchmarkMetrics(
                method_name="adapter",
                model_arch="gemma",
                total_params=2000000000,
                trainable_params=40000000,
                param_efficiency=0.02,
                peak_memory_mb=240.0,
                training_time_sec=5.6,
                batches_per_second=8.9,
                inference_latency_ms=2.4,
            ),
            "prefix": BenchmarkMetrics(
                method_name="prefix",
                model_arch="gemma",
                total_params=2000000000,
                trainable_params=24000000,
                param_efficiency=0.012,
                peak_memory_mb=128.0,
                training_time_sec=7.9,
                batches_per_second=6.4,
                inference_latency_ms=3.8,
            ),
            "bitfit": BenchmarkMetrics(
                method_name="bitfit",
                model_arch="gemma",
                total_params=2000000000,
                trainable_params=2000000,
                param_efficiency=0.0001,
                peak_memory_mb=64.0,
                training_time_sec=4.9,
                batches_per_second=10.3,
                inference_latency_ms=1.6,
            ),
        },
    }


# ==============================================================================
# TESTS: MethodRecommendation
# ==============================================================================


class TestMethodRecommendation:
    """Test method recommendation generation."""

    def test_recommendation_creation(self):
        """Test creating a recommendation."""
        rec = MethodRecommendation(
            method="lora",
            score=85.0,
            advantages=["Good balance", "Fast"],
            disadvantages=["Requires tuning"],
            best_for="General fine-tuning",
        )

        assert rec.method == "lora"
        assert rec.score == 85.0
        assert len(rec.advantages) == 2
        assert len(rec.disadvantages) == 1

    def test_recommendation_summary(self):
        """Test summary string generation."""
        rec = MethodRecommendation(
            method="adapter",
            score=80.0,
            advantages=["Modular", "Composable"],
            disadvantages=["Extra latency"],
            best_for="Multi-task learning",
        )

        summary = rec.summary()
        assert "ADAPTER" in summary
        assert "80.0" in summary
        assert "Multi-task" in summary
        assert "Modular" in summary


# ==============================================================================
# TESTS: BenchmarkAnalysis
# ==============================================================================


class TestBenchmarkAnalysis:
    """Test benchmark analysis."""

    def test_analysis_creation(self, mock_benchmark_results):
        """Test creating analysis."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        assert analysis.results == mock_benchmark_results

    def test_analysis_compute_method_statistics(self, mock_benchmark_results):
        """Test method statistics computation."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis._compute_method_statistics()

        assert len(analysis.method_stats) > 0
        assert "lora" in analysis.method_stats

        lora_stats = analysis.method_stats["lora"]
        assert "avg_param_efficiency" in lora_stats
        assert "avg_peak_memory" in lora_stats
        assert "avg_throughput" in lora_stats

    def test_analysis_compute_model_statistics(self, mock_benchmark_results):
        """Test model statistics computation."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis._compute_model_statistics()

        assert len(analysis.model_stats) > 0
        assert "llama" in analysis.model_stats
        assert "gemma" in analysis.model_stats

    def test_analysis_generate_recommendations(self, mock_benchmark_results):
        """Test recommendation generation."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis.analyze()

        assert len(analysis.recommendations) > 0
        assert "lora" in analysis.recommendations
        assert "bitfit" in analysis.recommendations

    def test_analysis_full_workflow(self, mock_benchmark_results):
        """Test full analysis workflow."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis.analyze()

        # Check all components populated
        assert len(analysis.method_stats) > 0
        assert len(analysis.model_stats) > 0
        assert len(analysis.recommendations) > 0

        # Check specific values
        for method_name, stats in analysis.method_stats.items():
            assert stats["num_benchmarks"] > 0
            assert stats["avg_param_efficiency"] > 0
            assert stats["avg_throughput"] > 0


# ==============================================================================
# TESTS: REPORT GENERATION
# ==============================================================================


class TestReportGeneration:
    """Test report and guide generation."""

    def test_comparison_report_generation(self, mock_benchmark_results):
        """Test comparison report generation."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis.analyze()

        report = analysis.generate_comparison_report()
        assert isinstance(report, str)
        assert "COMPARISON" in report
        assert "METHOD" in report

    def test_comparison_report_contains_methods(self, mock_benchmark_results):
        """Test that report contains all methods."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis.analyze()

        report = analysis.generate_comparison_report()
        assert "lora" in report or "LoRA" in report
        assert "adapter" in report or "Adapter" in report

    def test_quick_reference_generation(self, mock_benchmark_results):
        """Test quick reference generation."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis.analyze()

        quick_ref = analysis.generate_quick_reference()
        assert isinstance(quick_ref, str)
        assert "QUICK REFERENCE" in quick_ref
        assert "Recommendation" in quick_ref

    def test_run_week11_analysis(self, mock_benchmark_results):
        """Test run_week11_analysis function."""
        comparison, quick_ref = run_week11_analysis(mock_benchmark_results)

        assert isinstance(comparison, str)
        assert isinstance(quick_ref, str)
        assert "COMPARISON" in comparison
        assert "QUICK REFERENCE" in quick_ref


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


class TestAnalysisIntegration:
    """Integration tests for analysis."""

    def test_analysis_identifies_best_methods(self, mock_benchmark_results):
        """Test that analysis correctly identifies best methods."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis.analyze()

        # LoRA should have good score
        assert analysis.recommendations["lora"].score >= 75

        # BitFit should be memory efficient
        assert (
            analysis.method_stats["bitfit"]["avg_peak_memory"]
            < analysis.method_stats["lora"]["avg_peak_memory"]
        )

    def test_analysis_trade_offs(self, mock_benchmark_results):
        """Test trade-off analysis."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis.analyze()

        # BitFit should have lowest params
        bitfit_params = analysis.method_stats["bitfit"]["avg_param_efficiency"]
        lora_params = analysis.method_stats["lora"]["avg_param_efficiency"]

        assert bitfit_params < lora_params

    def test_all_methods_have_recommendations(self, mock_benchmark_results):
        """Test that all methods have recommendations."""
        analysis = BenchmarkAnalysis(results=mock_benchmark_results)
        analysis.analyze()

        methods = list(analysis.method_stats.keys())
        recommendations = list(analysis.recommendations.keys())

        for method in methods:
            assert method in recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
