"""
PEFT Benchmarking Analysis & Recommendations for Phase 3 Week 11.

Analyzes benchmark results and generates:
- Performance comparisons
- Trade-off analysis (speed vs quality vs memory)
- Method recommendations for different use cases
- Optimization suggestions
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from circuitkit.applications.finetuning.benchmark_peft import BenchmarkMetrics

logger = logging.getLogger(__name__)


@dataclass
class MethodRecommendation:
    """Recommendation for a PEFT method."""

    method: str
    score: float  # 0-100
    use_cases: List[str] = field(default_factory=list)
    advantages: List[str] = field(default_factory=list)
    disadvantages: List[str] = field(default_factory=list)
    best_for: str = ""
    not_recommended_for: str = ""

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"{self.method.upper()} (Score: {self.score:.1f}/100)",
            f"Best For: {self.best_for}",
            "Advantages:",
        ]
        for adv in self.advantages:
            lines.append(f"  • {adv}")
        lines.append("Disadvantages:")
        for dis in self.disadvantages:
            lines.append(f"  • {dis}")
        return "\n".join(lines)


@dataclass
class BenchmarkAnalysis:
    """Analysis of benchmark results."""

    results: Dict[str, Dict[str, BenchmarkMetrics]] = field(default_factory=dict)

    # Aggregated statistics
    method_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Recommendations
    recommendations: Dict[str, MethodRecommendation] = field(default_factory=dict)

    def analyze(self) -> None:
        """Perform analysis on benchmark results."""
        logger.info("Analyzing benchmark results...")

        # Compute statistics per method
        self._compute_method_statistics()

        # Compute statistics per model
        self._compute_model_statistics()

        # Generate recommendations
        self._generate_recommendations()

        logger.info("Analysis complete")

    def _compute_method_statistics(self) -> None:
        """Compute statistics for each PEFT method."""
        methods = {}

        # Collect metrics per method
        for model_results in self.results.values():
            for method_name, metrics in model_results.items():
                if method_name not in methods:
                    methods[method_name] = []
                methods[method_name].append(metrics)

        # Compute statistics
        for method_name, metrics_list in methods.items():
            self.method_stats[method_name] = {
                "avg_param_efficiency": statistics.mean(m.param_efficiency for m in metrics_list),
                "avg_peak_memory": statistics.mean(m.peak_memory_mb for m in metrics_list),
                "avg_throughput": statistics.mean(m.batches_per_second for m in metrics_list),
                "avg_latency": statistics.mean(m.inference_latency_ms for m in metrics_list),
                "min_memory": min(m.peak_memory_mb for m in metrics_list),
                "max_memory": max(m.peak_memory_mb for m in metrics_list),
                "num_benchmarks": len(metrics_list),
            }

            logger.info(
                f"{method_name}: Params={self.method_stats[method_name]['avg_param_efficiency']:.2%}, "
                f"Memory={self.method_stats[method_name]['avg_peak_memory']:.1f}MB, "
                f"Throughput={self.method_stats[method_name]['avg_throughput']:.2f}b/s"
            )

    def _compute_model_statistics(self) -> None:
        """Compute statistics for each model."""
        for model_name, model_results in self.results.items():
            metrics_list = list(model_results.values())

            self.model_stats[model_name] = {
                "avg_memory": statistics.mean(m.peak_memory_mb for m in metrics_list),
                "avg_throughput": statistics.mean(m.batches_per_second for m in metrics_list),
                "min_param_efficiency": min(m.param_efficiency for m in metrics_list),
                "max_param_efficiency": max(m.param_efficiency for m in metrics_list),
            }

    def _generate_recommendations(self) -> None:
        """Generate recommendations for each PEFT method."""

        # LoRA Recommendation
        if "lora" in self.method_stats:
            stats = self.method_stats["lora"]
            self.recommendations["lora"] = MethodRecommendation(
                method="LoRA",
                score=85.0,
                advantages=[
                    "Good balance of efficiency and quality",
                    f"~{stats['avg_param_efficiency']:.1%} parameter efficiency",
                    f"~{stats['avg_peak_memory']:.0f}MB memory usage",
                    f"High throughput: {stats['avg_throughput']:.1f} b/s",
                    "Well-studied method with good results",
                ],
                disadvantages=[
                    "Requires careful rank selection",
                    "Slightly higher memory than BitFit",
                    "May need tuning per task",
                ],
                best_for="General-purpose fine-tuning with good quality",
                use_cases=["General fine-tuning", "Multi-task adaptation"],
            )

        # Adapter Recommendation
        if "adapter" in self.method_stats:
            stats = self.method_stats["adapter"]
            self.recommendations["adapter"] = MethodRecommendation(
                method="Adapter",
                score=80.0,
                advantages=[
                    "Similar efficiency to LoRA",
                    f"~{stats['avg_param_efficiency']:.1%} parameter efficiency",
                    f"~{stats['avg_peak_memory']:.0f}MB memory usage",
                    "Modular design, easy to compose",
                    "Good generalization",
                ],
                disadvantages=[
                    "Slightly less studied than LoRA",
                    "Extra forward passes add latency",
                    "Similar memory to LoRA",
                ],
                best_for="Modular adaptation with composable methods",
                use_cases=["Multi-task learning", "Modular training"],
            )

        # Prefix Tuning Recommendation
        if "prefix" in self.method_stats:
            stats = self.method_stats["prefix"]
            self.recommendations["prefix"] = MethodRecommendation(
                method="Prefix Tuning",
                score=70.0,
                advantages=[
                    f"~{stats['avg_param_efficiency']:.1%} parameter efficiency",
                    f"~{stats['avg_peak_memory']:.0f}MB memory usage",
                    "Lowest memory footprint",
                    "Interesting for prompt-based methods",
                ],
                disadvantages=[
                    f"Lower throughput: {stats['avg_throughput']:.1f} b/s",
                    "Slower inference due to prefix processing",
                    "Less proven than LoRA/Adapter",
                    "Quality can vary by task",
                ],
                best_for="Ultra-low memory scenarios with quality trade-off",
                use_cases=["Memory-constrained devices", "Prompt engineering"],
            )

        # BitFit Recommendation
        if "bitfit" in self.method_stats:
            stats = self.method_stats["bitfit"]
            self.recommendations["bitfit"] = MethodRecommendation(
                method="BitFit",
                score=75.0,
                advantages=[
                    f"Minimal parameters: {stats['avg_param_efficiency']:.3%}",
                    f"Lowest memory: ~{stats['avg_peak_memory']:.0f}MB",
                    f"Highest throughput: {stats['avg_throughput']:.1f} b/s",
                    "Simplest method to implement",
                    "Easiest to integrate",
                ],
                disadvantages=[
                    "Quality highly task-dependent",
                    "May not work well for all tasks",
                    "Less research compared to LoRA",
                    "Not suitable for complex fine-tuning",
                ],
                best_for="Extreme parameter efficiency with acceptable quality",
                use_cases=["Edge devices", "Very limited compute"],
            )

    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report."""
        lines = [
            "=" * 80,
            "PEFT METHODS COMPREHENSIVE COMPARISON REPORT",
            "=" * 80,
            "",
        ]

        # Overall statistics
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Models tested: {len(self.results)}")
        lines.append(f"Methods benchmarked: {len(self.method_stats)}")
        lines.append(f"Total benchmarks: {sum(len(v) for v in self.results.values())}")
        lines.append("")

        # Method comparison table
        lines.append("METHOD PERFORMANCE COMPARISON")
        lines.append("-" * 80)

        if self.method_stats:
            header = (
                f"{'Method':<12} | {'Params':<10} | {'Memory':<10} | "
                f"{'Throughput':<12} | {'Latency':<10}"
            )
            lines.append(header)
            lines.append("-" * len(header))

            for method_name, stats in sorted(self.method_stats.items()):
                lines.append(
                    f"{method_name:<12} | {stats['avg_param_efficiency']:>8.2%} | "
                    f"{stats['avg_peak_memory']:>8.1f}MB | "
                    f"{stats['avg_throughput']:>10.2f}b/s | "
                    f"{stats['avg_latency']:>8.2f}ms"
                )

        lines.append("")

        # Trade-off analysis
        lines.append("TRADE-OFF ANALYSIS")
        lines.append("-" * 80)

        if self.method_stats:
            # Efficiency leaders
            param_leader = max(
                self.method_stats.items(),
                key=lambda x: x[1]["avg_param_efficiency"],
            )
            memory_leader = min(
                self.method_stats.items(),
                key=lambda x: x[1]["avg_peak_memory"],
            )
            speed_leader = max(
                self.method_stats.items(),
                key=lambda x: x[1]["avg_throughput"],
            )

            lines.append(f"Parameter Efficiency Leader: {param_leader[0].upper()}")
            lines.append(f"  → {param_leader[1]['avg_param_efficiency']:.2%} trainable parameters")
            lines.append("")

            lines.append(f"Memory Efficiency Leader: {memory_leader[0].upper()}")
            lines.append(f"  → {memory_leader[1]['avg_peak_memory']:.1f}MB peak memory")
            lines.append("")

            lines.append(f"Speed Leader: {speed_leader[0].upper()}")
            lines.append(f"  → {speed_leader[1]['avg_throughput']:.2f} batches/sec")
            lines.append("")

        # Recommendations
        lines.append("METHOD RECOMMENDATIONS")
        lines.append("-" * 80)

        for method_name in sorted(self.recommendations.keys()):
            rec = self.recommendations[method_name]
            lines.append(rec.summary())
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_quick_reference(self) -> str:
        """Generate quick reference guide."""
        lines = [
            "QUICK REFERENCE: CHOOSE YOUR PEFT METHOD",
            "=" * 60,
            "",
            "SCENARIO 1: Maximum Quality (Default Choice)",
            "  Recommendation: LoRA or Adapter",
            "  Why: Best balance of quality and efficiency",
            "",
            "SCENARIO 2: Memory-Constrained",
            "  Recommendation: BitFit",
            "  Why: Minimal parameters and memory",
            "",
            "SCENARIO 3: Speed-Critical",
            "  Recommendation: LoRA or BitFit",
            "  Why: High throughput, low latency",
            "",
            "SCENARIO 4: Research/Experimentation",
            "  Recommendation: Adapter",
            "  Why: Modular, composable, easy to modify",
            "",
            "SCENARIO 5: Production Deployment",
            "  Recommendation: LoRA",
            "  Why: Well-studied, reliable, good quality",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)


def run_week11_analysis(results: Dict[str, Dict[str, BenchmarkMetrics]]) -> Tuple[str, str]:
    """
    Run Week 11 analysis on benchmark results.

    Args:
        results: Benchmark results from Week 10 framework

    Returns:
        Tuple of (comparison_report, quick_reference)
    """
    analysis = BenchmarkAnalysis(results=results)
    analysis.analyze()

    return analysis.generate_comparison_report(), analysis.generate_quick_reference()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Example usage (with mock results)
    from circuitkit.applications.finetuning.benchmark_peft import BenchmarkMetrics

    results = {
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
    }

    comparison, quick_ref = run_week11_analysis(results)
    logger.info(comparison)
    logger.info("\n\n")
    logger.info(quick_ref)
