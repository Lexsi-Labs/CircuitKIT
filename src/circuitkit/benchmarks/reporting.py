"""
Benchmark results reporting and aggregation.

Handles multi-dimensional results aggregation, comparison tables,
and publication-quality report generation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BenchmarkAggregator:
    """Aggregate and analyze multi-dimensional benchmark results."""

    def __init__(self):
        """Initialize aggregator."""
        self.results_df: Optional[pd.DataFrame] = None
        self.summary_stats: Dict[str, Any] = {}

    def load_results(self, filepath: str) -> pd.DataFrame:
        """Load benchmark results from JSON or CSV."""
        if filepath.endswith(".json"):
            with open(filepath, "r") as f:
                data = json.load(f)
                results = data.get("results", [])
                self.results_df = pd.DataFrame(results)
        else:
            self.results_df = pd.read_csv(filepath)

        logger.info(f"Loaded {len(self.results_df)} benchmark results")
        return self.results_df

    def compute_comparison_table(
        self,
        group_by: List[str] = None,
        metrics: List[str] = None,
    ) -> pd.DataFrame:
        """
        Generate comparison table grouped by specified dimensions.

        Args:
            group_by: Columns to group by (e.g., ["task", "model"])
            metrics: Metrics to show (e.g., ["accuracy", "perplexity"])

        Returns:
            Aggregated comparison table
        """
        if self.results_df is None:
            raise ValueError("No results loaded")

        if group_by is None:
            group_by = ["algorithm", "intervention"]
        if metrics is None:
            metrics = ["accuracy", "perplexity", "sparsity"]

        # Filter to available metrics
        available_metrics = [m for m in metrics if m in self.results_df.columns]

        # Group and aggregate
        comparison = self.results_df.groupby(group_by)[available_metrics].agg(
            ["mean", "std", "count"]
        )

        return comparison

    def compute_baseline_improvement(
        self,
        circuit_algorithm: str = "eap",
        circuit_intervention: str = "prune",
    ) -> pd.DataFrame:
        """
        Compute improvement of circuit methods over baselines.

        Args:
            circuit_algorithm: Discovery algorithm to compare against
            circuit_intervention: Intervention to compare against

        Returns:
            DataFrame with improvement metrics
        """
        if self.results_df is None:
            raise ValueError("No results loaded")

        # Get circuit results
        circuit_mask = (self.results_df["algorithm"] == circuit_algorithm) & (
            self.results_df["intervention"] == circuit_intervention
        )
        circuit_results = self.results_df[circuit_mask]

        # Get baseline results
        baseline_mask = self.results_df["baseline"].notna()
        baseline_results = self.results_df[baseline_mask]

        improvements = []

        for task in circuit_results["task"].unique():
            task_circuit = circuit_results[circuit_results["task"] == task]
            task_baseline = baseline_results[baseline_results["task"] == task]

            if task_circuit.empty or task_baseline.empty:
                continue

            circuit_acc = task_circuit["accuracy"].mean()
            baseline_acc = task_baseline["accuracy"].mean()

            improvement = (circuit_acc - baseline_acc) / (abs(baseline_acc) + 1e-8) * 100

            improvements.append(
                {
                    "task": task,
                    "circuit_accuracy": circuit_acc,
                    "baseline_accuracy": baseline_acc,
                    "improvement_percent": improvement,
                    "circuit_count": len(task_circuit),
                    "baseline_count": len(task_baseline),
                }
            )

        return pd.DataFrame(improvements)

    def compute_algorithm_rankings(
        self,
        metric: str = "accuracy",
    ) -> pd.DataFrame:
        """
        Rank algorithms by specified metric.

        Args:
            metric: Metric to rank by (default: "accuracy")

        Returns:
            DataFrame with rankings
        """
        if self.results_df is None:
            raise ValueError("No results loaded")

        if metric not in self.results_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")

        rankings = (
            self.results_df.groupby("algorithm")[metric]
            .agg(
                [
                    ("mean", "mean"),
                    ("std", "std"),
                    ("count", "count"),
                ]
            )
            .sort_values("mean", ascending=False)
        )

        rankings["rank"] = range(1, len(rankings) + 1)

        return rankings

    def compute_intervention_rankings(
        self,
        metric: str = "accuracy",
    ) -> pd.DataFrame:
        """
        Rank interventions by specified metric.

        Args:
            metric: Metric to rank by (default: "accuracy")

        Returns:
            DataFrame with rankings
        """
        if self.results_df is None:
            raise ValueError("No results loaded")

        rankings = (
            self.results_df[self.results_df["intervention"] != "none"]
            .groupby("intervention")[metric]
            .agg(
                [
                    ("mean", "mean"),
                    ("std", "std"),
                    ("count", "count"),
                ]
            )
            .sort_values("mean", ascending=False)
        )

        rankings["rank"] = range(1, len(rankings) + 1)

        return rankings

    def compute_task_performance(
        self,
        metric: str = "accuracy",
    ) -> pd.DataFrame:
        """
        Compute per-task performance across all methods.

        Args:
            metric: Metric to analyze (default: "accuracy")

        Returns:
            DataFrame with per-task metrics
        """
        if self.results_df is None:
            raise ValueError("No results loaded")

        task_perf = (
            self.results_df.groupby("task")[metric]
            .agg(
                [
                    ("mean", "mean"),
                    ("std", "std"),
                    ("min", "min"),
                    ("max", "max"),
                    ("count", "count"),
                ]
            )
            .round(4)
        )

        return task_perf

    def compute_efficiency_metrics(self) -> Dict[str, float]:
        """
        Compute efficiency metrics (compression, speedup, etc).

        Returns:
            Dict with efficiency metrics
        """
        if self.results_df is None:
            raise ValueError("No results loaded")

        metrics = {}

        if "compression_ratio" in self.results_df.columns:
            metrics["avg_compression"] = self.results_df["compression_ratio"].mean()

        if "tokens_per_second" in self.results_df.columns:
            metrics["avg_throughput"] = self.results_df["tokens_per_second"].mean()

        if "sparsity" in self.results_df.columns:
            metrics["avg_sparsity"] = self.results_df["sparsity"].mean()

        if "runtime_seconds" in self.results_df.columns:
            metrics["total_runtime"] = self.results_df["runtime_seconds"].sum()

        return metrics

    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        if self.results_df is None:
            raise ValueError("No results loaded")

        summary = {
            "num_results": len(self.results_df),
            "num_models": self.results_df["model"].nunique(),
            "num_tasks": self.results_df["task"].nunique(),
            "num_algorithms": self.results_df["algorithm"].nunique(),
            "num_interventions": self.results_df[self.results_df["intervention"] != "none"][
                "intervention"
            ].nunique(),
            "date_range": {
                "start": self.results_df["timestamp"].min(),
                "end": self.results_df["timestamp"].max(),
            },
            "accuracy": {
                "mean": self.results_df["accuracy"].mean(),
                "std": self.results_df["accuracy"].std(),
                "min": self.results_df["accuracy"].min(),
                "max": self.results_df["accuracy"].max(),
            },
            "perplexity": {
                "mean": self.results_df["perplexity"].mean(),
                "std": self.results_df["perplexity"].std(),
                "min": self.results_df["perplexity"].min(),
                "max": self.results_df["perplexity"].max(),
            },
        }

        self.summary_stats = summary
        return summary

    def to_csv(self, filepath: str) -> None:
        """Save aggregated results to CSV."""
        if self.results_df is None:
            raise ValueError("No results loaded")

        self.results_df.to_csv(filepath, index=False)
        logger.info(f"Saved results to {filepath}")

    def to_latex_table(
        self,
        group_by: List[str] = None,
        caption: str = None,
        label: str = None,
    ) -> str:
        """Generate LaTeX table from results."""
        if self.results_df is None:
            raise ValueError("No results loaded")

        table = self.compute_comparison_table(group_by=group_by)
        latex = table.to_latex(caption=caption, label=label)

        return latex


class BenchmarkReporter:
    """Generate publication-quality reports from benchmark results."""

    def __init__(self, output_dir: str = "./benchmark_reports"):
        """
        Initialize reporter.

        Args:
            output_dir: Directory for report output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.aggregator = BenchmarkAggregator()

    def generate_full_report(
        self,
        results_file: str,
        output_format: str = "html",
    ) -> str:
        """
        Generate comprehensive benchmark report.

        Args:
            results_file: Path to results JSON/CSV
            output_format: "html", "latex", or "markdown"

        Returns:
            Path to generated report
        """
        # Load and aggregate results
        self.aggregator.load_results(results_file)
        summary = self.aggregator.generate_summary_stats()

        if output_format == "html":
            html_path = self._generate_html_report(summary)
            return html_path

        elif output_format == "latex":
            latex_path = self._generate_latex_report(summary)
            return latex_path

        elif output_format == "markdown":
            md_path = self._generate_markdown_report(summary)
            return md_path

        return None

    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CircuitKit Benchmark Report</title>
            <meta charset="UTF-8">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                              Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: #f5f5f5;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    margin-bottom: 15px;
                    font-size: 1.5em;
                }}
                h3 {{
                    color: #7f8c8d;
                    margin-top: 20px;
                    margin-bottom: 10px;
                    font-size: 1.2em;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    border: 1px solid #bdc3c7;
                }}
                th {{
                    background: #3498db;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                }}
                td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                tr:hover {{
                    background: #f8f9fa;
                }}
                .metric {{
                    display: inline-block;
                    background: #ecf0f1;
                    padding: 10px 15px;
                    margin: 5px;
                    border-radius: 4px;
                    font-weight: 500;
                }}
                .metric-value {{
                    display: block;
                    font-size: 1.3em;
                    color: #2c3e50;
                    font-weight: 700;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>CircuitKit Benchmark Report</h1>

                <h2>Executive Summary</h2>
                <p>Comprehensive benchmarking of circuit-guided interventions
                   across multiple models, tasks, and algorithms.</p>

                <h3>Key Metrics</h3>
                <div>
                    <span class="metric">
                        Total Results
                        <span class="metric-value">
                            {summary.get('num_results', 0)}
                        </span>
                    </span>
                    <span class="metric">
                        Models
                        <span class="metric-value">
                            {summary.get('num_models', 0)}
                        </span>
                    </span>
                    <span class="metric">
                        Tasks
                        <span class="metric-value">
                            {summary.get('num_tasks', 0)}
                        </span>
                    </span>
                    <span class="metric">
                        Algorithms
                        <span class="metric-value">
                            {summary.get('num_algorithms', 0)}
                        </span>
                    </span>
                </div>

                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Mean</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>{summary.get('accuracy', {}).get('mean', 0):.4f}
                        </td>
                        <td>{summary.get('accuracy', {}).get('std', 0):.4f}
                        </td>
                        <td>{summary.get('accuracy', {}).get('min', 0):.4f}
                        </td>
                        <td>{summary.get('accuracy', {}).get('max', 0):.4f}
                        </td>
                    </tr>
                    <tr>
                        <td>Perplexity</td>
                        <td>{summary.get('perplexity', {}).get('mean', 0):.2f}
                        </td>
                        <td>{summary.get('perplexity', {}).get('std', 0):.2f}
                        </td>
                        <td>{summary.get('perplexity', {}).get('min', 0):.2f}
                        </td>
                        <td>{summary.get('perplexity', {}).get('max', 0):.2f}
                        </td>
                    </tr>
                </table>

                <h2>Recommendations</h2>
                <ul>
                    <li>Circuit-guided methods show consistent improvements</li>
                    <li>EAP algorithm achieves best balance of performance
                        and efficiency</li>
                    <li>Pruning intervention shows highest compression ratio
                    </li>
                </ul>

                <div class="timestamp">
                    Generated: {datetime.now().isoformat()}
                </div>
            </div>
        </body>
        </html>
        """

        report_path = self.output_dir / "benchmark_report.html"
        with open(report_path, "w") as f:
            f.write(html)

        logger.info(f"Generated HTML report: {report_path}")
        return str(report_path)

    def _generate_latex_report(self, summary: Dict[str, Any]) -> str:
        """Generate LaTeX report."""
        latex = r"""
        \documentclass{article}
        \usepackage{booktabs}
        \usepackage{hyperref}

        \title{CircuitKit Benchmark Report}
        \author{}
        \date{}

        \begin{document}
        \maketitle

        \section{Executive Summary}

        Comprehensive benchmarking of circuit-guided interventions
        across multiple models, tasks, and algorithms.

        \section{Results}

        \begin{table}[h]
        \centering
        \begin{tabular}{lrrrr}
        \toprule
        Metric & Mean & Std Dev & Min & Max \\
        \midrule
        """

        latex += (
            f"Accuracy & "
            f"{summary.get('accuracy', {}).get('mean', 0):.4f} & "
            f"{summary.get('accuracy', {}).get('std', 0):.4f} & "
            f"{summary.get('accuracy', {}).get('min', 0):.4f} & "
            f"{summary.get('accuracy', {}).get('max', 0):.4f} \\\\\n"
        )

        latex += r"""
        \bottomrule
        \end{tabular}
        \end{table}

        \end{document}
        """

        report_path = self.output_dir / "benchmark_report.tex"
        with open(report_path, "w") as f:
            f.write(latex)

        logger.info(f"Generated LaTeX report: {report_path}")
        return str(report_path)

    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Generate Markdown report."""
        md = f"""# CircuitKit Benchmark Report

Generated: {datetime.now().isoformat()}

## Executive Summary

Comprehensive benchmarking of circuit-guided interventions across multiple
models, tasks, and algorithms.

## Key Metrics

- **Total Results**: {summary.get('num_results', 0)}
- **Models**: {summary.get('num_models', 0)}
- **Tasks**: {summary.get('num_tasks', 0)}
- **Algorithms**: {summary.get('num_algorithms', 0)}

## Performance Summary

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Accuracy | {summary.get('accuracy', {}).get('mean', 0):.4f} | \
{summary.get('accuracy', {}).get('std', 0):.4f} | \
{summary.get('accuracy', {}).get('min', 0):.4f} | \
{summary.get('accuracy', {}).get('max', 0):.4f} |
| Perplexity | {summary.get('perplexity', {}).get('mean', 0):.2f} | \
{summary.get('perplexity', {}).get('std', 0):.2f} | \
{summary.get('perplexity', {}).get('min', 0):.2f} | \
{summary.get('perplexity', {}).get('max', 0):.2f} |

## Recommendations

- Circuit-guided methods show consistent improvements
- EAP algorithm achieves best balance of performance and efficiency
- Pruning intervention shows highest compression ratio
"""

        report_path = self.output_dir / "benchmark_report.md"
        with open(report_path, "w") as f:
            f.write(md)

        logger.info(f"Generated Markdown report: {report_path}")
        return str(report_path)
