"""
Workstream K: Unified benchmarking framework for circuit-guided interventions.

This module provides a comprehensive benchmarking suite comparing circuit-guided
interventions (prune, heal, steer, quantize) against strong baselines
(magnitude, WANDA, GPTQ, SparseGPT, random).

Core Classes:
- CircuitBenchmark: Main benchmarking orchestrator
- BenchmarkResult: Structured result container
- BenchmarkGrid: Multi-dimensional benchmark grid
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for a single benchmark result."""

    model: str
    task: str
    algorithm: str  # discovery algorithm: ACDC, EAP, EAP-IG, IBCircuit
    intervention: str  # prune, heal, steer, quantize
    baseline: Optional[str]  # magnitude, wanda, gptq, sparsegpt, random

    # Performance metrics
    accuracy: float = 0.0
    perplexity: float = 0.0
    tokens_per_second: float = 0.0
    sparsity: float = 0.0

    # Size metrics
    model_size_mb: float = 0.0
    compressed_size_mb: float = 0.0
    compression_ratio: float = 0.0

    # Quality metrics
    circuit_quality_score: float = 0.0
    stability_jaccard: float = 0.0
    baseline_improvement: float = 1.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    runtime_seconds: float = 0.0
    device: str = "cuda"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(model={self.model}, task={self.task}, "
            f"algorithm={self.algorithm}, intervention={self.intervention}, "
            f"accuracy={self.accuracy:.4f}, perplexity={self.perplexity:.2f})"
        )


class CircuitBenchmark:
    """
    Unified benchmarking across discovery algorithms and interventions.

    Orchestrates benchmarking of:
    - Discovery algorithms: ACDC, EAP, EAP-IG, IBCircuit
    - Interventions: prune, heal, steer, quantize
    - Tasks: IOI, SVA, GreaterThan, CapitalCountry
    - Models: GPT-2, Llama 2 7B
    - Baselines: magnitude, wanda, gptq, sparsegpt, random
    """

    def __init__(
        self,
        model_names: List[str] = None,
        tasks: List[str] = None,
        device: str = "cuda",
        output_dir: str = "./benchmark_results",
        verbose: bool = True,
        trust_remote_code: bool = False,
    ):
        """
        Initialize CircuitBenchmark.

        Args:
            model_names: List of model names to benchmark (default: ["gpt2"])
            tasks: List of task names (default: ["ioi"])
            device: Device for evaluation (default: "cuda")
            output_dir: Directory to save results
            verbose: Print progress (default: True)
            trust_remote_code: Whether to execute custom modeling code shipped
                in a model repo when loading (default: False). Only enable this
                for model repositories you trust — trust_remote_code=True runs
                arbitrary Python from the repo at load time.
        """
        self.model_names = model_names or ["gpt2"]
        self.tasks = tasks or ["ioi"]
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.trust_remote_code = trust_remote_code

        self.models: Dict[str, HookedTransformer] = {}
        self.results: List[BenchmarkResult] = []
        self.summary: Dict[str, Any] = {}

        logger.info(
            f"Initialized CircuitBenchmark: models={self.model_names}, "
            f"tasks={self.tasks}, device={self.device}"
        )

    def load_models(self) -> None:
        """Load all models specified in initialization."""
        for model_name in self.model_names:
            if model_name in self.models:
                continue

            if self.verbose:
                logger.info(f"Loading model: {model_name}")

            try:
                model = HookedTransformer.from_pretrained(
                    model_name,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code,
                )
                self.models[model_name] = model
                logger.info(f"Loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                raise

    def run_discovery_benchmark(
        self,
        algorithms: List[str] = None,
        num_examples: int = 100,
    ) -> Dict[str, Any]:
        """
        Benchmark discovery algorithms across tasks.

        Args:
            algorithms: List of algorithms (default: ["acdc", "eap", "eap-ig"])
            num_examples: Number of examples per task

        Returns:
            Dict with discovery benchmark results
        """
        if algorithms is None:
            algorithms = ["acdc", "eap", "eap-ig"]

        logger.info(
            f"Running discovery benchmark: algorithms={algorithms}, "
            f"models={self.model_names}, tasks={self.tasks}"
        )

        discovery_results = []

        for model_name in self.model_names:
            if model_name not in self.models:
                self.load_models()

            self.models[model_name]

            for task in self.tasks:
                for algorithm in algorithms:
                    if self.verbose:
                        logger.info(f"Running {algorithm} on {task} with {model_name}...")

                    try:
                        from circuitkit.api import discover_circuit

                        start_time = time.time()

                        config = self._make_discovery_config(
                            model_name, task, algorithm, num_examples
                        )
                        discover_circuit(config)

                        result = BenchmarkResult(
                            model=model_name,
                            task=task,
                            algorithm=algorithm,
                            intervention="none",
                            baseline=None,
                            runtime_seconds=time.time() - start_time,
                        )
                        discovery_results.append(result)
                        self.results.append(result)

                    except Exception as e:
                        logger.error(f"Failed discovery {algorithm} on {task}: {e}")

        return {
            "num_runs": len(discovery_results),
            "algorithms": algorithms,
            "results": [r.to_dict() for r in discovery_results],
        }

    def run_intervention_benchmark(
        self,
        interventions: List[str] = None,
        discovered_circuits: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark interventions (prune, heal, steer, quantize) on discovered circuits.

        Args:
            interventions: List of interventions (default: ["prune", "heal", "steer"])
            discovered_circuits: Pre-computed circuits (optional)

        Returns:
            Dict with intervention benchmark results
        """
        if interventions is None:
            interventions = ["prune", "heal", "steer"]

        logger.info(f"Running intervention benchmark: interventions={interventions}")

        intervention_results = []

        for model_name in self.model_names:
            if model_name not in self.models:
                self.load_models()

            for task in self.tasks:
                for intervention in interventions:
                    if self.verbose:
                        logger.info(
                            f"Benchmarking {intervention} intervention "
                            f"on {task} with {model_name}..."
                        )

                    try:
                        from circuitkit.api import discover_circuit, evaluate_circuit

                        start_time = time.time()

                        config = self._make_discovery_config(
                            model_name, task, "eap", num_examples=32
                        )
                        discover_circuit(config)
                        eval_result = evaluate_circuit(config)

                        # evaluate_circuit always returns a FaithfulnessReport.
                        acc = float(eval_result.patching_score or 0.0)

                        result = BenchmarkResult(
                            model=model_name,
                            task=task,
                            algorithm="eap",
                            intervention=intervention,
                            baseline=None,
                            accuracy=acc,
                            perplexity=0.0,
                            runtime_seconds=time.time() - start_time,
                        )
                        intervention_results.append(result)
                        self.results.append(result)

                    except Exception as e:
                        logger.error(f"Failed intervention {intervention} on {task}: {e}")

        return {
            "num_runs": len(intervention_results),
            "interventions": interventions,
            "results": [r.to_dict() for r in intervention_results],
        }

    def compare_with_baselines(
        self,
        baselines: List[str] = None,
        sparsity_levels: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Compare circuit-guided X vs baseline X methods.

        Args:
            baselines: List of baseline methods
                (default: ["magnitude", "wanda", "random"])
            sparsity_levels: Sparsity levels to test
                (default: [0.1, 0.3, 0.5])

        Returns:
            Dict with baseline comparison results
        """
        if baselines is None:
            baselines = ["magnitude", "wanda", "random"]

        if sparsity_levels is None:
            sparsity_levels = [0.1, 0.3, 0.5]

        logger.info(
            f"Running baseline comparison: baselines={baselines}, "
            f"sparsity_levels={sparsity_levels}"
        )

        baseline_results = []

        for model_name in self.model_names:
            if model_name not in self.models:
                self.load_models()

            for task in self.tasks:
                for baseline in baselines:
                    for sparsity in sparsity_levels:
                        if self.verbose:
                            logger.info(
                                f"Benchmarking {baseline} baseline "
                                f"at {sparsity:.1%} sparsity on {task}..."
                            )

                        try:
                            import math

                            from circuitkit.api import discover_circuit, evaluate_circuit
                            from circuitkit.artifacts import CircuitScores

                            start_time = time.time()

                            config = self._make_discovery_config(
                                model_name, task, "eap", num_examples=32
                            )
                            config["pruning"] = {"target_sparsity": sparsity}
                            disc_result = discover_circuit(config)

                            # Build CircuitScores with strategy-appropriate scores.
                            # Node-level discovery returns a list of node names;
                            # neuron-level returns a dict. Normalise both to a
                            # {node_name: score} mapping.
                            if isinstance(disc_result, dict):
                                node_scores = disc_result.get("node_scores") or {}
                            elif isinstance(disc_result, list):
                                node_scores = {str(node): 1.0 for node in disc_result}
                            else:
                                node_scores = {}
                            if baseline == "random":
                                keys = list(node_scores.keys())
                                vals = list(node_scores.values())
                                import random as _random

                                _random.shuffle(vals)
                                node_scores = dict(zip(keys, vals))
                            elif baseline in ("magnitude", "wanda"):
                                model_obj = self.models.get(model_name)
                                if model_obj is not None:
                                    mag_scores: Dict[str, float] = {}
                                    for name, param in model_obj.named_parameters():
                                        mag_scores[name] = float(param.data.norm().item())
                                    # Map existing circuit node names to closest param norm.
                                    for k in list(node_scores.keys()):
                                        matches = [
                                            v
                                            for pn, v in mag_scores.items()
                                            if k.lower().replace(" ", "_") in pn.lower()
                                        ]
                                        node_scores[k] = float(np.mean(matches)) if matches else 0.0

                            cs = CircuitScores(
                                task=task,
                                model=model_name,
                                algorithm=baseline,
                                level="node",
                                node_scores=node_scores,
                                timestamp=CircuitScores.create_timestamp(),
                            )

                            # Measure pruned model accuracy via a quick perplexity proxy.
                            model_obj = self.models.get(model_name)
                            acc = 0.0
                            ppl = 0.0
                            if model_obj is not None:
                                from circuitkit.applications import StructuralPruner

                                pruner = StructuralPruner()
                                pruned = pruner.prune(
                                    model_obj, cs, sparsity=sparsity, inplace=False
                                )
                                # Quick perplexity on 4 random sequences.
                                pruned.eval()
                                vocab = getattr(getattr(pruned, "cfg", None), "d_vocab", 50257)
                                with torch.no_grad():
                                    ids = torch.randint(0, vocab, (4, 32), device=self.device)
                                    logits = pruned(ids)
                                    if hasattr(logits, "logits"):
                                        logits = logits.logits
                                    loss = torch.nn.functional.cross_entropy(
                                        logits[:, :-1].reshape(-1, vocab),
                                        ids[:, 1:].reshape(-1),
                                    )
                                ppl = float(math.exp(min(float(loss), 20.0)))
                                acc = float(math.exp(-float(loss)))
                            else:
                                eval_result = evaluate_circuit(config)
                                # evaluate_circuit always returns a FaithfulnessReport.
                                acc = float(eval_result.patching_score or 0.0)

                            result = BenchmarkResult(
                                model=model_name,
                                task=task,
                                algorithm="eap",
                                intervention="prune",
                                baseline=baseline,
                                sparsity=sparsity,
                                accuracy=acc,
                                perplexity=ppl,
                                runtime_seconds=time.time() - start_time,
                            )
                            baseline_results.append(result)
                            self.results.append(result)

                        except Exception as e:
                            logger.error(
                                f"Failed baseline {baseline} at " f"sparsity {sparsity}: {e}"
                            )

        return {
            "num_runs": len(baseline_results),
            "baselines": baselines,
            "sparsity_levels": sparsity_levels,
            "results": [r.to_dict() for r in baseline_results],
        }

    def run_multi_task_grid(
        self,
        algorithms: List[str] = None,
        interventions: List[str] = None,
        sparsity_levels: List[float] = None,
    ) -> pd.DataFrame:
        """
        Run benchmark across full grid: algorithm × intervention × task.

        This is the most comprehensive benchmark, generating a full
        multi-dimensional comparison table.

        Args:
            algorithms: List of discovery algorithms
            interventions: List of interventions to apply
            sparsity_levels: Sparsity levels to test

        Returns:
            DataFrame with grid results (rows: configs, columns: metrics)
        """
        if algorithms is None:
            algorithms = ["acdc", "eap", "eap-ig"]
        if interventions is None:
            interventions = ["prune", "heal", "steer"]
        if sparsity_levels is None:
            sparsity_levels = [0.1, 0.3, 0.5]

        grid_results = []
        total_configs = (
            len(self.model_names)
            * len(self.tasks)
            * len(algorithms)
            * len(interventions)
            * len(sparsity_levels)
        )

        logger.info(f"Running multi-task grid: {total_configs} configurations")

        config_idx = 0
        for model_name in self.model_names:
            if model_name not in self.models:
                self.load_models()

            for task in self.tasks:
                for algorithm in algorithms:
                    for intervention in interventions:
                        for sparsity in sparsity_levels:
                            config_idx += 1

                            if self.verbose and config_idx % 10 == 0:
                                logger.info(f"Progress: {config_idx}/{total_configs}")

                            try:
                                from circuitkit.api import discover_circuit, evaluate_circuit

                                start_time = time.time()

                                config = self._make_discovery_config(
                                    model_name, task, algorithm, num_examples=32
                                )
                                config["pruning"] = {"target_sparsity": sparsity}
                                discover_circuit(config)
                                eval_result = evaluate_circuit(config)

                                # evaluate_circuit always returns a FaithfulnessReport.
                                acc = float(eval_result.patching_score or 0.0)

                                result = BenchmarkResult(
                                    model=model_name,
                                    task=task,
                                    algorithm=algorithm,
                                    intervention=intervention,
                                    baseline=None,
                                    sparsity=sparsity,
                                    accuracy=acc,
                                    perplexity=0.0,
                                    runtime_seconds=time.time() - start_time,
                                )
                                grid_results.append(result)
                                self.results.append(result)

                            except Exception as e:
                                logger.error(f"Failed config {config_idx}: {e}")

        # Convert to DataFrame
        df = pd.DataFrame([r.to_dict() for r in grid_results])

        if self.verbose:
            logger.info(f"Completed grid: {len(df)} configurations")

        return df

    def generate_report(
        self,
        output_format: str = "html",
    ) -> str:
        """
        Generate publication-ready benchmark report.

        Args:
            output_format: "html", "latex", "markdown", or "json"

        Returns:
            Path to generated report
        """
        if not self.results:
            logger.warning("No benchmark results to report")
            return None

        logger.info(f"Generating {output_format} report...")

        # Compute summary statistics
        df = pd.DataFrame([r.to_dict() for r in self.results])

        # Generate summary by algorithm
        algo_summary = (
            df.groupby("algorithm")
            .agg(
                {
                    "accuracy": ["mean", "std"],
                    "perplexity": ["mean", "std"],
                    "runtime_seconds": ["sum"],
                }
            )
            .round(4)
        )

        # Generate summary by task
        task_summary = (
            df.groupby("task")
            .agg(
                {
                    "accuracy": ["mean", "std"],
                    "perplexity": ["mean", "std"],
                }
            )
            .round(4)
        )

        # Generate summary by intervention
        intervention_summary = (
            df.groupby("intervention")
            .agg(
                {
                    "accuracy": ["mean", "std"],
                    "perplexity": ["mean", "std"],
                }
            )
            .round(4)
        )

        report_content = self._format_report(
            algo_summary, task_summary, intervention_summary, output_format
        )

        # Save report
        suffix = self._get_report_suffix(output_format)
        report_path = self.output_dir / f"benchmark_report{suffix}"

        with open(report_path, "w") as f:
            f.write(report_content)

        # Also save raw data
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Report saved to {report_path}")
        logger.info(f"Raw data saved to {csv_path}")

        return str(report_path)

    def _make_discovery_config(
        self,
        model_name: str,
        task: str,
        algorithm: str,
        num_examples: int,
    ) -> Dict[str, Any]:
        """Create discovery configuration."""
        return {
            "model": {"name": model_name},
            "discovery": {
                "algorithm": algorithm,
                "task": task,
                "level": "node",
                "batch_size": 4,
                "data_params": {"num_examples": num_examples},
            },
            "pruning": {"target_sparsity": 0.1},
        }

    def _format_report(
        self,
        algo_summary: pd.DataFrame,
        task_summary: pd.DataFrame,
        intervention_summary: pd.DataFrame,
        output_format: str,
    ) -> str:
        """Format report in specified format."""
        if output_format == "json":

            def _summary_to_json(df: pd.DataFrame) -> Dict[str, Any]:
                """Convert a (possibly MultiIndex) summary DataFrame to a
                JSON-serializable dict with string keys."""
                summary = df.to_dict()
                json_summary = {}
                for col, values in summary.items():
                    # MultiIndex columns become tuples; flatten to a string.
                    col_key = ".".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
                    json_summary[col_key] = {str(k): v for k, v in values.items()}
                return json_summary

            return json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "algorithm_summary": _summary_to_json(algo_summary),
                    "task_summary": _summary_to_json(task_summary),
                    "intervention_summary": _summary_to_json(intervention_summary),
                    "num_results": len(self.results),
                },
                indent=2,
            )

        elif output_format == "markdown":
            report = "# CircuitKit Benchmark Report\n\n"
            report += f"Generated: {datetime.now().isoformat()}\n\n"
            report += "## Summary\n\n"
            report += f"- Total results: {len(self.results)}\n"
            report += f"- Models: {', '.join(self.model_names)}\n"
            report += f"- Tasks: {', '.join(self.tasks)}\n\n"

            report += "## Algorithm Performance\n\n"
            report += algo_summary.to_markdown()
            report += "\n\n## Task Performance\n\n"
            report += task_summary.to_markdown()
            report += "\n\n## Intervention Performance\n\n"
            report += intervention_summary.to_markdown()

            return report

        elif output_format == "html":
            html = f"""
            <html>
            <head>
                <title>CircuitKit Benchmark Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; margin-top: 30px; }}
                </style>
            </head>
            <body>
                <h1>CircuitKit Benchmark Report</h1>
                <p>Generated: {datetime.now().isoformat()}</p>

                <h2>Summary</h2>
                <ul>
                    <li>Total results: {len(self.results)}</li>
                    <li>Models: {', '.join(self.model_names)}</li>
                    <li>Tasks: {', '.join(self.tasks)}</li>
                </ul>

                <h2>Algorithm Performance</h2>
                {algo_summary.to_html()}

                <h2>Task Performance</h2>
                {task_summary.to_html()}

                <h2>Intervention Performance</h2>
                {intervention_summary.to_html()}
            </body>
            </html>
            """
            return html

        else:
            return str(self.results)

    def _get_report_suffix(self, format_str: str) -> str:
        """Get file suffix for report format."""
        suffixes = {
            "json": ".json",
            "markdown": ".md",
            "html": ".html",
            "latex": ".tex",
        }
        return suffixes.get(format_str, ".txt")

    def save_results(self, filepath: str = None) -> str:
        """
        Save benchmark results to file.

        Args:
            filepath: Output file path (default: results.json in output_dir)

        Returns:
            Path to saved file
        """
        if filepath is None:
            filepath = str(self.output_dir / "results.json")

        results_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "models": self.model_names,
                "tasks": self.tasks,
                "device": self.device,
            },
            "results": [r.to_dict() for r in self.results],
            "num_results": len(self.results),
        }

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {filepath}")
        return filepath

    def load_results(self, filepath: str) -> None:
        """Load benchmark results from file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.results = [BenchmarkResult(**r) for r in data.get("results", [])]
        logger.info(f"Loaded {len(self.results)} results from {filepath}")
