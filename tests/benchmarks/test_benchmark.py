"""
Tests for CircuitBenchmark and BenchmarkResult classes.
"""

import json
import tempfile
from pathlib import Path

from circuitkit.benchmarks import BenchmarkResult, CircuitBenchmark


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_create_result(self):
        """Test creating a benchmark result."""
        result = BenchmarkResult(
            model="gpt2",
            task="ioi",
            algorithm="eap",
            intervention="prune",
            baseline=None,
            accuracy=0.85,
            perplexity=15.5,
        )

        assert result.model == "gpt2"
        assert result.task == "ioi"
        assert result.algorithm == "eap"
        assert result.intervention == "prune"
        assert result.accuracy == 0.85
        assert result.perplexity == 15.5

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = BenchmarkResult(
            model="gpt2",
            task="ioi",
            algorithm="eap",
            intervention="prune",
            baseline=None,
            accuracy=0.85,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["model"] == "gpt2"
        assert result_dict["accuracy"] == 0.85

    def test_result_repr(self):
        """Test result string representation."""
        result = BenchmarkResult(
            model="gpt2",
            task="ioi",
            algorithm="eap",
            intervention="prune",
            baseline=None,
            accuracy=0.85,
            perplexity=15.5,
        )

        repr_str = repr(result)
        assert "gpt2" in repr_str
        assert "0.85" in repr_str


class TestCircuitBenchmark:
    """Test CircuitBenchmark class."""

    def test_init(self):
        """Test benchmarking initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(
                model_names=["gpt2"],
                tasks=["ioi"],
                output_dir=tmpdir,
            )

            assert bench.model_names == ["gpt2"]
            assert bench.tasks == ["ioi"]
            assert bench.output_dir == Path(tmpdir)
            assert len(bench.results) == 0

    def test_init_defaults(self):
        """Test default initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(output_dir=tmpdir)

            assert bench.model_names == ["gpt2"]
            assert bench.tasks == ["ioi"]

    def test_output_dir_creation(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            assert not output_dir.exists()

            CircuitBenchmark(output_dir=str(output_dir))
            assert output_dir.exists()

    def test_discovery_benchmark(self):
        """Test discovery benchmarking (without loading models)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(
                model_names=["gpt2"],
                tasks=["ioi"],
                output_dir=tmpdir,
                verbose=False,
            )

            # Mock load_models to avoid actual model loading
            bench.models = {"gpt2": None}

            results = bench.run_discovery_benchmark(
                algorithms=["eap", "eap-ig"],
                num_examples=10,
            )

            assert results["num_runs"] == 2  # 1 model × 1 task × 2 algorithms
            assert "results" in results
            assert len(results["results"]) == 2

    def test_intervention_benchmark(self):
        """Test intervention benchmarking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(
                model_names=["gpt2"],
                tasks=["ioi"],
                output_dir=tmpdir,
                verbose=False,
            )

            bench.models = {"gpt2": None}

            results = bench.run_intervention_benchmark(
                interventions=["prune", "heal"],
            )

            assert results["num_runs"] == 2  # 1 model × 1 task × 2 interventions
            assert "results" in results

    def test_baseline_comparison(self):
        """Test baseline comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(
                model_names=["gpt2"],
                tasks=["ioi"],
                output_dir=tmpdir,
                verbose=False,
            )

            bench.models = {"gpt2": None}

            results = bench.compare_with_baselines(
                baselines=["magnitude", "random"],
                sparsity_levels=[0.1, 0.3],
            )

            # 1 model × 1 task × 2 baselines × 2 sparsity levels = 4
            assert results["num_runs"] == 4
            assert "results" in results

    def test_multi_task_grid(self):
        """Test multi-task grid benchmarking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(
                model_names=["gpt2"],
                tasks=["ioi"],
                output_dir=tmpdir,
                verbose=False,
            )

            bench.models = {"gpt2": None}

            df = bench.run_multi_task_grid(
                algorithms=["eap"],
                interventions=["prune"],
                sparsity_levels=[0.1],
            )

            # 1 model × 1 task × 1 alg × 1 intervention × 1 sparsity = 1
            assert len(df) == 1
            assert "model" in df.columns
            assert "task" in df.columns
            assert "algorithm" in df.columns

    def test_save_results(self):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(
                model_names=["gpt2"],
                tasks=["ioi"],
                output_dir=tmpdir,
                verbose=False,
            )

            # Add a result
            result = BenchmarkResult(
                model="gpt2",
                task="ioi",
                algorithm="eap",
                intervention="prune",
                baseline=None,
                accuracy=0.85,
            )
            bench.results.append(result)

            # Save
            output_file = str(Path(tmpdir) / "results.json")
            saved_path = bench.save_results(output_file)

            assert Path(saved_path).exists()

            # Load and verify
            with open(saved_path, "r") as f:
                data = json.load(f)

            assert len(data["results"]) == 1
            assert data["results"][0]["accuracy"] == 0.85

    def test_load_results(self):
        """Test loading results from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(
                model_names=["gpt2"],
                tasks=["ioi"],
                output_dir=tmpdir,
            )

            # Create a sample results file
            sample_results = {
                "timestamp": "2024-01-01T00:00:00",
                "config": {"models": ["gpt2"], "tasks": ["ioi"]},
                "results": [
                    {
                        "model": "gpt2",
                        "task": "ioi",
                        "algorithm": "eap",
                        "intervention": "prune",
                        "baseline": None,
                        "accuracy": 0.85,
                        "perplexity": 15.5,
                        "tokens_per_second": 100.0,
                        "sparsity": 0.1,
                        "model_size_mb": 100.0,
                        "compressed_size_mb": 90.0,
                        "compression_ratio": 1.11,
                        "circuit_quality_score": 0.9,
                        "stability_jaccard": 0.85,
                        "baseline_improvement": 1.2,
                        "timestamp": "2024-01-01T00:00:00",
                        "runtime_seconds": 10.0,
                        "device": "cuda",
                        "notes": "test",
                    }
                ],
            }

            results_file = Path(tmpdir) / "results.json"
            with open(results_file, "w") as f:
                json.dump(sample_results, f)

            # Load
            bench.load_results(str(results_file))

            assert len(bench.results) == 1
            assert bench.results[0].accuracy == 0.85

    def test_generate_report(self):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(
                model_names=["gpt2"],
                tasks=["ioi"],
                output_dir=tmpdir,
                verbose=False,
            )

            # Add results
            for i in range(3):
                result = BenchmarkResult(
                    model="gpt2",
                    task="ioi",
                    algorithm="eap",
                    intervention="prune",
                    baseline=None,
                    accuracy=0.80 + i * 0.05,
                    perplexity=20.0 - i * 2,
                )
                bench.results.append(result)

            # Generate reports
            for fmt in ["json", "markdown", "html"]:
                report_path = bench.generate_report(output_format=fmt)
                assert report_path is not None
                assert Path(report_path).exists()

    def test_make_discovery_config(self):
        """Test config generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(output_dir=tmpdir)
            config = bench._make_discovery_config("gpt2", "ioi", "eap", 100)

            assert config["model"]["name"] == "gpt2"
            assert config["discovery"]["algorithm"] == "eap"
            assert config["discovery"]["task"] == "ioi"
            assert config["discovery"]["data_params"]["num_examples"] == 100


class TestBenchmarkIntegration:
    """Integration tests for benchmarking suite."""

    def test_full_benchmark_workflow(self):
        """Test complete benchmark workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = CircuitBenchmark(
                model_names=["gpt2"],
                tasks=["ioi"],
                output_dir=tmpdir,
                verbose=False,
            )

            bench.models = {"gpt2": None}

            # Run discovery
            bench.run_discovery_benchmark(
                algorithms=["eap"],
                num_examples=10,
            )

            # Run interventions
            bench.run_intervention_benchmark(interventions=["prune"])

            # Run baselines
            bench.compare_with_baselines(
                baselines=["magnitude"],
                sparsity_levels=[0.1],
            )

            # Save and report
            bench.save_results()
            report = bench.generate_report()

            assert len(bench.results) > 0
            assert report is not None
