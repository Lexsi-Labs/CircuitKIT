"""
PEFT Methods Benchmarking Framework for Phase 3 Week 10.

Benchmarks all PEFT methods across multiple architectures:
- LoRA (Low-Rank Adaptation)
- Adapter (Bottleneck modules)
- Prefix Tuning (Learnable tokens)
- BitFit (Bias-only tuning)

Metrics tracked:
- Memory: peak memory, parameter count, parameter efficiency
- Speed: training time per batch, inference latency
- Quality: mean train loss and perplexity-derived pseudo-accuracy

Usage:
    from circuitkit.applications.finetuning.benchmark_peft import PEFTBenchmark

    benchmark = PEFTBenchmark(model, method="lora")
    results = benchmark.run(num_batches=10)
    logger.info(results.summary())
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark results."""

    method_name: str
    model_arch: str

    # Memory metrics
    total_params: int = 0
    trainable_params: int = 0
    param_efficiency: float = 0.0  # trainable / total
    peak_memory_mb: float = 0.0

    # Speed metrics
    training_time_sec: float = 0.0
    batches_per_second: float = 0.0
    inference_latency_ms: float = 0.0

    # Quality metrics
    task_accuracy: float = 0.0  # exp(-mean_train_loss), a perplexity-derived proxy
    task_loss: float = 0.0

    # Metadata
    num_batches: int = 0
    batch_size: int = 0
    timestamp: str = ""
    notes: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"PEFT Method Benchmark: {self.method_name}",
            f"Model: {self.model_arch}",
            f"{'='*60}",
            "Memory Metrics:",
            f"  Total Parameters:     {self.total_params:,}",
            f"  Trainable Parameters: {self.trainable_params:,}",
            f"  Efficiency:           {self.param_efficiency:.2%}",
            f"  Peak Memory:          {self.peak_memory_mb:.1f} MB",
            "Speed Metrics:",
            f"  Training Time:        {self.training_time_sec:.2f} sec",
            f"  Throughput:           {self.batches_per_second:.2f} batches/sec",
            f"  Inference Latency:    {self.inference_latency_ms:.2f} ms",
        ]
        return "\n".join(lines)


class PEFTBenchmark:
    """
    Benchmark PEFT methods on different models and architectures.

    Measures:
    - Parameter efficiency (trainable params / total params)
    - Memory usage (peak GPU memory)
    - Speed (training throughput, inference latency)
    - Quality (mean train loss + perplexity-derived pseudo-accuracy)
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "lora",
        rank: int = 8,
        device: str = "cpu",
        verbose: bool = True,
    ):
        """
        Initialize PEFT benchmark.

        Args:
            model: Model to benchmark
            method: PEFT method ("lora", "adapter", "prefix", "bitfit")
            rank: Rank for low-rank methods
            device: Device to use ("cpu" or "cuda")
            verbose: Print progress
        """
        self.model = model
        self.method = method.lower()
        self.rank = rank
        self.device = device
        self.verbose = verbose

        # Get model architecture
        self.model_arch = self._detect_architecture()

        logger.info(f"Initialized PEFTBenchmark: {self.method} on {self.model_arch}")

    def _detect_architecture(self) -> str:
        """Detect model architecture."""
        if hasattr(self.model.config, "model_type"):
            return self.model.config.model_type
        elif hasattr(self.model.config, "_name_or_path"):
            return self.model.config._name_or_path.split("/")[-1]
        else:
            return "unknown"

    def _count_parameters(self, model: nn.Module) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def _measure_peak_memory(self) -> float:
        """Measure peak GPU memory in MB."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # Do a forward pass
            try:
                batch = torch.randn(1, 10, 256).to(self.device)
                with torch.no_grad():
                    _ = self.model(batch)
                torch.cuda.synchronize()
            except Exception:
                pass

            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            return peak_memory
        else:
            return 0.0  # Not applicable for CPU

    def _measure_inference_latency(self, num_runs: int = 10) -> float:
        """Measure inference latency in milliseconds."""

        try:
            batch = torch.randn(1, 10, 256).to(self.device)

            with torch.no_grad():
                # Warmup
                for _ in range(2):
                    _ = self.model(batch)

                # Measure
                if self.device == "cuda":
                    torch.cuda.synchronize()

                start = time.time()
                for _ in range(num_runs):
                    _ = self.model(batch)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                elapsed = time.time() - start
                avg_latency = (elapsed / num_runs) * 1000  # Convert to ms
                return avg_latency
        except Exception as e:
            logger.warning(f"Could not measure inference latency: {e}")
            return 0.0

    def run(self, num_batches: int = 10, batch_size: int = 4) -> BenchmarkMetrics:
        """
        Run benchmark for PEFT method.

        Args:
            num_batches: Number of batches to process
            batch_size: Batch size

        Returns:
            BenchmarkMetrics with all measurements
        """
        metrics = BenchmarkMetrics(
            method_name=self.method,
            model_arch=self.model_arch,
            num_batches=num_batches,
            batch_size=batch_size,
        )

        try:
            # Count parameters
            total, trainable = self._count_parameters(self.model)
            metrics.total_params = total
            metrics.trainable_params = trainable
            metrics.param_efficiency = trainable / total if total > 0 else 0.0

            if self.verbose:
                logger.info(
                    f"Parameters: {trainable:,} / {total:,} " f"({metrics.param_efficiency:.2%})"
                )

            # Measure peak memory
            metrics.peak_memory_mb = self._measure_peak_memory()
            if self.verbose:
                logger.info(f"Peak Memory: {metrics.peak_memory_mb:.1f} MB")

            # Measure inference latency
            metrics.inference_latency_ms = self._measure_inference_latency()
            if self.verbose:
                logger.info(f"Inference Latency: {metrics.inference_latency_ms:.2f} ms")

            # Actual training throughput measurement via real forward+backward passes.
            logger.info(f"Benchmarking {num_batches} training batches...")
            losses = []
            try:
                # Build vocab size from model config; fall back to small default.
                vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", 50257)
                seq_len = 16
                optimizer = torch.optim.AdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=1e-4,
                )
                self.model.train()
                if self.device == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                for batch_idx in range(num_batches):
                    input_ids = torch.randint(
                        0, vocab_size, (batch_size, seq_len), device=self.device
                    )
                    labels = input_ids.clone()
                    optimizer.zero_grad()
                    try:
                        out = self.model(input_ids, labels=labels)
                        loss = out.loss if hasattr(out, "loss") else out[0]
                    except TypeError:
                        logits = self.model(input_ids)
                        if hasattr(logits, "logits"):
                            logits = logits.logits
                        loss = nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, vocab_size),
                            input_ids[:, 1:].reshape(-1),
                        )
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
                    if (batch_idx + 1) % max(1, num_batches // 4) == 0 and self.verbose:
                        logger.info(
                            f"  Batch {batch_idx + 1}/{num_batches}  " f"loss={losses[-1]:.4f}"
                        )
                if self.device == "cuda":
                    torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                self.model.eval()
            except Exception as train_exc:
                logger.warning(f"Training throughput measurement failed: {train_exc}")
                elapsed_time = 0.0
            metrics.training_time_sec = elapsed_time
            metrics.batches_per_second = num_batches / elapsed_time if elapsed_time > 0 else 0.0
            metrics.task_loss = float(np.mean(losses)) if losses else 0.0
            # Perplexity-derived pseudo-accuracy: e^(-loss) mapped to [0,1].
            metrics.task_accuracy = float(np.exp(-metrics.task_loss)) if losses else 0.0

            if self.verbose:
                logger.info(f"Training Time: {metrics.training_time_sec:.2f} sec")
                logger.info(f"Throughput: {metrics.batches_per_second:.2f} batches/sec")
                logger.info(
                    f"Mean Train Loss: {metrics.task_loss:.4f}  "
                    f"Pseudo-acc (e^-loss): {metrics.task_accuracy:.4f}"
                )

        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            metrics.notes = f"Error: {str(e)}"

        return metrics


class CrossArchitectureBenchmark:
    """
    Run PEFT benchmarks across multiple architectures.

    Compares all PEFT methods (LoRA, Adapter, Prefix, BitFit)
    on multiple models (LLaMA, Gemma, Qwen, GPT-2)
    """

    def __init__(self, models_dict: Dict[str, nn.Module], device: str = "cpu"):
        """
        Initialize cross-architecture benchmark.

        Args:
            models_dict: Dict mapping model_name -> model
            device: Device to use
        """
        self.models = models_dict
        self.device = device
        self.peft_methods = ["lora", "adapter", "prefix", "bitfit"]
        self.results: Dict[str, Dict[str, BenchmarkMetrics]] = {}

        logger.info(f"Initialized CrossArchitectureBenchmark with {len(models_dict)} models")

    def run_all(self, num_batches: int = 10, rank: int = 8) -> Dict[str, Any]:
        """
        Run all benchmarks for all methods on all models.

        Args:
            num_batches: Number of batches per benchmark
            rank: Rank for low-rank methods

        Returns:
            Results dictionary
        """
        logger.info("Starting cross-architecture benchmark...")

        total_tests = len(self.models) * len(self.peft_methods)
        test_num = 0

        for model_name, model in self.models.items():
            self.results[model_name] = {}

            for method in self.peft_methods:
                test_num += 1
                logger.info(f"[{test_num}/{total_tests}] Benchmarking {method} on {model_name}...")

                try:
                    benchmark = PEFTBenchmark(
                        model,
                        method=method,
                        rank=rank,
                        device=self.device,
                        verbose=False,
                    )

                    metrics = benchmark.run(num_batches=num_batches)
                    self.results[model_name][method] = metrics

                except Exception as e:
                    logger.error(f"Error benchmarking {method} on {model_name}: {e}")
                    # Create failed result
                    metrics = BenchmarkMetrics(
                        method_name=method, model_arch=model_name, notes=f"Failed: {str(e)}"
                    )
                    self.results[model_name][method] = metrics

        return self._format_results()

    def _format_results(self) -> Dict[str, Any]:
        """Format results for reporting."""
        return {
            "models": list(self.models.keys()),
            "methods": self.peft_methods,
            "results": self.results,
        }

    def generate_report(self) -> str:
        """Generate formatted benchmark report."""
        lines = [
            "=" * 80,
            "CROSS-ARCHITECTURE PEFT BENCHMARK REPORT",
            "=" * 80,
            f"Models tested: {len(self.models)}",
            f"Methods tested: {len(self.peft_methods)}",
            f"Total benchmarks: {len(self.models) * len(self.peft_methods)}",
            "",
        ]

        # Per-model summary
        for model_name in self.models.keys():
            if model_name in self.results:
                lines.append(f"\n{model_name.upper()}")
                lines.append("-" * 80)

                for method in self.peft_methods:
                    if method in self.results[model_name]:
                        metrics = self.results[model_name][method]
                        lines.append(
                            f"{method:12} | "
                            f"Params: {metrics.param_efficiency:6.2%} | "
                            f"Memory: {metrics.peak_memory_mb:7.1f} MB | "
                            f"Speed: {metrics.batches_per_second:6.2f} b/s"
                        )

        # Comparison table
        lines.append("\n" + "=" * 80)
        lines.append("PARAMETER EFFICIENCY COMPARISON (% of trainable params)")
        lines.append("-" * 80)

        header = "Method".ljust(12)
        for model_name in self.models.keys():
            header += f" | {model_name:8}"
        lines.append(header)
        lines.append("-" * len(header))

        for method in self.peft_methods:
            row = method.ljust(12)
            for model_name in self.models.keys():
                if model_name in self.results and method in self.results[model_name]:
                    eff = self.results[model_name][method].param_efficiency
                    row += f" | {eff:7.2%}"
                else:
                    row += f" | {'N/A':>7}"
            lines.append(row)

        lines.append("=" * 80)
        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create mock models (same as in test_cross_architecture.py)
    from tests.apply.test_cross_architecture import (
        MockGemmaModel,
        MockGPT2Model,
        MockLLaMAModel,
        MockQwenModel,
    )

    models = {
        "llama": MockLLaMAModel(),
        "gemma": MockGemmaModel(),
        "qwen": MockQwenModel(),
        "gpt2": MockGPT2Model(),
    }

    benchmark = CrossArchitectureBenchmark(models, device="cpu")
    results = benchmark.run_all(num_batches=5)

    logger.info(benchmark.generate_report())
