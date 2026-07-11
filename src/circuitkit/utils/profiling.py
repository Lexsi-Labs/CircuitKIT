"""
Advanced profiling and performance monitoring utilities for CircuitKit.
Provides detailed performance analysis, memory tracking, and optimization recommendations.
"""

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import psutil
import torch
import torch.profiler

from circuitkit.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    name: str
    start_time: float
    end_time: float
    duration: float
    memory_start: float
    memory_end: float
    memory_peak: float
    cpu_percent: float
    gpu_memory_start: float = 0.0
    gpu_memory_end: float = 0.0
    gpu_memory_peak: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Advanced performance profiler."""

    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def start_profile(self, name: str, **kwargs) -> None:
        """
        Start profiling a section.

        Args:
            name: Name of the profile section
            **kwargs: Additional parameters to track
        """
        with self.lock:
            start_time = time.time()
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            gpu_memory_start = 0.0
            if torch.cuda.is_available():
                gpu_memory_start = torch.cuda.memory_allocated() / 1024 / 1024  # MB

            self.active_profiles[name] = {
                "start_time": start_time,
                "memory_start": memory_start,
                "gpu_memory_start": gpu_memory_start,
                "parameters": kwargs,
            }

    def end_profile(self, name: str) -> PerformanceMetrics:
        """
        End profiling a section.

        Args:
            name: Name of the profile section

        Returns:
            Performance metrics
        """
        with self.lock:
            if name not in self.active_profiles:
                raise ValueError(f"Profile '{name}' not found")

            profile_data = self.active_profiles[name]
            end_time = time.time()
            memory_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            gpu_memory_end = 0.0
            gpu_memory_peak = 0.0
            if torch.cuda.is_available():
                gpu_memory_end = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            metrics = PerformanceMetrics(
                name=name,
                start_time=profile_data["start_time"],
                end_time=end_time,
                duration=end_time - profile_data["start_time"],
                memory_start=profile_data["memory_start"],
                memory_end=memory_end,
                memory_peak=self._get_memory_peak(profile_data),
                cpu_percent=psutil.cpu_percent(),
                gpu_memory_start=profile_data["gpu_memory_start"],
                gpu_memory_end=gpu_memory_end,
                gpu_memory_peak=gpu_memory_peak,
                parameters=profile_data["parameters"],
            )

            self.metrics.append(metrics)
            del self.active_profiles[name]

            logger.info(
                f"Profile '{name}': {metrics.duration:.3f}s, "
                f"Memory: {metrics.memory_end - metrics.memory_start:.1f}MB, "
                f"GPU: {metrics.gpu_memory_end - metrics.gpu_memory_start:.1f}MB"
            )

            return metrics

    @contextmanager
    def profile(self, name: str, **kwargs):
        """Context manager for profiling."""
        self.start_profile(name, **kwargs)
        try:
            yield
        finally:
            self.end_profile(name)

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.metrics:
            return {}

        total_duration = sum(m.duration for m in self.metrics)
        total_memory = sum(m.memory_end - m.memory_start for m in self.metrics)
        total_gpu_memory = sum(m.gpu_memory_end - m.gpu_memory_start for m in self.metrics)

        return {
            "total_duration": total_duration,
            "total_memory_mb": total_memory,
            "total_gpu_memory_mb": total_gpu_memory,
            "num_profiles": len(self.metrics),
            "profiles": [
                {
                    "name": m.name,
                    "duration": m.duration,
                    "memory_delta": m.memory_end - m.memory_start,
                    "gpu_memory_delta": m.gpu_memory_end - m.gpu_memory_start,
                }
                for m in self.metrics
            ],
        }

    def save_report(self, file_path: Union[str, Path]) -> None:
        """Save profiling report to file."""
        report = {
            "summary": self.get_summary(),
            "detailed_metrics": [
                {
                    "name": m.name,
                    "start_time": m.start_time,
                    "end_time": m.end_time,
                    "duration": m.duration,
                    "memory_start": m.memory_start,
                    "memory_end": m.memory_end,
                    "memory_peak": m.memory_peak,
                    "cpu_percent": m.cpu_percent,
                    "gpu_memory_start": m.gpu_memory_start,
                    "gpu_memory_end": m.gpu_memory_end,
                    "gpu_memory_peak": m.gpu_memory_peak,
                    "parameters": m.parameters,
                }
                for m in self.metrics
            ],
        }

        with open(file_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Profiling report saved to {file_path}")


class MemoryMonitor:
    """Memory monitoring utilities."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history: List[Dict[str, float]] = []

    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")

    def _monitor_loop(self) -> None:
        """Memory monitoring loop."""
        while self.monitoring:
            try:
                # System memory
                system_memory = psutil.virtual_memory()
                process_memory = psutil.Process().memory_info()

                memory_data = {
                    "timestamp": time.time(),
                    "system_total": system_memory.total / 1024 / 1024,  # MB
                    "system_used": system_memory.used / 1024 / 1024,  # MB
                    "system_available": system_memory.available / 1024 / 1024,  # MB
                    "process_rss": process_memory.rss / 1024 / 1024,  # MB
                    "process_vms": process_memory.vms / 1024 / 1024,  # MB
                }

                # GPU memory
                if torch.cuda.is_available():
                    memory_data.update(
                        {
                            "gpu_allocated": torch.cuda.memory_allocated() / 1024 / 1024,  # MB
                            "gpu_reserved": torch.cuda.memory_reserved() / 1024 / 1024,  # MB
                            "gpu_max_allocated": torch.cuda.max_memory_allocated()
                            / 1024
                            / 1024,  # MB
                        }
                    )

                self.memory_history.append(memory_data)

                # Keep only last 1000 entries
                if len(self.memory_history) > 1000:
                    self.memory_history = self.memory_history[-1000:]

            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")

            time.sleep(self.interval)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        if not self.memory_history:
            return {}

        latest = self.memory_history[-1]
        peak_memory = max(entry["process_rss"] for entry in self.memory_history)
        peak_gpu = 0.0
        if torch.cuda.is_available():
            peak_gpu = max(entry.get("gpu_allocated", 0) for entry in self.memory_history)

        return {
            "current_memory_mb": latest["process_rss"],
            "peak_memory_mb": peak_memory,
            "current_gpu_mb": latest.get("gpu_allocated", 0),
            "peak_gpu_mb": peak_gpu,
            "monitoring_duration": len(self.memory_history) * self.interval,
        }

    def _get_memory_peak(self, profile_data: Dict[str, Any]) -> float:
        """Get the peak memory usage during profiling."""
        # Get current memory usage
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # If we have tracked peak during execution, use that
        if "memory_peak_tracked" in profile_data:
            return profile_data["memory_peak_tracked"]

        # Otherwise, estimate peak as max of start, end, and current
        start_memory = profile_data.get("memory_start", 0)
        end_memory = profile_data.get("memory_end", current_memory)

        # Estimate peak as the maximum observed
        estimated_peak = max(start_memory, end_memory, current_memory)

        # Add some buffer for peak estimation (10% buffer)
        return estimated_peak * 1.1


class OptimizationRecommender:
    """Optimization recommendation system."""

    def __init__(self):
        self.recommendations: List[str] = []

    def analyze_performance(
        self, profiler: PerformanceProfiler, memory_monitor: MemoryMonitor
    ) -> List[str]:
        """
        Analyze performance and provide optimization recommendations.

        Args:
            profiler: Performance profiler instance
            memory_monitor: Memory monitor instance

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        # Analyze profiling data
        summary = profiler.get_summary()
        if summary:
            if summary["total_duration"] > 300:  # 5 minutes
                recommendations.append(
                    "Consider using gradient checkpointing to reduce memory usage"
                )

            if summary["total_gpu_memory_mb"] > 8000:  # 8GB
                recommendations.append("Consider using mixed precision training (FP16/BF16)")

            if summary["total_memory_mb"] > 16000:  # 16GB
                recommendations.append("Consider using CPU offloading for large models")

        # Analyze memory usage
        memory_stats = memory_monitor.get_memory_stats()
        if memory_stats:
            if memory_stats["peak_memory_mb"] > 32000:  # 32GB
                recommendations.append("High memory usage detected - consider model sharding")

            if memory_stats["peak_gpu_mb"] > 20000:  # 20GB
                recommendations.append("High GPU memory usage - consider gradient accumulation")

        # Check for long-running operations
        long_operations = [m for m in profiler.metrics if m.duration > 60]  # 1 minute
        if long_operations:
            recommendations.append(
                f"Found {len(long_operations)} long-running operations - consider optimization"
            )

        self.recommendations = recommendations
        return recommendations

    def get_recommendations(self) -> List[str]:
        """Get current optimization recommendations."""
        return self.recommendations


class PyTorchProfiler:
    """PyTorch-specific profiler integration."""

    def __init__(self, output_dir: str = "./profiler_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    @contextmanager
    def profile_training(self, model, dataloader, num_steps: int = 10):
        """Profile training loop with PyTorch profiler."""
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            yield prof

    def save_trace(self, profiler, filename: str = "trace.json") -> None:
        """Save profiler trace to file."""
        trace_path = self.output_dir / filename
        profiler.export_chrome_trace(str(trace_path))
        logger.info(f"Profiler trace saved to {trace_path}")


# Global instances
performance_profiler = PerformanceProfiler()
memory_monitor = MemoryMonitor()
optimization_recommender = OptimizationRecommender()
pytorch_profiler = PyTorchProfiler()


def profile_function(func: Callable) -> Callable:
    """Decorator to profile a function."""

    def wrapper(*args, **kwargs):
        with performance_profiler.profile(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def start_performance_monitoring() -> None:
    """Start comprehensive performance monitoring."""
    memory_monitor.start_monitoring()
    logger.info("Performance monitoring started")


def stop_performance_monitoring() -> None:
    """Stop performance monitoring."""
    memory_monitor.stop_monitoring()
    logger.info("Performance monitoring stopped")


def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report."""
    return {
        "profiling": performance_profiler.get_summary(),
        "memory": memory_monitor.get_memory_stats(),
        "recommendations": optimization_recommender.get_recommendations(),
    }
