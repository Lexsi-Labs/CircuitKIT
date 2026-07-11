"""
Debugging and profiling utilities for CircuitKit.
"""

import gc
import json
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import psutil
import torch

from .logging import get_logger

logger = get_logger(__name__)


class PerformanceProfiler:
    """Performance profiler for tracking execution time and memory usage."""

    def __init__(self):
        self.measurements: Dict[str, Dict[str, Any]] = {}
        self.start_times: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, Dict[str, float]] = {}

    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self._take_memory_snapshot(f"{operation}_start")

    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            raise ValueError(f"Timer for '{operation}' was not started")

        duration = time.time() - self.start_times[operation]
        self._take_memory_snapshot(f"{operation}_end")

        # Calculate memory delta
        start_memory = self.memory_snapshots.get(f"{operation}_start", {})
        end_memory = self.memory_snapshots.get(f"{operation}_end", {})

        memory_delta = {
            "rss_delta": end_memory.get("rss", 0) - start_memory.get("rss", 0),
            "vms_delta": end_memory.get("vms", 0) - start_memory.get("vms", 0),
            "peak_delta": end_memory.get("peak", 0) - start_memory.get("peak", 0),
        }

        self.measurements[operation] = {
            "duration": duration,
            "memory_start": start_memory,
            "memory_end": end_memory,
            "memory_delta": memory_delta,
        }

        del self.start_times[operation]
        return duration

    def _take_memory_snapshot(self, name: str):
        """Take a memory snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()

        self.memory_snapshots[name] = {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "peak": memory_info.peak_wss / 1024 / 1024 if hasattr(memory_info, "peak_wss") else 0,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_duration = sum(m["duration"] for m in self.measurements.values())

        return {
            "total_duration": total_duration,
            "operations": self.measurements,
            "memory_peak": max(
                (m["memory_end"]["rss"] for m in self.measurements.values()), default=0
            ),
        }

    def save_report(self, filepath: str):
        """Save performance report to file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "measurements": self.measurements,
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report saved to {filepath}")


def profile_function(operation_name: Optional[str] = None):
    """Decorator to profile function execution."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"

            profiler = PerformanceProfiler()
            profiler.start_timer(name)

            try:
                result = func(*args, **kwargs)
                duration = profiler.end_timer(name)
                logger.log_performance(name, duration)
                return result
            except Exception as e:
                profiler.end_timer(name)
                logger.error(f"Error in {name}", exception=e)
                raise

        return wrapper

    return decorator


@contextmanager
def profile_operation(operation_name: str):
    """Context manager for profiling operations."""
    profiler = PerformanceProfiler()
    profiler.start_timer(operation_name)

    try:
        yield profiler
    finally:
        duration = profiler.end_timer(operation_name)
        logger.log_performance(operation_name, duration)


class MemoryTracker:
    """Track memory usage and detect leaks."""

    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
        self.torch_memory_snapshots: List[Dict[str, Any]] = []

    def take_snapshot(self, name: str):
        """Take a memory snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()

        snapshot = {
            "name": name,
            "timestamp": time.time(),
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
        }

        # PyTorch memory info
        if torch.cuda.is_available():
            snapshot["torch_cuda_allocated"] = torch.cuda.memory_allocated() / 1024 / 1024
            snapshot["torch_cuda_reserved"] = torch.cuda.memory_reserved() / 1024 / 1024

        self.snapshots.append(snapshot)
        logger.debug(f"Memory snapshot taken: {name}", **snapshot)

    def detect_memory_leak(self, threshold_mb: float = 100) -> bool:
        """Detect potential memory leak."""
        if len(self.snapshots) < 2:
            return False

        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]

        memory_increase = last_snapshot["rss_mb"] - first_snapshot["rss_mb"]

        if memory_increase > threshold_mb:
            logger.warning(f"Potential memory leak detected: {memory_increase:.2f}MB increase")
            return True

        return False

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.snapshots:
            return {"error": "No snapshots available"}

        first = self.snapshots[0]
        last = self.snapshots[-1]

        return {
            "total_snapshots": len(self.snapshots),
            "memory_increase_mb": last["rss_mb"] - first["rss_mb"],
            "peak_memory_mb": max(s["rss_mb"] for s in self.snapshots),
            "current_memory_mb": last["rss_mb"],
            "torch_cuda_allocated_mb": last.get("torch_cuda_allocated", 0),
            "torch_cuda_reserved_mb": last.get("torch_cuda_reserved", 0),
        }


class Debugger:
    """Main debugging class for CircuitKit."""

    def __init__(self, enable_profiling: bool = True, enable_memory_tracking: bool = True):
        self.enable_profiling = enable_profiling
        self.enable_memory_tracking = enable_memory_tracking

        self.profiler = PerformanceProfiler() if enable_profiling else None
        self.memory_tracker = MemoryTracker() if enable_memory_tracking else None

        self.debug_info: Dict[str, Any] = {}
        self.checkpoints: List[Dict[str, Any]] = []

    def set_debug_info(self, **kwargs):
        """Set debug information."""
        self.debug_info.update(kwargs)
        logger.debug("Debug info updated", **kwargs)

    def add_checkpoint(self, name: str, **data):
        """Add a debug checkpoint."""
        checkpoint = {
            "name": name,
            "timestamp": time.time(),
            "data": data,
            "debug_info": self.debug_info.copy(),
        }

        if self.memory_tracker:
            self.memory_tracker.take_snapshot(f"checkpoint_{name}")

        self.checkpoints.append(checkpoint)
        logger.debug(f"Checkpoint added: {name}", **data)

    def get_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint by name."""
        for checkpoint in self.checkpoints:
            if checkpoint["name"] == name:
                return checkpoint
        return None

    def get_all_checkpoints(self) -> List[Dict[str, Any]]:
        """Get all checkpoints."""
        return self.checkpoints.copy()

    def clear_checkpoints(self):
        """Clear all checkpoints."""
        self.checkpoints.clear()
        logger.debug("All checkpoints cleared")

    def start_operation(self, operation: str):
        """Start profiling an operation."""
        if self.profiler:
            self.profiler.start_timer(operation)

        if self.memory_tracker:
            self.memory_tracker.take_snapshot(f"{operation}_start")

    def end_operation(self, operation: str) -> float:
        """End profiling an operation."""
        duration = 0

        if self.profiler:
            duration = self.profiler.end_timer(operation)

        if self.memory_tracker:
            self.memory_tracker.take_snapshot(f"{operation}_end")

        return duration

    def get_debug_report(self) -> Dict[str, Any]:
        """Get comprehensive debug report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "debug_info": self.debug_info,
            "checkpoints": self.checkpoints,
        }

        if self.profiler:
            report["performance"] = self.profiler.get_summary()

        if self.memory_tracker:
            report["memory"] = self.memory_tracker.get_memory_summary()

        return report

    def save_debug_report(self, filepath: str):
        """Save debug report to file."""
        report = self.get_debug_report()

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Debug report saved to {filepath}")


# Global debugger instance
debugger = Debugger()


def debug_operation(operation_name: str):
    """Decorator for debugging operations."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            debugger.start_operation(operation_name)

            try:
                result = func(*args, **kwargs)
                duration = debugger.end_operation(operation_name)
                logger.log_performance(operation_name, duration)
                return result
            except Exception as e:
                debugger.end_operation(operation_name)
                logger.error(f"Error in {operation_name}", exception=e)
                raise

        return wrapper

    return decorator


@contextmanager
def debug_context(operation_name: str, **debug_data):
    """Context manager for debugging operations."""
    debugger.set_debug_info(**debug_data)
    debugger.start_operation(operation_name)

    try:
        yield debugger
    finally:
        debugger.end_operation(operation_name)


def enable_torch_debugging():
    """Enable PyTorch debugging features."""
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = False
    logger.info("PyTorch debugging enabled")


def disable_torch_debugging():
    """Disable PyTorch debugging features."""
    torch.autograd.set_detect_anomaly(False)
    torch.backends.cudnn.benchmark = True
    logger.info("PyTorch debugging disabled")


def cleanup_memory():
    """Clean up memory and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("Memory cleanup completed")
