"""
Debugging utilities for CircuitKit.
"""

import json
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable

import psutil
import torch

from .logging import get_logger

logger = get_logger("circuitkit.debug")


class Debugger:
    """Main debugging class for CircuitKit."""

    def __init__(self, enabled: bool = True, log_level: str = "DEBUG"):
        self.enabled = enabled
        self.log_level = log_level
        self.logger = get_logger("debug")
        self.breakpoints = set()
        self.watch_vars = {}
        self.call_stack = []
        self.performance_data = {}

    def debug_print(self, message: str, **kwargs):
        """Print debug message if debugging is enabled."""
        if self.enabled:
            self.logger.debug(f"DEBUG: {message}", **kwargs)

    def set_breakpoint(self, function_name: str):
        """Set a breakpoint on a function."""
        self.breakpoints.add(function_name)
        self.debug_print(f"Breakpoint set on function: {function_name}")

    def clear_breakpoint(self, function_name: str):
        """Clear a breakpoint."""
        self.breakpoints.discard(function_name)
        self.debug_print(f"Breakpoint cleared on function: {function_name}")

    def watch_variable(self, name: str, value: Any):
        """Watch a variable for changes."""
        self.watch_vars[name] = {
            "value": value,
            "type": type(value).__name__,
            "timestamp": time.time(),
        }
        self.debug_print(f"Watching variable: {name} = {value}")

    def check_watch_vars(self, **vars):
        """Check watched variables for changes."""
        for name, value in vars.items():
            if name in self.watch_vars:
                old_value = self.watch_vars[name]["value"]
                if old_value != value:
                    self.debug_print(f"Variable {name} changed: {old_value} -> {value}")
                    self.watch_vars[name]["value"] = value
                    self.watch_vars[name]["timestamp"] = time.time()

    def log_call_stack(self, max_depth: int = 10):
        """Log the current call stack."""
        if self.enabled:
            stack = traceback.extract_stack()
            self.debug_print("Call stack:")
            for i, frame in enumerate(stack[-max_depth:]):
                self.debug_print(f"  {i}: {frame.filename}:{frame.lineno} in {frame.name}()")

    def log_memory_usage(self):
        """Log current memory usage."""
        if self.enabled:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.debug_print(
                "Memory usage:",
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
            )

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                self.debug_print(
                    "GPU memory usage:", allocated_mb=gpu_memory, max_allocated_mb=gpu_max_memory
                )

    def log_tensor_info(self, tensor: torch.Tensor, name: str = "tensor"):
        """Log detailed tensor information."""
        if self.enabled and isinstance(tensor, torch.Tensor):
            self.debug_print(
                f"Tensor {name}:",
                shape=tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
                requires_grad=tensor.requires_grad,
                memory_mb=tensor.element_size() * tensor.nelement() / 1024 / 1024,
            )

    def log_model_info(self, model):
        """Log model information."""
        if self.enabled:
            if hasattr(model, "cfg"):
                self.debug_print("Model config:", config=model.cfg)
            if hasattr(model, "parameters"):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.debug_print(
                    "Model parameters:", total=total_params, trainable=trainable_params
                )

    def profile_function(self, func: Callable, *args, **kwargs):
        """Profile a function's execution."""
        if not self.enabled:
            return func(*args, **kwargs)

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            self.debug_print(
                f"Function {func.__name__} profiled:",
                execution_time=f"{end_time - start_time:.3f}s",
                memory_delta_mb=(end_memory - start_memory) / 1024 / 1024,
            )

            return result
        except Exception as e:
            end_time = time.time()
            self.debug_print(
                f"Function {func.__name__} failed after {end_time - start_time:.3f}s: {e}"
            )
            raise

    def save_debug_info(self, filepath: str):
        """Save debug information to file."""
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "breakpoints": list(self.breakpoints),
            "watch_vars": {
                k: {"value": str(v["value"]), "type": v["type"]} for k, v in self.watch_vars.items()
            },
            "performance_data": self.performance_data,
            "system_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            },
        }

        with open(filepath, "w") as f:
            json.dump(debug_info, f, indent=2)

        self.debug_print(f"Debug info saved to: {filepath}")


# Global debugger instance
debugger = Debugger()


def debug_function(func: Callable) -> Callable:
    """Decorator to add debugging to a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if debugger.enabled:
            debugger.debug_print(f"Entering function: {func.__name__}")
            debugger.log_call_stack()
            debugger.log_memory_usage()

            # Check for breakpoints
            if func.__name__ in debugger.breakpoints:
                debugger.debug_print(f"BREAKPOINT HIT: {func.__name__}")
                # In a real debugger, this would pause execution
                input("Press Enter to continue...")

        try:
            result = debugger.profile_function(func, *args, **kwargs)
            if debugger.enabled:
                debugger.debug_print(f"Exiting function: {func.__name__}")
            return result
        except Exception as e:
            if debugger.enabled:
                debugger.debug_print(f"Exception in {func.__name__}: {e}")
                debugger.log_call_stack()
            raise

    return wrapper


@contextmanager
def debug_context(operation: str, **context_vars):
    """Context manager for debugging operations."""
    if debugger.enabled:
        debugger.debug_print(f"Starting operation: {operation}")
        debugger.log_memory_usage()

        # Watch context variables
        for name, value in context_vars.items():
            debugger.watch_variable(name, value)

    try:
        yield debugger
    except Exception as e:
        if debugger.enabled:
            debugger.debug_print(f"Operation {operation} failed: {e}")
            debugger.log_call_stack()
        raise
    finally:
        if debugger.enabled:
            debugger.debug_print(f"Completed operation: {operation}")
            debugger.log_memory_usage()


def debug_tensor(tensor: torch.Tensor, name: str = "tensor"):
    """Debug a tensor with detailed information."""
    if debugger.enabled:
        debugger.log_tensor_info(tensor, name)

        # Check for common issues
        if torch.isnan(tensor).any():
            debugger.debug_print(f"WARNING: {name} contains NaN values")
        if torch.isinf(tensor).any():
            debugger.debug_print(f"WARNING: {name} contains Inf values")
        if tensor.grad is not None and torch.isnan(tensor.grad).any():
            debugger.debug_print(f"WARNING: {name}.grad contains NaN values")


def debug_model(model, name: str = "model"):
    """Debug a model with detailed information."""
    if debugger.enabled:
        debugger.log_model_info(model)

        # Check for common issues
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                debugger.debug_print(f"WARNING: Parameter {name} contains NaN values")
            if torch.isinf(param).any():
                debugger.debug_print(f"WARNING: Parameter {name} contains Inf values")


def debug_gradient_flow(model):
    """Debug gradient flow in a model."""
    if debugger.enabled:
        debugger.debug_print("Checking gradient flow...")

        total_norm = 0
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                debugger.debug_print(f"Gradient norm for {name}: {param_norm:.6f}")
            else:
                debugger.debug_print(f"No gradient for {name}")

        total_norm = total_norm ** (1.0 / 2)
        debugger.debug_print(f"Total gradient norm: {total_norm:.6f}")
        debugger.debug_print(f"Parameters with gradients: {param_count}")


def debug_memory_leaks():
    """Debug for potential memory leaks."""
    if debugger.enabled:
        debugger.debug_print("Checking for memory leaks...")

        # Check GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            debugger.debug_print(
                f"GPU memory - Allocated: {allocated / 1024 / 1024:.2f}MB, "
                f"Cached: {cached / 1024 / 1024:.2f}MB"
            )

        # Check CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        debugger.debug_print(
            f"CPU memory - RSS: {memory_info.rss / 1024 / 1024:.2f}MB, "
            f"VMS: {memory_info.vms / 1024 / 1024:.2f}MB"
        )


def enable_debugging(enabled: bool = True):
    """Enable or disable debugging."""
    debugger.enabled = enabled
    logger.info(f"Debugging {'enabled' if enabled else 'disabled'}")


def set_debug_level(level: str):
    """Set debug level."""
    debugger.log_level = level
    logger.info(f"Debug level set to: {level}")


def save_debug_session(filepath: str):
    """Save current debug session."""
    debugger.save_debug_info(filepath)
    logger.info(f"Debug session saved to: {filepath}")


# Convenience functions
def debug_print(message: str, **kwargs):
    """Print debug message."""
    debugger.debug_print(message, **kwargs)


def set_breakpoint(function_name: str):
    """Set a breakpoint."""
    debugger.set_breakpoint(function_name)


def watch_variable(name: str, value: Any):
    """Watch a variable."""
    debugger.watch_variable(name, value)


def log_call_stack():
    """Log call stack."""
    debugger.log_call_stack()


def log_memory_usage():
    """Log memory usage."""
    debugger.log_memory_usage()


def profile_function(func: Callable):
    """Profile a function."""
    return debugger.profile_function(func)
