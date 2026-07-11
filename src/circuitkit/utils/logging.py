"""
Comprehensive logging utilities for CircuitKit.
"""

import json
import logging
import sys
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional


# Suppress common warnings that clutter output
def configure_warning_filters():
    """Configure warning filters to reduce noise."""
    # Suppress TransformerLens precision warnings
    warnings.filterwarnings("ignore", message=".*reduced precision.*")
    warnings.filterwarnings("ignore", message=".*from_pretrained_no_processing.*")

    # Suppress lm-eval model warnings
    warnings.filterwarnings("ignore", message=".*pretrained.*model kwarg is not of type.*")
    warnings.filterwarnings("ignore", message=".*Passed an already-initialized model.*")
    warnings.filterwarnings("ignore", message=".*Overwriting default num_fewshot.*")

    # Suppress IOI dataset warnings (these are expected)
    warnings.filterwarnings("ignore", message=".*S2 index has been computed.*")
    warnings.filterwarnings("ignore", message=".*Some groups have less than 5 prompts.*")

    # Suppress common torch warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="circuitkit.data")

    # Suppress via logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("lm_eval").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)

    # Suppress root logger warnings from TransformerLens
    logging.getLogger().setLevel(logging.ERROR)


class CircuitKitFormatter(logging.Formatter):
    """Custom formatter for cleaner CircuitKit logs."""

    # Color codes for terminal
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BOLD": "\033[1m",  # Bold
    }

    # Icons for different message types
    ICONS = {
        "step": "→",
        "complete": "✓",
        "error": "✗",
        "performance": "⏱",
        "model": "🔧",
        "config": "⚙",
        "start": "▶",
        "info": "•",
    }

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record):
        # Extract message
        msg = record.getMessage()

        # Determine icon and formatting
        icon = self.ICONS["info"]
        color = self.COLORS["INFO"] if self.use_colors else ""
        reset = self.COLORS["RESET"] if self.use_colors else ""
        self.COLORS["BOLD"] if self.use_colors else ""

        if "Starting operation" in msg:
            icon = self.ICONS["start"]
            color = self.COLORS["INFO"] if self.use_colors else ""
        elif "Step" in msg:
            icon = self.ICONS["step"]
        elif "Completed" in msg or "complete" in msg.lower():
            icon = self.ICONS["complete"]
        elif "Performance" in msg:
            icon = self.ICONS["performance"]
        elif "Model" in msg:
            icon = self.ICONS["model"]
        elif "Config" in msg:
            icon = self.ICONS["config"]
        elif record.levelno >= logging.ERROR:
            icon = self.ICONS["error"]
            color = self.COLORS["ERROR"] if self.use_colors else ""
        elif record.levelno >= logging.WARNING:
            color = self.COLORS["WARNING"] if self.use_colors else ""

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")

        # Clean up the message - remove verbose JSON context for console
        if "| Context:" in msg:
            msg = msg.split("| Context:")[0].strip()

        # Format final message
        formatted = f"{color}{timestamp} {icon} {msg}{reset}"

        return formatted


class CircuitKitLogger:
    """Enhanced logger for CircuitKit with structured logging capabilities."""

    def __init__(self, name: str = "circuitkit", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self._context = {}

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self.logger.setLevel(level)
            self._setup_handlers()

        else:
            # MODIFY: If handlers exist (e.g., from the global logger), ensure we update their levels
            self.setLevel(level)

    def _setup_handlers(self):
        """Setup console and file handlers."""
        # Avoid propagating to parent loggers to prevent duplicate logs
        self.logger.propagate = False

        # Console handler with custom formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.logger.level)
        console_handler.setFormatter(CircuitKitFormatter(use_colors=True))
        self.logger.addHandler(console_handler)

        # File handler for detailed logs (with JSON context)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"circuitkit_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def setLevel(self, level: int):
        """Update the level for the logger and its console handler."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            # Only update the console stream handler, keep the file handler at DEBUG
            if (
                isinstance(handler, logging.StreamHandler)
                and getattr(handler, "stream", None) == sys.stdout
            ):
                handler.setLevel(level)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        self._log_with_context(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        self._log_with_context(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with optional context."""
        self._log_with_context(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message with optional context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)

    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with additional context."""
        if kwargs:
            context = json.dumps(kwargs, default=str)
            message = f"{message} | Context: {context}"
        self.logger.log(level, message)

    def log_function_call(self, func_name: str, args: tuple, kwargs: dict, result: Any = None):
        """Log function call details."""
        self.debug(
            f"Function call: {func_name}",
            args=str(args)[:200],
            kwargs=str(kwargs)[:200],
            result_type=type(result).__name__ if result is not None else None,
        )

    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        # Simple format for console
        self.info(f"{operation}: {duration:.2f}s")

    def log_model_info(self, model_name: str, **model_details):
        """Log model information."""
        params = model_details.get("parameters", 0)
        if params > 1e9:
            params_str = f"{params/1e9:.1f}B"
        elif params > 1e6:
            params_str = f"{params/1e6:.1f}M"
        else:
            params_str = f"{params:,}"
        self.info(
            f"Model: {model_name} ({params_str} params, {model_details.get('device', 'unknown')} device)"
        )

    def log_config(self, config: Dict[str, Any]):
        """Log configuration details - simplified for console."""
        algo = config.get("discovery", {}).get("algorithm", "unknown")
        task = config.get("discovery", {}).get("task", "unknown")
        level = config.get("discovery", {}).get("level", "node")
        sparsity = config.get("pruning", {}).get("target_sparsity", 0)
        self.info(f"Config: {algo.upper()} on {task} task, {level} level, {sparsity:.0%} sparsity")

    def log_error_with_traceback(self, message: str, exception: Exception):
        """Log error with full traceback."""
        self.error(f"{message}: {str(exception)}")
        self.debug("Full traceback:", traceback=traceback.format_exc())


# Global logger instance
logger = CircuitKitLogger()

# Configure warning filters on module import
configure_warning_filters()


def get_logger(name: Optional[str] = None) -> CircuitKitLogger:
    """Get logger instance."""
    if name:
        return CircuitKitLogger(name)
    return logger


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logger = CircuitKitLogger(level=level)

    # Reconfigure warning filters
    configure_warning_filters()

    if log_file:
        # Add custom file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.logger.addHandler(file_handler)

    return logger


@contextmanager
def log_execution_time(operation: str, logger: Optional[CircuitKitLogger] = None):
    """Context manager to log execution time."""
    if logger is None:
        logger = get_logger()

    start_time = datetime.now()

    try:
        yield
        duration = (datetime.now() - start_time).total_seconds()
        logger.log_performance(operation, duration)
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Failed: {operation} (took {duration:.3f}s)", error=str(e))
        raise


def log_function_calls(logger: Optional[CircuitKitLogger] = None):
    """Decorator to log function calls."""
    if logger is None:
        logger = get_logger()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log_function_call(func.__name__, args, kwargs)
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.log_error_with_traceback(f"Function {func.__name__} failed", e)
                raise

        return wrapper

    return decorator


class ProgressLogger:
    """Logger for progress tracking with structured output."""

    def __init__(self, logger: Optional[CircuitKitLogger] = None):
        self.logger = logger or get_logger()
        self.steps = []
        self.current_step = 0
        self.start_time = None

    def start_operation(self, operation: str, total_steps: int = 1):
        """Start a new operation."""
        self.operation = operation
        self.total_steps = total_steps
        self.current_step = 0
        self.steps = []
        self.start_time = datetime.now()
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Starting: {operation}")
        self.logger.info(f"{'='*50}")

    def step(self, step_name: str, **context):
        """Log a step in the operation."""
        self.current_step += 1
        self.steps.append(step_name)
        # Format context nicely if present
        if context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
            self.logger.info(f"[{self.current_step}/{self.total_steps}] {step_name} ({ctx_str})")
        else:
            self.logger.info(f"[{self.current_step}/{self.total_steps}] {step_name}")

    def complete(self, **summary):
        """Complete the operation."""
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        summary_str = ", ".join(f"{k}={v}" for k, v in summary.items()) if summary else ""
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Completed: {self.operation} in {duration:.1f}s")
        if summary_str:
            self.logger.info(f"Summary: {summary_str}")
        self.logger.info(f"{'='*50}")

    def fail(self, error: str, **context):
        """Log operation failure."""
        self.logger.error(f"Operation failed: {self.operation}")
        self.logger.error(f"Error: {error}")


# Convenience functions
def debug(message: str, **kwargs):
    """Log debug message."""
    logger.debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log info message."""
    logger.info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message."""
    logger.warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log error message."""
    logger.error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log critical message."""
    logger.critical(message, **kwargs)
