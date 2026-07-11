"""
CircuitKit: A comprehensive toolkit for circuit discovery in transformer models.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Pratinav Seth, Hem Gosalia, Aditya Kasliwal, Vinay Kumar Sankarapu"
__description__ = "Unified Discover, Evaluate, Intervene toolkit for mechanistic interpretability"
_API_EXPORTS = {"discover_circuit", "evaluate_circuit", "load_circuit"}

# Flat front-door API (circuitkit.quick) — lazily imported so `import circuitkit`
# stays fast and torch-free until one of these is actually accessed.
_QUICK_EXPORTS = {
    "load_model",
    "discover",
    "faithfulness",
    "prune",
    "quantize",
    "export_checkpoint",
    "benchmark",
    "load_scores",
    "selective_finetune",
    "visualize_circuit",
}
_CIRCUIT_EXPORTS = {"Circuit"}
_PIPELINE_EXPORTS = {"Pipeline"}

# Main exports
__all__ = [
    # Core dict-config API
    "discover_circuit",
    "evaluate_circuit",
    "load_circuit",
    # Flat front-door API
    "load_model",
    "discover",
    "faithfulness",
    "prune",
    "quantize",
    "export_checkpoint",
    "benchmark",
    "Circuit",
    "load_scores",
    "selective_finetune",
    "visualize_circuit",
    "Pipeline",
    # Task management
    "get_task",
    "list_tasks",
    "register_task",
    # Version info
    "__version__",
    "__author__",
    "__description__",
]


def __getattr__(name):
    """Lazily import heavy API helpers when accessed from the package root."""
    if name in _API_EXPORTS:
        from . import api

        value = getattr(api, name)
        globals()[name] = value
        return value
    if name in _QUICK_EXPORTS:
        from . import quick

        value = getattr(quick, name)
        globals()[name] = value
        return value
    if name in _CIRCUIT_EXPORTS:
        from .circuit import Circuit

        globals()["Circuit"] = Circuit
        return Circuit
    if name in _PIPELINE_EXPORTS:
        from .pipeline import Pipeline

        globals()["Pipeline"] = Pipeline
        return Pipeline
    if name == "visualize":
        # Deprecated pre-1.0 alias (H1 in the 1.0.0 audit): ``visualize`` was
        # renamed to ``visualize_circuit`` in 1.0.0 and is now a subpackage.
        # Return the (callable) subpackage so ``ck.visualize(...)`` keeps working
        # and warns on use — see circuitkit/visualize/__init__.py. The warning
        # fires on the call, not on attribute access, so it survives the
        # submodule being imported (which shadows any parent __getattr__ shim).
        from . import visualize as _visualize

        return _visualize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose lazily-imported names for autocomplete / dir()."""
    return sorted(set(globals()) | set(__all__))


def _ensure_builtin_tasks():
    """Register built-in tasks only when task helpers are used."""
    from .tasks.bootstrap import _bootstrap_builtin_tasks

    return _bootstrap_builtin_tasks()


def get_task(name):
    """Get a built-in or registered task by name."""
    _ensure_builtin_tasks()
    from .tasks.registry import get_task as _get_task

    return _get_task(name)


def list_tasks():
    """List registered task names."""
    _ensure_builtin_tasks()
    from .tasks.registry import list_tasks as _list_tasks

    return _list_tasks()


def register_task(spec):
    """Register a custom task specification."""
    _ensure_builtin_tasks()
    from .tasks.registry import register_task as _register_task

    return _register_task(spec)
