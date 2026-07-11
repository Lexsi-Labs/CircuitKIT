"""
Unit tests for circuitkit/__init__.py lazy loading and public exports.

Verifies that the new names (load_scores, selective_finetune, visualize_circuit,
Pipeline) appear in dir() and __all__, that lazy imports resolve to the
correct objects, and that __getattr__ raises AttributeError for unknown names.

All tests are pure import-time checks — no model loading, no GPU.
"""

import pytest


# ---------------------------------------------------------------------------
# Public surface — dir() and __all__
# ---------------------------------------------------------------------------

class TestPublicSurface:
    """Verify that new and existing names are reachable via the package root."""

    NEW_NAMES = ["load_scores", "selective_finetune", "visualize_circuit", "Pipeline"]
    ORIGINAL_QUICK_NAMES = [
        "load_model", "discover", "faithfulness",
        "prune", "quantize", "export_checkpoint", "benchmark",
    ]
    DICT_API_NAMES = ["discover_circuit", "evaluate_circuit", "load_circuit"]
    TASK_HELPER_NAMES = ["get_task", "list_tasks", "register_task"]

    def test_new_names_in_dir(self):
        import circuitkit
        names = dir(circuitkit)
        for name in self.NEW_NAMES:
            assert name in names, f"{name!r} missing from dir(circuitkit)"

    def test_new_names_in_all(self):
        import circuitkit
        for name in self.NEW_NAMES:
            assert name in circuitkit.__all__, f"{name!r} missing from circuitkit.__all__"

    def test_original_quick_names_still_in_dir(self):
        import circuitkit
        names = dir(circuitkit)
        for name in self.ORIGINAL_QUICK_NAMES:
            assert name in names, f"Regression: {name!r} missing from dir(circuitkit)"

    def test_dict_api_names_in_dir(self):
        import circuitkit
        names = dir(circuitkit)
        for name in self.DICT_API_NAMES:
            assert name in names, f"Regression: {name!r} missing from dir(circuitkit)"

    def test_task_helpers_in_dir(self):
        import circuitkit
        names = dir(circuitkit)
        for name in self.TASK_HELPER_NAMES:
            assert name in names, f"{name!r} missing from dir(circuitkit)"

    def test_circuit_in_dir(self):
        import circuitkit
        assert "Circuit" in dir(circuitkit)

    def test_circuit_in_all(self):
        import circuitkit
        assert "Circuit" in circuitkit.__all__


# ---------------------------------------------------------------------------
# Version and package metadata
# ---------------------------------------------------------------------------

class TestPackageMetadata:
    def test_version_is_string(self):
        import circuitkit
        assert isinstance(circuitkit.__version__, str)

    def test_version_is_1_0_0(self):
        import circuitkit
        assert circuitkit.__version__ == "1.0.0"

    def test_author_is_string(self):
        import circuitkit
        assert isinstance(circuitkit.__author__, str)
        assert len(circuitkit.__author__) > 0


# ---------------------------------------------------------------------------
# Lazy import correctness — object identity
# ---------------------------------------------------------------------------

class TestLazyImportIdentity:
    """Accessing a name via the package root must return the canonical object."""

    def test_pipeline_identity(self):
        import circuitkit
        from circuitkit.pipeline import Pipeline
        assert circuitkit.Pipeline is Pipeline

    def test_circuit_identity(self):
        import circuitkit
        from circuitkit.circuit import Circuit
        assert circuitkit.Circuit is Circuit

    def test_load_scores_identity(self):
        import circuitkit
        from circuitkit.quick import load_scores
        assert circuitkit.load_scores is load_scores

    def test_visualize_identity(self):
        import circuitkit
        from circuitkit.quick import visualize_circuit
        assert circuitkit.visualize_circuit is visualize_circuit

    def test_selective_finetune_identity(self):
        import circuitkit
        from circuitkit.quick import selective_finetune
        assert circuitkit.selective_finetune is selective_finetune

    def test_discover_circuit_identity(self):
        import circuitkit
        from circuitkit.api import discover_circuit
        assert circuitkit.discover_circuit is discover_circuit

    def test_load_circuit_identity(self):
        import circuitkit
        from circuitkit.api import load_circuit
        assert circuitkit.load_circuit is load_circuit


# ---------------------------------------------------------------------------
# Lazy import callability
# ---------------------------------------------------------------------------

class TestLazyImportCallable:
    """Names resolved via __getattr__ must be callable functions/classes."""

    @pytest.mark.parametrize("name", [
        "load_model", "discover", "faithfulness", "prune",
        "quantize", "export_checkpoint", "benchmark",
        "load_scores", "selective_finetune", "visualize_circuit",
        "discover_circuit", "evaluate_circuit", "load_circuit",
    ])
    def test_name_is_callable(self, name):
        import circuitkit
        obj = getattr(circuitkit, name)
        assert callable(obj), f"circuitkit.{name} is not callable"

    def test_pipeline_is_a_class(self):
        import circuitkit
        import inspect
        assert inspect.isclass(circuitkit.Pipeline)

    def test_circuit_is_a_class(self):
        import circuitkit
        import inspect
        assert inspect.isclass(circuitkit.Circuit)


# ---------------------------------------------------------------------------
# __getattr__ for unknown names
# ---------------------------------------------------------------------------

class TestGetAttrUnknown:
    def test_unknown_name_raises_attribute_error(self):
        import circuitkit
        with pytest.raises(AttributeError):
            _ = circuitkit.this_name_does_not_exist_xyz

    def test_attribute_error_mentions_module_name(self):
        import circuitkit
        with pytest.raises(AttributeError, match="circuitkit"):
            _ = circuitkit.totally_bogus_name_abc


# ---------------------------------------------------------------------------
# Task helpers are functional (light smoke)
# ---------------------------------------------------------------------------

class TestTaskHelpers:
    def test_list_tasks_returns_list(self):
        import circuitkit
        tasks = circuitkit.list_tasks()
        assert isinstance(tasks, list)

    def test_list_tasks_contains_ioi(self):
        """ioi is a built-in task and must always be registered."""
        import circuitkit
        assert "ioi" in circuitkit.list_tasks()

    def test_get_task_ioi(self):
        import circuitkit
        task = circuitkit.get_task("ioi")
        assert task is not None

    def test_get_task_unknown_raises(self):
        import circuitkit
        with pytest.raises(Exception):  # KeyError or ValueError depending on registry impl
            circuitkit.get_task("nonexistent_task_xyz_abc")
