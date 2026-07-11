"""
Regression tests for the public API.

Ensures the restructuring (Phases 1–8) did not silently remove, rename, or
break any previously working public surface: the dict-config API, the flat
quick API, the Circuit class, applications exports, and the Phase 6/7
applications/__init__.py and deprecation-warning changes.
"""

import warnings

import pytest


# ---------------------------------------------------------------------------
# Flat quick API — all original functions still callable
# ---------------------------------------------------------------------------

class TestOriginalQuickAPICallable:
    NAMES = [
        "load_model",
        "discover",
        "faithfulness",
        "prune",
        "quantize",
        "export_checkpoint",
        "benchmark",
    ]

    @pytest.mark.parametrize("name", NAMES)
    def test_callable_via_package_root(self, name):
        import circuitkit as ck
        fn = getattr(ck, name, None)
        assert fn is not None, f"circuitkit.{name} is missing"
        assert callable(fn), f"circuitkit.{name} is not callable"


# ---------------------------------------------------------------------------
# Dict-config API — discover_circuit, evaluate_circuit, load_circuit
# ---------------------------------------------------------------------------

class TestDictAPICallable:
    NAMES = ["discover_circuit", "evaluate_circuit", "load_circuit"]

    @pytest.mark.parametrize("name", NAMES)
    def test_callable_via_package_root(self, name):
        import circuitkit as ck
        fn = getattr(ck, name, None)
        assert fn is not None, f"circuitkit.{name} is missing"
        assert callable(fn), f"circuitkit.{name} is not callable"

    def test_discover_circuit_importable_from_api(self):
        from circuitkit.api import discover_circuit
        assert callable(discover_circuit)

    def test_evaluate_circuit_importable_from_api(self):
        from circuitkit.api import evaluate_circuit
        assert callable(evaluate_circuit)

    def test_load_circuit_importable_from_api(self):
        from circuitkit.api import load_circuit
        assert callable(load_circuit)


# ---------------------------------------------------------------------------
# Circuit class — basic operations unchanged
# ---------------------------------------------------------------------------

class TestCircuitClassUnchanged:
    def test_constructible_with_node_list_and_scores(self):
        from circuitkit import Circuit
        c = Circuit(["A0.1", "MLP 3"], {"A0.1": 0.9, "MLP 3": 0.5})
        assert len(c) == 2

    def test_contains_operator(self):
        from circuitkit import Circuit
        c = Circuit(["A0.1", "MLP 3"], {"A0.1": 0.9, "MLP 3": 0.5})
        assert "A0.1" in c
        assert "X9.9" not in c

    def test_iter(self):
        from circuitkit import Circuit
        c = Circuit(["A0.1", "MLP 3"])
        assert list(c) == ["A0.1", "MLP 3"]

    def test_top_nodes(self):
        from circuitkit import Circuit
        c = Circuit(["A0.1", "MLP 3"], {"A0.1": 0.9, "MLP 3": 0.5})
        top = c.top_nodes(1)
        assert list(top.keys()) == ["A0.1"]

    def test_repr_is_string(self):
        from circuitkit import Circuit
        c = Circuit(["A0.1"])
        assert isinstance(repr(c), str)

    def test_save_and_from_artifact_roundtrip(self, tmp_path):
        import torch
        from circuitkit import Circuit
        c = Circuit(["A0.1", "MLP 3"], {"A0.1": 0.9, "MLP 3": 0.5})
        path = c.save(tmp_path / "circuit.pt")
        loaded = Circuit.from_artifact(path)
        assert list(loaded.nodes) == ["A0.1", "MLP 3"]


# ---------------------------------------------------------------------------
# applications/__init__.py — Phase 6 export restrictions
# ---------------------------------------------------------------------------

class TestApplicationsExports:
    def test_steering_not_exported(self):
        """steering was removed from applications.__all__ in Phase 6.

        Check __all__ (the export contract) rather than hasattr — importing the
        steering submodule elsewhere in the suite sets it as a package attribute,
        which would make a hasattr check spuriously fail.
        """
        from circuitkit import applications
        assert "steering" not in getattr(applications, "__all__", []), \
            "steering must not be in applications.__all__ in v1.0"

    def test_editing_not_exported(self):
        from circuitkit import applications
        assert "editing" not in getattr(applications, "__all__", []), \
            "editing must not be in applications.__all__ in v1.0"

    def test_finetuning_not_in_all(self):
        """finetuning is a subpackage on disk so hasattr() returns True,
        but it must not appear in __all__ (removed in Phase 6)."""
        from circuitkit import applications
        assert "finetuning" not in applications.__all__

    def test_selective_finetuning_not_in_all(self):
        """selective_finetuning is intentionally excluded from __all__
        (conditional import, star-import safety) per Phase 6 handoff."""
        from circuitkit import applications
        assert "selective_finetuning" not in applications.__all__

    def test_pruning_in_all(self):
        from circuitkit import applications
        assert "pruning" in applications.__all__

    def test_quantization_in_all(self):
        from circuitkit import applications
        assert "quantization" in applications.__all__

    def test_common_in_all(self):
        from circuitkit import applications
        assert "common" in applications.__all__

    def test_arch_registry_exports_present(self):
        """arch_registry re-exports are untouched per Phase 6 handoff."""
        from circuitkit import applications
        assert hasattr(applications, "get_model_family")
        assert hasattr(applications, "detect_model_architecture")

    def test_selective_finetuning_accessible_directly(self):
        """selective_finetuning is reachable via direct submodule import
        even though it's excluded from __all__."""
        import circuitkit.applications.selective_finetuning  # must not raise


# ---------------------------------------------------------------------------
# Phase 7 — Deprecation warnings on score_extractor.run_discovery()
# ---------------------------------------------------------------------------

class TestDeprecationWarnings:
    """Both score_extractor files must emit DeprecationWarning on run_discovery()."""

    def _assert_deprecation_on_run_discovery(self, module_path: str) -> None:
        """Helper: import the module and verify run_discovery raises DeprecationWarning.

        We call the function inside catch_warnings so we never actually run
        discovery — the DeprecationWarning is emitted at function entry, before
        any heavy computation, so we let it raise naturally and catch it.
        """
        import importlib
        mod = importlib.import_module(module_path)
        assert hasattr(mod, "run_discovery"), \
            f"{module_path} must expose run_discovery()"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                # Call with obviously wrong args — we only need the warning,
                # not a successful run.
                mod.run_discovery(None, None, None)
            except Exception:
                pass  # expected — we only care about the warning

        dep_warnings = [
            w for w in caught
            if issubclass(w.category, DeprecationWarning)
        ]
        assert dep_warnings, (
            f"run_discovery() in {module_path} must emit DeprecationWarning. "
            f"Caught: {[str(w.message) for w in caught]}"
        )

    def test_pruning_score_extractor_deprecation(self):
        self._assert_deprecation_on_run_discovery(
            "circuitkit.applications.pruning.score_extractor"
        )

    def test_quantization_score_extractor_deprecation(self):
        self._assert_deprecation_on_run_discovery(
            "circuitkit.applications.quantization.score_extractor"
        )


# ---------------------------------------------------------------------------
# New Phase 3 extensions still present
# ---------------------------------------------------------------------------

class TestPhase3Extensions:
    """load_scores, selective_finetune, visualize_circuit, Pipeline were added in
    Phase 3/4 — verify they remain intact as a regression guard."""

    NEW_NAMES = ["load_scores", "selective_finetune", "visualize_circuit", "Pipeline"]

    @pytest.mark.parametrize("name", NEW_NAMES)
    def test_new_name_reachable(self, name):
        import circuitkit as ck
        obj = getattr(ck, name, None)
        assert obj is not None, f"circuitkit.{name} is missing"
        assert callable(obj), f"circuitkit.{name} is not callable"

    def test_pipeline_is_correct_class(self):
        import circuitkit as ck
        from circuitkit.pipeline import Pipeline
        assert ck.Pipeline is Pipeline
