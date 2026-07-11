"""
Basic integration tests for D5 and D7 implementations.

Verifies that:
1. Pillar4_Robustness can be imported
2. Pillar6_Generalization can be imported
3. Class methods exist and have correct signatures
4. Classes are properly registered in __all__
"""

import sys
import unittest
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestPillar4Robustness(unittest.TestCase):
    """Test Pillar 4 (Robustness) implementation."""

    def test_import(self):
        """Verify Pillar4_Robustness can be imported."""
        from circuitkit.evaluation.pillars import Pillar4_Robustness

        self.assertIsNotNone(Pillar4_Robustness)

    def test_run_method_exists(self):
        """Verify run() static method exists."""
        from circuitkit.evaluation.pillars import Pillar4_Robustness

        self.assertTrue(hasattr(Pillar4_Robustness, "run"))
        self.assertTrue(callable(getattr(Pillar4_Robustness, "run")))

    def test_compare_corruption_variants_exists(self):
        """Verify compare_corruption_variants() static method exists."""
        from circuitkit.evaluation.pillars import Pillar4_Robustness

        self.assertTrue(hasattr(Pillar4_Robustness, "compare_corruption_variants"))
        self.assertTrue(callable(getattr(Pillar4_Robustness, "compare_corruption_variants")))

    def test_compare_with_baseline_exists(self):
        """Verify compare_with_baseline() static method exists."""
        from circuitkit.evaluation.pillars import Pillar4_Robustness

        self.assertTrue(hasattr(Pillar4_Robustness, "compare_with_baseline"))
        self.assertTrue(callable(getattr(Pillar4_Robustness, "compare_with_baseline")))

    def test_docstring(self):
        """Verify docstrings are present."""
        from circuitkit.evaluation.pillars import Pillar4_Robustness

        self.assertIsNotNone(Pillar4_Robustness.__doc__)
        self.assertIn("Robustness", Pillar4_Robustness.__doc__)


class TestPillar6Generalization(unittest.TestCase):
    """Test Pillar 6 (Generalization) implementation."""

    def test_import(self):
        """Verify Pillar6_Generalization can be imported."""
        from circuitkit.evaluation.pillars import Pillar6_Generalization

        self.assertIsNotNone(Pillar6_Generalization)

    def test_run_method_exists(self):
        """Verify run() static method exists."""
        from circuitkit.evaluation.pillars import Pillar6_Generalization

        self.assertTrue(hasattr(Pillar6_Generalization, "run"))
        self.assertTrue(callable(getattr(Pillar6_Generalization, "run")))

    def test_build_transfer_matrix_exists(self):
        """Verify build_transfer_matrix() static method exists."""
        from circuitkit.evaluation.pillars import Pillar6_Generalization

        self.assertTrue(hasattr(Pillar6_Generalization, "build_transfer_matrix"))
        self.assertTrue(callable(getattr(Pillar6_Generalization, "build_transfer_matrix")))

    def test_summarize_transfer_matrix_exists(self):
        """Verify summarize_transfer_matrix() static method exists."""
        from circuitkit.evaluation.pillars import Pillar6_Generalization

        self.assertTrue(hasattr(Pillar6_Generalization, "summarize_transfer_matrix"))
        self.assertTrue(callable(getattr(Pillar6_Generalization, "summarize_transfer_matrix")))

    def test_docstring(self):
        """Verify docstrings are present."""
        from circuitkit.evaluation.pillars import Pillar6_Generalization

        self.assertIsNotNone(Pillar6_Generalization.__doc__)
        self.assertIn("Generalization", Pillar6_Generalization.__doc__)


class TestPillarRegistration(unittest.TestCase):
    """Test that pillars are properly registered in __all__."""

    def test_pillar4_in_all(self):
        """Verify Pillar4_Robustness is in __all__."""
        from circuitkit.evaluation import pillars

        self.assertIn("Pillar4_Robustness", pillars.__all__)

    def test_pillar6_in_all(self):
        """Verify Pillar6_Generalization is in __all__."""
        from circuitkit.evaluation import pillars

        self.assertIn("Pillar6_Generalization", pillars.__all__)

    def test_all_pillars_present(self):
        """Verify all 6 pillars are in __all__."""
        from circuitkit.evaluation import pillars

        expected_pillars = [
            "Pillar1_CausalPatching",
            "Pillar2_Ablation",
            "Pillar3_Stability",
            "Pillar4_Robustness",
            "Pillar5_Baselines",
            "Pillar6_Generalization",
        ]
        for pillar in expected_pillars:
            self.assertIn(pillar, pillars.__all__, f"{pillar} missing from pillars.__all__")


class TestPillar4Methods(unittest.TestCase):
    """Test Pillar4_Robustness method signatures."""

    def test_run_signature(self):
        """Verify run() has expected parameters."""
        import inspect

        from circuitkit.evaluation.pillars import Pillar4_Robustness

        sig = inspect.signature(Pillar4_Robustness.run)
        params = list(sig.parameters.keys())

        # Check required/important parameters
        self.assertIn("model", params)
        self.assertIn("graph", params)
        self.assertIn("original_dataloader", params)
        self.assertIn("metric_fn", params)
        self.assertIn("corruption_variant", params)

    def test_compare_corruption_variants_signature(self):
        """Verify compare_corruption_variants() has expected parameters."""
        import inspect

        from circuitkit.evaluation.pillars import Pillar4_Robustness

        sig = inspect.signature(Pillar4_Robustness.compare_corruption_variants)
        params = list(sig.parameters.keys())

        self.assertIn("model", params)
        self.assertIn("graph", params)
        self.assertIn("original_dataloader", params)
        self.assertIn("corruption_dataloaders", params)
        self.assertIn("metric_fn", params)


class TestPillar6Methods(unittest.TestCase):
    """Test Pillar6_Generalization method signatures."""

    def test_run_signature(self):
        """Verify run() has expected parameters."""
        import inspect

        from circuitkit.evaluation.pillars import Pillar6_Generalization

        sig = inspect.signature(Pillar6_Generalization.run)
        params = list(sig.parameters.keys())

        # Check required/important parameters
        self.assertIn("model", params)
        self.assertIn("graph", params)
        self.assertIn("source_dataloader", params)
        self.assertIn("target_dataloader", params)
        self.assertIn("metric_fn", params)

    def test_build_transfer_matrix_signature(self):
        """Verify build_transfer_matrix() has expected parameters."""
        import inspect

        from circuitkit.evaluation.pillars import Pillar6_Generalization

        sig = inspect.signature(Pillar6_Generalization.build_transfer_matrix)
        params = list(sig.parameters.keys())

        self.assertIn("model", params)
        self.assertIn("circuits", params)
        self.assertIn("task_dataloaders", params)
        self.assertIn("metric_fn", params)

    def test_summarize_transfer_matrix_signature(self):
        """Verify summarize_transfer_matrix() has expected parameters."""
        import inspect

        from circuitkit.evaluation.pillars import Pillar6_Generalization

        sig = inspect.signature(Pillar6_Generalization.summarize_transfer_matrix)
        params = list(sig.parameters.keys())

        self.assertIn("transfer_matrix", params)


if __name__ == "__main__":
    unittest.main(verbosity=2)


def test_visualize_deprecated_alias_warns_and_delegates(monkeypatch):
    """H1 (1.0.0 audit): ck.visualize was renamed to ck.visualize_circuit.

    ``visualize`` is now a subpackage, so the deprecated *call* site
    ``ck.visualize(circuit, ...)`` is what must keep working through 1.x: it
    stays callable, emits a DeprecationWarning on use, delegates to
    ``visualize_circuit``, and stays out of the public ``__all__``.

    This must hold *regardless of whether the ``circuitkit.visualize``
    subpackage has already been imported* — importing it (directly or via
    ``mock.patch("circuitkit.visualize....")``) used to shadow the old
    attribute shim and silently drop the warning.
    """
    import warnings

    import circuitkit as ck
    import circuitkit.visualize  # noqa: F401 — historically shadowed the shim

    sentinel = object()
    recorded = {}

    def _fake_visualize_circuit(*args, **kwargs):
        recorded["call"] = (args, kwargs)
        return sentinel

    monkeypatch.setattr("circuitkit.quick.visualize_circuit", _fake_visualize_circuit)

    assert callable(ck.visualize)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ck.visualize("circuit", foo=1)

    assert result is sentinel
    assert recorded["call"] == (("circuit",), {"foo": 1})
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert "visualize" not in ck.__all__
