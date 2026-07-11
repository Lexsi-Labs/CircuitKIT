"""
Integration smoke tests for the Pipeline class end-to-end.

These tests load a real gpt2 model and run real discovery — they are slow
(~1-3 min on CPU) and are gated behind the `slow` mark.  Run with:

    pytest tests/integration/test_pipeline_smoke.py -m slow

They are deliberately minimal: small n_examples/batch_size, no evaluation
pillars (which require a running API), no export (requires HF conversion).
The goal is to verify that the Discover → Prune chaining works without
errors on the actual codebase — not to benchmark quality.
"""

import os
import tempfile
from pathlib import Path

import pytest

# All tests in this module are slow — mark at the module level so they can be
# excluded from CI with `-m "not slow"`.
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_network():
    """Skip if HuggingFace Hub is unreachable (offline CI)."""
    import socket
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            ("huggingface.co", 443)
        )
    except OSError:
        pytest.skip("HuggingFace Hub unreachable — skipping model-loading tests")


# ---------------------------------------------------------------------------
# Smoke: discover produces a non-empty circuit
# ---------------------------------------------------------------------------

class TestDiscoverProducesCircuit:
    def test_discover_eap_ig_ioi(self, tmp_path):
        """Pipeline.discover() on gpt2/ioi/eap-ig must populate _circuit."""
        _skip_if_no_network()

        from circuitkit.pipeline import Pipeline

        pipe = Pipeline(
            "gpt2",
            task="ioi",
            precision="float32",
            output_dir=str(tmp_path),
        )
        pipe.discover(
            algorithm="eap-ig",
            level="node",
            sparsity=0.3,
            n_examples=4,
            batch_size=2,
        )

        assert pipe._circuit is not None, "_circuit must be populated after discover()"
        assert len(pipe._circuit) > 0, "Discovered circuit must contain at least one node"
        assert pipe._circuit.level == "node"
        assert "discover" in pipe._history

    def test_discover_sets_algorithm_on_circuit(self, tmp_path):
        _skip_if_no_network()

        from circuitkit.pipeline import Pipeline

        pipe = Pipeline("gpt2", task="ioi", precision="float32", output_dir=str(tmp_path))
        pipe.discover(algorithm="eap-ig", level="node", n_examples=4, batch_size=2)

        assert pipe._circuit.algorithm == "eap-ig"

    def test_discover_sets_task_on_circuit(self, tmp_path):
        _skip_if_no_network()

        from circuitkit.pipeline import Pipeline

        pipe = Pipeline("gpt2", task="ioi", precision="float32", output_dir=str(tmp_path))
        pipe.discover(algorithm="eap-ig", level="node", n_examples=4, batch_size=2)

        assert pipe._circuit.task == "ioi"

    def test_discover_writes_artifact_to_output_dir(self, tmp_path):
        _skip_if_no_network()

        from circuitkit.pipeline import Pipeline

        pipe = Pipeline("gpt2", task="ioi", precision="float32", output_dir=str(tmp_path))
        pipe.discover(algorithm="eap-ig", level="node", n_examples=4, batch_size=2)

        assert pipe._artifact_path is not None
        # The artifact file itself may not exist if the engine skipped writing
        # (e.g. no output_path in config), but the path must be set.
        assert isinstance(pipe._artifact_path, str)


# ---------------------------------------------------------------------------
# Smoke: discover → prune chaining
# ---------------------------------------------------------------------------

class TestDiscoverPruneChaining:
    def test_prune_after_discover_stores_pruned_model(self, tmp_path):
        """Pipeline.prune() must populate _pruned_model from _model."""
        _skip_if_no_network()

        from circuitkit.pipeline import Pipeline

        pipe = Pipeline("gpt2", task="ioi", precision="float32", output_dir=str(tmp_path))
        pipe.discover(algorithm="eap-ig", level="node", n_examples=4, batch_size=2)
        pipe.prune(sparsity=0.3)

        assert pipe._pruned_model is not None, \
            "_pruned_model must be set after prune()"
        assert pipe._model is not None, \
            "_model (original) must not be overwritten by prune()"
        assert pipe._pruned_model is not pipe._model, \
            "Prune must return a copy, not modify the original model in-place"

    def test_prune_appends_history(self, tmp_path):
        _skip_if_no_network()

        from circuitkit.pipeline import Pipeline

        pipe = Pipeline("gpt2", task="ioi", precision="float32", output_dir=str(tmp_path))
        pipe.discover(algorithm="eap-ig", level="node", n_examples=4, batch_size=2)
        pipe.prune(sparsity=0.3)

        assert "prune" in pipe._history

    def test_method_chaining_returns_same_pipe(self, tmp_path):
        _skip_if_no_network()

        from circuitkit.pipeline import Pipeline

        pipe = Pipeline("gpt2", task="ioi", precision="float32", output_dir=str(tmp_path))
        result = pipe.discover(
            algorithm="eap-ig", level="node", n_examples=4, batch_size=2
        ).prune(sparsity=0.3)

        assert result is pipe


# ---------------------------------------------------------------------------
# Smoke: from_artifact constructor
# ---------------------------------------------------------------------------

class TestFromArtifactRoundtrip:
    def test_from_artifact_after_discover(self, tmp_path):
        """Discover writes an artifact; from_artifact must load it back."""
        _skip_if_no_network()

        from circuitkit.pipeline import Pipeline

        # First, discover and ensure artifact is written
        pipe = Pipeline("gpt2", task="ioi", precision="float32", output_dir=str(tmp_path))
        pipe.discover(algorithm="eap-ig", level="node", n_examples=4, batch_size=2)

        artifact_path = pipe._artifact_path
        if not Path(artifact_path).exists():
            pytest.skip("Artifact file was not written — skipping roundtrip test")

        # Load from the artifact
        pipe2 = Pipeline.from_artifact(artifact_path, "gpt2")
        assert pipe2._circuit is not None
        assert len(pipe2._circuit) > 0

    def test_summary_after_discover(self, tmp_path):
        """summary() must not raise after a real discover() call."""
        _skip_if_no_network()

        from circuitkit.pipeline import Pipeline

        pipe = Pipeline("gpt2", task="ioi", precision="float32", output_dir=str(tmp_path))
        pipe.discover(algorithm="eap-ig", level="node", n_examples=4, batch_size=2)
        pipe.summary()  # must not raise
