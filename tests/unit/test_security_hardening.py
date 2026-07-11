"""Regression tests for the security-hardening fixes.

Covers:
  * SEC-1 — pickle-RCE via ``torch.load`` without ``weights_only=True`` on the
    public circuit-artifact loading path (CWE-502).
  * SEC-2 — ``CircuitBenchmark`` no longer forces ``trust_remote_code=True``.
  * SEC-3 — ``Pipeline`` artifact filenames can't be used for path traversal.

The tests are model-free (plain ``torch.save``/attribute checks) so they run in
milliseconds.
"""

import os

import pytest
import torch

from circuitkit.circuit import Circuit


# A module-level (picklable) object whose __reduce__ would run a shell command
# on unpickling. With weights_only=True the safe unpickler must refuse it
# *before* the reduce executes, so the sentinel file is never created.
class _PickleExploit:
    def __init__(self, sentinel_path: str) -> None:
        self.sentinel_path = sentinel_path

    def __reduce__(self):
        return (os.system, (f"touch {self.sentinel_path}",))


class TestArtifactDeserialization:
    """SEC-1: circuit artifacts are untrusted input; loading must not execute
    embedded pickle payloads."""

    def test_benign_artifact_loads(self, tmp_path):
        """A legitimate node-list artifact still round-trips through
        from_artifact after the weights_only=True hardening."""
        art = tmp_path / "circuit.pt"
        torch.save(["a0.h0", "a1.h3", "m2"], art)

        circuit = Circuit.from_artifact(art)

        assert list(circuit.nodes) == ["a0.h0", "a1.h3", "m2"]

    def test_malicious_pickle_is_blocked(self, tmp_path):
        """A malicious artifact must raise on load and must NOT execute its
        embedded payload."""
        art = tmp_path / "evil.pt"
        sentinel = tmp_path / "pwned"
        torch.save(_PickleExploit(str(sentinel)), art)

        with pytest.raises(Exception):
            Circuit.from_artifact(art)

        assert not sentinel.exists(), "pickle payload executed — weights_only bypassed"

    def test_malicious_scores_sidecar_is_blocked(self, tmp_path):
        """The auto-discovered *_scores.pt side-car is loaded the same safe way."""
        art = tmp_path / "circuit.pt"
        torch.save(["a0.h0"], art)
        sentinel = tmp_path / "pwned_sidecar"
        torch.save(_PickleExploit(str(sentinel)), tmp_path / "circuit_scores.pt")

        with pytest.raises(Exception):
            Circuit.from_artifact(art)

        assert not sentinel.exists()


class TestBenchmarkTrustRemoteCode:
    """SEC-2: trust_remote_code must default to False (opt-in), not be forced."""

    def test_default_is_false(self, tmp_path):
        from circuitkit.benchmarks.benchmark import CircuitBenchmark

        bench = CircuitBenchmark(output_dir=str(tmp_path / "out"))
        assert bench.trust_remote_code is False

    def test_opt_in_is_respected(self, tmp_path):
        from circuitkit.benchmarks.benchmark import CircuitBenchmark

        bench = CircuitBenchmark(
            output_dir=str(tmp_path / "out"), trust_remote_code=True
        )
        assert bench.trust_remote_code is True


class TestPipelinePathToken:
    """SEC-3: task / model names interpolated into artifact filenames must be
    sanitized so they can't escape the output directory."""

    @pytest.mark.parametrize(
        "raw",
        [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "/abs/path",
            "..",
            "a/b/c",
        ],
    )
    def test_no_separators_or_leading_dots(self, raw):
        from circuitkit.pipeline import _safe_path_token

        token = _safe_path_token(raw)
        assert "/" not in token
        assert "\\" not in token
        assert not token.startswith("."), token
        # Joining under a base dir must not escape it.
        base = "/tmp/out"
        joined = os.path.normpath(os.path.join(base, f"eap_{token}_node.pt"))
        assert joined.startswith(base + os.sep)

    def test_normal_task_name_preserved(self):
        from circuitkit.pipeline import _safe_path_token

        assert _safe_path_token("ioi") == "ioi"
        assert _safe_path_token("greater_than-v2") == "greater_than-v2"
