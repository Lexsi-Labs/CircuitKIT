"""
test_covariance.py
==================
Unit tests for circuitkit.apply._covariance.

Covers:
    solve_with_C    — linear algebra correctness, dtype, shape
    get_covariance  — accumulation, hook firing, shape, PSD, warnings
    Caching         — write/read, cache hit speed, corpus_id isolation,
                      corrupt file recovery, use_cache=False bypass

Tiers:
    Tier 1 — solve_with_C: pure tensor math, no model.
    Tier 2 — get_covariance: lightweight mock model.
    Tier 3 — Real model (GPT-2, marked @pytest.mark.slow).

Usage:
    Unit only:   pytest test_covariance.py -v -m "not slow"
    All tests:   pytest test_covariance.py -v
"""

from __future__ import annotations

import os
import sys
import time

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import gate
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_IMPORT_ERROR = ""
try:
    from circuitkit.applications.common_utils._covariance import (
        _FALLBACK_TEXTS,
        _cache_key,
        get_covariance,
        solve_with_C,
    )

    _MODULES_AVAILABLE = True
except ImportError as exc:
    _MODULES_AVAILABLE = False
    _IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    not _MODULES_AVAILABLE,
    reason=f"_covariance not importable: {_IMPORT_ERROR}",
)


# ===========================================================================
# Mock infrastructure (self-contained — does NOT import from other test files)
# ===========================================================================


class _MockHookPoint:
    """Minimal stand-in for transformer_lens HookPoint."""

    def __init__(self, name: str):
        self.name = name


class _CovMockConfig:
    """Minimal config for covariance tests."""

    def __init__(self, n_layers: int = 4, d_model: int = 32, device: str = "cpu"):
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = device
        # No model_name — get_covariance falls back to "unknown"


class _CovMockModel(nn.Module):
    """
    Lightweight mock for get_covariance tests.

    Key properties:
      * to_tokens() is character-level (deterministic, always produces tokens).
      * run_with_hooks() fires each hook with shape [B, S, d] where d is
        d_mlp if 'hook_post' appears in the hook name, else d_model.
      * reset_hooks() is a no-op.
      * Has at least one nn.Parameter so next(model.parameters()).device works.
    """

    def __init__(
        self,
        n_layers: int = 4,
        d_model: int = 32,
        d_mlp: int = 64,
        d_vocab: int = 200,
    ):
        super().__init__()
        self.cfg = _CovMockConfig(n_layers=n_layers, d_model=d_model)
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_vocab = d_vocab
        # One real parameter so that next(model.parameters()).device resolves.
        self._dummy = nn.Parameter(torch.zeros(1))

    def to_tokens(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        ids = [max(1, ord(c) % self.d_vocab) for c in text[:80]]
        if prepend_bos:
            ids = [0] + ids
        return torch.tensor([ids])

    def run_with_hooks(self, tokens: torch.Tensor, fwd_hooks=None, **kwargs):
        batch, seq = tokens.shape
        if fwd_hooks:
            for hook_name, hook_fn in fwd_hooks:
                d = self.d_mlp if "hook_post" in hook_name else self.d_model
                act = torch.randn(batch, seq, d)
                hook_fn(act, _MockHookPoint(hook_name))
        return torch.randn(batch, seq, self.d_vocab)

    def reset_hooks(self, **kwargs):
        pass

    def parameters(self, recurse=True):
        return super().parameters(recurse=recurse)


@pytest.fixture
def mock_model() -> _CovMockModel:
    torch.manual_seed(42)
    return _CovMockModel()


# ===========================================================================
# Tier 1 — solve_with_C (pure tensor math)
# ===========================================================================


class TestSolveWithC:
    """
    solve_with_C(C, k, lam) solves (C + λI) v = k for v.
    All tests are analytical — no model needed.
    """

    def test_known_diagonal_solution(self):
        """
        C = diag(2, 4), k = [6, 8], λ = 1.
        (C + I) = diag(3, 5) → v = [6/3, 8/5] = [2.0, 1.6].
        """
        C = torch.diag(torch.tensor([2.0, 4.0]))
        k = torch.tensor([6.0, 8.0])
        v = solve_with_C(C, k, lam=1.0)
        expected = torch.tensor([2.0, 1.6])
        assert torch.allclose(v, expected, atol=1e-5), f"Expected {expected}, got {v}"

    def test_identity_C(self):
        """C = I → v = k / (1 + λ)."""
        d = 8
        lam = 0.5
        C = torch.eye(d)
        k = torch.randn(d)
        v = solve_with_C(C, k, lam=lam)
        expected = k / (1.0 + lam)
        assert torch.allclose(v, expected, atol=1e-5)

    def test_dtype_preservation_float32(self):
        """Input k as float32 → output must be float32."""
        C = torch.eye(4, dtype=torch.float32)
        k = torch.randn(4, dtype=torch.float32)
        v = solve_with_C(C, k)
        assert v.dtype == torch.float32, f"Expected float32, got {v.dtype}"

    def test_dtype_preservation_float64(self):
        """Input k as float64 → output must be float64."""
        C = torch.eye(4, dtype=torch.float64)
        k = torch.randn(4, dtype=torch.float64)
        v = solve_with_C(C, k)
        assert v.dtype == torch.float64, f"Expected float64, got {v.dtype}"

    def test_shape_preservation_1d(self):
        """1-D k in → 1-D v out."""
        C = torch.eye(5)
        k = torch.randn(5)
        v = solve_with_C(C, k)
        assert v.shape == k.shape, f"Shape mismatch: {v.shape} vs {k.shape}"

    def test_shape_preservation_2d_column(self):
        """2-D column k [d, 1] in → [d, 1] out."""
        C = torch.eye(5)
        k = torch.randn(5, 1)
        v = solve_with_C(C, k)
        assert v.shape == k.shape, f"Shape mismatch: {v.shape} vs {k.shape}"

    def test_zero_C_with_regularization(self):
        """
        C = 0 → (0 + λI) v = k → v = k / λ.
        Proves regularization makes a singular system solvable.
        """
        d = 6
        lam = 0.1
        C = torch.zeros(d, d)
        k = torch.randn(d)
        v = solve_with_C(C, k, lam=lam)
        expected = k / lam
        assert torch.allclose(
            v, expected, atol=1e-4
        ), "With C=0, expected k/λ but got different result"

    def test_large_lambda_dominates(self):
        """When λ >> ||C||, result ≈ k / λ regardless of C."""
        d = 10
        lam = 1e6
        C = torch.randn(d, d)
        C = C @ C.T  # make PSD
        k = torch.randn(d)
        v = solve_with_C(C, k, lam=lam)
        expected = k / lam
        assert torch.allclose(v, expected, atol=1e-3), "Large λ should make C negligible"


# ===========================================================================
# Tier 2 — get_covariance (mock model)
# ===========================================================================


class TestGetCovariance:
    """
    get_covariance(model, layer, hook_name, ...) → [d, d] tensor.
    Tests accumulation logic, shape, mathematical properties, and edge cases.
    """

    def test_happy_path_returns_correct_shape(self, mock_model):
        """C at mlp.hook_post should be [d_mlp, d_mlp]."""
        C = get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=["Hello world", "Test sentence"],
            n_samples=2,
            use_cache=False,
        )
        assert C is not None
        assert C.shape == (
            mock_model.d_mlp,
            mock_model.d_mlp,
        ), f"Expected ({mock_model.d_mlp}, {mock_model.d_mlp}), got {C.shape}"

    def test_non_hook_post_returns_d_model_shape(self, mock_model):
        """A hook name without 'hook_post' yields [d_model, d_model]."""
        C = get_covariance(
            mock_model,
            layer=0,
            hook_name="hook_mlp_out",
            texts=["Hello world"],
            n_samples=1,
            use_cache=False,
        )
        assert C is not None
        assert C.shape == (mock_model.d_model, mock_model.d_model)

    def test_result_is_symmetric(self, mock_model):
        """C = E[k k^T] is symmetric by construction."""
        C = get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:5],
            n_samples=5,
            use_cache=False,
        )
        assert C is not None
        assert torch.allclose(C, C.T, atol=1e-5), "C is not symmetric"

    def test_result_is_psd(self, mock_model):
        """All eigenvalues of C must be >= 0 (positive semi-definite)."""
        C = get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:5],
            n_samples=5,
            use_cache=False,
        )
        assert C is not None
        eigenvalues = torch.linalg.eigvalsh(C)
        assert (
            eigenvalues >= -1e-5
        ).all(), f"C has negative eigenvalues: min={eigenvalues.min().item():.6f}"

    def test_diagonal_is_nonnegative(self, mock_model):
        """Diagonal of E[k k^T] = E[k_i^2] ≥ 0."""
        C = get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:5],
            n_samples=5,
            use_cache=False,
        )
        assert C is not None
        assert (C.diag() >= -1e-6).all(), "Diagonal has negative entries"

    def test_result_is_float32(self, mock_model):
        """C is accumulated in float64 but returned as float32."""
        C = get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=["test"],
            n_samples=1,
            use_cache=False,
        )
        assert C is not None
        assert C.dtype == torch.float32

    def test_n_samples_limits_texts_consumed(self, mock_model, monkeypatch):
        """With 5 texts but n_samples=2, only 2 should be processed."""
        call_count = 0
        original_run = mock_model.run_with_hooks

        def counting_run(tokens, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_run(tokens, **kwargs)

        monkeypatch.setattr(mock_model, "run_with_hooks", counting_run)

        texts = ["a", "b", "c", "d", "e"]
        C = get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=texts,
            n_samples=2,
            use_cache=False,
        )
        assert C is not None
        assert call_count == 2, f"Expected 2 run_with_hooks calls, got {call_count}"

    @pytest.mark.filterwarnings("default::UserWarning")
    def test_empty_texts_returns_none_with_warning(self, mock_model):
        """texts=[] should warn and return None."""
        with pytest.warns(UserWarning, match="[Ee]mpty"):
            C = get_covariance(
                mock_model,
                layer=0,
                hook_name="mlp.hook_post",
                texts=[],
                n_samples=10,
                use_cache=False,
            )
        assert C is None

    def test_none_texts_uses_fallback_corpus(self, mock_model):
        """texts=None should silently use _FALLBACK_TEXTS and succeed."""
        C = get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=None,
            n_samples=5,
            use_cache=False,
        )
        assert C is not None
        assert C.shape == (mock_model.d_mlp, mock_model.d_mlp)

    def test_fallback_corpus_too_small_warning(self, mock_model, caplog):
        """texts=None with n_samples >> len(fallback) should log a warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="circuitkit.applications.common_utils._covariance"):
            get_covariance(
                mock_model,
                layer=0,
                hook_name="mlp.hook_post",
                texts=None,
                n_samples=999_999,
                use_cache=False,
            )
        assert any("approximate" in r.message for r in caplog.records)

    def test_more_samples_changes_result(self, mock_model):
        """
        C estimated from 2 samples vs 10 samples should differ.
        Proves accumulation actually depends on the data, not a fixed value.
        """
        torch.manual_seed(0)
        C2 = get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:2],
            n_samples=2,
            use_cache=False,
        )
        torch.manual_seed(1)
        C10 = get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:10],
            n_samples=10,
            use_cache=False,
        )
        assert C2 is not None and C10 is not None
        assert not torch.allclose(C2, C10, atol=1e-3), "C from 2 vs 10 samples should differ"

    def test_max_seq_len_truncation(self, mock_model, monkeypatch):
        """
        Tokens longer than max_seq_len should be truncated.
        Verify via a very short max_seq_len that limits processing.
        """
        captured_shapes = []
        original_run = mock_model.run_with_hooks

        def shape_capturing_run(tokens, **kwargs):
            captured_shapes.append(tokens.shape)
            return original_run(tokens, **kwargs)

        monkeypatch.setattr(mock_model, "run_with_hooks", shape_capturing_run)

        long_text = "A" * 200  # will produce ~200 tokens
        get_covariance(
            mock_model,
            layer=0,
            hook_name="mlp.hook_post",
            texts=[long_text],
            n_samples=1,
            max_seq_len=10,
            use_cache=False,
        )
        assert len(captured_shapes) == 1
        assert (
            captured_shapes[0][1] <= 10
        ), f"Tokens should be truncated to max_seq_len=10, got {captured_shapes[0][1]}"


# ===========================================================================
# Tier 2 — Caching (mock model + tmp_path)
# ===========================================================================


class TestCovarianceCache:
    """Tests for the on-disk caching layer of get_covariance."""

    @pytest.fixture(autouse=True)
    def _redirect_cache(self, tmp_path, monkeypatch):
        """Redirect _cache_dir to tmp_path so tests never touch ~/.cache."""
        import circuitkit.applications.common_utils._covariance as _cov_mod

        monkeypatch.setattr(_cov_mod, "_cache_dir", lambda: tmp_path)
        self._cache_path = tmp_path

    def test_cache_write_then_read(self, mock_model):
        """First call computes, second call loads from cache. Results match."""
        kwargs = dict(
            layer=0,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:3],
            n_samples=3,
            use_cache=True,
            corpus_id="test_rw",
        )
        C1 = get_covariance(mock_model, **kwargs)
        C2 = get_covariance(mock_model, **kwargs)
        assert C1 is not None and C2 is not None
        assert torch.equal(C1, C2), "Cached result differs from computed result"

    def test_cache_hit_is_fast(self, mock_model):
        """Second (cached) call must complete in < 100ms."""
        kwargs = dict(
            layer=0,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:3],
            n_samples=3,
            use_cache=True,
            corpus_id="test_speed",
        )
        # First call: compute and cache
        get_covariance(mock_model, **kwargs)

        # Second call: should be a cache hit
        t0 = time.time()
        C = get_covariance(mock_model, **kwargs)
        elapsed_ms = (time.time() - t0) * 1000
        assert C is not None
        assert elapsed_ms < 100, f"Cache hit took {elapsed_ms:.1f}ms — expected < 100ms"

    def test_use_cache_false_forces_recompute(self, mock_model, monkeypatch):
        """use_cache=False should call run_with_hooks even if cache file exists."""
        kwargs = dict(
            layer=0,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:3],
            n_samples=3,
            use_cache=True,
            corpus_id="test_bypass",
        )
        # Populate cache
        get_covariance(mock_model, **kwargs)

        # Now force recompute with use_cache=False and count calls
        call_count = 0
        original_run = mock_model.run_with_hooks

        def counting_run(tokens, **kw):
            nonlocal call_count
            call_count += 1
            return original_run(tokens, **kw)

        monkeypatch.setattr(mock_model, "run_with_hooks", counting_run)

        get_covariance(mock_model, **{**kwargs, "use_cache": False})
        assert call_count > 0, "use_cache=False did not trigger recomputation"

    def test_different_corpus_id_different_cache(self, mock_model):
        """Two calls with same n_samples but different corpus_id use separate caches."""
        common = dict(
            layer=0,
            hook_name="mlp.hook_post",
            n_samples=3,
            use_cache=True,
        )
        C_a = get_covariance(
            mock_model,
            texts=_FALLBACK_TEXTS[:3],
            corpus_id="corpus_a",
            **common,
        )
        C_b = get_covariance(
            mock_model,
            texts=_FALLBACK_TEXTS[3:6],
            corpus_id="corpus_b",
            **common,
        )
        assert C_a is not None and C_b is not None

        # Verify two separate cache files exist
        cache_files = list(self._cache_path.glob("*.pt"))
        assert len(cache_files) >= 2, (
            f"Expected at least 2 cache files for different corpus_ids, "
            f"found {len(cache_files)}"
        )

    @pytest.mark.filterwarnings("default::UserWarning")
    def test_corrupt_cache_triggers_recompute(self, mock_model):
        """
        If the cache file contains garbage, get_covariance should warn
        and recompute successfully instead of crashing.
        """
        kwargs = dict(
            layer=0,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:3],
            n_samples=3,
            use_cache=True,
            corpus_id="test_corrupt",
        )
        # Compute once to get the correct cache key/path
        C_good = get_covariance(mock_model, **kwargs)
        assert C_good is not None

        # Corrupt the cache file

        model_name = getattr(getattr(mock_model, "cfg", None), "model_name", "unknown")
        key = _cache_key(model_name, 0, "mlp.hook_post", 3, "test_corrupt")
        cache_file = self._cache_path / f"{key}.pt"
        assert cache_file.exists(), "Cache file should have been created"
        cache_file.write_bytes(b"this is not a valid torch file")

        # Re-call: should warn about load failure and recompute
        with pytest.warns(UserWarning, match="[Ff]ailed to load"):
            C_recovered = get_covariance(mock_model, **kwargs)

        assert C_recovered is not None
        assert C_recovered.shape == C_good.shape

    def test_same_params_same_cache_key(self):
        """Identical parameters produce the same cache key."""
        k1 = _cache_key("gpt2", 5, "mlp.hook_post", 1000, "default")
        k2 = _cache_key("gpt2", 5, "mlp.hook_post", 1000, "default")
        assert k1 == k2

    def test_different_layer_different_cache_key(self):
        """Different layer produces a different cache key."""
        k1 = _cache_key("gpt2", 5, "mlp.hook_post", 1000, "default")
        k2 = _cache_key("gpt2", 6, "mlp.hook_post", 1000, "default")
        assert k1 != k2

    def test_different_n_samples_different_cache_key(self):
        """Different n_samples produces a different cache key."""
        k1 = _cache_key("gpt2", 5, "mlp.hook_post", 1000, "default")
        k2 = _cache_key("gpt2", 5, "mlp.hook_post", 2000, "default")
        assert k1 != k2


# ===========================================================================
# Tier 3 — Real model tests (marked slow)
# ===========================================================================


@pytest.fixture(scope="module")
def gpt2_model():
    tl = pytest.importorskip("transformer_lens")
    return tl.HookedTransformer.from_pretrained("gpt2", device=("cuda" if torch.cuda.is_available() else "cpu"))


@pytest.mark.slow
class TestGetCovarianceRealModel:
    """
    Real GPT-2 integration tests for get_covariance.
    Verifies that the accumulation logic works with a real model's hooks.
    """

    def test_real_model_shape_and_symmetry(self, gpt2_model, tmp_path, monkeypatch):
        """C on GPT-2 layer 5 has correct shape and is symmetric."""
        import circuitkit.applications.common_utils._covariance as _cov_mod

        monkeypatch.setattr(_cov_mod, "_cache_dir", lambda: tmp_path)

        d_mlp = gpt2_model.cfg.d_mlp
        C = get_covariance(
            gpt2_model,
            layer=5,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:10],
            n_samples=10,
            use_cache=False,
        )
        assert C is not None
        assert C.shape == (d_mlp, d_mlp), f"Expected ({d_mlp},{d_mlp}), got {C.shape}"
        assert torch.allclose(C, C.T, atol=1e-4), "C is not symmetric on real model"

    def test_real_model_psd_and_nonzero_diagonal(self, gpt2_model, tmp_path, monkeypatch):
        """C from real activations must be PSD with positive diagonal mean."""
        import circuitkit.applications.common_utils._covariance as _cov_mod

        monkeypatch.setattr(_cov_mod, "_cache_dir", lambda: tmp_path)

        C = get_covariance(
            gpt2_model,
            layer=5,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:10],
            n_samples=10,
            use_cache=False,
        )
        assert C is not None
        eigenvalues = torch.linalg.eigvalsh(C)
        assert (
            eigenvalues >= -1e-4
        ).all(), f"Negative eigenvalue on real model: min={eigenvalues.min().item():.6f}"
        assert C.diag().mean().item() > 0, "Diagonal mean should be positive on real activations"

    def test_real_model_cache_roundtrip(self, gpt2_model, tmp_path, monkeypatch):
        """Compute, cache, reload — result must be identical."""
        import circuitkit.applications.common_utils._covariance as _cov_mod

        monkeypatch.setattr(_cov_mod, "_cache_dir", lambda: tmp_path)

        kwargs = dict(
            layer=3,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:5],
            n_samples=5,
            use_cache=True,
            corpus_id="real_roundtrip",
        )
        C1 = get_covariance(gpt2_model, **kwargs)
        C2 = get_covariance(gpt2_model, **kwargs)
        assert C1 is not None and C2 is not None
        assert torch.equal(C1, C2), "Cache roundtrip produced different result"

    def test_solve_with_real_C(self, gpt2_model, tmp_path, monkeypatch):
        """
        End-to-end: estimate C from real model, then solve_with_C.
        Result should be finite and well-conditioned.
        """
        import circuitkit.applications.common_utils._covariance as _cov_mod

        monkeypatch.setattr(_cov_mod, "_cache_dir", lambda: tmp_path)

        C = get_covariance(
            gpt2_model,
            layer=5,
            hook_name="mlp.hook_post",
            texts=_FALLBACK_TEXTS[:10],
            n_samples=10,
            use_cache=False,
        )
        assert C is not None
        d = C.shape[0]
        k = torch.randn(d)
        v = solve_with_C(C, k, lam=1e-2)
        assert torch.isfinite(v).all(), "solve_with_C produced non-finite values on real C"
        assert v.shape == (d,)
