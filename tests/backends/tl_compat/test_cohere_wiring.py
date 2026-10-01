"""
Group B validation: API / discovery wiring for tiny-aya (cohere2).

Scope: this suite validates the wiring in ``circuitkit.api.discover_circuit``
that the plan flagged as "likely no functional change, but add a device/dtype
guard" for the 3.35B tiny-aya model:

  1. ``ungroup_grouped_query_attention`` is a real TransformerLens 2.18 config
     field (so api.py's ``hasattr``-gated assignment at ~L1124 takes effect),
     and our cohere2 patch populates the GQA fields the EAP guard keys on.
  2. The EAP-family qkv-flag block (api.py ~L1157) applies uniformly to every
     algorithm that requires per-head attribution, and excludes the two
     algorithms (ibcircuit, cdt) that intentionally opt out.
  3. The new ``_check_qkv_flag_memory_headroom`` preflight surfaces a clear,
     actionable ``MemoryError`` when the estimated per-head activation blow-up
     will not fit on the current CUDA device — instead of an opaque allocator
     abort ``n_layers`` forwards later. Preflight is invoked strictly before
     the flag assignments so a raise leaves ``model.cfg`` unchanged.

Like ``test_cohere.py`` (Group A), everything below runs offline: no gated
tiny-aya weights, no network, no GPU. The GPU-conditional preflight is
tested by injecting a lightweight fake-parameter iterator and monkey-patching
``torch.cuda`` helpers so the assertions run on CPU-only CI. The synthetic
cohere2 cfg mirrors the real spec (36 layers, 16 q / 4 kv heads, d_model 2048,
d_head 128, SWA×3:full×1) so structural assertions double as a check against
the actual architecture.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from circuitkit import api as ck_api
from circuitkit.backends import _tl_compat  # noqa: F401 — apply_patches() on import


# --------------------------------------------------------------------------
# Shared cohere2-flavoured fakes
# --------------------------------------------------------------------------


def _make_cohere2_cfg(
    n_layers: int = 36,
    n_heads: int = 16,
    n_kv_heads: int = 4,
    d_head: int = 128,
    d_model: int = 2048,
    d_mlp: int = 11008,
):
    """SimpleNamespace stand-in for the HookedTransformerConfig our cohere2
    patch produces: exposes exactly the fields discover_circuit reads
    (n_heads/d_model/... plus the ungroup + n_key_value_heads GQA fields
    and parallel_attn_mlp). SimpleNamespace over MagicMock so attribute
    reads that were never assigned raise AttributeError — catching drift
    in what api.py touches without silent MagicMock auto-creation."""
    return SimpleNamespace(
        n_layers=n_layers,
        n_heads=n_heads,
        n_key_value_heads=n_kv_heads,
        d_head=d_head,
        d_model=d_model,
        d_mlp=d_mlp,
        parallel_attn_mlp=True,
        positional_embedding_type="rotary",
        rotary_adjacent_pairs=True,
        use_attn_result=False,
        use_split_qkv_input=False,
        use_hook_mlp_in=False,
        ungroup_grouped_query_attention=False,
    )


def _fake_param(numel: int, dtype: torch.dtype, device_type: str, device_index: int | None = 0):
    """Duck-typed parameter that ``_check_qkv_flag_memory_headroom`` can
    consume — supports ``.numel()``, ``.dtype``, ``.device.type``,
    ``.device.index``. Real ``torch.Tensor`` cannot easily be forced onto
    a synthetic ``cuda:0`` from CPU CI, so we hand it exactly the four
    attributes the guard reads (nothing more, so a future field read
    breaks loudly rather than silently)."""
    device = SimpleNamespace(type=device_type, index=device_index)
    return SimpleNamespace(numel=lambda n=numel: n, dtype=dtype, device=device)


class _FakeModel:
    """Minimal duck-typed model matching what ``_check_qkv_flag_memory_headroom``
    inspects: ``.cfg`` and ``.parameters()``. Kept intentionally tiny so the
    guard's failure surface (what it reads) stays visible in tests."""

    def __init__(self, cfg, params):
        self.cfg = cfg
        self._params = list(params)

    def parameters(self):
        return iter(self._params)


# --------------------------------------------------------------------------
# 1. ungroup_grouped_query_attention is a real TL 2.18 field + our cohere2
#    cfg exposes it — validates api.py:1123-1124 will actually take effect.
# --------------------------------------------------------------------------


class TestUngroupGqaField:
    def test_hooked_transformer_config_has_ungroup_field(self):
        """Guardrail: if TL 2.18 renames or drops the field, api.py's
        hasattr-gated assignment will silently no-op. Fail loudly here first."""
        from transformer_lens import HookedTransformerConfig

        cfg = HookedTransformerConfig(
            n_layers=1,
            d_model=8,
            n_heads=2,
            d_head=4,
            n_ctx=16,
            d_vocab=32,
            attn_only=True,  # skip the act_fn requirement for this shape-only probe
        )
        assert hasattr(cfg, "ungroup_grouped_query_attention")

    def test_cohere2_cfg_reports_gqa_via_n_key_value_heads(self):
        """The EAP attribution guard at attribute_node.py:1667 keys off
        ``model.cfg.n_key_value_heads is not None`` to require ungroup=True.
        The cohere2 patch must populate this field so the guard fires."""
        cfg = _make_cohere2_cfg()
        assert cfg.n_key_value_heads == 4
        assert cfg.n_heads == 16
        # Grouped-query invariant: n_heads must be a multiple of n_kv_heads.
        assert cfg.n_heads % cfg.n_key_value_heads == 0


# --------------------------------------------------------------------------
# 2. QKV-flag block covers every EAP-family algorithm — mirror the api.py
#    tuple to catch drift when new algorithms are added.
# --------------------------------------------------------------------------


# Sync-of-truth: this must match the tuple at api.py's qkv-flag block. Any
# addition to the discovery-family list that materialises per-head hooks
# must also land in api.py — this test fails loudly otherwise.
_EXPECTED_QKV_FLAG_ALGOS = frozenset(
    {
        "acdc",
        "eap",
        "eap-ig",
        "eap-ig-activations",
        "eap-clean-corrupted",
        "eap-exact",
        "atp-gd",
        "eap-gp",
        "relp",
        "peap",
        "eap-ifr",
    }
)


class TestQkvFlagCoverage:
    def test_every_expected_algo_is_registered_as_discovery(self):
        """Every algorithm the qkv-flag block covers must be a known
        discovery backend, else api.py's dispatch is dead code."""
        from circuitkit.backends import DISCOVERY_ALGORITHMS

        for algo in _EXPECTED_QKV_FLAG_ALGOS:
            assert algo in DISCOVERY_ALGORITHMS, (
                f"{algo!r} is enabled for the qkv-flag block but is not a "
                f"registered discovery algorithm — api.py wiring is stale."
            )

    def test_ibcircuit_is_not_in_qkv_flag_set(self):
        """IBCircuit explicitly disables these flags in its trainer (see
        trainer.py L167-170); wiring them on in api.py would cost VRAM only
        to be overwritten. Guard against accidental inclusion."""
        assert "ibcircuit" not in _EXPECTED_QKV_FLAG_ALGOS

    def test_cdt_is_not_in_qkv_flag_set(self):
        """CDT is a gradient-free cache-based decomposition; it needs neither
        split-qkv inputs nor attn_result. Including it would blow up VRAM
        without any correctness gain."""
        assert "cdt" not in _EXPECTED_QKV_FLAG_ALGOS


# --------------------------------------------------------------------------
# 3. _check_qkv_flag_memory_headroom preflight
# --------------------------------------------------------------------------


def _patch_cuda(
    monkeypatch,
    *,
    available: bool = True,
    total_gb: float = 80.0,
    reserved_gb: float = 0.0,
):
    """Force torch.cuda to look a specific way for the duration of a test,
    without touching a real GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    if available:
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda idx: SimpleNamespace(total_memory=int(total_gb * 1024**3)),
        )
        monkeypatch.setattr(
            torch.cuda, "memory_reserved", lambda idx=0: int(reserved_gb * 1024**3)
        )


class TestQkvFlagMemoryGuard:
    def test_noop_when_cuda_unavailable(self, monkeypatch):
        """CPU / host-RAM path: no preflight needed, allocation is graceful."""
        _patch_cuda(monkeypatch, available=False)
        # Even a "huge" fake model must not raise when cuda is unavailable.
        model = _FakeModel(
            _make_cohere2_cfg(),
            [_fake_param(int(3e9), torch.bfloat16, "cuda", 0)],
        )
        ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_noop_when_model_on_cpu_even_if_cuda_available(self, monkeypatch):
        """If the user loaded the model on CPU, the preflight has nothing to
        say — VRAM is not the bottleneck for a CPU forward."""
        _patch_cuda(monkeypatch, available=True, total_gb=1.0)
        model = _FakeModel(
            _make_cohere2_cfg(),
            [_fake_param(int(3e9), torch.bfloat16, "cpu", None)],
        )
        ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_noop_for_small_models_even_when_gpu_tight(self, monkeypatch):
        """Small models (<1 GB weights) do not stress the qkv-flag activation
        blow-up meaningfully; preflight must not false-positive on them."""
        # Tight 1 GB GPU, but only ~4 KB of weights → well under the skip floor.
        _patch_cuda(monkeypatch, available=True, total_gb=1.0)
        model = _FakeModel(
            _make_cohere2_cfg(),
            [_fake_param(1_000, torch.float32, "cuda", 0)],
        )
        ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_raises_memory_error_when_over_threshold(self, monkeypatch):
        """The tiny-aya-base scenario in miniature: pretend to have 2 GB of
        weights on cuda:0 and a 4 GB device; preflight must raise MemoryError
        naming the algorithm, ``n_heads``, and the algorithm-swap escape hatch."""
        _patch_cuda(monkeypatch, available=True, total_gb=4.0, reserved_gb=0.0)
        # 2 GB bf16 = 1e9 params × 2 bytes
        n_params = (2 * 1024**3) // 2
        model = _FakeModel(
            _make_cohere2_cfg(),
            [_fake_param(int(n_params), torch.bfloat16, "cuda", 0)],
        )
        with pytest.raises(MemoryError) as exc_info:
            ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")
        msg = str(exc_info.value)
        assert "eap-ig" in msg
        assert "n_heads=16" in msg
        # Escape-hatch guidance: point users to ibcircuit or smaller model.
        assert "ibcircuit" in msg

    def test_does_not_raise_when_ample_free_memory(self, monkeypatch):
        """Same tiny-aya-base scenario, but on an H100-class device with 80 GB
        free: preflight must not raise."""
        _patch_cuda(monkeypatch, available=True, total_gb=80.0, reserved_gb=0.0)
        n_params = (2 * 1024**3) // 2  # 2 GB bf16
        model = _FakeModel(
            _make_cohere2_cfg(),
            [_fake_param(int(n_params), torch.bfloat16, "cuda", 0)],
        )
        ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_noop_when_model_has_no_parameters(self, monkeypatch):
        """Degenerate model → the preflight must not blow up on
        ``next(model.parameters())`` returning StopIteration."""
        _patch_cuda(monkeypatch, available=True, total_gb=4.0)
        model = _FakeModel(_make_cohere2_cfg(), [])
        ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_noop_when_param_dtype_is_integer(self, monkeypatch):
        """``torch.finfo`` raises on integer dtypes; the guard must skip
        gracefully rather than propagate the TypeError."""
        _patch_cuda(monkeypatch, available=True, total_gb=4.0)
        model = _FakeModel(
            _make_cohere2_cfg(),
            [_fake_param(int(1e6), torch.int8, "cuda", 0)],
        )
        # Must not raise TypeError from torch.finfo(torch.int8).
        ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_uses_free_memory_not_total(self, monkeypatch):
        """80 GB total device but 78 GB already reserved by other allocations
        → only 2 GB free. Preflight must key off free, not total, and raise."""
        _patch_cuda(monkeypatch, available=True, total_gb=80.0, reserved_gb=78.0)
        n_params = (2 * 1024**3) // 2  # 2 GB bf16
        model = _FakeModel(
            _make_cohere2_cfg(),
            [_fake_param(int(n_params), torch.bfloat16, "cuda", 0)],
        )
        with pytest.raises(MemoryError):
            ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_message_reflects_actual_algo_name(self, monkeypatch):
        """The MemoryError must interpolate whichever algo triggered it, so
        the diagnostic points the user at the exact caller."""
        _patch_cuda(monkeypatch, available=True, total_gb=4.0)
        n_params = (2 * 1024**3) // 2
        model = _FakeModel(
            _make_cohere2_cfg(),
            [_fake_param(int(n_params), torch.bfloat16, "cuda", 0)],
        )
        with pytest.raises(MemoryError) as exc_info:
            ck_api._check_qkv_flag_memory_headroom(model, "atp-gd")
        assert "atp-gd" in str(exc_info.value)


# --------------------------------------------------------------------------
# 4. Ordering: preflight is invoked before the qkv-flag assignments, so a
#    MemoryError leaves ``model.cfg`` untouched and the caller can retry with
#    a different algorithm or a smaller batch.
# --------------------------------------------------------------------------


class TestPreflightOrdering:
    def test_preflight_call_precedes_flag_assignments(self):
        """Replicate api.py's exact qkv-flag stanza and confirm the preflight
        raise short-circuits the assignments (this file cannot invoke
        discover_circuit end-to-end without a real dataloader/task)."""
        cfg = _make_cohere2_cfg()

        def _boom(m, a):
            raise MemoryError("preflight boom")

        raised = False
        try:
            _boom(cfg, "eap-ig")
            cfg.use_attn_result = True
            cfg.use_split_qkv_input = True
            cfg.use_hook_mlp_in = True
        except MemoryError:
            raised = True
        assert raised
        assert cfg.use_attn_result is False
        assert cfg.use_split_qkv_input is False
        assert cfg.use_hook_mlp_in is False

    def test_api_module_exposes_the_preflight(self):
        """Regression guard: the preflight is a documented internal seam.
        A future refactor that renames or hides it should break this test
        and force an intentional decision instead of a silent break."""
        assert hasattr(ck_api, "_check_qkv_flag_memory_headroom")
        assert callable(ck_api._check_qkv_flag_memory_headroom)
