"""Fast, offline tests for the qkv-flag preflight guard and flag scoping.

No model download, no GPU: ``torch.cuda`` is monkey-patched and parameters are
duck-typed, so the guard's branches run on CPU-only CI.
"""

from types import SimpleNamespace

import pytest
import torch

from circuitkit import api as ck_api

_GB_BF16_PARAMS = (2 * 1024**3) // 2  # parameter count of 2 GB of bf16 weights


def _cfg(**flags):
    return SimpleNamespace(n_heads=16, **flags)


def _fake_param(numel, dtype, device_type, device_index=0):
    device = SimpleNamespace(type=device_type, index=device_index)
    return SimpleNamespace(numel=lambda n=numel: n, dtype=dtype, device=device)


class _FakeModel:
    def __init__(self, params, cfg=None):
        self.cfg = cfg or _cfg()
        self._params = list(params)

    def parameters(self):
        return iter(self._params)


def _patch_cuda(monkeypatch, *, available=True, total_gb=80.0, reserved_gb=0.0):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    if available:
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda idx: SimpleNamespace(total_memory=int(total_gb * 1024**3)),
        )
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda idx=0: int(reserved_gb * 1024**3))


def _two_gb_model():
    return _FakeModel([_fake_param(_GB_BF16_PARAMS, torch.bfloat16, "cuda")])


class TestQkvFlagMemoryGuard:
    def test_noop_when_cuda_unavailable(self, monkeypatch):
        _patch_cuda(monkeypatch, available=False)
        ck_api._check_qkv_flag_memory_headroom(_two_gb_model(), "eap-ig")

    def test_noop_when_model_on_cpu(self, monkeypatch):
        _patch_cuda(monkeypatch, total_gb=1.0)
        model = _FakeModel([_fake_param(int(3e9), torch.bfloat16, "cpu", None)])
        ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_noop_for_small_models_on_a_tight_gpu(self, monkeypatch):
        _patch_cuda(monkeypatch, total_gb=1.0)
        model = _FakeModel([_fake_param(1_000, torch.float32, "cuda")])
        ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_noop_when_model_has_no_parameters(self, monkeypatch):
        _patch_cuda(monkeypatch, total_gb=4.0)
        ck_api._check_qkv_flag_memory_headroom(_FakeModel([]), "eap-ig")

    def test_noop_for_integer_dtype(self, monkeypatch):
        _patch_cuda(monkeypatch, total_gb=4.0)
        model = _FakeModel([_fake_param(int(1e6), torch.int8, "cuda")])
        ck_api._check_qkv_flag_memory_headroom(model, "eap-ig")

    def test_raises_when_estimate_exceeds_free_memory(self, monkeypatch):
        _patch_cuda(monkeypatch, total_gb=4.0)
        with pytest.raises(MemoryError) as exc:
            ck_api._check_qkv_flag_memory_headroom(_two_gb_model(), "atp-gd")
        msg = str(exc.value)
        assert "atp-gd" in msg and "n_heads=16" in msg and "ibcircuit" in msg

    def test_passes_with_ample_free_memory(self, monkeypatch):
        _patch_cuda(monkeypatch, total_gb=80.0)
        ck_api._check_qkv_flag_memory_headroom(_two_gb_model(), "eap-ig")

    def test_uses_free_not_total_memory(self, monkeypatch):
        _patch_cuda(monkeypatch, total_gb=80.0, reserved_gb=78.0)
        with pytest.raises(MemoryError):
            ck_api._check_qkv_flag_memory_headroom(_two_gb_model(), "eap-ig")


class TestQkvFlagsEnabled:
    def test_sets_flags_then_restores_previous_values(self):
        model = _FakeModel(
            [],
            _cfg(use_attn_result=False, use_split_qkv_input=True, use_hook_mlp_in=False),
        )
        with ck_api._qkv_flags_enabled(model):
            assert all(getattr(model.cfg, f) for f in ck_api._QKV_FLAGS)
        assert model.cfg.use_attn_result is False
        assert model.cfg.use_split_qkv_input is True
        assert model.cfg.use_hook_mlp_in is False

    def test_restores_on_exception(self):
        model = _FakeModel(
            [],
            _cfg(use_attn_result=False, use_split_qkv_input=False, use_hook_mlp_in=False),
        )
        with pytest.raises(RuntimeError):
            with ck_api._qkv_flags_enabled(model):
                raise RuntimeError("boom")
        assert not any(getattr(model.cfg, f) for f in ck_api._QKV_FLAGS)

    def test_removes_flags_the_config_did_not_have(self):
        model = _FakeModel([], _cfg())
        with ck_api._qkv_flags_enabled(model):
            assert model.cfg.use_attn_result is True
        assert not any(hasattr(model.cfg, f) for f in ck_api._QKV_FLAGS)
