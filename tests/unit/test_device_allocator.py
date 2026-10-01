"""CUDA allocator policy is opt-in at model-load time, never package import."""

import importlib

from circuitkit.utils import device


def test_importing_device_module_does_not_configure_allocator(monkeypatch):
    calls = []
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        device.torch.cuda.memory,
        "_set_allocator_settings",
        lambda value: calls.append(value),
        raising=False,
    )
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("CIRCUITKIT_NO_EXPANDABLE_SEGMENTS", raising=False)

    importlib.reload(device)

    assert calls == []
