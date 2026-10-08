"""enable_expandable_segments() must not run as an import-time side effect,
and must be skipped on Windows and under the opt-out env var. CPU-only,
offline -- no GPU needed, since every CPU-relevant branch returns False
before touching any CUDA API.
"""

import subprocess
import sys

import pytest

from circuitkit.utils import device as ck_device


@pytest.fixture(autouse=True)
def _reset_applied_flag(monkeypatch):
    monkeypatch.setattr(ck_device, "_expandable_segments_applied", None)


class TestNotAppliedAtImportTime:
    def test_importing_device_module_does_not_apply_it(self):
        """A fresh subprocess importing circuitkit.utils.device alone must not
        have applied the setting -- it is not a module-level call."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from circuitkit.utils import device; "
                "print(device._expandable_segments_applied)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "None"


class TestOptOutsAndPlatformGuard:
    def test_opt_out_env_var_skips_it(self, monkeypatch):
        monkeypatch.setenv("CIRCUITKIT_NO_EXPANDABLE_SEGMENTS", "1")
        assert ck_device.enable_expandable_segments() is False
        assert ck_device._expandable_segments_applied is False

    def test_skipped_on_windows(self, monkeypatch):
        monkeypatch.setattr(ck_device.sys, "platform", "win32")
        monkeypatch.setattr(ck_device.torch.cuda, "is_available", lambda: True)
        assert ck_device.enable_expandable_segments() is False

    def test_skipped_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr(ck_device.torch.cuda, "is_available", lambda: False)
        assert ck_device.enable_expandable_segments() is False

    def test_skipped_when_user_already_configured_it(self, monkeypatch):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
        monkeypatch.setattr(ck_device.torch.cuda, "is_available", lambda: True)
        assert ck_device.enable_expandable_segments() is False

    def test_idempotent_second_call_returns_cached_result(self, monkeypatch):
        monkeypatch.setattr(ck_device.torch.cuda, "is_available", lambda: False)
        first = ck_device.enable_expandable_segments()
        monkeypatch.setattr(ck_device.torch.cuda, "is_available", lambda: True)
        second = ck_device.enable_expandable_segments()
        assert first == second == False  # noqa: E712 -- explicit bool, not truthiness


class TestCalledFromTheCentralLoader:
    def test_from_pretrained_calls_it(self, monkeypatch):
        """quick._from_pretrained is the shared loader; it must trigger the
        allocator setting before the first CUDA allocation a load would
        cause, rather than relying on import-time side effects."""
        from circuitkit import quick as ck_quick

        calls = []
        # _from_pretrained does `from .utils.device import
        # enable_expandable_segments` locally, resolved at call time, so
        # patching the attribute on the device module is what it picks up.
        monkeypatch.setattr(ck_device, "enable_expandable_segments", lambda: calls.append(1))
        with pytest.raises(Exception):
            # No real checkpoint named this; it's fine if HookedTransformer
            # itself fails further in -- we only care that our call happened
            # before that failure.
            ck_quick._from_pretrained("circuitkit-test-nonexistent-checkpoint-xyz")
        assert calls == [1]
